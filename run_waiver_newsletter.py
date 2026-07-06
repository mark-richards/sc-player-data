"""
run_waiver_newsletter.py — Main orchestrator for the Waiver Wire Newsletter.

Usage
-----
  # Full run (Markdown + GSheets):
  py run_waiver_newsletter.py

  # Dry run — Markdown only, printed to stdout (no GSheets write):
  py run_waiver_newsletter.py --dry-run

  # Override auto-detected round number:
  py run_waiver_newsletter.py --round 5

  # Dry run for a specific round:
  py run_waiver_newsletter.py --dry-run --round 3
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("waiver")


def _detect_round(fanfooty_processed_dir) -> int | None:
    """
    Infer the current AFL round from the most recent fanfooty processed file.
    Returns None if no current-season (SEASON_YEAR) data exists yet (pre-season).
    """
    import glob, re
    from waiver.config import SEASON_YEAR, FANFOOTY_PROCESSED_DIR

    pattern = str(FANFOOTY_PROCESSED_DIR / f"{SEASON_YEAR}_round_*_fanfooty_data.csv")
    files = glob.glob(pattern)
    if not files:
        return None  # No current-season data = pre-season

    year_round_re = re.compile(r"(\d{4})_round_(\d+)_fanfooty_data\.csv$")
    rounds = []
    for f in files:
        m = year_round_re.search(f)
        if m and int(m.group(1)) == SEASON_YEAR:
            rounds.append(int(m.group(2)))
    return max(rounds) if rounds else None


def _ensure_fanfooty_rounds(year: int, up_to_round: int, probe_round: int | None = None) -> list[str]:
    """
    Ensures processed fanfooty CSVs exist for all rounds 1..up_to_round.

    For each missing round:
      - If raw files already exist → process and save immediately.
      - Otherwise → scrape from FanFooty first, then process.

    probe_round: if set, missing match IDs for this round are silently skipped
                 (used when probing one round ahead that may not have been played yet).

    Returns a list of warning strings for any rounds that failed.
    """
    from build_features import process_round_data
    from ingest_fanfooty import get_match_ids_for_round, download_match_files
    from waiver.config import FANFOOTY_PROCESSED_DIR

    warnings: list[str] = []
    raw_base = Path("data/raw/fanfooty")

    for r in range(1, up_to_round + 1):
        processed_path = FANFOOTY_PROCESSED_DIR / f"{year}_round_{r}_fanfooty_data.csv"
        if processed_path.exists():
            continue

        raw_dir = raw_base / str(year) / f"round_{r}"
        if not raw_dir.exists() or not list(raw_dir.glob("*.txt")):
            log.info("Scraping FanFooty raw data for %d R%d...", year, r)
            match_ids = get_match_ids_for_round(year, r)
            if not match_ids:
                if r == probe_round:
                    log.info("No match IDs for %d R%d — round not yet played, skipping.", year, r)
                else:
                    msg = f"FanFooty scrape failed: no match IDs found for {year} R{r}."
                    log.warning(msg)
                    warnings.append(msg)
                continue
            download_match_files(match_ids, year, r)

        log.info("Processing FanFooty data for %d R%d...", year, r)
        round_df = process_round_data(year, r)
        if round_df is None or round_df.empty:
            msg = f"FanFooty processing failed for {year} R{r} — fanfooty metrics may be incomplete."
            log.warning(msg)
            warnings.append(msg)
            continue

        FANFOOTY_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        round_df.to_csv(processed_path, index=False)
        log.info("Saved %s", processed_path)

    return warnings


def _sync_master_fanfooty(year: int, up_to_round: int) -> None:
    """
    Appends any per-round fanfooty CSVs for year/rounds 1..up_to_round that are
    not yet represented in master_fanfooty_data.csv (deduped on Fanfooty Match ID).
    """
    from waiver.config import FANFOOTY_PROCESSED_DIR
    master_path = FANFOOTY_PROCESSED_DIR / "master_fanfooty_data.csv"

    master_df = pd.read_csv(master_path, low_memory=False) if master_path.exists() else pd.DataFrame()
    existing_ids = set(master_df["Fanfooty Match ID"].dropna().unique()) if not master_df.empty else set()

    new_chunks = []
    for r in range(1, up_to_round + 1):
        p = FANFOOTY_PROCESSED_DIR / f"{year}_round_{r}_fanfooty_data.csv"
        if not p.exists():
            continue
        rdf = pd.read_csv(p, low_memory=False)
        if "Fanfooty Match ID" not in rdf.columns:
            continue
        new_rows = rdf[~rdf["Fanfooty Match ID"].isin(existing_ids)]
        if not new_rows.empty:
            new_chunks.append(new_rows)
            existing_ids.update(new_rows["Fanfooty Match ID"].dropna().unique())

    if new_chunks:
        updated = pd.concat([master_df] + new_chunks, ignore_index=True) if not master_df.empty else pd.concat(new_chunks, ignore_index=True)
        updated.to_csv(master_path, index=False)
        added = sum(len(c) for c in new_chunks)
        log.info("master_fanfooty_data.csv: appended %d new rows for %d R1–R%d", added, year, up_to_round)
    else:
        log.info("master_fanfooty_data.csv already up to date for %d R1–R%d", year, up_to_round)


def validate_round_data(year: int, round_num: int) -> list[str]:
    """
    Validates that fanfooty and SC data for round_num have the correct number
    of player rows based on the AFL fixture.

    FanFooty:
      - Fetches expected match count from the FanFooty archive page.
      - Checks that raw .txt files match that count.
      - Checks processed CSV rows >= n_games * MIN_ROWS_PER_GAME (23 players
        per team * 2 teams = 46; threshold is 44 to allow 1 DNP per side).
      - Auto-repairs by deleting partial data and re-scraping if incomplete.

    SC player_match_results:
      - Checks that the round has >= N_TEAMS * 19 rows (on-field starters).
      - Logs a warning if short; SC data cannot be auto-repaired here.

    Returns a list of warning strings (also logged at WARNING level).
    """
    import shutil
    import pandas as pd
    from build_features import process_round_data
    from ingest_fanfooty import get_match_ids_for_round, download_match_files
    from waiver.config import FANFOOTY_PROCESSED_DIR

    MIN_ROWS_PER_GAME  = 44   # 23 players × 2 teams = 46; allow 1 DNP per side
    N_TEAMS            = 8
    MIN_SC_ROWS        = N_TEAMS * 19   # at minimum on-field starters per round

    raw_base       = Path("data/raw/fanfooty")
    raw_dir        = raw_base / str(year) / f"round_{round_num}"
    processed_path = FANFOOTY_PROCESSED_DIR / f"{year}_round_{round_num}_fanfooty_data.csv"
    warnings: list[str] = []

    # ── AFL fixture: expected game count from FanFooty archive ───────────
    expected_ids = get_match_ids_for_round(year, round_num)
    n_expected   = len(expected_ids)

    if n_expected == 0:
        w = f"VALIDATION: Could not fetch AFL fixture for {year} R{round_num} — skipping coverage check."
        log.warning(w)
        warnings.append(w)
        return warnings

    # ── Check raw file count ──────────────────────────────────────────────
    n_raw = len(list(raw_dir.glob("*.txt"))) if raw_dir.exists() else 0
    needs_rescrape = False

    if n_raw < n_expected:
        w = (f"VALIDATION: FanFooty R{round_num} has {n_raw}/{n_expected} raw files. "
             f"Incomplete scrape — will re-download.")
        log.warning(w)
        warnings.append(w)
        needs_rescrape = True

    # ── Check processed row count ─────────────────────────────────────────
    if not needs_rescrape and processed_path.exists():
        n_rows        = len(pd.read_csv(processed_path, usecols=[0]))
        expected_rows = n_expected * MIN_ROWS_PER_GAME
        if n_rows < expected_rows:
            w = (f"VALIDATION: FanFooty R{round_num} processed has {n_rows} rows — "
                 f"expected ≥{expected_rows} ({n_expected} games × {MIN_ROWS_PER_GAME}). "
                 f"Will re-scrape.")
            log.warning(w)
            warnings.append(w)
            needs_rescrape = True

    # ── Auto-repair fanfooty if incomplete ────────────────────────────────
    if needs_rescrape:
        if processed_path.exists():
            processed_path.unlink()
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        log.info("Re-scraping FanFooty for %d R%d (%d games)...", year, round_num, n_expected)
        download_match_files(expected_ids, year, round_num)
        round_df = process_round_data(year, round_num)
        if round_df is not None and not round_df.empty:
            FANFOOTY_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            round_df.to_csv(processed_path, index=False)
            log.info("Repaired: saved %s (%d rows)", processed_path, len(round_df))
        else:
            w = f"VALIDATION: Re-scrape of {year} R{round_num} still produced no data."
            log.warning(w)
            warnings.append(w)

    # ── SC player_match_results coverage ─────────────────────────────────
    pm_path = Path(__file__).resolve().parent / "data" / "live" / "player_match_results.csv"
    if pm_path.exists():
        pm        = pd.read_csv(pm_path, usecols=["round_x"])
        n_sc_rows = int((pd.to_numeric(pm["round_x"], errors="coerce") == round_num).sum())
        if n_sc_rows < MIN_SC_ROWS:
            w = (f"VALIDATION: SC player_match_results R{round_num} has {n_sc_rows} rows — "
                 f"expected ≥{MIN_SC_ROWS} ({N_TEAMS} teams × 19 on-field). "
                 f"Re-run refresh_2026_data() if the round is complete.")
            log.warning(w)
            warnings.append(w)

    return warnings


def _assert_refresh_succeeded(label: str = "refresh") -> list[str]:
    """
    Checks that the key live data files were updated within the last 10 minutes.
    Returns error strings for any file that wasn't updated, so callers can surface
    a loud warning before the stale data flows into newsletter content.
    """
    import time
    project_root = Path(__file__).resolve().parent
    max_age = 10 * 60  # 10 minutes
    now = time.time()
    files = [
        project_root / "data" / "live" / "player_stats_current.csv",
        project_root / "data" / "live" / "player_match_results.csv",
        project_root / "data" / "live" / "ladder.csv",
    ]
    errors = []
    for path in files:
        if not path.exists():
            errors.append(f"DATA ERROR ({label}): {path.name} missing after refresh.")
        elif now - path.stat().st_mtime > max_age:
            age_min = (now - path.stat().st_mtime) / 60
            errors.append(
                f"DATA ERROR ({label}): {path.name} is {age_min:.0f} min old after refresh "
                f"— refresh likely failed silently."
            )
    return errors


def _check_data_freshness(round_num: int) -> list[str]:
    """
    Checks modification times of key live data files and fanfooty round coverage.
    Returns warning strings for files older than 4 days or missing rounds.
    """
    import time
    from waiver.config import FANFOOTY_PROCESSED_DIR, SEASON_YEAR

    warnings: list[str] = []
    now = time.time()
    max_age_seconds = 4 * 24 * 3600  # 4 days

    project_root = Path(__file__).resolve().parent
    files_to_check = [
        (project_root / "data" / "live" / "player_stats_current.csv",    "player_stats_current.csv"),
        (project_root / "data" / "live" / "ladder.csv",                  "ladder.csv"),
        (project_root / "data" / "processed" / "master_player_data.csv", "master_player_data.csv"),
    ]
    for path, label in files_to_check:
        if not path.exists():
            warnings.append(f"DATA WARNING: {label} is missing.")
        elif now - path.stat().st_mtime > max_age_seconds:
            age_days = (now - path.stat().st_mtime) / 86400
            warnings.append(
                f"DATA WARNING: {label} is {age_days:.1f} days old (threshold: 4 days)."
            )

    for r in range(1, round_num + 1):
        p = FANFOOTY_PROCESSED_DIR / f"{SEASON_YEAR}_round_{r}_fanfooty_data.csv"
        if not p.exists():
            warnings.append(
                f"DATA WARNING: Missing fanfooty processed data for {SEASON_YEAR} R{r}."
            )

    return warnings


def _refresh_coach_list(round_num: int = 1) -> None:
    """
    Fetches current league teams from the SC API ladder and writes data/live/coach_list.csv.
    Silently skips on failure so the newsletter can still run with the cached file.
    """
    from waiver.config import LADDER_LEAGUE_ID, SEASON_YEAR
    from waiver import league_api

    teams = league_api.fetch_league_teams(
        year=SEASON_YEAR,
        ladder_league_id=LADDER_LEAGUE_ID,
        token=league_api.get_valid_token(year=SEASON_YEAR),
        round_num=round_num,
    )
    if not teams:
        log.warning("Could not refresh coach_list.csv from API — cached file will be used.")
        return

    coach_list_path = Path(__file__).resolve().parent / "data" / "live" / "coach_list.csv"
    df = pd.DataFrame(teams, columns=["coach_team_id", "coach_id", "coach_first_name", "coach_team_name"])
    df.to_csv(coach_list_path, index=False)
    log.info("coach_list.csv refreshed with %d teams → %s", len(df), coach_list_path)


def run_pipeline(round_override: int | None = None, dry_run: bool = False) -> Path | None:
    """
    Executes the full waiver newsletter pipeline.

    Parameters
    ----------
    round_override : int | None
        If set, use this round number instead of auto-detecting.
    dry_run : bool
        If True, print Markdown to stdout and skip GSheets upload.

    Returns
    -------
    Path to the generated Markdown file, or None on critical failure.

    ORDERING CONSTRAINT — do not load live CSVs before the SC refresh runs.
    The required sequence for steps that touch live data is:
      1. Fanfooty scrape (4b) — before load_historical_scores / validate fanfooty
      2. SC refresh / refresh_2026_data (4e) — before load_season_scores / validate SC
      3. All data loads (scores_df, hist_df) — after their respective refresh steps
    Breaking this order causes the newsletter to use stale data with no visible error.
    _assert_refresh_succeeded() enforces step 2 at runtime.
    """
    from waiver.config import LADDER_LEAGUE_ID, SC_API_TOKEN, SEASON_YEAR
    from waiver import league_api, data_loader, metrics, newsletter
    from waiver import fixture_schedule, emailer

    log.info("=" * 60)
    log.info("Waiver Newsletter Pipeline — Season %d", SEASON_YEAR)
    log.info("=" * 60)

    pipeline_warnings: list[str] = []

    # ── Step 0a: Refresh AFL news (runs scraper if DB is >20h old) ────────
    try:
        from afl_news_scraper import run_if_stale
        run_if_stale(max_age_hours=20.0)
    except Exception as _news_exc:
        log.warning("News scraper step failed (non-fatal): %s", _news_exc)

    # ── Step 0: Token health check + auto-refresh ────────────────────────────
    for w in league_api.check_token_health():
        pipeline_warnings.append(f"⚠️ TOKEN: {w}")
    token = league_api.get_valid_token(year=SEASON_YEAR)
    if not token:
        log.warning("No SC bearer token available — ownership/injury data will be unavailable.")

    # ── Step 1: Fetch player ownership from SC in-season league API ───────
    log.info("[1/9] Fetching player ownership from SC in-season league API...")
    status_dict = league_api.fetch_player_status(
        year=SEASON_YEAR,
        league_id=LADDER_LEAGUE_ID,
        token=token,
    )
    if not status_dict:
        msg = "SC API unavailable — free agent identification may be inaccurate. Check SC_API_TOKEN."
        log.warning(msg)
        pipeline_warnings.append(msg)

    # ── Step 2: Fetch live injury / suspension data ───────────────────────
    log.info("[2/9] Fetching live injury and suspension data...")
    injury_dict = league_api.fetch_player_injuries(
        year=SEASON_YEAR,
        token=token,
    )
    _injury_cache_path = Path(__file__).resolve().parent / "data" / "live" / "injury_cache.json"
    if not injury_dict:
        msg = "Injury data unavailable — Injury Report section may be incomplete."
        log.warning(msg)
        pipeline_warnings.append(msg)
    else:
        import json
        active_count = sum(1 for v in injury_dict.values() if v.get("status"))
        if active_count == 0 and _injury_cache_path.exists():
            try:
                cached = json.loads(_injury_cache_path.read_text(encoding="utf-8"))
                cached_active = sum(1 for v in cached.values() if v.get("status"))
                if cached_active > 0:
                    msg = (
                        f"SC returned 0 injuries — likely a data lag. "
                        f"Using cached injury data ({cached_active} active) from last run."
                    )
                    log.warning(msg)
                    pipeline_warnings.append(msg)
                    injury_dict = cached
            except Exception as exc:
                log.warning("Could not load injury cache: %s", exc)
        if active_count > 0:
            try:
                _injury_cache_path.write_text(
                    json.dumps(injury_dict, ensure_ascii=False), encoding="utf-8"
                )
                log.info("Injury cache updated: %d active injuries.", active_count)
            except Exception as exc:
                log.warning("Could not write injury cache: %s", exc)

    # ── Step 3: Load player list ──────────────────────────────────────────
    log.info("[3/9] Loading SC player list...")
    player_list_df = data_loader.load_player_list(year=SEASON_YEAR)
    if player_list_df.empty:
        log.error("Player list is empty. Aborting.")
        return None

    # Auto-detect round from fanfooty processed files (not from player_stats_current.csv
    # which spans multiple seasons and would give a misleading round number)
    from waiver.config import FANFOOTY_PROCESSED_DIR
    round_num = round_override if round_override is not None else _detect_round(FANFOOTY_PROCESSED_DIR)
    log.info("Using round: %s", round_num if round_num else "Pre-season")

    # ── Step 4b: Scrape/process fanfooty data for all played rounds ──────────
    # Probe one round ahead of locally available data so a newly completed round
    # gets scraped automatically (avoids chicken-and-egg with _detect_round).
    scrape_up_to = (round_num or 0) + 1
    log.info("[4b] Scraping/processing fanfooty data for %d R1–R%d (probing +1)...", SEASON_YEAR, scrape_up_to)
    fanfooty_warnings = _ensure_fanfooty_rounds(SEASON_YEAR, scrape_up_to, probe_round=scrape_up_to)
    pipeline_warnings.extend(fanfooty_warnings)
    # Re-detect round now that fresh data may have been scraped
    if round_override is None:
        round_num = _detect_round(FANFOOTY_PROCESSED_DIR) or round_num
    log.info("Round confirmed after scrape: %s", round_num if round_num else "Pre-season")

    # Sync any newly processed rounds into master_fanfooty_data.csv
    if round_num:
        _sync_master_fanfooty(SEASON_YEAR, round_num)

    # ── Step 4d-pre: Download SC player-round files for all played rounds ────
    # One request per round (bulk all-player endpoint) — fast, covers all players.
    # Must run before load_season_scores() so scores_2026 is current.
    log.info("[4d-pre] Downloading SC player-round data for %d R1–R%d...", SEASON_YEAR, round_num or 0)
    if round_num:
        try:
            from download_2026_statspacks import download_player_rounds as _dl_rounds
            _dl_rounds(
                year=SEASON_YEAR,
                token=token,
                rounds=list(range(1, round_num + 1)),
                force_refresh=True,
            )
        except Exception as exc:
            msg = f"SC player-round download failed: {exc}"
            log.warning(msg)
            pipeline_warnings.append(msg)
    else:
        log.info("[4d-pre] Pre-season — no rounds to download.")

    # ── Step 4e (early): Refresh SC data before validation so coverage check ─
    # sees the current round's player_match_results, not stale data.
    from waiver.fixture_schedule import round_is_complete
    round_complete = round_is_complete(round_num or 1)
    if round_complete:
        log.info("[4e] Refreshing website outputs (ladder, fixtures, player match results)...")
        try:
            from website.data.api_ingest import refresh_2026_data
            refresh_2026_data()
        except Exception as exc:
            msg = f"Website data refresh failed (non-fatal): {exc}"
            log.warning(msg)
            pipeline_warnings.append(msg)

        for err in _assert_refresh_succeeded("step 4e"):
            log.error(err)
            pipeline_warnings.append(err)

        # Clear the Flask website cache so the running app picks up the new round data.
        try:
            import requests as _requests
            _requests.post("http://localhost:5000/api/cache-clear", timeout=3)
            log.info("[4e] Website cache cleared.")
        except Exception:
            pass  # Flask app not running — no action needed

        # ── Step 4f: Ingest current-season SC data into master_player_data.csv ─
        log.info("[4f] Updating master_player_data.csv with %d current-season data...", SEASON_YEAR)
        try:
            from build_master_dataset import update_current_season as _update_master
            n = _update_master(year=SEASON_YEAR)
            log.info("[4f] master_player_data.csv: %d rows added for %d.", n, SEASON_YEAR)
        except Exception as exc:
            msg = f"SC master data update failed (non-fatal): {exc}"
            log.warning(msg)
            pipeline_warnings.append(msg)
    else:
        msg = f"Round {round_num} still in progress — skipping data refresh."
        log.info("[4e/4f] %s", msg)
        pipeline_warnings.append(msg)

    # ── Step 4g: Auto-generate predictions for upcoming round ─────────────────
    upcoming_round = (round_num or 0) + 1
    pred_path = (
        Path(__file__).resolve().parent
        / "data" / "predictions"
        / f"{SEASON_YEAR}_round_{upcoming_round}_predictions.csv"
    )
    if not pred_path.exists():
        log.info("[4g] Generating Round %d predictions...", upcoming_round)
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "run_predictions.py",
                 "--year", str(SEASON_YEAR), "--round", str(upcoming_round)],
                capture_output=True, text=True, timeout=120,
                cwd=Path(__file__).resolve().parent,
            )
            if result.returncode != 0:
                msg = f"Predictions generation failed (non-fatal): {result.stderr[-500:]}"
                log.warning(msg)
                pipeline_warnings.append(msg)
            else:
                log.info("[4g] Round %d predictions generated.", upcoming_round)
        except Exception as exc:
            msg = f"Predictions generation error (non-fatal): {exc}"
            log.warning(msg)
            pipeline_warnings.append(msg)
    else:
        log.info("[4g] Round %d predictions already exist — skipping generation.", upcoming_round)

    # ── Step 4: Load 2026 season scores (after refresh so R7 data is current) ─
    log.info("[4/9] Loading current-season scores...")
    scores_df = data_loader.load_season_scores()

    # ── Step 4b-post: Validate round coverage ────────────────────────────────
    if round_num:
        log.info("[4b-validate] Validating round %d data coverage...", round_num)
        validation_warnings = validate_round_data(SEASON_YEAR, round_num)
        pipeline_warnings.extend(validation_warnings)
        # Re-detect in case auto-repair changed available data
        if round_override is None:
            round_num = _detect_round(FANFOOTY_PROCESSED_DIR) or round_num

    # ── Step 4b-post: Data freshness check ───────────────────────────────────
    if round_num:
        freshness_warnings = _check_data_freshness(round_num)
        for w in freshness_warnings:
            log.warning(w)
        pipeline_warnings.extend(freshness_warnings)

    # ── Step 4c: Fetch my lineup (on-field vs bench) from statsPlayers API ─
    from waiver.config import MY_TEAM_ID
    lineup_round = (round_num or 0) + 1  # upcoming round = currently configured lineup
    log.info("[4c] Fetching lineup for round %d...", lineup_round)
    lineup_dict = league_api.fetch_my_lineup(
        year=SEASON_YEAR,
        user_team_id=MY_TEAM_ID,
        round_num=lineup_round,
        token=token,
    )
    if not lineup_dict:
        msg = "Could not fetch lineup — on-field/bench split will be inaccurate."
        log.warning(msg)
        pipeline_warnings.append(msg)

    # ── Step 4d: Refresh coach list from SC API ladder ────────────────────
    log.info("[4d] Refreshing coach list from SC API ladder...")
    _refresh_coach_list(round_num or 1)

    # ── Step 5: Build unified player table ───────────────────────────────
    log.info("[5/9] Building player table...")
    player_table = data_loader.build_player_table(
        player_list_df, scores_df, status_dict,
        injury_dict=injury_dict,
        lineup_dict=lineup_dict,
    )
    if player_table.empty:
        log.error("Player table is empty. Aborting.")
        return None

    # ── Step 6: Load historical data + compute metrics ───────────────────
    log.info("[6/9] Loading historical fanfooty data and computing metrics...")
    hist_df = data_loader.load_historical_scores()
    metrics_df = metrics.score_all_players(player_table, hist_df)

    # ── Step 7: Merge model predictions for upcoming round ────────────────
    log.info("[7/9] Loading model predictions...")
    predictions_df = data_loader.load_predictions(year=SEASON_YEAR, round_num=round_num)
    if not predictions_df.empty:
        pred_round = int(predictions_df["pred_round"].iloc[0])
        metrics_df = metrics_df.merge(predictions_df.drop(columns=["pred_round"]),
                                      on="feed_id", how="left")
        n_with_proj = metrics_df["projected_score"].notna().sum()
        log.info(
            "Predictions for Round %d merged: %d players have projected scores.",
            pred_round, n_with_proj,
        )
    else:
        pred_round = None
        msg = (
            f"No model predictions available — projections omitted. "
            f"Run: py run_predictions.py --year {SEASON_YEAR} --round {(round_num or 0) + 1}"
        )
        log.info(msg)
        pipeline_warnings.append(msg)

    # ── Step 7b: Correct prediction outliers against historical data ───────
    if "probability_gt_100" in metrics_df.columns:
        log.info("[7b] Applying historical sanity checks to model predictions...")

        def _cap_p100(row) -> float:
            """
            Caps model P(100+) at 2.5× the player's historical 100+ rate.
            Players who have never scored 100 are capped at 10%.
            Only applied when we have sufficient historical data.
            """
            p = row.get("probability_gt_100")
            if p is None or (hasattr(p, "__float__") and p != p):  # None or NaN
                return p
            if not row.get("sufficient_data", False):
                return p
            scores = row.get("effective_scores") or []
            if not scores:
                return p
            hist_rate = sum(1 for s in scores if s >= 100) / len(scores) * 100
            if hist_rate == 0:
                return min(float(p), 10.0)   # never scored 100 → cap at 10%
            return min(float(p), hist_rate * 2.5)  # allow 2.5× historical rate

        mask = metrics_df["projected_score"].notna()
        metrics_df.loc[mask, "probability_gt_100"] = metrics_df[mask].apply(
            _cap_p100, axis=1
        )
        n_capped = (
            metrics_df.loc[mask, "probability_gt_100"]
            < predictions_df.set_index("feed_id")
              .reindex(metrics_df.loc[mask, "feed_id"].values)["probability_gt_100"]
              .values
        ).sum() if not predictions_df.empty else 0
        log.info("P(100+) corrected for %d players based on historical ceiling.", n_capped)

    if "probability_gt_80" in metrics_df.columns:
        def _cap_p80(row) -> float:
            p = row.get("probability_gt_80")
            if p is None or (hasattr(p, "__float__") and p != p):
                return p
            if not row.get("sufficient_data", False):
                return p
            avg_2026 = float(row.get("avg_fanfooty_2026") or 0)
            if avg_2026 > 0:
                implied_cap = min(99.0, max(10.0, (avg_2026 / 80) * 95))
                p = min(float(p), implied_cap)
            scores = row.get("effective_scores") or []
            if scores:
                hist_rate = sum(1 for s in scores if s >= 80) / len(scores) * 100
                if hist_rate > 0:
                    p = min(float(p), hist_rate * 2.5)
            return p

        mask80 = metrics_df["projected_score"].notna() if "projected_score" in metrics_df.columns else pd.Series(False, index=metrics_df.index)
        metrics_df.loc[mask80, "probability_gt_80"] = metrics_df[mask80].apply(_cap_p80, axis=1)
        log.info("P(80+) sanity cap applied to players with 2026 actual data.")

    # Quick sanity check
    fa_count = metrics_df["is_free_agent"].sum()
    injured_on_field = metrics_df[metrics_df["is_on_field"] & (metrics_df["is_injured"] | metrics_df["is_suspended"])]
    log.info(
        "Metrics ready: %d players total, %d free agents, %d on my roster, %d injured on-field.",
        len(metrics_df),
        fa_count,
        metrics_df["is_mine"].sum(),
        len(injured_on_field),
    )

    # ── Step 7b-pre: Build DVP table (shared by jumper scorer + section 8c) ──
    dvp_table = None
    fixture_next = {}
    dvp_ranks_by_feed_id: dict = {}
    if round_num:
        try:
            from waiver.dvp import build_dvp_table, compute_dvp_ranks
            from run_predictions import get_fixture_for_round
            _t = build_dvp_table(SEASON_YEAR, lookback_rounds=6)
            if not _t.empty:
                dvp_table = _t
                fixture_next = get_fixture_for_round(SEASON_YEAR, round_num + 1) or {}
                dvp_ranks_by_feed_id = compute_dvp_ranks(dvp_table, metrics_df, fixture_next)
                log.info("[7b-pre] DVP table ready (%d entries), %d player ranks computed.", len(dvp_table), len(dvp_ranks_by_feed_id))
        except Exception as exc:
            log.warning("DVP pre-build failed (non-fatal): %s", exc)

    # ── Step 7c: Score mid-year jumpers and weekly streamers ─────────────
    from waiver.config import (
        JUMPER_AVG_MAX, JUMPER_MIN_CAREER_AVG, JUMPER_MIN_AVG3,
        JUMPER_MIN_GAMES, JUMPER_MIN_ROUNDS,
        JUMPER_SCORE_THRESHOLD, JUMPER_TOP_N,
        STREAMER_MIN_AVG3, STREAMER_SIGNAL_THRESHOLD, STREAMER_TOP_N,
    )
    jumpers_df, streamers_df = None, None
    if round_num and round_num >= JUMPER_MIN_ROUNDS:
        log.info("[7c] Scoring mid-year jumper and streamer candidates...")
        try:
            from waiver.jumper_scorer import load_2026_fanfooty, load_2026_cbas, score_free_agents
            ff_2026 = load_2026_fanfooty(FANFOOTY_PROCESSED_DIR, season_year=SEASON_YEAR)
            cbas_2026 = load_2026_cbas()
            jumpers_raw, streamers_raw = score_free_agents(
                metrics_df=metrics_df,
                hist_df=hist_df,
                fanfooty_2026_df=ff_2026,
                cbas_2026_df=cbas_2026,
                min_games=JUMPER_MIN_GAMES,
                season_avg_max=JUMPER_AVG_MAX,
                current_round=round_num,
                jumper_threshold=JUMPER_SCORE_THRESHOLD,
                streamer_threshold=STREAMER_SIGNAL_THRESHOLD,
                min_career_avg=JUMPER_MIN_CAREER_AVG,
                min_avg3_jumper=JUMPER_MIN_AVG3,
                min_avg3_streamer=STREAMER_MIN_AVG3,
                dvp_ranks=dvp_ranks_by_feed_id or None,
            )
            jumpers_df = jumpers_raw.head(JUMPER_TOP_N) if not jumpers_raw.empty else None
            streamers_df = streamers_raw.head(STREAMER_TOP_N) if not streamers_raw.empty else None
            log.info(
                "Jumper/streamer scoring done: %d mover candidates, %d streamer candidates.",
                len(jumpers_df) if jumpers_df is not None else 0,
                len(streamers_df) if streamers_df is not None else 0,
            )
        except Exception as exc:
            msg = f"Jumper/streamer scoring failed (non-fatal): {exc}"
            log.warning(msg)
            pipeline_warnings.append(msg)
    else:
        log.info("[7c] Skipping jumper/streamer scoring (round %s < %d).", round_num, JUMPER_MIN_ROUNDS)

    # ── Step 8b: Refresh fixture schedule ─────────────────────────────────
    log.info("[8b] Refreshing AFL fixture schedule...")
    schedule_updated = fixture_schedule.update_schedule()
    if schedule_updated:
        log.info("waiver_schedule.bat updated — newsletter will prompt re-registration.")

    # ── Step 8c: DVP (Defence vs Position) — matchup intel + schedule streamers ──
    dvp_targets_df, schedule_streamers_df = None, None
    if round_num and dvp_table is not None:
        log.info("[8c] Computing DVP matchup intel (reusing pre-built table)...")
        try:
            from waiver.dvp import get_dvp_targets, get_schedule_streamers
            fa_df = metrics_df[metrics_df["is_free_agent"]].copy() if "is_free_agent" in metrics_df.columns else pd.DataFrame()

            dvp_targets_df = get_dvp_targets(dvp_table, fa_df, fixture_next)

            fixture_maps = []
            for r in range(round_num + 1, round_num + 4):
                fm = get_fixture_for_round(SEASON_YEAR, r)
                if fm:
                    fixture_maps.append(fm)
            schedule_streamers_df = get_schedule_streamers(dvp_table, fa_df, fixture_maps)

            log.info(
                "DVP done: %d matchup targets, %d schedule streamers.",
                len(dvp_targets_df) if dvp_targets_df is not None else 0,
                len(schedule_streamers_df) if schedule_streamers_df is not None else 0,
            )
        except Exception as exc:
            log.warning("DVP computation failed (non-fatal): %s", exc)

    # ── Step 8d: Positional scarcity meter ────────────────────────────────
    scarcity_dict = None
    try:
        from waiver.metrics import calc_positional_scarcity
        scarcity_dict = calc_positional_scarcity(metrics_df)
    except Exception as exc:
        log.warning("Positional scarcity calc failed (non-fatal): %s", exc)

    # ── Step 8b: Fetch AFL injury list ────────────────────────────────────
    afl_injury_df = None
    try:
        from ingest_afl_injuries import run as fetch_afl_injuries
        log.info("[8b] Fetching AFL official injury list...")
        afl_injury_df = fetch_afl_injuries()
        log.info("AFL injury list loaded: %d players", len(afl_injury_df))
    except Exception as exc:
        log.warning("AFL injury list fetch failed (non-fatal): %s", exc)

    # ── Step 9: Generate newsletter ───────────────────────────────────────
    log.info("[9/9] Generating newsletter...")
    md_path = newsletter.generate(
        metrics_df=metrics_df,
        round_num=round_num,
        pred_round=pred_round,
        warnings=pipeline_warnings or None,
        schedule_updated=schedule_updated,
        jumpers_df=jumpers_df,
        streamers_df=streamers_df,
        dvp_targets_df=dvp_targets_df,
        schedule_streamers_df=schedule_streamers_df,
        scarcity_dict=scarcity_dict,
        afl_injury_df=afl_injury_df,
        dvp_ranks=dvp_ranks_by_feed_id or None,
    )

    round_label = f"Round {round_num}, {SEASON_YEAR}" if round_num else f"Pre-Season {SEASON_YEAR}"

    if dry_run:
        log.info("--- DRY RUN: printing Markdown to stdout ---")
        content = md_path.read_text(encoding="utf-8")
        # Use sys.stdout.buffer to avoid Windows cp1252 encoding issues with emojis
        sys.stdout.buffer.write(("\n" + content).encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
        log.info("Email skipped (dry run).")
    else:
        log.info("Emailing newsletter to %s...", emailer.RECIPIENT)
        emailer.send_newsletter(md_path, round_label)

    # ── Step 10: Rebuild static site and push to GitHub Pages ────────────────
    if not dry_run:
        log.info("[10/10] Deploying static site to GitHub Pages...")
        try:
            from deploy_github_pages import deploy as _deploy_pages
            ok = _deploy_pages(round_num=round_num)
            if not ok:
                log.warning("GitHub Pages deploy failed — site may be stale. Run: py deploy_github_pages.py")
        except Exception as exc:
            log.warning("GitHub Pages deploy error (non-fatal): %s", exc)

    log.info("Done. Newsletter saved to: %s", md_path)
    return md_path


# ── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the weekly SuperCoach Draft Waiver Wire Newsletter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Markdown to stdout and skip email.",
    )
    parser.add_argument(
        "--round",
        type=int,
        metavar="N",
        dest="round_num",
        default=None,
        help="Override the auto-detected AFL round number.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_pipeline(round_override=args.round_num, dry_run=args.dry_run)
    sys.exit(0 if result else 1)
