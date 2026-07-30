"""
data_loader.py — Loads all data sources and builds the unified player table.

Sources
-------
1. draft_prep/SC {year}/{year}_SC_Player_list.csv
   Maps SC player id → fanfooty feed_id, name, position, team.

2. data/raw/supercoach/{year}/players_round_N.json
   Bulk all-player stats per round — downloaded weekly by the pipeline.
   Primary source for scores_2026: covers ALL players (owned and free agents).
   Supplemented by fanfooty 2026 CSVs for any missing rounds.

3. data/processed/YYYY_round_N_fanfooty_data.csv
   Historical match-level data: SC scores + Tag Notes for Role Watch.

4. status_dict from league_api.fetch_player_status()
   Tells us which players are free_agent / owner / waiver.
"""

import glob
import logging
import re
from pathlib import Path

import pandas as pd

from waiver.config import (
    CURRENT_SEASON_MIN_ROUNDS,
    DEFAULT_SEASON_WEIGHT,
    FANFOOTY_PROCESSED_DIR,
    MY_TEAM_ID,
    PLAYER_LIST_PATH,
    PREDICTIONS_DIR,
    ROLE_WATCH_ROUNDS,
    SEASON_WEIGHTS,
    SEASON_YEAR,
)

_SC_RAW_DIR = PLAYER_LIST_PATH.parent.parent.parent / "data" / "raw" / "supercoach"

log = logging.getLogger(__name__)


# ── 1. SC Player List ──────────────────────────────────────────────────────

def load_player_list(year: int = SEASON_YEAR) -> pd.DataFrame:
    """
    Loads the SC player list CSV for the given season.

    Returns a DataFrame with columns:
        sc_id, feed_id, name, position, team
    """
    path = PLAYER_LIST_PATH
    if not path.exists():
        log.error("Player list not found: %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Rename the first unnamed index column to 'row_idx' if present
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "row_idx"})

    required = {"id", "feed_id", "first_name", "last_name"}
    missing = required - set(df.columns)
    if missing:
        log.error("Player list CSV missing columns: %s", missing)
        return pd.DataFrame()

    df = df.rename(columns={"id": "sc_id"})
    df["name"] = df["first_name"].str.strip() + " " + df["last_name"].str.strip()
    df["sc_id"] = pd.to_numeric(df["sc_id"], errors="coerce")
    df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce")
    df = df.dropna(subset=["sc_id", "feed_id"])
    df["sc_id"] = df["sc_id"].astype(int)
    df["feed_id"] = df["feed_id"].astype(int)

    # Normalise position to DEF / MID / RUC / FWD
    pos_col = "position" if "position" in df.columns else None
    if pos_col:
        df["position"] = df[pos_col].str.strip().str.upper()
    else:
        df["position"] = "UNK"

    team_col = "long_team" if "long_team" in df.columns else "team"
    df["team"] = df[team_col].fillna("").str.strip()

    log.info("Loaded %d players from player list.", len(df))
    return df[["sc_id", "feed_id", "name", "position", "team"]]


# ── 2. Current-season SC scores ────────────────────────────────────────────

def _load_sc_round_files(year: int) -> tuple[dict[int, dict[int, int]], set[int]]:
    """
    Reads data/raw/supercoach/{year}/players_round_N.json files.

    Returns:
        scores_by_feed_id: {feed_id: {round_num: sc_score}}  — only rounds where games=1
        rounds_found: set of round numbers with SC data
    """
    import json as _json
    sc_dir = _SC_RAW_DIR / str(year)
    pattern = str(sc_dir / "players_round_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}, set()

    round_re = re.compile(r"players_round_(\d+)\.json$")
    scores: dict[int, dict[int, int]] = {}
    rounds_found: set[int] = set()

    for fpath in files:
        m = round_re.search(fpath)
        if not m:
            continue
        round_num = int(m.group(1))
        try:
            players = _json.loads(Path(fpath).read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read %s: %s", fpath, exc)
            continue

        rounds_found.add(round_num)
        for player in players:
            fid = player.get("feed_id")
            if not fid:
                continue
            try:
                fid = int(fid)
            except (ValueError, TypeError):
                continue

            stats_list = player.get("player_stats") or []
            for stat in stats_list:
                if int(stat.get("games", 0)) == 1:
                    pts = stat.get("points")
                    if pts is not None and int(pts) > 0:
                        scores.setdefault(fid, {})[round_num] = int(pts)

    return scores, rounds_found


def _load_fanfooty_round_scores(year: int) -> dict[int, dict[int, int]]:
    """
    Reads fanfooty 2026 processed CSVs as a supplement to SC round files.
    Returns {feed_id: {round_num: sc_score}}.
    """
    pattern = str(FANFOOTY_PROCESSED_DIR / f"{year}_round_*_fanfooty_data.csv")
    files = sorted(glob.glob(pattern))
    year_round_re = re.compile(r"(\d{4})_round_(\d+)_fanfooty_data\.csv$")
    scores: dict[int, dict[int, int]] = {}

    for fpath in files:
        m = year_round_re.search(fpath)
        if not m:
            continue
        round_num = int(m.group(2))
        try:
            chunk = pd.read_csv(fpath, usecols=["Player ID", "SC"], low_memory=False)
        except Exception as exc:
            log.warning("Could not read %s: %s", fpath, exc)
            continue
        chunk.columns = ["feed_id", "sc_score"]
        chunk["feed_id"] = pd.to_numeric(chunk["feed_id"], errors="coerce")
        chunk["sc_score"] = pd.to_numeric(chunk["sc_score"], errors="coerce")
        chunk = chunk.dropna()
        chunk = chunk[chunk["sc_score"] > 0]
        for _, row in chunk.iterrows():
            fid = int(row["feed_id"])
            scores.setdefault(fid, {})[round_num] = int(row["sc_score"])

    return scores


def load_season_scores(year: int = SEASON_YEAR) -> pd.DataFrame:
    """
    Returns one row per player with:
        feed_id, scores_2026 (list of ints, ordered by round), avg_2026, rounds_played_2026

    Primary source: data/raw/supercoach/{year}/players_round_N.json
      — bulk all-player endpoint, one file per round, covers owned and free agents.
      — only includes rounds where the player actually played (games=1).

    Supplement: fanfooty 2026 processed CSVs fill any rounds missing from SC files
      — covers edge cases where a round file wasn't downloaded yet.

    Raises FileNotFoundError if no SC round files exist (run download first).
    """
    sc_scores, sc_rounds = _load_sc_round_files(year)

    if not sc_rounds:
        raise FileNotFoundError(
            f"No SC player round files found in {_SC_RAW_DIR / str(year)}. "
            f"Run: py download_2026_statspacks.py --year {year} --rounds <current_round>"
        )

    # Supplement with fanfooty for any player/round absent from SC files
    ff_scores = _load_fanfooty_round_scores(year)
    supplemented = 0
    for fid, round_scores in ff_scores.items():
        for rnd, score in round_scores.items():
            if fid not in sc_scores or rnd not in sc_scores.get(fid, {}):
                sc_scores.setdefault(fid, {})[rnd] = score
                supplemented += 1

    if supplemented:
        log.info("Fanfooty supplement: %d player-round scores added to SC data.", supplemented)

    # Build output DataFrame — scores ordered by round
    records = []
    for fid, round_scores in sc_scores.items():
        ordered = [round_scores[r] for r in sorted(round_scores)]
        records.append({
            "feed_id": fid,
            "scores_2026": ordered,
            "round_scores_2026": {r: round_scores[r] for r in sorted(round_scores)},
            "rounds_played_2026": len(ordered),
            "avg_2026": round(sum(ordered) / len(ordered), 1) if ordered else 0.0,
        })

    grouped = pd.DataFrame(records)
    log.info(
        "Loaded %d season scores from SC API (%d rounds) + fanfooty supplement: %d players.",
        year, len(sc_rounds), len(grouped),
    )
    return grouped




# ── 3. Historical fanfooty data ────────────────────────────────────────────

def _season_weight(year: int) -> float:
    return SEASON_WEIGHTS.get(year, DEFAULT_SEASON_WEIGHT)


def load_historical_scores(
    base_dir: Path = FANFOOTY_PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Loads all available fanfooty processed CSVs (2011–2025) and returns
    a long DataFrame with one row per player-game:
        feed_id, year, round_num, sc_score, weight

    Used for computing weighted Ceiling Index and CV.
    """
    pattern = str(base_dir / "*_round_*_fanfooty_data.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        log.warning("No historical fanfooty CSVs found at %s", base_dir)
        return pd.DataFrame(columns=["feed_id", "year", "round_num", "sc_score", "weight"])

    frames = []
    year_round_re = re.compile(r"(\d{4})_round_(\d+)_fanfooty_data\.csv$")

    for fpath in files:
        m = year_round_re.search(fpath)
        if not m:
            continue
        year, round_num = int(m.group(1)), int(m.group(2))
        weight = _season_weight(year)

        try:
            chunk = pd.read_csv(fpath, low_memory=False)
        except Exception as exc:
            log.warning("Could not read %s: %s", fpath, exc)
            continue

        # The SC score column is labelled 'SC' in fanfooty processed files
        if "SC" not in chunk.columns or "Player ID" not in chunk.columns:
            continue

        chunk = chunk[["Player ID", "SC"]].copy()
        chunk.columns = ["feed_id", "sc_score"]
        chunk["feed_id"] = pd.to_numeric(chunk["feed_id"], errors="coerce")
        chunk["sc_score"] = pd.to_numeric(chunk["sc_score"], errors="coerce")
        chunk = chunk.dropna()
        chunk = chunk[chunk["sc_score"] > 0]
        chunk["feed_id"] = chunk["feed_id"].astype(int)
        chunk["sc_score"] = chunk["sc_score"].astype(int)
        chunk["year"] = year
        chunk["round_num"] = round_num
        chunk["weight"] = weight
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["feed_id", "year", "round_num", "sc_score", "weight"])

    out = pd.concat(frames, ignore_index=True)
    log.info(
        "Loaded %d historical game records across %d seasons.",
        len(out),
        out["year"].nunique(),
    )
    return out


def load_fanfooty_recent_rounds(
    base_dir: Path = FANFOOTY_PROCESSED_DIR,
    n_rounds: int = ROLE_WATCH_ROUNDS,
) -> pd.DataFrame:
    """
    Loads the most recent N rounds of fanfooty data for Role Watch.

    Returns columns:
        feed_id, year, round_num, tag_notes, tag2_notes, tog, clearances, name
    """
    pattern = str(base_dir / "*_round_*_fanfooty_data.csv")
    files = sorted(glob.glob(pattern))
    year_round_re = re.compile(r"(\d{4})_round_(\d+)_fanfooty_data\.csv$")

    # Find the most recent (year, round_num) tuples
    file_meta = []
    for fpath in files:
        m = year_round_re.search(fpath)
        if m:
            file_meta.append((int(m.group(1)), int(m.group(2)), fpath))

    if not file_meta:
        return pd.DataFrame()

    file_meta.sort(key=lambda x: (x[0], x[1]))

    # Only use current-season data — never fall back to prior seasons for Role Watch
    current_season = [f for f in file_meta if f[0] == SEASON_YEAR]
    if not current_season:
        log.info("No %d fanfooty data available yet — Role Watch will be empty.", SEASON_YEAR)
        return pd.DataFrame()

    recent = current_season[-n_rounds:]

    frames = []
    for year, round_num, fpath in recent:
        try:
            chunk = pd.read_csv(fpath, low_memory=False)
        except Exception as exc:
            log.warning("Could not read %s: %s", fpath, exc)
            continue

        wanted = {
            "Player ID": "feed_id",
            "First Name": "first_name",
            "Surname": "surname",
            "Tag Notes": "tag_notes",
            "Tag 2 Notes": "tag2_notes",
            "Time on ground": "tog",
            "Clearances": "clearances",
        }
        chunk = chunk.rename(columns=wanted)
        available = [c for c in wanted.values() if c in chunk.columns]
        chunk = chunk[available].copy()

        chunk["feed_id"] = pd.to_numeric(chunk.get("feed_id", pd.Series(dtype=float)), errors="coerce")
        chunk = chunk.dropna(subset=["feed_id"])
        chunk["feed_id"] = chunk["feed_id"].astype(int)

        if "first_name" in chunk.columns and "surname" in chunk.columns:
            chunk["name"] = chunk["first_name"].str.strip() + " " + chunk["surname"].str.strip()
        else:
            chunk["name"] = ""

        chunk["year"] = year
        chunk["round_num"] = round_num
        frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ── 4. Model predictions for upcoming round ────────────────────────────────

_PRED_RE = re.compile(r"(\d{4})_round_(\d+)_predictions\.csv$")


def load_predictions(
    year: int = SEASON_YEAR,
    round_num: int | None = None,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> pd.DataFrame:
    """
    Loads XGBoost model predictions for the upcoming round.

    Looks for data/predictions/{year}_round_{round_num+1}_predictions.csv.
    Falls back to the highest available round if the specific file doesn't exist.

    Returns a DataFrame with columns:
        feed_id, projected_score, probability_gt_80, probability_gt_100,
        opponent, pred_round

    Returns an empty DataFrame if no predictions are found (pipeline continues
    without them and newsletter sections gracefully omit prediction columns).
    """
    if not predictions_dir.exists():
        log.info("No predictions directory at %s — run run_predictions.py first.", predictions_dir)
        return pd.DataFrame()

    # Predictions are generated *for* the upcoming round, so target = last_played + 1
    target_round = (round_num + 1) if round_num is not None else 1

    target_path = predictions_dir / f"{year}_round_{target_round}_predictions.csv"
    if target_path.exists():
        chosen, pred_round = str(target_path), target_round
    else:
        # Fall back to the latest available predictions file for this year
        files = sorted(glob.glob(str(predictions_dir / f"{year}_round_*_predictions.csv")))
        candidates = []
        for f in files:
            m = _PRED_RE.search(f)
            if m:
                candidates.append((int(m.group(2)), f))
        if not candidates:
            log.info("No %d predictions files found — run run_predictions.py first.", year)
            return pd.DataFrame()
        pred_round, chosen = max(candidates, key=lambda x: x[0])
        log.warning(
            "Round %d predictions not found; falling back to Round %d predictions.",
            target_round, pred_round,
        )

    try:
        df = pd.read_csv(chosen, low_memory=False)
    except Exception as exc:
        log.warning("Could not read predictions file %s: %s", chosen, exc)
        return pd.DataFrame()

    if "Player ID" not in df.columns or "projected_score" not in df.columns:
        log.warning("Predictions file missing required columns.")
        return pd.DataFrame()

    df = df.rename(columns={"Player ID": "feed_id", "Opponent": "opponent"})
    df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce")
    df = df.dropna(subset=["feed_id"])
    df["feed_id"] = df["feed_id"].astype(int)
    df["pred_round"] = pred_round

    keep = [c for c in ("feed_id", "projected_score", "probability_gt_80",
                         "probability_gt_100", "probability_gt_120",
                         "opponent", "pred_round") if c in df.columns]
    df = df[keep]

    log.info(
        "Loaded Round %d predictions: %d players, avg projected score = %.1f.",
        pred_round, len(df), df["projected_score"].mean(),
    )
    return df


# ── 5. Build unified player table ──────────────────────────────────────────

def build_player_table(
    player_list_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    status_dict: dict[str, str],
    injury_dict: dict[str, dict] | None = None,
    lineup_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Merges all sources into a single player table.

    Returns one row per player with columns:
        sc_id, feed_id, name, position, team,
        scores_2026, avg_2026, rounds_played_2026,
        trade_status, is_free_agent, is_mine, is_on_field,
        injury_status, injury_text, is_injured, is_suspended

    lineup_dict: sc_player_id (str) → picked status from statsPlayers API
        "true" = on field, "false" = bench, "emerg" = emergency bench
        When provided, is_on_field is derived from this.
        Falls back to is_mine when absent (safe default: treat all mine as on field).
    """
    if player_list_df.empty:
        log.error("Player list is empty; cannot build player table.")
        return pd.DataFrame()

    # Attach trade status and team ownership from API
    player_list_df = player_list_df.copy()
    player_list_df["sc_id_str"] = player_list_df["sc_id"].astype(str)
    player_list_df["trade_status"] = player_list_df["sc_id_str"].map(
        lambda sid: (status_dict.get(sid) or {}).get("trade_status", "unknown")
    )
    player_list_df["user_team_id"] = player_list_df["sc_id_str"].map(
        lambda sid: (status_dict.get(sid) or {}).get("user_team_id")
    )
    # Pre-season: undrafted players have trade_status="waiver".
    # In-season:  "free_agent" = immediately claimable;
    #             "waiver"     = recently dropped, claimable after waiver deadline.
    # Both are pickup targets; we flag both as free agents in the newsletter.
    player_list_df["is_free_agent"] = player_list_df["trade_status"].isin(
        ["free_agent", "waiver"]
    )

    # Attach live injury / suspension data
    if injury_dict:
        player_list_df["injury_status"] = player_list_df["sc_id_str"].map(
            lambda sid: (injury_dict.get(sid) or {}).get("status") or ""
        )
        player_list_df["injury_text"] = player_list_df["sc_id_str"].map(
            lambda sid: (injury_dict.get(sid) or {}).get("text") or ""
        )
    else:
        player_list_df["injury_status"] = ""
        player_list_df["injury_text"] = ""

    player_list_df["is_injured"]   = player_list_df["injury_status"] == "Injury"
    player_list_df["is_suspended"] = player_list_df["injury_status"] == "Suspended"

    # Attach 2026 scores
    if not scores_df.empty:
        merged = player_list_df.merge(scores_df, on="feed_id", how="left")
    else:
        merged = player_list_df.copy()
        merged["scores_2026"] = [[] for _ in range(len(merged))]
        merged["round_scores_2026"] = [{} for _ in range(len(merged))]
        merged["avg_2026"] = 0.0
        merged["rounds_played_2026"] = 0

    # Fill NaN score lists (players with no 2026 data yet)
    merged["scores_2026"] = merged["scores_2026"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    if "round_scores_2026" in merged.columns:
        merged["round_scores_2026"] = merged["round_scores_2026"].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
    else:
        merged["round_scores_2026"] = [{} for _ in range(len(merged))]
    merged["rounds_played_2026"] = merged["rounds_played_2026"].fillna(0).astype(int)
    merged["avg_2026"] = merged["avg_2026"].fillna(0.0)

    # Flag roster membership from live API data
    merged["is_mine"] = merged["user_team_id"] == MY_TEAM_ID

    # Derive on-field status from the live lineup (statsPlayers API)
    if lineup_dict:
        merged["picked"] = merged["sc_id_str"].map(lineup_dict).fillna("")
        merged["is_on_field"] = merged["picked"] == "true"
    else:
        # Fallback: treat all my players as on field if lineup unavailable
        log.warning("No lineup data — treating all roster players as on field.")
        merged["is_on_field"] = merged["is_mine"]

    injured_mine = merged[merged["is_mine"] & (merged["is_injured"] | merged["is_suspended"])]
    log.info(
        "Player table built: %d total, %d free agents, %d on my roster, "
        "%d on my roster with injury/suspension.",
        len(merged),
        merged["is_free_agent"].sum(),
        merged["is_mine"].sum(),
        len(injured_mine),
    )
    if not injured_mine.empty:
        for _, row in injured_mine.iterrows():
            log.warning(
                "ROSTER INJURY: %s — %s: %s",
                row["name"], row["injury_status"], row["injury_text"]
            )
    return merged


# ── 6. News sentiment from SQLite (afl_news_scraper) ──────────────────────

_NEWS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "afl_news.db"


def load_news_sentiment(
    feed_ids: "list[int] | None" = None,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Load recent player mention sentiment from the news scraper SQLite DB.

    Returns a DataFrame with columns:
        feed_id, sentiment_score, keyword_tags, context_window, published_at, headline

    One row per mention (multiple rows per player if mentioned in several articles).
    Returns an empty DataFrame if the DB doesn't exist or has no data.
    """
    try:
        import sqlite3
        from datetime import datetime, timedelta, timezone
    except ImportError:
        return pd.DataFrame()

    if not _NEWS_DB_PATH.exists():
        log.debug("News DB not found at %s — skipping sentiment load", _NEWS_DB_PATH)
        return pd.DataFrame()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    try:
        conn = sqlite3.connect(_NEWS_DB_PATH)
        query = """
            SELECT
                pm.feed_id,
                pm.sentiment_score,
                pm.keyword_tags,
                pm.context_window,
                a.published_at,
                a.headline
            FROM player_mentions pm
            JOIN articles a ON pm.article_id = a.id
            WHERE a.published_at >= ?
              AND pm.feed_id IS NOT NULL
              AND pm.sentiment_score IS NOT NULL
        """
        params = [cutoff]
        if feed_ids:
            placeholders = ",".join("?" * len(feed_ids))
            query += f" AND pm.feed_id IN ({placeholders})"
            params.extend(str(f) for f in feed_ids)

        query += " ORDER BY a.published_at DESC"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce").astype("Int64")
        log.info("Loaded %d news mentions (last %d days)", len(df), lookback_days)
        return df

    except Exception as exc:
        log.warning("Could not load news sentiment: %s", exc)
        return pd.DataFrame()
