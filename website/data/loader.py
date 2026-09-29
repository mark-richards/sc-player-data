"""
loader.py — Single data-access layer for the website.
All heavy CSV reads are wrapped in flask-caching memoize calls.
Call cache.clear() via POST /api/refresh to force a reload.

Every load_* function takes an optional `year` — defaults to the live
SEASON_YEAR. flask-caching's cache.memoize() keys on function arguments, so
passing a different year automatically gets its own cache entry.
"""
import base64
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Cache is injected from app.py after creation
cache = None


def _memoize(fn):
    """Lazy wrapper: if cache is available use it, otherwise call directly."""
    def wrapper(*args, **kwargs):
        if cache is not None:
            return cache.memoize()(fn)(*args, **kwargs)
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


# ── Available seasons / coach roster ────────────────────────────────────────

def list_available_seasons() -> list[int]:
    """SEASON_YEAR plus every year with a data/seasons/{year} directory, newest first."""
    from website.config import SEASON_YEAR, SEASONS_DIR
    years = {SEASON_YEAR}
    if SEASONS_DIR.exists():
        for p in SEASONS_DIR.iterdir():
            if p.is_dir() and p.name.isdigit():
                years.add(int(p.name))
    return sorted(years, reverse=True)


def list_season_coaches(year: int | None = None) -> list[str]:
    """
    Unique coach_first_name values for this season, from coach_list.csv (falling
    back to ladder.csv), in config.COACH_ORDER where possible. Falls back to
    COACH_ORDER outright for the current season if neither file is available.
    """
    from website.config import SEASON_YEAR, COACH_ORDER, coach_list_csv_path, ladder_csv_path
    year = year or SEASON_YEAR

    names: list[str] = []
    try:
        df = pd.read_csv(coach_list_csv_path(year), usecols=["coach_first_name"])
        names = df["coach_first_name"].dropna().unique().tolist()
    except Exception:
        try:
            df = pd.read_csv(ladder_csv_path(year), usecols=["coach_first_name"])
            names = df["coach_first_name"].dropna().unique().tolist()
        except Exception:
            names = []

    if not names:
        return list(COACH_ORDER) if year == SEASON_YEAR else []

    ordered = [c for c in COACH_ORDER if c in names]
    ordered += sorted(c for c in names if c not in COACH_ORDER)
    return ordered


# ── Ladder ─────────────────────────────────────────────────────────────────

def load_ladder(year: int | None = None) -> pd.DataFrame:
    """
    Reads ladder.csv for the given season.
    Deduplicates on (round, coach_first_name) — pipeline appends duplicates
    on the final round. Returns cumulative standings per round.
    """
    from website.config import SEASON_YEAR, ladder_csv_path
    year = year or SEASON_YEAR
    try:
        df = pd.read_csv(ladder_csv_path(year), low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or df.columns.empty:
        return pd.DataFrame()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"])
    df["round"] = df["round"].astype(int)
    df = df.drop_duplicates(subset=["round", "coach_first_name"], keep="first")
    return df.reset_index(drop=True)


# ── Fixtures ───────────────────────────────────────────────────────────────

def load_fixtures(year: int | None = None) -> pd.DataFrame:
    """
    Reads fixture_results_by_team.csv — one row per team per round.
    Adds derived columns: win (bool), draw (bool).
    """
    from website.config import SEASON_YEAR, fixture_team_csv_path
    year = year or SEASON_YEAR
    try:
        df = pd.read_csv(fixture_team_csv_path(year), low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or df.columns.empty:
        return pd.DataFrame()
    df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
    df["team_points"] = pd.to_numeric(df["team_points"], errors="coerce")
    df["opposition_team_points"] = pd.to_numeric(df["opposition_team_points"], errors="coerce")
    df = df.dropna(subset=["round_number", "team_points"])
    df["round_number"] = df["round_number"].astype(int)
    df["win"] = df["team_points"] > df["opposition_team_points"]
    df["draw"] = df["team_points"] == df["opposition_team_points"]
    # Normalise coach column name → coach_first_name for consistency
    if "team_coach" in df.columns and "coach_first_name" not in df.columns:
        df = df.rename(columns={"team_coach": "coach_first_name"})
    return df.reset_index(drop=True)


# ── Player match results ────────────────────────────────────────────────────

def load_player_matches(year: int | None = None) -> pd.DataFrame:
    """
    Reads player_match_results.csv. Normalises key column names. Filters to
    rows with a valid round.
    Note: round column is 'round_x', points is 'points_x', position is 'played_position'.
    """
    from website.config import SEASON_YEAR, player_match_csv_path
    year = year or SEASON_YEAR
    try:
        df = pd.read_csv(player_match_csv_path(year), low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or df.columns.empty:
        return pd.DataFrame()
    df["round_x"] = pd.to_numeric(df["round_x"], errors="coerce")
    df["points_x"] = pd.to_numeric(df["points_x"], errors="coerce")
    df = df.dropna(subset=["round_x", "points_x", "coach_first_name"])
    df["round_x"] = df["round_x"].astype(int)
    # Normalise on_field to bool
    if df["on_field"].dtype == object:
        df["on_field"] = df["on_field"].map(
            {"True": True, "False": False, True: True, False: False}
        ).fillna(False)
    return df.reset_index(drop=True)


# ── Fanfooty per-round data ─────────────────────────────────────────────────

def load_fanfooty_season(year: int | None = None) -> pd.DataFrame:
    """
    Loads all per-round fanfooty CSVs for the given season year and concatenates
    them. Parses Round ("R1" → 1) into round_num. Player ID cast to Int64 for
    joining on feed_id.
    """
    import glob
    from website.config import PROCESSED_DATA_DIR, SEASON_YEAR
    year = year or SEASON_YEAR
    pattern = str(PROCESSED_DATA_DIR / f"{year}_round_*_fanfooty_data.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["round_num"] = pd.to_numeric(
        df["Round"].str.replace(r"[^0-9]", "", regex=True), errors="coerce"
    )
    df = df.dropna(subset=["round_num", "Player ID"])
    df["round_num"] = df["round_num"].astype(int)
    df["Player ID"] = pd.to_numeric(df["Player ID"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


# ── SC bulk all-player round data (owned + free agents) ─────────────────────

def load_sc_round_files(year: int | None = None) -> pd.DataFrame:
    """
    Reads data/raw/supercoach/{year}/players_round_N.json — the bulk SC API
    endpoint covering EVERY player each round (owned and free agents alike),
    unlike player_stats_current.csv/master_player_data.csv which are built from
    fantasy team rosters and are missing any round a player wasn't drafted/held.

    Returns one row per player per round: feed_id, round, team_abbrev, played
    (games==1), points.
    """
    import glob
    import json
    import re
    from website.config import SEASON_YEAR
    year = year or SEASON_YEAR
    sc_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "supercoach" / str(year)
    pattern = str(sc_dir / "players_round_*.json")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    round_re = re.compile(r"players_round_(\d+)\.json$")
    rows = []
    for fpath in files:
        m = round_re.search(fpath)
        if not m:
            continue
        round_num = int(m.group(1))
        try:
            players = json.loads(Path(fpath).read_text(encoding="utf-8"))
        except Exception:
            continue
        for player in players:
            fid = player.get("feed_id")
            if not fid:
                continue
            team_abbrev = (player.get("team") or {}).get("abbrev", "")
            for stat in (player.get("player_stats") or []):
                rows.append({
                    "feed_id":     fid,
                    "round":       round_num,
                    "team_abbrev": team_abbrev,
                    "played":      int(stat.get("games", 0) or 0),
                    "points":      stat.get("points", 0) or 0,
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce").astype("Int64")
    return df.dropna(subset=["feed_id"]).reset_index(drop=True)


# ── SC current-season per-round scores ─────────────────────────────────────

def load_sc_current(year: int | None = None) -> pd.DataFrame:
    """
    Reads data/processed/master_player_data.csv — SC per-round scores, built
    from completeStatspack JSONs via build_master_dataset.py.

    Filters to the given season year and normalises column names for
    _compute_effective_avg: feed_id, round, points, team_abbrev, pos_1, pos_2, played
    """
    from website.config import SC_CURRENT_CSV, SEASON_YEAR
    year = year or SEASON_YEAR
    try:
        df = pd.read_csv(SC_CURRENT_CSV, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or df.columns.empty:
        return pd.DataFrame()
    # Filter to the requested season only
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"] == year].copy()
    if df.empty:
        return pd.DataFrame()
    # Normalise column names
    df = df.rename(columns={
        "Player ID":   "feed_id",
        "SC":          "points",
        "sc_position": "pos_1",
        "Round_Num":   "round",
    })
    df["pos_2"]   = pd.NA
    df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce").astype("Int64")
    df["round"]   = pd.to_numeric(df["round"],   errors="coerce")
    df["points"]  = pd.to_numeric(df["points"],  errors="coerce").fillna(0)
    df["played"]  = pd.to_numeric(df["played"],  errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["feed_id", "round"]).reset_index(drop=True)


# ── Historical league scoring prior ────────────────────────────────────────

def load_historical_scores() -> "np.ndarray":
    """
    Returns a NumPy array of all historical team game scores from
    data/processed/league_master.csv (2016–2021, ~1004 scores).
    Used to build a robust prior for the simulation's score distributions.
    Returns empty array if file not found.
    """
    import numpy as np
    from website.config import LEAGUE_MASTER_CSV
    try:
        df = pd.read_csv(LEAGUE_MASTER_CSV, low_memory=False, usecols=["team_score"])
    except Exception:
        return np.array([])
    scores = pd.to_numeric(df["team_score"], errors="coerce").dropna().values
    return scores.astype(float)


# ── Fixture schedule (all rounds, played + future) ─────────────────────────

def load_fixture_schedule(year: int | None = None) -> pd.DataFrame:
    """
    Reads fixture_schedule_{year}.csv — one row per matchup per round.
    Columns: round_number, home_coach, away_coach.
    Returns empty DataFrame if the file doesn't exist.
    """
    from website.config import SEASON_YEAR, fixture_schedule_csv_path
    year = year or SEASON_YEAR
    try:
        df = pd.read_csv(fixture_schedule_csv_path(year), low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or df.columns.empty:
        return pd.DataFrame()
    df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
    df = df.dropna(subset=["round_number", "home_coach", "away_coach"])
    df["round_number"] = df["round_number"].astype(int)
    return df.reset_index(drop=True)


# ── Current rosters ─────────────────────────────────────────────────────────

def load_current_teams(year: int | None = None) -> pd.DataFrame:
    from website.config import SEASON_YEAR, current_teams_csv_path
    year = year or SEASON_YEAR
    try:
        df = pd.read_csv(current_teams_csv_path(year), low_memory=False)
    except Exception:
        return pd.DataFrame()
    return df


# ── Historical league master (2016–2021) ────────────────────────────────────

def load_league_master() -> pd.DataFrame:
    from website.config import LEAGUE_MASTER_CSV
    if not LEAGUE_MASTER_CSV.exists():
        log.warning("league_master.csv not found — honour board will have limited history.")
        return pd.DataFrame()
    df = pd.read_csv(LEAGUE_MASTER_CSV, low_memory=False)
    return df


# ── Coach portrait images (base64-encoded for Plotly scatter) ───────────────

def load_coach_portraits() -> dict[str, str]:
    """
    Returns {coach_name: 'data:image/png;base64,...'} for each available portrait.
    """
    from website.config import COACH_IMAGES_DIR, COACH_PORTRAITS
    portraits = {}
    for name, filename in COACH_PORTRAITS.items():
        path = COACH_IMAGES_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            portraits[name] = f"data:image/png;base64,{encoded}"
        else:
            log.warning("Portrait not found: %s", path)
    return portraits
