"""
loader.py — Single data-access layer for the website.
All heavy CSV reads are wrapped in flask-caching memoize calls.
Call cache.clear() via POST /api/refresh to force a reload.
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


# ── Ladder ─────────────────────────────────────────────────────────────────

def load_ladder() -> pd.DataFrame:
    """
    Reads data/live/ladder.csv.
    Deduplicates on (round, coach_first_name) — pipeline appends duplicates
    on the final round. Returns cumulative standings per round.
    """
    from website.config import LADDER_CSV
    try:
        df = pd.read_csv(LADDER_CSV, low_memory=False)
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

def load_fixtures() -> pd.DataFrame:
    """
    Reads fixture_results_by_team.csv — one row per team per round.
    Adds derived columns: win (bool), draw (bool).
    """
    from website.config import FIXTURE_TEAM_CSV
    try:
        df = pd.read_csv(FIXTURE_TEAM_CSV, low_memory=False)
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

def load_player_matches() -> pd.DataFrame:
    """
    Reads player_match_results.csv (3.3 MB — cached aggressively).
    Normalises key column names. Filters to rows with a valid round.
    Note: round column is 'round_x', points is 'points_x', position is 'played_position'.
    """
    from website.config import PLAYER_MATCH_CSV
    try:
        df = pd.read_csv(PLAYER_MATCH_CSV, low_memory=False)
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

def load_fanfooty_season() -> pd.DataFrame:
    """
    Loads all per-round fanfooty CSVs for the current season year and concatenates them.
    Parses Round ("R1" → 1) into round_num. Player ID cast to Int64 for joining on feed_id.
    """
    import glob
    from website.config import PROCESSED_DATA_DIR, SEASON_YEAR
    pattern = str(PROCESSED_DATA_DIR / f"{SEASON_YEAR}_round_*_fanfooty_data.csv")
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

def load_sc_round_files() -> pd.DataFrame:
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
    sc_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "supercoach" / str(SEASON_YEAR)
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

def load_sc_current() -> pd.DataFrame:
    """
    Reads data/processed/master_player_data.csv — SC per-round scores for the
    current season, built from completeStatspack JSONs via build_master_dataset.py.

    Filters to SEASON_YEAR and normalises column names for _compute_effective_avg:
      feed_id, round, points, team_abbrev, pos_1, pos_2, played
    """
    from website.config import SC_CURRENT_CSV, SEASON_YEAR
    try:
        df = pd.read_csv(SC_CURRENT_CSV, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or df.columns.empty:
        return pd.DataFrame()
    # Filter to current season only
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"] == SEASON_YEAR].copy()
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

def load_fixture_schedule() -> pd.DataFrame:
    """
    Reads data/live/fixture_schedule_2026.csv — one row per matchup per round.
    Columns: round_number, home_coach, away_coach.
    Returns empty DataFrame if the file doesn't exist.
    """
    from website.config import FIXTURE_SCHEDULE_CSV
    try:
        df = pd.read_csv(FIXTURE_SCHEDULE_CSV, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or df.columns.empty:
        return pd.DataFrame()
    df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
    df = df.dropna(subset=["round_number", "home_coach", "away_coach"])
    df["round_number"] = df["round_number"].astype(int)
    return df.reset_index(drop=True)


# ── Current rosters ─────────────────────────────────────────────────────────

def load_current_teams() -> pd.DataFrame:
    from website.config import CURRENT_TEAMS_CSV
    df = pd.read_csv(CURRENT_TEAMS_CSV, low_memory=False)
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
