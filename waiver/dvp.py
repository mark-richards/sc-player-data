"""
dvp.py — Defence vs. Position (DVP) analytics.

Builds a table of how many SC points each AFL team concedes per position,
using rolling recent-round fanfooty data. Used for two newsletter sections:

  1. Matchup Intel — FAs facing weak defences THIS round
  2. Schedule Streamer — FAs with 3 consecutive soft matchups ahead
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Map SC player list long_team names → 2-letter fanfooty codes
_LONG_TO_ABBR = {
    "Adelaide": "AD",
    "Brisbane": "BL",
    "Carlton": "CA",
    "Collingwood": "CO",
    "Essendon": "ES",
    "Fremantle": "FR",
    "GWS Giants": "WS",
    "Geelong": "GE",
    "Gold Coast": "GC",
    "Hawthorn": "HW",
    "Melbourne": "ME",
    "North Melbourne": "NM",
    "Port Adelaide": "PA",
    "Richmond": "RI",
    "St Kilda": "SK",
    "Sydney": "SY",
    "West Coast": "WC",
    "Western Bulldogs": "WB",
}

# Map fanfooty Position strings → SC position buckets
_POS_MAP = {
    "Midfielder": "MID",
    "Midfielder Forward": "MID",
    "Midfielder/Forward": "MID",
    "Forward": "FWD",
    "Back": "DEF",
    "Back Midfielder": "DEF",
    "Defender": "DEF",
    "Defender/Forward": "DEF",
    "Ruck": "RUC",
    "Ruckman": "RUC",
}


def _load_fanfooty_for_dvp(
    year: int,
    max_rounds: int | None = None,
    lookback_rounds: int = 6,
) -> pd.DataFrame:
    """
    Load recent fanfooty CSVs and add an 'opponent' column by pairing teams
    within each match (every match has exactly 2 teams).

    Returns columns: feed_id, SC, pos_bucket, team, opponent, round_num, year
    """
    pattern = f"{year}_round_*_fanfooty_data.csv"
    files = sorted(_PROCESSED_DIR.glob(pattern))
    if not files:
        log.warning("No fanfooty CSVs found for %d in %s", year, _PROCESSED_DIR)
        return pd.DataFrame()

    # Restrict to the most recent lookback_rounds
    if lookback_rounds and len(files) > lookback_rounds:
        files = files[-lookback_rounds:]
    if max_rounds:
        files = files[:max_rounds]

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as exc:
            log.warning("Could not read %s: %s", f.name, exc)
            continue

        required = {"Fanfooty Match ID", "Player ID", "SC", "Team", "Position"}
        if not required.issubset(df.columns):
            log.warning("Skipping %s — missing columns", f.name)
            continue

        df = df.rename(columns={"Player ID": "feed_id", "Fanfooty Match ID": "match_id"})
        df["SC"] = pd.to_numeric(df["SC"], errors="coerce")
        df = df.dropna(subset=["SC", "Team", "Position"])
        df["pos_bucket"] = df["Position"].map(_POS_MAP)
        df = df.dropna(subset=["pos_bucket"])

        # Derive opponent: for each match, each player's opponent = the other team
        match_teams = (
            df.groupby("match_id")["Team"]
            .apply(lambda s: list(s.unique()))
            .to_dict()
        )

        def _opp(row):
            teams = match_teams.get(row["match_id"], [])
            others = [t for t in teams if t != row["Team"]]
            return others[0] if others else None

        df["opponent"] = df.apply(_opp, axis=1)
        df = df.dropna(subset=["opponent"])

        # Extract round number from filename
        try:
            round_num = int(f.stem.split("_round_")[1].split("_")[0])
        except (IndexError, ValueError):
            round_num = None
        df["round_num"] = round_num
        df["year"] = year

        keep = ["feed_id", "SC", "pos_bucket", "Team", "opponent", "round_num", "year"]
        frames.append(df[[c for c in keep if c in df.columns]])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"Team": "team"})
    return combined


def build_dvp_table(year: int, lookback_rounds: int = 6) -> pd.DataFrame:
    """
    Build a DVP (Defence vs Position) table from recent fanfooty data.

    For each (opponent_team, pos_bucket) pair, calculates the average SC score
    conceded and ranks teams within each position (rank 1 = leaks most points = easiest).

    Returns columns: team, pos_bucket, avg_sc_conceded, games, dvp_rank
    """
    data = _load_fanfooty_for_dvp(year, lookback_rounds=lookback_rounds)
    if data.empty:
        return pd.DataFrame()

    # Group by opponent team (the team conceding) and position
    dvp = (
        data.groupby(["opponent", "pos_bucket"])["SC"]
        .agg(avg_sc_conceded="mean", games="count")
        .reset_index()
        .rename(columns={"opponent": "team"})
    )

    # Rank within each position: rank 1 = leaks most (easiest matchup)
    dvp["dvp_rank"] = dvp.groupby("pos_bucket")["avg_sc_conceded"].rank(
        ascending=False, method="min"
    ).astype(int)

    log.info(
        "DVP table built: %d team-position entries from last %d rounds",
        len(dvp), lookback_rounds,
    )
    return dvp


def compute_dvp_ranks(
    dvp_table: pd.DataFrame,
    player_df: pd.DataFrame,
    fixture_map: dict[str, str],
) -> "dict[int, int]":
    """
    Return {feed_id: dvp_rank} for each player whose next opponent is in dvp_table.
    dvp_rank 1 = softest matchup (leaks most points). Uses _LONG_TO_ABBR to handle
    full team names in player_df["team"].
    """
    if dvp_table.empty or player_df.empty or not fixture_map:
        return {}
    df = player_df[["feed_id", "team", "pos_bucket"]].copy()
    df["_abbr"] = df["team"].map(_LONG_TO_ABBR).fillna(df["team"])
    df["opponent"] = df["_abbr"].map(fixture_map)
    df = df.dropna(subset=["opponent"])
    merged = df.merge(
        dvp_table.rename(columns={"team": "opponent"}),
        on=["opponent", "pos_bucket"],
        how="left",
    )
    return {
        int(r["feed_id"]): int(r["dvp_rank"])
        for _, r in merged.iterrows()
        if pd.notna(r.get("dvp_rank"))
    }


def get_dvp_targets(
    dvp_table: pd.DataFrame,
    fa_df: pd.DataFrame,
    fixture_map: dict[str, str],
    top_n_teams: int = 4,
    top_n_players: int = 3,
) -> pd.DataFrame:
    """
    Return top FA pickup targets for the current round based on DVP matchups.

    Parameters
    ----------
    dvp_table   : from build_dvp_table()
    fa_df       : metrics_df filtered to free agents (must have pos_bucket, team, effective_avg)
    fixture_map : {team_abbr: opponent_abbr} for the current round
    top_n_teams : how many weakest defences to consider per position
    top_n_players : how many FAs to show per position

    Returns a DataFrame with columns:
        name, pos_bucket, team, opponent, avg_sc_conceded, dvp_rank, effective_avg
    """
    if dvp_table.empty or fa_df.empty or not fixture_map:
        return pd.DataFrame()

    fa = fa_df.copy()
    fa["_team_abbr"] = fa["team"].map(_LONG_TO_ABBR).fillna(fa["team"])
    fa["opponent"] = fa["_team_abbr"].map(fixture_map)
    fa = fa.dropna(subset=["opponent"])

    # Merge DVP info for each player's opponent
    merged = fa.merge(
        dvp_table.rename(columns={"team": "opponent"}),
        on=["opponent", "pos_bucket"],
        how="left",
    )

    # Keep only FAs facing top_n_teams weakest defences at their position
    soft_matchup = merged[merged["dvp_rank"] <= top_n_teams].copy()

    if soft_matchup.empty:
        return pd.DataFrame()

    avg_col = "avg_fanfooty_2026" if "avg_fanfooty_2026" in soft_matchup.columns else "effective_avg"

    # Drop players with no meaningful 2026 data
    soft_matchup = soft_matchup[soft_matchup[avg_col].fillna(0) >= 40]

    soft_matchup = (
        soft_matchup
        .sort_values(["pos_bucket", avg_col], ascending=[True, False])
        .groupby("pos_bucket", group_keys=False)
        .head(top_n_players)
        .sort_values(["pos_bucket", avg_col], ascending=[True, False])
    )

    keep = [c for c in ["name", "pos_bucket", "team", "opponent",
                         "avg_sc_conceded", "dvp_rank", avg_col, "projected_score"]
            if c in soft_matchup.columns]
    return soft_matchup[keep].reset_index(drop=True)


def get_schedule_streamers(
    dvp_table: pd.DataFrame,
    fa_df: pd.DataFrame,
    fixture_maps: "list[dict[str, str]]",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify FAs with the best 3-round schedule window by DVP.

    Parameters
    ----------
    dvp_table    : from build_dvp_table()
    fa_df        : metrics_df filtered to free agents
    fixture_maps : list of fixture dicts for the next N rounds (index 0 = soonest)
    top_n        : how many streamers to return

    Returns a DataFrame with columns:
        name, pos_bucket, team, avg_dvp_rank, rounds_checked, effective_avg
    """
    if dvp_table.empty or fa_df.empty or not fixture_maps:
        return pd.DataFrame()

    fa = fa_df.copy()
    avg_col = "avg_fanfooty_2026" if "avg_fanfooty_2026" in fa.columns else "effective_avg"
    fa = fa[fa[avg_col].fillna(0) >= 40]
    rank_scores = []

    fa["_team_abbr"] = fa["team"].map(_LONG_TO_ABBR).fillna(fa["team"])

    for rnd_fixture in fixture_maps:
        fa_rnd = fa.copy()
        fa_rnd["opponent"] = fa_rnd["_team_abbr"].map(rnd_fixture)
        fa_rnd = fa_rnd.dropna(subset=["opponent"])
        merged = fa_rnd.merge(
            dvp_table.rename(columns={"team": "opponent"}),
            on=["opponent", "pos_bucket"],
            how="left",
        )
        merged = merged.set_index("feed_id" if "feed_id" in merged.columns else merged.index)
        rank_scores.append(merged[["dvp_rank"]])

    # Average DVP rank across the next N rounds (lower = easier schedule)
    combined = pd.concat(rank_scores, axis=1)
    combined.columns = [f"dvp_rank_r{i}" for i in range(len(fixture_maps))]
    combined["avg_dvp_rank"] = combined.mean(axis=1)
    combined["rounds_checked"] = combined.notna().sum(axis=1)

    fa = fa.copy()
    if "feed_id" in fa.columns:
        fa = fa.set_index("feed_id")
    fa = fa.join(combined[["avg_dvp_rank", "rounds_checked"]], how="left")
    fa = fa.dropna(subset=["avg_dvp_rank"])
    fa = fa.sort_values("avg_dvp_rank").head(top_n)

    avg_col = "avg_fanfooty_2026" if "avg_fanfooty_2026" in fa.columns else "effective_avg"
    keep = [c for c in ["name", "pos_bucket", "team", "avg_dvp_rank",
                         "rounds_checked", avg_col]
            if c in fa.columns]
    return fa[keep].reset_index(drop=True)
