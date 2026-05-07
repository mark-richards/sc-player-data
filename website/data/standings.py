"""
standings.py — Compute the season summary standings table from ladder.csv.
"""
import pandas as pd


def get_current_round(ladder_df: pd.DataFrame) -> int:
    return int(ladder_df["round"].max())


def compute_standings(ladder_df: pd.DataFrame, current_round: int) -> pd.DataFrame:
    """
    Returns one row per coach for the given round, with derived columns.
    Sorted by league_points DESC, then avg_for DESC (SC tiebreaker).
    """
    df = ladder_df[ladder_df["round"] == current_round].copy()

    games = df["wins"] + df["draws"] + df["losses"]
    df["games"] = games
    df["avg_for"] = (df["points_for"] / games.replace(0, 1)).round(1)
    df["avg_against"] = (df["points_against"] / games.replace(0, 1)).round(1)
    df["win_pct"] = ((df["wins"] / games.replace(0, 1)) * 100).round(1)
    df["wdl"] = df["wins"].astype(str) + "-" + df["draws"].astype(str) + "-" + df["losses"].astype(str)

    df = df.sort_values(["league_points", "avg_for"], ascending=[False, False])
    df["ladder_pos"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)
