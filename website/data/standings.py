"""
standings.py — Compute the season summary standings table from ladder.csv.
"""
import pandas as pd

# The league's own tiebreak rule changed partway through its history: checked
# against every completed season's real finals bracket (the literal saved
# game record, not ladder.csv's own "position" column — that turned out to
# be unreliable for the 2016-2019 legacy-sourced years, see 2018 below), the
# true 2nd-sort key was percentage (points_for / points_against) through
# 2021, then switched to raw total points_for from 2022 on. ladder.csv's own
# "position" column isn't used as the source of truth here: for 2020+ (live
# API-sourced years) it agrees with this; for 2016-2019 (legacy workbook-
# sourced) it doesn't look independently platform-sourced at all — it matches
# raw avg_for everywhere, including 2018, where real_finalists prove that's
# wrong (Jordan played that year's semi-final, not Lester, who ladder.csv's
# own numbers plus "position" column both incorrectly rank above Jordan).
PERCENTAGE_TIEBREAK_LAST_YEAR = 2021


def get_current_round(ladder_df: pd.DataFrame) -> int:
    if ladder_df.empty or "round" not in ladder_df.columns:
        return 0
    return int(ladder_df["round"].max())


def compute_standings(ladder_df: pd.DataFrame, current_round: int, year: int | None = None) -> pd.DataFrame:
    """
    Returns one row per coach for the given round, with derived columns.
    Sorted by league_points DESC, then by whichever tiebreak the league
    actually used that year (see PERCENTAGE_TIEBREAK_LAST_YEAR above) —
    percentage for year <= 2021, raw avg_for (total points) for 2022+.
    year defaults to whatever's in ladder_df's own "year" column if present,
    else PERCENTAGE_TIEBREAK_LAST_YEAR + 1 (i.e. the newer/current rule) so
    an un-dated call doesn't silently fall back to the older formula.
    """
    if ladder_df.empty or "round" not in ladder_df.columns:
        return pd.DataFrame(columns=[
            "coach_first_name", "games", "avg_for", "avg_against",
            "win_pct", "wdl", "league_points", "ladder_pos",
        ])

    df = ladder_df[ladder_df["round"] == current_round].copy()

    games = df["wins"] + df["draws"] + df["losses"]
    df["games"] = games
    df["avg_for"] = (df["points_for"] / games.replace(0, 1)).round(1)
    df["avg_against"] = (df["points_against"] / games.replace(0, 1)).round(1)
    df["win_pct"] = ((df["wins"] / games.replace(0, 1)) * 100).round(1)
    df["wdl"] = df["wins"].astype(str) + "-" + df["draws"].astype(str) + "-" + df["losses"].astype(str)

    if year is None:
        year = int(df["year"].iloc[0]) if "year" in df.columns and not df.empty else PERCENTAGE_TIEBREAK_LAST_YEAR + 1

    if year <= PERCENTAGE_TIEBREAK_LAST_YEAR:
        df["tiebreak"] = df["points_for"] / df["points_against"].replace(0, 1) * 100
    else:
        df["tiebreak"] = df["avg_for"]

    df = df.sort_values(["league_points", "tiebreak"], ascending=[False, False])
    df["ladder_pos"] = range(1, len(df) + 1)
    return df.drop(columns=["tiebreak"]).reset_index(drop=True)
