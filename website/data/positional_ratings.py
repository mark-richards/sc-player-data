"""
positional_ratings.py — Compute 0-10 positional strength ratings per coach.

Algorithm:
  1. Filter player_match_results to on_field == True
  2. Group by (round, coach, position) → sum points
  3. Per coach per position: rolling 7-game mean
  4. At current_round: normalise each position 0-10 across all coaches
  5. ALL = mean of the four position ratings
"""
import pandas as pd
import numpy as np

from website.config import POSITIONAL_WINDOW

POSITIONS = ["DEF", "MID", "RUC", "FWD"]


def compute_positional_ratings(
    player_matches: pd.DataFrame,
    current_round: int,
    window: int = POSITIONAL_WINDOW,
) -> pd.DataFrame:
    """
    Returns DataFrame: coach_first_name | DEF | MID | RUC | FWD | ALL
    All values are floats 0.0–10.0.
    """
    df = player_matches[player_matches["on_field"] == True].copy()

    # Only keep known positions
    df = df[df["played_position"].isin(POSITIONS)]

    # Total on-field points per coach per round per position
    agg = (
        df.groupby(["round_x", "coach_first_name", "played_position"])["points_x"]
        .sum()
        .reset_index()
    )
    agg.columns = ["round", "coach", "position", "points"]

    # Pivot: one column per position
    pivot = agg.pivot_table(index=["round", "coach"], columns="position", values="points", fill_value=0)
    pivot = pivot.reset_index()

    # Ensure all position columns exist
    for pos in POSITIONS:
        if pos not in pivot.columns:
            pivot[pos] = 0.0

    # Build full round × coach grid so rolling doesn't skip rounds
    coaches = pivot["coach"].unique()
    all_rounds = range(1, current_round + 1)
    full_idx = pd.MultiIndex.from_product([all_rounds, coaches], names=["round", "coach"])
    pivot = pivot.set_index(["round", "coach"]).reindex(full_idx, fill_value=0).reset_index()

    # Rolling mean per coach per position
    pivot = pivot.sort_values(["coach", "round"])
    for pos in POSITIONS:
        pivot[f"{pos}_roll"] = (
            pivot.groupby("coach")[pos]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # Extract values at current_round
    current = pivot[pivot["round"] == current_round][
        ["coach"] + [f"{pos}_roll" for pos in POSITIONS]
    ].copy()
    current.columns = ["coach_first_name"] + POSITIONS

    # Normalise 0-10 per position across coaches
    for pos in POSITIONS:
        col = current[pos]
        mn, mx = col.min(), col.max()
        if mx > mn:
            current[pos] = ((col - mn) / (mx - mn) * 10).round(1)
        else:
            current[pos] = 5.0

    current["ALL"] = current[POSITIONS].mean(axis=1).round(1)
    return current.reset_index(drop=True)
