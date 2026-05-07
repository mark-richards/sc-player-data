"""
honour_board.py — Build the all-time ASL honour board.

Data sources (in priority order):
  1. _HARDCODED below — all completed seasons (2013–2022)
  2. data/processed/league_master.csv — fallback for 2016–2021
     (overridden by _HARDCODED)

Simulation CSVs are NOT used: they contain probabilities, not actual outcomes.
Add new seasons to _HARDCODED once results are known.
"""
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Actual season results, 2013–2022.
# finals = top-4 finalists in order [champion, runner_up, 3rd, 4th].
# spoon  = last-place coach (None if the last-place coach is no longer in the league).
_HARDCODED: dict[int, dict] = {
    2013: {"champion": "Simon",   "runner_up": "Mark",   "finals": ["Simon",   "Mark",   "Lester", "Jordan"], "spoon": None},
    2014: {"champion": "Simon",   "runner_up": "Lester", "finals": ["Simon",   "Lester", "Anthony","Jordan"], "spoon": "Paul"},
    2015: {"champion": "Anthony", "runner_up": "Paul",   "finals": ["Anthony", "Paul",   "Simon",  "James"],  "spoon": "Mark"},
    2016: {"champion": "Anthony", "runner_up": "Simon",  "finals": ["Anthony", "Simon",  "Luke",   "Lester"], "spoon": "Jordan"},
    2017: {"champion": "Paul",    "runner_up": "Jordan", "finals": ["Paul",    "Jordan", "Luke",   "Lester"], "spoon": "Simon"},
    2018: {"champion": "Luke",    "runner_up": "Simon",  "finals": ["Luke",    "Simon",  "Anthony","Jordan"], "spoon": "Mark"},
    2019: {"champion": "Luke",    "runner_up": "Paul",   "finals": ["Luke",    "Paul",   "Anthony","Jordan"], "spoon": "Simon"},
    2020: {"champion": "Mark",    "runner_up": "Simon",  "finals": ["Mark",    "Simon",  "Lester", "Jordan"], "spoon": "Anthony"},
    2021: {"champion": "Paul",    "runner_up": "James",  "finals": ["Paul",    "James",  "Anthony","Luke"],   "spoon": "Simon"},
    2022: {"champion": "Paul",    "runner_up": "Mark",   "finals": ["Paul",    "Mark",   "Lester", "Luke"],   "spoon": "Simon"},
    2023: {"champion": "Luke",    "runner_up": "Anthony","finals": ["Luke",    "Anthony","Lester", "James"],  "spoon": "Jordan"},
    2024: {"champion": "Paul",    "runner_up": "Jordan", "finals": ["Paul",    "Jordan", "Simon",  "Luke"],   "spoon": "Mark"},
    2025: {"champion": "Paul",    "runner_up": "Luke",   "finals": ["Paul",    "Luke",   "Lester", "James"],  "spoon": "Jordan"},
}


def _results_from_master(master_df: pd.DataFrame) -> dict[int, dict]:
    """Derive season results from league_master.csv (2016-2021)."""
    results = {}
    if master_df.empty:
        return results

    # Find the final round per season
    for season, sdf in master_df.groupby("season"):
        final_round = sdf["round"].max()
        final = sdf[sdf["round"] == final_round].copy()

        # Sort by position (ascending = 1st is best)
        if "position" in final.columns:
            final = final.sort_values("position")
        elif "points" in final.columns:
            final = final.sort_values("points", ascending=False)

        coaches = final["coach"].tolist()
        if not coaches:
            continue

        results[int(season)] = {
            "champion":   coaches[0] if len(coaches) > 0 else None,
            "runner_up":  coaches[1] if len(coaches) > 1 else None,
            "finals":     coaches[:4],
            "spoon":      coaches[-1] if len(coaches) >= 8 else None,
        }
    return results



def build_honour_board(
    master_df: pd.DataFrame,
    fantasy_banter_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns:
      all_time_df  — DataFrame: coach | Championships | GF | Finals | Spoons | Seasons
      season_list  — list of dicts: year, champion, runner_up, finalist_3, finalist_4, spoon
    """
    # Gather all season results; _HARDCODED takes priority over master CSV
    season_results: dict[int, dict] = {}
    season_results.update(_results_from_master(master_df))
    season_results.update(_HARDCODED)

    all_coaches = ["Anthony", "James", "Jordan", "Lester", "Luke", "Mark", "Paul", "Simon"]
    stats = {c: {"Championships": 0, "GF": 0, "Finals": 0, "Spoons": 0, "Seasons": 0} for c in all_coaches}

    season_list = []
    for year in sorted(season_results.keys(), reverse=True):
        r = season_results[year]
        finals = r.get("finals", [])
        champ  = r.get("champion")
        runner = r.get("runner_up")
        spoon  = r.get("spoon")

        for coach in finals:
            if coach in stats:
                stats[coach]["Finals"] += 1
                stats[coach]["Seasons"] += 1

        # Count seasons for non-finalists too
        for coach in all_coaches:
            if coach not in finals and coach in stats:
                stats[coach]["Seasons"] += 1

        if champ and champ in stats:
            stats[champ]["Championships"] += 1
            stats[champ]["GF"] += 1

        if runner and runner in stats:
            stats[runner]["GF"] += 1

        if spoon and spoon in stats:
            stats[spoon]["Spoons"] += 1

        season_list.append({
            "year":       year,
            "champion":   champ,
            "runner_up":  runner,
            "finalist_3": finals[2] if len(finals) > 2 else "",
            "finalist_4": finals[3] if len(finals) > 3 else "",
            "spoon":      spoon,
        })

    rows = []
    for coach, s in stats.items():
        rows.append({"coach": coach, **s})
    all_time_df = pd.DataFrame(rows).sort_values("Championships", ascending=False)

    return all_time_df, season_list
