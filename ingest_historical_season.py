"""
ingest_historical_season.py — One-off converter: archived league_data.RDS /
player_data.RDS (from the old R-based pipeline, archive/legacy/fantasy_banter_data/
{year}/) into the data/seasons/{year}/ CSV shape the website reads for historical
seasons (same file shape as data/live/, see website/config.py).

Usage: py ingest_historical_season.py 2022 2023 2024 2025

Source schemas (verified by hand against 2022-2025):
  league_data.RDS (data.frame): round, fixture, team_id, team, coach, team_score,
    opponent_team_id, opponent_team, opponent_coach, opponent_score, differential,
    win, draw, loss, position, points, wins, draws, losses, points_for,
    points_against, pcnt
    -> one row per team per round (both sides of each fixture appear separately).
  player_data.RDS (data.frame): feed_id, player_id, first_name, last_name,
    team_abbrev, pos_1, pos_2, round, projected_points, points, played, avg,
    avg3, avg5, price, user_team_id, team, coach, picked, type, position
    -> team_id (league_data) and user_team_id (player_data) are the same ID space.

NOT covered by this pass (missing/unreliable source data — see project memory):
  - transactions.csv (trades/waivers) — raw RDS dumps exist for some years but
    need a user_id -> user_team_id reconciliation not yet done.
  - draft CSVs — format/coverage varies by year (2024 has none at all).
  - finals bracket json — not available for these years.
Historical pages that depend on those (Acquisition detail, Draft Heat Map,
Finals bracket) will simply show reduced/no data for these years; this is
expected and handled gracefully by the website already.
"""
import sys
from pathlib import Path

import pandas as pd
import pyreadr

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_DIR = PROJECT_ROOT / "archive" / "legacy" / "fantasy_banter_data"
SEASONS_DIR = PROJECT_ROOT / "data" / "seasons"


def _read_rds(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    return list(result.values())[0]


def build_ladder(league: pd.DataFrame) -> pd.DataFrame:
    df = league.dropna(subset=["wins", "draws", "losses", "points_for", "points_against", "points", "position"]).copy()
    df["round"] = df["round"].astype(int)
    out = pd.DataFrame({
        "round":             df["round"],
        "coach_id":          df["team_id"].astype(int),
        "coach_team_id":     df["team_id"].astype(int),
        "coach_first_name":  df["coach"],
        "coach_team_name":   df["team"],
        "wins":              df["wins"].astype(int),
        "draws":             df["draws"].astype(int),
        "losses":             df["losses"].astype(int),
        "points_for":        df["points_for"],
        "points_against":    df["points_against"],
        "league_points":     df["points"].astype(int),
        "position":          df["position"].astype(int),
    })
    return out.drop_duplicates(subset=["round", "coach_first_name"], keep="first").reset_index(drop=True)


def build_fixtures(league: pd.DataFrame) -> pd.DataFrame:
    df = league.dropna(subset=["team_score", "opponent_score", "position"]).copy()
    df["round"] = df["round"].astype(int)
    out = pd.DataFrame({
        "round_number":              df["round"],
        "match_name":                df["team"] + " vs " + df["opponent_team"],
        "team_id":                   df["team_id"].astype(int),
        "team_name":                 df["team"],
        "team_coach":                df["coach"],
        "team_points":               df["team_score"],
        "opposition_team_id":        df["opponent_team_id"].astype(int),
        "opposition_team_name":      df["opponent_team"],
        "opposition_team_coach":     df["opponent_coach"],
        "opposition_team_points":    df["opponent_score"],
        "Rank":                      df["position"].astype(int),
    })
    return out.reset_index(drop=True)


def build_player_matches(player: pd.DataFrame) -> pd.DataFrame:
    df = player.dropna(subset=["round", "coach", "feed_id"]).copy()
    df["round"] = df["round"].astype(int)
    out = pd.DataFrame({
        "round_x":           df["round"],
        "coach_id":          df["user_team_id"].astype(int),
        "coach_team_id":     df["user_team_id"].astype(int),
        "coach_first_name":  df["coach"],
        "coach_team_name":   df["team"],
        "player_id":         df["player_id"],
        "played_position":   df["position"],
        "points_x":          df["points"],
        "on_field":          df["picked"].astype("boolean").fillna(False).astype(bool),
        "picked":            df["picked"].astype("boolean").fillna(False).astype(bool),
        "feed_id":           df["feed_id"],
        "first_name":        df["first_name"],
        "last_name":         df["last_name"],
        "team":              df["team_abbrev"],
        "position":          df["pos_1"],
    })
    return out.reset_index(drop=True)


def build_current_teams(player: pd.DataFrame) -> pd.DataFrame:
    df = player.dropna(subset=["round", "coach", "feed_id"]).copy()
    df["round"] = df["round"].astype(int)
    last_round = df["round"].max()
    last = df[df["round"] == last_round].drop_duplicates(subset=["user_team_id", "player_id"], keep="last")
    out = pd.DataFrame({
        "coach_first_name": last["coach"],
        "first_name":       last["first_name"],
        "last_name":        last["last_name"],
        "team":             last["team_abbrev"],
        "position":         last["pos_1"],
        "avg":              last["avg"],
        "avg3":             last["avg3"],
        "price":            last["price"],
        "feed_id":          last["feed_id"],
    })
    return out.reset_index(drop=True)


def build_coach_list(league: pd.DataFrame) -> pd.DataFrame:
    df = league.drop_duplicates(subset=["team_id"], keep="last")
    out = pd.DataFrame({
        "coach_team_id":    df["team_id"].astype(int),
        "coach_id":         df["team_id"].astype(int),
        "coach_first_name": df["coach"],
        "coach_team_name":  df["team"],
    })
    return out.reset_index(drop=True)


def ingest_year(year: int) -> None:
    src_dir = LEGACY_DIR / str(year)
    league_path = src_dir / "league_data.RDS"
    player_path = src_dir / "player_data.RDS"
    if not league_path.exists() or not player_path.exists():
        print(f"  [{year}] SKIP — league_data.RDS/player_data.RDS not found in {src_dir}")
        return

    league = _read_rds(league_path)
    player = _read_rds(player_path)

    out_dir = SEASONS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    ladder = build_ladder(league)
    n_good_rounds = ladder["round"].nunique()
    if n_good_rounds < 2:
        print(f"  [{year}] SKIP — only {n_good_rounds} round(s) of usable ladder data (source mostly NaN)")
        return

    fixtures = build_fixtures(league)
    matches = build_player_matches(player)
    teams = build_current_teams(player)
    coaches = build_coach_list(league)

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    matches.to_csv(out_dir / "player_match_results.csv", index=False)
    teams.to_csv(out_dir / "current_teams.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)

    print(f"  [{year}] OK — {n_good_rounds} rounds (max round {ladder['round'].max()}), "
          f"{len(matches)} player-match rows -> {out_dir}")


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2022, 2023, 2024, 2025]
    print(f"Ingesting seasons: {years}")
    for y in years:
        ingest_year(y)
