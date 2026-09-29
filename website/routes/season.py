"""
routes/season.py — Season summary page (/ and /season/<year>).
"""
import json

from flask import Blueprint, render_template

from website.config import SEASON_YEAR
from website.data.loader import (
    load_ladder, load_fixtures, load_player_matches, load_coach_portraits,
    load_fixture_schedule, load_historical_scores,
)
from website.data.standings import compute_standings, get_current_round
from website.data.charts import ladder_journey_chart, score_boxplot, score_scatter
from website.data.positional_ratings import compute_positional_ratings
from website.data.simulation import run_monte_carlo
from website.data.fixture_strength import build_fixture_strength, build_monte_carlo_schedule_sim
from website.data.finals import build_finals_bracket
from website.data.honour_board import get_season_result

bp = Blueprint("season", __name__)


@bp.get("/")
@bp.get("/season/<int:year>")
def season_summary(year: int | None = None):
    year = year or SEASON_YEAR
    is_current_season = (year == SEASON_YEAR)

    ladder    = load_ladder(year)
    fixtures  = load_fixtures(year)
    players   = load_player_matches(year)
    portraits = load_coach_portraits()

    current_round = get_current_round(ladder)
    if is_current_season:
        from waiver.fixture_schedule import round_is_complete
        if not round_is_complete(current_round):
            current_round = max(1, current_round - 1)
    if not ladder.empty:
        ladder = ladder[ladder["round"] <= current_round]
    if not fixtures.empty:
        fixtures = fixtures[fixtures["round_number"] <= current_round]
    standings = compute_standings(ladder, current_round)
    pos_ratings = compute_positional_ratings(players, current_round)
    standings = standings.merge(pos_ratings, on="coach_first_name", how="left")

    # Forward-looking premiership simulation only makes sense for the season
    # still in progress — a completed season has an actual result instead.
    if is_current_season:
        schedule = load_fixture_schedule(year)
        historical_scores = load_historical_scores()
        sim_results = run_monte_carlo(
            fixtures, ladder, current_round,
            schedule_df=schedule,
            historical_scores=historical_scores,
        )
        standings = standings.merge(sim_results, on="coach_first_name", how="left")

    for col in ["finals_pct", "gf_pct", "champ_pct", "spoon_pct"]:
        if col not in standings.columns:
            standings[col] = 0.0

    try:
        fixture_strength = build_fixture_strength(fixtures)
    except Exception:
        fixture_strength = []

    try:
        monte_carlo_sim = build_monte_carlo_schedule_sim(fixtures)
    except Exception:
        monte_carlo_sim = []

    try:
        finals_bracket = build_finals_bracket(load_ladder(year), year=year)
    except Exception:
        finals_bracket = None

    season_result = None if is_current_season else get_season_result(year)

    return render_template(
        "season_summary.html",
        standings=standings.to_dict(orient="records"),
        ladder_chart=json.dumps(ladder_journey_chart(ladder)),
        boxplot_chart=json.dumps(score_boxplot(fixtures)),
        scatter_chart=json.dumps(score_scatter(fixtures, portraits)),
        current_round=current_round,
        season_year=year,
        is_current_season=is_current_season,
        fixture_strength=fixture_strength,
        monte_carlo_sim=monte_carlo_sim,
        finals_bracket=finals_bracket,
        season_result=season_result,
    )
