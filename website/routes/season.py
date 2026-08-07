"""
routes/season.py — Season summary page (/).
"""
import json

from flask import Blueprint, render_template

from website.data.loader import load_ladder, load_fixtures, load_player_matches, load_coach_portraits, load_fixture_schedule, load_historical_scores
from website.data.standings import compute_standings, get_current_round
from website.data.charts import ladder_journey_chart, score_boxplot, score_scatter
from website.data.positional_ratings import compute_positional_ratings
from website.data.simulation import run_monte_carlo
from website.data.fixture_strength import build_fixture_strength, build_monte_carlo_schedule_sim
from waiver.fixture_schedule import round_is_complete

bp = Blueprint("season", __name__)


@bp.get("/")
def season_summary():
    ladder   = load_ladder()
    fixtures = load_fixtures()
    players  = load_player_matches()
    portraits = load_coach_portraits()

    current_round = get_current_round(ladder)
    if not round_is_complete(current_round):
        current_round = max(1, current_round - 1)
    ladder   = ladder[ladder["round"] <= current_round]
    fixtures = fixtures[fixtures["round_number"] <= current_round]
    standings = compute_standings(ladder, current_round)
    pos_ratings = compute_positional_ratings(players, current_round)
    schedule = load_fixture_schedule()
    historical_scores = load_historical_scores()
    sim_results = run_monte_carlo(
        fixtures, ladder, current_round,
        schedule_df=schedule,
        historical_scores=historical_scores,
    )

    # Merge positional ratings and simulation into standings
    standings = standings.merge(pos_ratings, on="coach_first_name", how="left")
    standings = standings.merge(sim_results, on="coach_first_name", how="left")

    # Fill any missing simulation values
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

    return render_template(
        "season_summary.html",
        standings=standings.to_dict(orient="records"),
        ladder_chart=json.dumps(ladder_journey_chart(ladder)),
        boxplot_chart=json.dumps(score_boxplot(fixtures)),
        scatter_chart=json.dumps(score_scatter(fixtures, portraits)),
        current_round=current_round,
        season_year=2026,
        fixture_strength=fixture_strength,
        monte_carlo_sim=monte_carlo_sim,
    )
