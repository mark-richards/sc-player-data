"""
routes/power_rankings.py — Power Rankings page (/power-rankings and /power-rankings/<year>).
"""
import plotly.io as pio

from flask import Blueprint, render_template

from website.config import COACH_PORTRAITS, SEASON_YEAR
from website.data.loader import load_fixtures, load_player_matches, load_fanfooty_season
from website.data.power_rankings import (
    build_power_rankings_chart,
    build_power_rankings_table,
    build_player_rankings,
)

bp = Blueprint("power_rankings", __name__)


@bp.get("/")
@bp.get("/<int:year>")
def power_rankings(year: int | None = None):
    year = year or SEASON_YEAR
    is_current_season = (year == SEASON_YEAR)

    fixtures = load_fixtures(year)
    players  = load_player_matches(year)
    fanfooty = load_fanfooty_season(year)

    # Drop the current round if it hasn't fully completed yet (only relevant
    # for the season still in progress — a past season is always all-complete)
    if is_current_season and not fixtures.empty:
        from waiver.fixture_schedule import round_is_complete
        latest_round = int(fixtures["round_number"].max())
        if not round_is_complete(latest_round):
            fixtures = fixtures[fixtures["round_number"] < latest_round]
            if not players.empty:
                players = players[players["round_x"] < latest_round]

    pr_chart = pio.to_json(build_power_rankings_chart(fixtures, players, COACH_PORTRAITS))
    pr_table = build_power_rankings_table(fixtures, players, fanfooty)

    rankings = {
        "overall": build_player_rankings(players),
        "DEF":     build_player_rankings(players, "DEF"),
        "MID":     build_player_rankings(players, "MID"),
        "RUC":     build_player_rankings(players, "RUC"),
        "FWD":     build_player_rankings(players, "FWD"),
    }

    return render_template(
        "power_rankings.html",
        pr_chart=pr_chart,
        pr_table=pr_table,
        rankings=rankings,
        season_year=year,
        is_current_season=is_current_season,
    )
