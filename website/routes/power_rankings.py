"""
routes/power_rankings.py — Power Rankings page (/power-rankings).
"""
import plotly.io as pio

from flask import Blueprint, render_template

from website.config import COACH_PORTRAITS
from website.data.loader import load_fixtures, load_player_matches, load_fanfooty_season
from website.data.power_rankings import (
    build_power_rankings_chart,
    build_power_rankings_table,
    build_player_rankings,
)

bp = Blueprint("power_rankings", __name__)


@bp.get("/")
def power_rankings():
    from waiver.fixture_schedule import round_is_complete

    fixtures = load_fixtures()
    players  = load_player_matches()
    fanfooty = load_fanfooty_season()

    # Drop the current round if it hasn't fully completed yet
    if not fixtures.empty:
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
    )
