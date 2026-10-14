"""
routes/head_to_head.py — Rivalry drill-down page (/head-to-head/<coach_a>/<coach_b>).
"""
from flask import Blueprint, abort, render_template

from website.config import COACH_ORDER
from website.data.hall_of_fame import completed_years, build_h2h_games

bp = Blueprint("head_to_head", __name__)


@bp.get("/<coach_a>/<coach_b>")
def detail(coach_a: str, coach_b: str):
    if coach_a not in COACH_ORDER or coach_b not in COACH_ORDER or coach_a == coach_b:
        abort(404)

    years = completed_years()
    games = build_h2h_games(years, coach_a, coach_b)

    wins_a = sum(1 for g in games if g["winner"] == coach_a)
    wins_b = sum(1 for g in games if g["winner"] == coach_b)
    draws  = sum(1 for g in games if g["winner"] is None)

    return render_template(
        "head_to_head_detail.html",
        coach_a=coach_a,
        coach_b=coach_b,
        games=games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
    )
