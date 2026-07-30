"""
routes/records.py — Honour board page (/records).
"""
from flask import Blueprint, render_template

from website.data.loader import load_league_master
from website.data.honour_board import build_honour_board

bp = Blueprint("records", __name__)


@bp.get("/")
def records():
    master   = load_league_master()
    all_time_df, season_list = build_honour_board(master)

    return render_template(
        "records.html",
        all_time=all_time_df.to_dict(orient="records"),
        seasons=season_list,
    )
