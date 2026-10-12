"""
routes/records.py — Hall of Fame page (/records).
"""
from flask import Blueprint, render_template

from website.data.loader import load_league_master
from website.data.honour_board import build_honour_board
from website.data.hall_of_fame import (
    completed_years, build_trophy_cabinet, build_notable_records,
    build_championship_drought, build_bridesmaid_tally, build_head_to_head,
    build_favourite_players,
)

bp = Blueprint("records", __name__)


@bp.get("/")
def records():
    master   = load_league_master()
    all_time_df, season_list = build_honour_board(master)

    years = completed_years()
    trophy_cabinet = build_trophy_cabinet(years)
    notable_records = build_notable_records(years)
    drought = build_championship_drought(season_list)
    bridesmaid = build_bridesmaid_tally(season_list)
    head_to_head = build_head_to_head(years)
    favourite_players = build_favourite_players(years)

    return render_template(
        "records.html",
        all_time=all_time_df.to_dict(orient="records"),
        seasons=season_list,
        trophy_cabinet=trophy_cabinet,
        notable_records=notable_records,
        drought=drought,
        bridesmaid=bridesmaid,
        head_to_head=head_to_head,
        favourite_players=favourite_players,
    )
