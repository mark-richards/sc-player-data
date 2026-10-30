"""
routes/records.py — Hall of Fame page (/records).
"""
from flask import Blueprint, render_template

from website.data.loader import load_league_master
from website.data.honour_board import build_honour_board
from website.data.hall_of_fame import (
    completed_years, build_trophy_cabinet, build_notable_records, build_notable_season_records,
    build_championship_drought, build_bridesmaid_tally, build_head_to_head, build_h2h_droughts,
    build_favourite_players, build_all_time_ladder,
)

bp = Blueprint("records", __name__)


@bp.get("/")
def records():
    master   = load_league_master()
    all_time_df, season_list = build_honour_board(master)

    years = completed_years()
    all_time_ladder = build_all_time_ladder(years)
    trophy_cabinet = build_trophy_cabinet(years)

    # Minor Premierships aren't in honour_board's own tally (they're derived
    # from live ladder data the same way Trophy Cabinet computes every other
    # award) — pull the per-coach counts already computed there instead of
    # re-deriving them.
    minor_prem_counts = next(
        (r["coach_counts"] for r in trophy_cabinet["by_award"] if r["key"] == "minor_premier"), {}
    )
    all_time_df["MinorPrems"] = all_time_df["coach"].map(minor_prem_counts).fillna(0).astype(int)
    notable_records = build_notable_records(years)
    notable_season_records = build_notable_season_records(years)
    drought = build_championship_drought(season_list)
    bridesmaid = build_bridesmaid_tally(season_list)
    head_to_head = build_head_to_head(years)
    h2h_droughts = build_h2h_droughts(years)
    favourite_players = build_favourite_players(years)

    return render_template(
        "records.html",
        all_time=all_time_df.to_dict(orient="records"),
        seasons=season_list,
        all_time_ladder=all_time_ladder,
        trophy_cabinet=trophy_cabinet,
        notable_records=notable_records,
        notable_season_records=notable_season_records,
        drought=drought,
        bridesmaid=bridesmaid,
        head_to_head=head_to_head,
        h2h_droughts=h2h_droughts,
        favourite_players=favourite_players,
    )
