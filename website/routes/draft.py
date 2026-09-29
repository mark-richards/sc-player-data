"""
routes/draft.py — Draft Heat Map page (/draft-heat-map and /draft-heat-map/<year>).
"""
from flask import Blueprint, render_template

from website.config import SEASON_YEAR, DRAFT_ALIAS_MAP, draft_csv_path, player_list_csv_path
from website.data.loader import load_sc_current, load_sc_round_files, list_season_coaches
from website.data.draft_data import load_draft_board

bp = Blueprint("draft", __name__)


@bp.get("/")
@bp.get("/<int:year>")
def draft_heat_map(year: int | None = None):
    year = year or SEASON_YEAR

    sc_df = load_sc_current(year)
    sc_round_df = load_sc_round_files(year)
    summary, board, n_rounds, coaches = load_draft_board(
        draft_csv_path(year), player_list_csv_path(year), DRAFT_ALIAS_MAP,
        list_season_coaches(year), sc_df, sc_round_df,
    )

    return render_template(
        "draft.html",
        coach_summary=summary,
        board=board,
        coaches=coaches,
        n_rounds=n_rounds,
        season_year=year,
        is_current_season=(year == SEASON_YEAR),
    )
