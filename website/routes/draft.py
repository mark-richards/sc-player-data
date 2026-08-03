"""
routes/draft.py — Draft Heat Map page (/draft-heat-map).
"""
from flask import Blueprint, render_template

from website.config import DRAFT_CSV_2026, PLAYER_LIST_CSV_2026, DRAFT_ALIAS_MAP, COACH_ORDER
from website.data.loader import load_sc_current, load_sc_round_files
from website.data.draft_data import load_draft_board

bp = Blueprint("draft", __name__)


@bp.get("/")
def draft_heat_map():
    sc_df = load_sc_current()
    sc_round_df = load_sc_round_files()
    summary, board, n_rounds, coaches = load_draft_board(
        DRAFT_CSV_2026, PLAYER_LIST_CSV_2026, DRAFT_ALIAS_MAP, COACH_ORDER, sc_df, sc_round_df
    )

    return render_template(
        "draft.html",
        coach_summary=summary,
        board=board,
        coaches=coaches,
        n_rounds=n_rounds,
    )
