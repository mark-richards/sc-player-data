"""
routes/draft.py — Draft Heat Map page (/draft-heat-map and /draft-heat-map/<year>).
"""
from flask import Blueprint, render_template

from website.config import SEASON_YEAR, DRAFT_ALIAS_MAP, draft_csv_path, player_list_csv_path
from website.data.loader import load_sc_current, load_sc_round_files, list_season_coaches
from website.data.draft_data import load_draft_board, get_pick_value_extremes, build_round_value_baseline
from website.data.awards import EARLY_LATE_SPLIT_ROUND, draft_pick_award_cards

bp = Blueprint("draft", __name__)


@bp.get("/")
@bp.get("/<int:year>")
def draft_heat_map(year: int | None = None):
    year = year or SEASON_YEAR

    sc_df = load_sc_current(year)
    sc_round_df = load_sc_round_files(year)
    draft_csv, player_list_csv = draft_csv_path(year), player_list_csv_path(year)
    summary, board, n_rounds, coaches = load_draft_board(
        draft_csv, player_list_csv, DRAFT_ALIAS_MAP,
        list_season_coaches(year), sc_df, sc_round_df, year,
    )

    round_baseline = build_round_value_baseline(DRAFT_ALIAS_MAP)
    extremes = get_pick_value_extremes(draft_csv, player_list_csv, DRAFT_ALIAS_MAP, sc_df, sc_round_df,
                                        early_max_round=EARLY_LATE_SPLIT_ROUND,
                                        round_baseline=round_baseline, year=year)
    award_cards = draft_pick_award_cards(extremes)
    award_picks = {card["pick"]: card for card in award_cards}

    return render_template(
        "draft.html",
        coach_summary=summary,
        board=board,
        coaches=coaches,
        n_rounds=n_rounds,
        season_year=year,
        is_current_season=(year == SEASON_YEAR),
        award_picks=award_picks,
        award_cards=award_cards,
    )
