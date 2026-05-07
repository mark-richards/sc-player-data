"""
routes/acquisition.py — Acquisition Source Analysis page (/acquisition).
"""
from flask import Blueprint, render_template

from website.config import (
    DRAFT_CSV_2026, DRAFT_ALIAS_MAP, COACH_ORDER,
    TRANSACTIONS_CSV, COACH_LIST_CSV, PLAYER_MATCH_CSV, PROCESSED_WAIVERS_JSON,
)
from website.data.acquisition import (
    load_player_sources,
    build_acquisition_stats,
    get_acquisition_kings,
    build_acquisition_heatmap,
)

bp = Blueprint("acquisition", __name__)


@bp.get("/")
def acquisition():
    player_sources = load_player_sources(
        draft_csv=DRAFT_CSV_2026,
        transactions_csv=TRANSACTIONS_CSV,
        coach_list_csv=COACH_LIST_CSV,
        player_match_csv=PLAYER_MATCH_CSV,
        draft_alias_map=DRAFT_ALIAS_MAP,
        processed_waivers_json=PROCESSED_WAIVERS_JSON,
    )

    stats       = build_acquisition_stats(player_sources, PLAYER_MATCH_CSV)
    kings       = get_acquisition_kings(stats)
    heatmap     = build_acquisition_heatmap(stats, COACH_ORDER)

    return render_template(
        "acquisition.html",
        kings=kings,
        heatmap=heatmap,
        categories=["Draft", "Waiver", "FA", "Trade"],
    )
