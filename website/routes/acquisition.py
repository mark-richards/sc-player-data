"""
routes/acquisition.py — Acquisition Source Analysis page (/acquisition and /acquisition/<year>).
"""
from flask import Blueprint, render_template

from website.config import (
    SEASON_YEAR, DRAFT_ALIAS_MAP,
    draft_csv_path, transactions_csv_path, coach_list_csv_path,
    player_match_csv_path, processed_waivers_json_path,
)
from website.data.loader import list_season_coaches
from website.data.acquisition import (
    load_player_sources,
    build_acquisition_stats,
    get_acquisition_kings,
    build_acquisition_heatmap,
)

bp = Blueprint("acquisition", __name__)


@bp.get("/")
@bp.get("/<int:year>")
def acquisition(year: int | None = None):
    year = year or SEASON_YEAR
    player_match_csv = player_match_csv_path(year)

    player_sources = load_player_sources(
        draft_csv=draft_csv_path(year),
        transactions_csv=transactions_csv_path(year),
        coach_list_csv=coach_list_csv_path(year),
        player_match_csv=player_match_csv,
        draft_alias_map=DRAFT_ALIAS_MAP,
        processed_waivers_json=processed_waivers_json_path(year),
    )

    stats       = build_acquisition_stats(player_sources, player_match_csv)
    kings       = get_acquisition_kings(stats)
    heatmap     = build_acquisition_heatmap(stats, list_season_coaches(year))

    return render_template(
        "acquisition.html",
        kings=kings,
        heatmap=heatmap,
        categories=["Draft", "Waiver", "FA", "Trade"],
        season_year=year,
        is_current_season=(year == SEASON_YEAR),
    )
