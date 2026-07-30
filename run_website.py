import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from website.data.api_ingest import refresh_2026_data
from build_master_dataset import update_current_season
from run_waiver_newsletter import _detect_round, _ensure_fanfooty_rounds, validate_round_data
from waiver.config import FANFOOTY_PROCESSED_DIR, SEASON_YEAR
from website.app import create_app

print("Refreshing data to latest round...")
refresh_2026_data()
update_current_season(year=2026)

rnd = _detect_round(FANFOOTY_PROCESSED_DIR) or 0
_ensure_fanfooty_rounds(SEASON_YEAR, rnd + 1)
rnd = _detect_round(FANFOOTY_PROCESSED_DIR) or rnd
if rnd:
    validate_round_data(SEASON_YEAR, rnd)

app = create_app()
app.run(port=5000, debug=True)
