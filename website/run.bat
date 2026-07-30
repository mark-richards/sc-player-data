@echo off
cd /d "g:\My Drive\projects\sc-player-data"
call .venv\Scripts\activate.bat
echo Refreshing data to latest round...
python -c "from website.data.api_ingest import refresh_2026_data; refresh_2026_data()"
python -c "from build_master_dataset import update_current_season; update_current_season(year=2026)"
python -c "import sys; sys.path.insert(0,'.'); from run_waiver_newsletter import _detect_round, _ensure_fanfooty_rounds, validate_round_data; from waiver.config import FANFOOTY_PROCESSED_DIR, SEASON_YEAR; rnd = _detect_round(FANFOOTY_PROCESSED_DIR) or 0; _ensure_fanfooty_rounds(SEASON_YEAR, rnd + 1, probe_round=rnd + 1); rnd = _detect_round(FANFOOTY_PROCESSED_DIR) or rnd; validate_round_data(SEASON_YEAR, rnd) if rnd else None"
python -m flask --app website/app.py run --port 5000 --debug
pause
