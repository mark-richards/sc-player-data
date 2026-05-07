"""
config.py — Website-specific configuration.
Paths are absolute so the app works regardless of cwd.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Data source paths ──────────────────────────────────────────────────────
DATA_LIVE_DIR        = PROJECT_ROOT / "data" / "live"
PROCESSED_DATA_DIR   = PROJECT_ROOT / "data" / "processed"

LADDER_CSV           = DATA_LIVE_DIR / "ladder.csv"
FIXTURE_CSV          = DATA_LIVE_DIR / "fixture_results.csv"
FIXTURE_TEAM_CSV     = DATA_LIVE_DIR / "fixture_results_by_team.csv"
FIXTURE_SCHEDULE_CSV = DATA_LIVE_DIR / "fixture_schedule_2026.csv"
PLAYER_MATCH_CSV     = DATA_LIVE_DIR / "player_match_results.csv"
CURRENT_TEAMS_CSV    = DATA_LIVE_DIR / "current_teams.csv"
LEAGUE_MASTER_CSV    = PROCESSED_DATA_DIR / "league_master.csv"

COACH_IMAGES_DIR     = Path(__file__).resolve().parent / "static" / "images" / "coaches"

# ── Season settings ────────────────────────────────────────────────────────
SEASON_YEAR: int       = int(os.getenv("SEASON_YEAR", "2026"))
TOTAL_REGULAR_ROUNDS   = 21
FINALS_TOP_N           = 4
N_TEAMS                = 8
SIMULATION_RUNS        = 10_000
POSITIONAL_WINDOW      = 7   # rolling window for positional ratings
SCORE_FLOOR            = 1200  # minimum realistic team score for simulation

# ── Flask cache ────────────────────────────────────────────────────────────
CACHE_TTL = 300  # seconds

# ── Coach name → portrait filename mapping ─────────────────────────────────
COACH_PORTRAITS = {
    "Anthony": "Anthony_2022.png",
    "James":   "James_2022.png",
    "Jordan":  "Jordan_2022.png",
    "Lester":  "Lester_2022.png",
    "Luke":    "Luke_2022.png",
    "Mark":    "Mark_2022.png",
    "Paul":    "Paul_2022.png",
    "Simon":   "Simon_2022.png",
}

COACH_ORDER = ["Mark", "Simon", "Luke", "Lester", "Paul", "Jordan", "Anthony", "James"]

# Draft alias → coach first name mapping (used by Draft Heat Map page)
# Confirmed: RICHO, LESTER, PMAC, JMERC, CHIEF, KAPPAZ from cross-referencing draft CSVs.
# MELONS / GARTER → Jordan / Simon — update if incorrect.
DRAFT_ALIAS_MAP = {
    "RICHO":  "Mark",
    "LESTER": "Lester",
    "PMAC":   "Paul",
    "GARTER": "James",
    "MELONS": "Anthony",
    "JMERC":  "Jordan",
    "CHIEF":  "Simon",
    "KAPPAZ": "Luke",
}

DRAFT_CSV_2026        = PROJECT_ROOT / "draft_prep" / "SC 2026" / "draft_2026_result.csv"
PLAYER_LIST_CSV_2026  = PROJECT_ROOT / "draft_prep" / "SC 2026" / "2026_SC_Player_list.csv"
SC_CURRENT_CSV        = PROCESSED_DATA_DIR / "master_player_data.csv"
TRANSACTIONS_CSV      = DATA_LIVE_DIR / "transactions.csv"
COACH_LIST_CSV        = DATA_LIVE_DIR / "coach_list.csv"

_raw_sc_dir           = PROJECT_ROOT / "data" / "raw" / "supercoach" / str(SEASON_YEAR)
PROCESSED_WAIVERS_JSON = next(iter(sorted(_raw_sc_dir.glob("*_processedWaivers.json"))), None)
