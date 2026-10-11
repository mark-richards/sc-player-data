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
SEASONS_DIR           = PROJECT_ROOT / "data" / "seasons"

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

# Fallback coach ordering — used only when a season's own coach_list/ladder data
# is unavailable (see loader.list_season_coaches).
COACH_ORDER = ["Mark", "Simon", "Luke", "Lester", "Paul", "Jordan", "Anthony", "James"]

# Draft alias → coach first name mapping (used by Draft Heat Map page)
# Confirmed: RICHO, LESTER, PMAC, JMERC, CHIEF, KAPPAZ from cross-referencing draft CSVs.
# MELONS / GARTER → Jordan / Simon — update if incorrect.
# Identity entries (Mark->Mark etc.) let ingested historical draft CSVs use real
# coach names directly instead of guessing which year used which nickname.
DRAFT_ALIAS_MAP = {
    "RICHO":  "Mark",
    "LESTER": "Lester",
    "PMAC":   "Paul",
    "GARTER": "James",
    "MELONS": "Anthony",
    "JMERC":  "Jordan",
    "CHIEF":  "Simon",
    "KAPPAZ": "Luke",
    **{name: name for name in COACH_ORDER},
}

SC_CURRENT_CSV        = PROCESSED_DATA_DIR / "master_player_data.csv"


# ── Per-season path resolution ──────────────────────────────────────────────
# The current SEASON_YEAR is served live from data/live/ (auto-refreshed each
# pipeline run). Any other year is read from a frozen data/seasons/{year}/
# snapshot in the same file shape, populated by a separate ingestion process.

def _season_dir(year: int) -> Path:
    return DATA_LIVE_DIR if year == SEASON_YEAR else SEASONS_DIR / str(year)


def ladder_csv_path(year: int) -> Path:
    return _season_dir(year) / "ladder.csv"


def fixture_team_csv_path(year: int) -> Path:
    return _season_dir(year) / "fixture_results_by_team.csv"


def player_match_csv_path(year: int) -> Path:
    return _season_dir(year) / "player_match_results.csv"


def current_teams_csv_path(year: int) -> Path:
    return _season_dir(year) / "current_teams.csv"


def transactions_csv_path(year: int) -> Path:
    return _season_dir(year) / "transactions.csv"


def coach_list_csv_path(year: int) -> Path:
    return _season_dir(year) / "coach_list.csv"


def fixture_schedule_csv_path(year: int) -> Path:
    return _season_dir(year) / f"fixture_schedule_{year}.csv"


def finals_json_dir(year: int) -> Path:
    return _season_dir(year) / "json"


def draft_csv_path(year: int) -> Path:
    return PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"draft_{year}_result.csv"


def player_list_csv_path(year: int) -> Path:
    return PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"{year}_SC_Player_list.csv"


def processed_waivers_json_path(year: int) -> Path | None:
    d = PROJECT_ROOT / "data" / "raw" / "supercoach" / str(year)
    found = next(iter(sorted(d.glob("*_processedWaivers.json"))), None)
    if found is not None:
        return found
    # Years ingested from archive/legacy/fantasy_banter_data (see
    # ingest_historical_season.py) write their sidecar into data/seasons/
    # instead, to keep generated files out of the tracked archive/ tree.
    seasons_json = SEASONS_DIR / str(year) / "processedWaivers.json"
    return seasons_json if seasons_json.exists() else None


# ── Per-season league-structure overrides ───────────────────────────────────
# Most seasons share the same structure (8 teams, 21 regular rounds, top-4
# finals). Override here if a past season is found to differ once ingested.
_SEASON_META_DEFAULTS = {
    "total_regular_rounds": TOTAL_REGULAR_ROUNDS,
    "finals_top_n":         FINALS_TOP_N,
    "n_teams":              N_TEAMS,
    "finals_weeks":         3,  # 3 = McIntyre (QF+EF / PF / GF), 2 = straight semis -> GF
}
SEASON_META_OVERRIDES: dict[int, dict] = {
    # 2022's finals series was only 2 weeks (semis -> GF directly, no
    # preliminary final) — confirmed by hand: round 24 has no data anywhere
    # (not even in an independent backup), round 23's single fixture winner/
    # loser match honour_board.py's recorded champion/runner-up exactly.
    2022: {"finals_weeks": 2},
    # 2016-2019: confirmed directly from the archived fixture data (FINALS
    # FLAG column) — every year's finals are rounds 22-23 only, never 24.
    2016: {"finals_weeks": 2},
    2017: {"finals_weeks": 2},
    2018: {"finals_weeks": 2},
    2019: {"finals_weeks": 2},
    # 2020: COVID-shortened AFL season — only 16 regular rounds, confirmed
    # from the archived data (round 17 = 2 semis, round 18 = GF, Mark beat
    # Simon 1758-1700 matching honour_board.py exactly).
    2020: {"total_regular_rounds": 16, "finals_weeks": 2},
}


def season_meta(year: int) -> dict:
    return {**_SEASON_META_DEFAULTS, **SEASON_META_OVERRIDES.get(year, {})}
