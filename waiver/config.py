"""
config.py — Central configuration for the Waiver Newsletter system.
Reads secrets from a .env file; all other constants are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above this package)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── API / Auth ─────────────────────────────────────────────────────────────
SC_API_TOKEN: str = os.getenv("SC_API_TOKEN", "")
# Auth0 credentials for automatic bearer-token refresh (from secrets.R equivalents).
# SC_CLIENT_ID  : Auth0 client_id   (the `cid`  value from secrets.R)
# SC_SOCIAL_TOKEN: Auth0 social token (the `tkn`  value from secrets.R)
# If both are set, token is refreshed automatically on each pipeline run.
# If only SC_API_TOKEN is set, it is used directly (manual refresh required).
SC_CLIENT_ID: str = os.getenv("SC_CLIENT_ID", "")
SC_SOCIAL_TOKEN: str = os.getenv("SC_SOCIAL_TOKEN", "")
SC_REFRESH_TOKEN: str = os.getenv("SC_REFRESH_TOKEN", "")
# LEAGUE_ID: draft league — used for playersStatus (player ownership / waiver status)
LEAGUE_ID: str = os.getenv("LEAGUE_ID", "4199")
# LADDER_LEAGUE_ID: in-season scoring league — used for ladderAndFixtures (scores, standings)
LADDER_LEAGUE_ID: str = os.getenv("LADDER_LEAGUE_ID", "14889")
SEASON_YEAR: int = int(os.getenv("SEASON_YEAR", "2026"))
API_URL_TEMPLATE: str = (
    "https://www.supercoach.com.au/{year}/api/afl/draft/v1"
    "/leagues/{league_id}/playersStatus"
)

# ── League structure ───────────────────────────────────────────────────────
N_TEAMS: int = 8
# Starters per team (on-field): 5 DEF + 7 MID + 2 RUC + 5 FWD = 19
SLOTS: dict[str, int] = {"DEF": 5, "MID": 7, "RUC": 2, "FWD": 5}
# Bench: 3 regular + 1 emergency = 4 total
BENCH_SIZE: int = 4
# No lowest-score drop (emergencies fill gaps; all 19 on-field always score)
DROP_N: int = 0
CEILING_THRESHOLD: int = 120  # Score above which a game counts as a "ceiling" game
# Rounds where AFL multi-byes apply → Best 16 of 19 rule
BYE_ROUNDS: list[int] = [12, 13, 14, 15, 16]

# ── File paths ─────────────────────────────────────────────────────────────
PLAYER_LIST_PATH: Path = (
    _PROJECT_ROOT / f"draft_prep/SC {SEASON_YEAR}" / f"{SEASON_YEAR}_SC_Player_list.csv"
)
PLAYER_STATS_PATH: Path = _PROJECT_ROOT / "data" / "live" / "player_stats_current.csv"
FANFOOTY_PROCESSED_DIR: Path = _PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR: Path = _PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR: Path = _PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Historical data weighting (for Ceiling Index / CV) ────────────────────
# More recent seasons weighted higher when computing historical metrics.
SEASON_WEIGHTS: dict[int, float] = {
    2026: 4.0,
    2025: 3.0,
    2024: 2.0,
    2023: 1.5,
    2022: 1.0,
}
DEFAULT_SEASON_WEIGHT: float = 0.5  # For seasons older than 2022
# Minimum round samples before CI/CV are considered reliable
MIN_GAMES_FOR_METRICS: int = 5
# Use 2026 season data as primary once this many rounds have been played
CURRENT_SEASON_MIN_ROUNDS: int = 5
# How many recent fanfooty rounds to scan for Role Watch
ROLE_WATCH_ROUNDS: int = 3

# ── Email delivery ─────────────────────────────────────────────────────────
# Outlook SMTP (smtp-mail.outlook.com:587). Leave EMAIL_PASSWORD blank to disable.
# If 2FA is enabled, use an app password: account.microsoft.com → Security → App passwords.
EMAIL_SENDER: str = os.getenv("EMAIL_SENDER", "richardsmark@outlook.com")
EMAIL_RECIPIENT: str = os.getenv("EMAIL_RECIPIENT", "richardsmark@outlook.com")
EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")

# ── Jumper / Streamer scoring ─────────────────────────────────────────────
# Minimum rounds played in current season before jumper scoring activates
JUMPER_MIN_ROUNDS: int = 5
# Min games a player needs this season before being scored as a jumper candidate
JUMPER_MIN_GAMES: int = 5
# Maximum season avg for jumper/streamer candidates (only target underperformers)
# Set to 85 to capture the 80-85 "dead zone" players who are rising but not yet caught
# by Positional Targets (the Peter Wright / Josh Battle profile).
JUMPER_AVG_MAX: float = 85.0
# Jumper composite score threshold to appear in newsletter (0–1 scale)
JUMPER_SCORE_THRESHOLD: float = 0.55
# Minimum career avg (pre-2026) for a jumper candidate to be realistic for 85+ rest-of-season
JUMPER_MIN_CAREER_AVG: float = 75.0
# Minimum last-3-game avg for any candidate (jumper OR streamer) to be realistic for 85+ soon
JUMPER_MIN_AVG3: float = 75.0
# Minimum last-3-game avg specifically for streamers (higher bar — must be nearly there)
STREAMER_MIN_AVG3: float = 80.0
# Minimum streamer signal count to appear in newsletter
STREAMER_SIGNAL_THRESHOLD: int = 2
# Top N shown in each section
JUMPER_TOP_N: int = 5
STREAMER_TOP_N: int = 5

# ── My team identity (from SC in-season league API) ───────────────────────
# MY_TEAM_ID: user_team_id in LADDER_LEAGUE_ID (14889).
# On-field/bench status is derived live from the statsPlayers API — no
# hardcoded bench list needed.
MY_TEAM_ID: int = int(os.getenv("MY_TEAM_ID", "0"))
