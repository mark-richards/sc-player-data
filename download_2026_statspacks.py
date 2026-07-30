"""
Download SuperCoach player data for the current season.

Two endpoints:
  1. players-cf?embed=player_stats&round=N  — all players for one round (bulk, ~2 MB)
     Saves to data/raw/supercoach/{year}/players_round_{N}.json
     Used as the primary source for scores_2026 in the newsletter.

  2. completeStatspack?player_id={id}  — career history per player (historical, no 2026)
     Saves to data/raw/supercoach/{year}/completeStatspack_player_id={id}.json
     Used for historical metrics (Ceiling Index, CV, TPOR) via load_historical_scores().

Usage
-----
  # Download all rounds played so far (weekly, fast — ~10 requests):
  py download_2026_statspacks.py --year 2026 --rounds 9

  # Force re-download existing round files:
  py download_2026_statspacks.py --year 2026 --rounds 9 --force

  # Download historical statspacks (one-off, slow — 781 requests):
  py download_2026_statspacks.py --year 2026 --statspacks
"""

import argparse
import json
import logging
import os
import time
import urllib3
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Optional: auto-refresh the bearer token via Auth0 if credentials are available.
try:
    from waiver.league_api import get_valid_token as _get_valid_token
    _AUTO_REFRESH = True
except ImportError:
    _AUTO_REFRESH = False

# --- Configuration ---
_PLAYERS_ROUND_URL = (
    "https://www.supercoach.com.au/{year}/api/afl/draft/v1/players-cf"
    "?embed=player_stats&round={round_num}"
)
_STATSPACK_URL = (
    "https://www.supercoach.com.au/{year}/api/afl/draft/v1/completeStatspack"
    "?player_id={player_id}"
)
_PLAYER_LIST_TEMPLATE = "draft_prep/SC {year}/{year}_SC_Player_list.csv"
_OUTPUT_DIR_TEMPLATE = "data/raw/supercoach/{year}"
_RATE_LIMIT_DELAY = 0.3  # seconds between requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def download_player_rounds(
    year: int,
    token: str,
    rounds: list[int],
    output_dir: Path | None = None,
    force_refresh: bool = False,
) -> tuple[int, int, int]:
    """
    Download all-player stats for each round in `rounds` using the bulk endpoint.

    One request per round → all 794 players → saves to players_round_{N}.json.
    This is the primary source for scores_2026 in load_season_scores().

    Returns (downloaded, skipped, errors) counts.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if output_dir is None:
        output_dir = Path(_OUTPUT_DIR_TEMPLATE.format(year=year))
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    downloaded = skipped = errors = 0
    for round_num in rounds:
        output_file = output_dir / f"players_round_{round_num}.json"
        if not force_refresh and output_file.exists() and output_file.stat().st_size > 1000:
            skipped += 1
            continue

        url = _PLAYERS_ROUND_URL.format(year=year, round_num=round_num)
        try:
            resp = requests.get(url, headers=headers, verify=False, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                downloaded += 1
                log.info("  Round %d: %d players saved → %s", round_num, len(data), output_file.name)
            elif resp.status_code == 401:
                log.error("  Round %d — AUTH FAILED (401). Token may be expired.", round_num)
                errors += 1
                if errors >= 3:
                    log.error("Too many auth failures. Stopping.")
                    break
            else:
                log.warning("  Round %d — HTTP %d", round_num, resp.status_code)
                errors += 1
        except requests.exceptions.RequestException as e:
            log.warning("  Round %d — request error: %s", round_num, e)
            errors += 1

        time.sleep(0.5)

    log.info(
        "Player rounds download complete: %d downloaded, %d skipped, %d errors",
        downloaded, skipped, errors,
    )
    return downloaded, skipped, errors


def download_statspacks(
    year: int,
    token: str,
    output_dir: Path | None = None,
    player_list_path: Path | None = None,
    force_refresh: bool = False,
) -> tuple[int, int, int]:
    """
    Download completeStatspack JSON files for all players in the SC player list.

    Parameters
    ----------
    year              : AFL season year.
    token             : SC API Bearer token.
    output_dir        : Where to save JSON files. Defaults to data/raw/supercoach/{year}/.
    player_list_path  : Path to the SC player list CSV. Defaults to draft_prep/SC {year}/.
    force_refresh     : If True, re-download files that already exist (needed for weekly updates).

    Returns
    -------
    (downloaded, skipped, errors) counts.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if output_dir is None:
        output_dir = Path(_OUTPUT_DIR_TEMPLATE.format(year=year))
    if player_list_path is None:
        player_list_path = Path(_PLAYER_LIST_TEMPLATE.format(year=year))

    output_dir.mkdir(parents=True, exist_ok=True)

    if not player_list_path.exists():
        log.error(f"Player list not found: {player_list_path}")
        return 0, 0, 0

    df = pd.read_csv(player_list_path)
    player_ids = sorted(df["id"].unique())
    total = len(player_ids)
    log.info(f"Downloading statspacks for {total} players (year={year}, force_refresh={force_refresh})")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    downloaded = skipped = errors = 0
    error_ids: list[int] = []

    for i, player_id in enumerate(player_ids, start=1):
        output_file = output_dir / f"completeStatspack_player_id={player_id}.json"

        if not force_refresh and output_file.exists() and output_file.stat().st_size > 100:
            skipped += 1
            continue

        url = _API_URL_TEMPLATE.format(year=year, player_id=player_id)
        try:
            resp = requests.get(url, headers=headers, verify=False, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                downloaded += 1
                if downloaded % 50 == 0 or i == total:
                    log.info(
                        f"  [{i}/{total}] {downloaded} downloaded, "
                        f"{skipped} skipped, {errors} errors"
                    )
            elif resp.status_code == 401:
                log.error(f"  Player {player_id} — AUTH FAILED (401). Token may be expired.")
                errors += 1
                error_ids.append(player_id)
                if errors >= 3:
                    log.error("Too many auth failures. Token likely expired. Stopping.")
                    break
            else:
                log.warning(f"  Player {player_id} — HTTP {resp.status_code}")
                errors += 1
                error_ids.append(player_id)

        except requests.exceptions.RequestException as e:
            log.warning(f"  Player {player_id} — request error: {e}")
            errors += 1
            error_ids.append(player_id)

        time.sleep(_RATE_LIMIT_DELAY)

    log.info(
        f"Statspack download complete: {downloaded} downloaded, "
        f"{skipped} skipped, {errors} errors"
    )
    if error_ids:
        log.warning(f"Failed player IDs: {error_ids}")

    return downloaded, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Download SC player data.")
    parser.add_argument("--year", type=int, default=2026, help="AFL season year.")
    parser.add_argument(
        "--rounds", type=int, metavar="N",
        help="Download bulk player-stats for rounds 1..N (fast, ~N requests).",
    )
    parser.add_argument(
        "--statspacks", action="store_true",
        help="Download historical completeStatspacks (slow, ~781 requests).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()

    if not args.rounds and not args.statspacks:
        parser.error("Specify --rounds N (current season) or --statspacks (historical).")

    if _AUTO_REFRESH:
        token = _get_valid_token(year=args.year)
        if not token:
            log.error("Could not obtain a valid SC bearer token. Aborting.")
            return
    else:
        token = os.getenv("SC_API_TOKEN", "")
        if not token:
            log.error("SC_API_TOKEN not set in .env. Cannot authenticate.")
            return

    if args.rounds:
        download_player_rounds(
            year=args.year,
            token=token,
            rounds=list(range(1, args.rounds + 1)),
            force_refresh=args.force,
        )

    if args.statspacks:
        download_statspacks(year=args.year, token=token, force_refresh=args.force)


if __name__ == "__main__":
    main()
