"""
Ingest DFS Australia AFL Statistics
====================================
Fetches per-player CBA%, kick-in counts/PO%, and ruck contest counts
from DFS Australia WordPress AJAX JSON endpoints.

Player ID mapping: "CD_I992242" -> feed_id 992242 (strip "CD_I" prefix).

Output: data/raw/dfsaustralia/{year}/
  cbas_{year}.csv          — feed_id, round_num, cba_pct
  kickins_{year}.csv       — feed_id, round_num, kickin_count, kickin_po_pct
  ruck_contests_{year}.csv — feed_id, round_num, ruck_contest_count

Usage:
  py ingest_dfsaustralia.py --year 2026
  py ingest_dfsaustralia.py --years 2021-2026
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

AJAX_URL = "https://dfsaustralia.com/wp-admin/admin-ajax.php"
TEAMS = [
    "ADE", "BRL", "CAR", "COL", "ESS", "FRE", "GCS", "GEE",
    "GWS", "HAW", "MEL", "NTH", "PTA", "RIC", "STK", "SYD", "WBD", "WCE",
]
OUTPUT_DIR = Path("data/raw/dfsaustralia")
SLEEP = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def post_ajax(action: str, team: str, season: int) -> dict:
    resp = requests.post(
        AJAX_URL,
        data={"action": action, "team": team, "season": season},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_cbas(year: int) -> pd.DataFrame:
    rows = []
    for team in TEAMS:
        logging.info(f"  CBA: {team} {year}")
        try:
            payload = post_ajax("afl_cbas_call_new_mysql", team, year)
        except Exception as e:
            logging.warning(f"  CBA fetch failed {team} {year}: {e}")
            time.sleep(SLEEP)
            continue

        games = payload.get("games", [])
        for player in payload.get("cbas", []):
            pid = player.get("playerId") or ""
            if not (pid or "").startswith("CD_I"):
                continue
            feed_id = int(pid[4:])
            for i, game in enumerate(games):
                round_num = int(game["round"])
                val = player.get(f"G{i + 1}avg")
                if val is not None:
                    try:
                        rows.append({"feed_id": feed_id, "round_num": round_num, "cba_pct": float(val)})
                    except (ValueError, TypeError):
                        pass
        time.sleep(SLEEP)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["feed_id", "round_num"])


def fetch_kickins(year: int) -> pd.DataFrame:
    rows = []
    for team in TEAMS:
        logging.info(f"  Kick-ins: {team} {year}")
        try:
            payload = post_ajax("afl_kickins_new_call", team, year)
        except Exception as e:
            logging.warning(f"  Kick-in fetch failed {team} {year}: {e}")
            time.sleep(SLEEP)
            continue

        games = payload.get("games", [])
        for player in payload.get("kickins", []):
            pid = player.get("playerId") or ""
            if not (pid or "").startswith("CD_I"):
                continue
            feed_id = int(pid[4:])
            for i, game in enumerate(games):
                round_num = int(game["round"])
                ki_val = player.get(f"KI_G{i + 1}")
                po_val = player.get(f"PO_G{i + 1}")
                if ki_val is None:
                    continue
                try:
                    ki = float(ki_val)
                    po = float(po_val) if po_val is not None else 0.0
                    rows.append({
                        "feed_id": feed_id,
                        "round_num": round_num,
                        "kickin_count": ki,
                        "kickin_po_pct": (po / ki * 100) if ki > 0 else None,
                    })
                except (ValueError, TypeError):
                    pass
        time.sleep(SLEEP)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["feed_id", "round_num"])


def fetch_ruck_contests(year: int) -> pd.DataFrame:
    rows = []
    for team in TEAMS:
        logging.info(f"  Ruck contests: {team} {year}")
        try:
            payload = post_ajax("afl_ruckcontests_call_new_mysql", team, year)
        except Exception as e:
            logging.warning(f"  Ruck contest fetch failed {team} {year}: {e}")
            time.sleep(SLEEP)
            continue

        games = payload.get("games", [])
        for player in payload.get("ruckContests", []):
            pid = player.get("playerId") or ""
            if not (pid or "").startswith("CD_I"):
                continue
            feed_id = int(pid[4:])
            for i, game in enumerate(games):
                round_num = int(game["round"])
                val = player.get(f"G{i + 1}")
                if val is not None:
                    try:
                        rows.append({
                            "feed_id": feed_id,
                            "round_num": round_num,
                            "ruck_contest_count": float(val),
                        })
                    except (ValueError, TypeError):
                        pass
        time.sleep(SLEEP)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["feed_id", "round_num"])


def ingest_year(year: int):
    out_dir = OUTPUT_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"=== Fetching DFS Australia data for {year} ===")

    cbas = fetch_cbas(year)
    cbas_path = out_dir / f"cbas_{year}.csv"
    cbas.to_csv(cbas_path, index=False)
    logging.info(f"Saved {len(cbas)} CBA rows → {cbas_path}")

    kickins = fetch_kickins(year)
    kickins_path = out_dir / f"kickins_{year}.csv"
    kickins.to_csv(kickins_path, index=False)
    logging.info(f"Saved {len(kickins)} kick-in rows → {kickins_path}")

    ruck = fetch_ruck_contests(year)
    ruck_path = out_dir / f"ruck_contests_{year}.csv"
    ruck.to_csv(ruck_path, index=False)
    logging.info(f"Saved {len(ruck)} ruck contest rows → {ruck_path}")


def parse_years(arg: str) -> list[int]:
    if "-" in arg:
        start, end = arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(arg)]


def main():
    parser = argparse.ArgumentParser(description="Ingest DFS Australia AFL stats")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Single year to ingest")
    group.add_argument("--years", type=str, help="Year range e.g. 2021-2026")
    args = parser.parse_args()

    years = [args.year] if args.year else parse_years(args.years)
    for year in years:
        ingest_year(year)


if __name__ == "__main__":
    main()
