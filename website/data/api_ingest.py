"""
api_ingest.py — Fetches live 2026 data from the SC API and writes all output CSVs.

Writes 5 files:
  data/live/ladder.csv                  — round-by-round standings per coach
  data/live/fixture_results_by_team.csv — per-team per-round fixture results
  data/live/player_match_results.csv    — per-player per-round scores with on_field status
  data/live/current_teams.csv           — current rosters with season avg/price
  data/live/player_stats_current.csv    — per-player per-round scores (for newsletter metrics)

Confirmed API response structure (ladderAndFixtures?round=N&scores=true):
  {
    "fixtures": [
      {"fixture": 1, "round": N,
       "user_team1": {"id":..,"teamname":..,"user":{"first_name":..,"id":..},
                      "stats":[{"points":1810,...}]},
       "user_team2": {same}}
    ],
    "ladder": [
      {"draws":0,"losses":0,"wins":1,"points":4,"points_for":1813,"points_against":1630,
       "position":1,"round":N,"user_team_id":...,
       "userTeam": {
         "id":..,"teamname":..,"user":{"first_name":..,"id":..},
         "scores": {
           "score": 1813,
           "scoring":    [{player_id,points,position,round,
                           player:{feed_id,first_name,last_name,team:{abbrev},
                                   player_stats:[{points,round,avg,avg3,avg5,price,...}]}}],
           "nonscoring": [same; team contribution points=0, actual score in player.player_stats[0].points]
         }
       }}
    ]
  }
"""
import logging
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0"}


# ── Lazy imports ──────────────────────────────────────────────────────────────

def _get_config():
    from waiver.league_api import get_valid_token, fetch_player_status
    from waiver.config import SEASON_YEAR, LADDER_LEAGUE_ID, LEAGUE_ID
    from website.config import (
        LADDER_CSV, FIXTURE_TEAM_CSV, PLAYER_MATCH_CSV, CURRENT_TEAMS_CSV,
        TOTAL_REGULAR_ROUNDS,
    )
    project_root = Path(__file__).resolve().parent.parent.parent
    player_stats_csv = project_root / "data" / "live" / "player_stats_current.csv"
    coach_list_csv   = project_root / "data" / "live" / "coach_list.csv"
    json_dir         = project_root / "data" / "live" / "json"
    raw_dir          = project_root / "data" / "raw" / "supercoach" / str(SEASON_YEAR)
    return (get_valid_token, fetch_player_status, SEASON_YEAR, LADDER_LEAGUE_ID, LEAGUE_ID,
            LADDER_CSV, FIXTURE_TEAM_CSV, PLAYER_MATCH_CSV, CURRENT_TEAMS_CSV,
            TOTAL_REGULAR_ROUNDS, player_stats_csv, coach_list_csv, json_dir, raw_dir)


# ── SC API helpers ────────────────────────────────────────────────────────────

def _json_path(json_dir: Path, round_num: int) -> Path:
    return json_dir / f"match_json_rd{round_num}.json"


def _raw_filename(url: str) -> str:
    """
    Derive the R-convention filename from a SC API URL.
    Mirrors: nme <- gsub('/','_', str_extract(url, '(?<=/v1/).+'))
    e.g. .../v1/leagues/14889/ladderAndFixtures?round=1&scores=true
      → leagues_14889_ladderAndFixtures_round=1&scores=true.json
    """
    import re
    m = re.search(r'/v1/(.+)', url)
    stem = m.group(1) if m else url.split('/')[-1]
    # Replace / then sanitize chars invalid in Windows filenames (?&=*:|"<>)
    stem = stem.replace('/', '_')
    stem = re.sub(r'[?&=*:|"<>]', '_', stem)
    return stem + '.json'


def _fetch_round(
    year: int, league_id: str, token: str, round_num: int,
    raw_dir: Path | None = None,
) -> dict:
    """
    Fetches one round from the SC API.
    Optionally archives the raw response to data/raw/supercoach/{year}/ (always safe to write).
    Does NOT write to data/live/json/ — only played rounds should go there (see _detect_and_fetch_all).
    Returns the parsed dict, or {} on failure.
    """
    import json as _json
    url = (
        f"https://www.supercoach.com.au/{year}/api/afl/draft/v1"
        f"/leagues/{league_id}/ladderAndFixtures?round={round_num}&scores=true"
    )
    try:
        r = requests.get(
            url,
            headers={**_HEADERS, "Authorization": f"Bearer {token}"},
            verify=False,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("Round %d API fetch failed: %s", round_num, exc)
        return {}

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / _raw_filename(url)
        with open(raw_path, "w", encoding="utf-8") as f:
            _json.dump(data, f)
        log.debug("Archived round %d → %s", round_num, raw_path)

    return data


def _fetch_and_save_round(
    year: int, league_id: str, token: str, round_num: int,
    json_dir: Path, raw_dir: Path | None = None,
) -> dict:
    """Fetch a round and save to data/live/json/ only if it was played in the current season."""
    import json as _json
    data = _fetch_round(year, league_id, token, round_num, raw_dir)
    if not data:
        return {}
    if _round_was_played(data, round_num):
        json_dir.mkdir(parents=True, exist_ok=True)
        out_path = _json_path(json_dir, round_num)
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(data, f)
        log.debug("Saved live JSON for round %d → %s", round_num, out_path)
    return data


def _fetch_and_save_endpoint(
    year: int, league_id: str, path: str, token: str, raw_dir: Path
) -> dict:
    """
    Fetch a league endpoint (e.g. 'processedWaivers', 'teamtrades', 'trades',
    'playersStatus') and save to raw_dir using the R naming convention.
    Returns the parsed JSON or {} on failure.
    """
    import json as _json
    url = (
        f"https://www.supercoach.com.au/{year}/api/afl/draft/v1"
        f"/leagues/{league_id}/{path}"
    )
    try:
        r = requests.get(
            url,
            headers={**_HEADERS, "Authorization": f"Bearer {token}"},
            verify=False,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("Endpoint '%s' fetch failed: %s", path, exc)
        return {}

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / _raw_filename(url)
    with open(raw_path, "w", encoding="utf-8") as f:
        _json.dump(data, f)
    log.info("Saved %s → %s", path, raw_path)
    return data


def _fetch_top_level_endpoint(
    year: int, path: str, token: str, raw_dir: Path
) -> dict | list:
    """
    Fetch a top-level (non-league) SC API endpoint and save to raw_dir.
    e.g. path='me', 'real_fixture', 'players?cf_embed=...&round=1'
    """
    import json as _json
    import re as _re
    url = f"https://www.supercoach.com.au/{year}/api/afl/draft/v1/{path}"
    try:
        r = requests.get(
            url,
            headers={**_HEADERS, "Authorization": f"Bearer {token}"},
            verify=False,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("Endpoint '%s' fetch failed: %s", path, exc)
        return {}

    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _re.sub(r'[<>:"/\\|?*]', '_', path) + ".json"
    raw_path = raw_dir / safe_name
    with open(raw_path, "w", encoding="utf-8") as f:
        _json.dump(data, f)
    log.info("Saved %s → %s", path.split("?")[0], raw_path.name)
    return data


def _load_round_json(json_dir: Path, round_num: int) -> dict:
    """Load a previously saved round JSON. Returns {} if not found."""
    import json as _json
    path = _json_path(json_dir, round_num)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return _json.load(f)


def _has_scoring_data(raw: dict) -> bool:
    """Returns True if the round response contains actual player scores."""
    for entry in (raw.get("ladder") or []):
        team = entry.get("userTeam") or {}
        if (team.get("scores") or {}).get("scoring"):
            return True
    return False


def _round_was_played(raw: dict, round_num: int) -> bool:
    """
    Returns True only if this round has been played in the CURRENT season.

    The SC API returns 2025 historical data for future rounds even in 2026,
    so we can't just check for scoring presence. Instead we verify that the
    cumulative games played for at least one team equals round_num — meaning
    they've actually played that many games this season.
    """
    for entry in (raw.get("ladder") or []):
        games = entry.get("wins", 0) + entry.get("draws", 0) + entry.get("losses", 0)
        if games == round_num:
            return True
    return False


def _detect_and_fetch_all(
    year: int, league_id: str, token: str, total_rounds: int,
    json_dir: Path, raw_dir: Path | None = None,
) -> dict[int, dict]:
    """
    Fetches all rounds 1..(total_rounds+1) from the SC API, saving each to disk.
    Returns a dict of {round_num: raw_response} for rounds that have scoring data.

    Probes one beyond total_rounds to catch a newly completed round.
    Stops early once we reach rounds with no scoring data (working backwards).
    """
    fetched: dict[int, dict] = {}

    # Probe downward from total_rounds+1 to find the highest round actually
    # played in the current season. We check wins+draws+losses == round_num
    # because the API serves historical (prior season) data for unplayed rounds.
    current_round = 0
    for rnd in range(total_rounds + 1, 0, -1):
        raw = _fetch_and_save_round(year, league_id, token, rnd, json_dir, raw_dir)
        if raw and _round_was_played(raw, rnd):
            fetched[rnd] = raw
            if current_round == 0:
                current_round = rnd
                log.info("Detected current round: %d", rnd)
            break  # found the top — all earlier rounds are also played

    if current_round == 0:
        log.warning("No completed rounds detected; defaulting to round 1")
        return fetched

    # Fetch rounds 1..(current_round-1)
    for rnd in range(1, current_round):
        raw = _fetch_and_save_round(year, league_id, token, rnd, json_dir, raw_dir)
        if raw and _round_was_played(raw, rnd):
            fetched[rnd] = raw

    return fetched


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_ladder_entry(entry: dict, rnd: int) -> dict:
    team = entry.get("userTeam") or {}
    user = team.get("user") or {}
    return {
        "round":            rnd,
        "coach_id":         user.get("id") or entry.get("user_id"),
        "coach_team_id":    team.get("id") or entry.get("user_team_id"),
        "coach_first_name": user.get("first_name", ""),
        "coach_team_name":  team.get("teamname", ""),
        "wins":             entry.get("wins", 0),
        "draws":            entry.get("draws", 0),
        "losses":           entry.get("losses", 0),
        "points_for":       entry.get("points_for", 0),
        "points_against":   entry.get("points_against", 0),
        "league_points":    entry.get("points", 0),
        "position":         entry.get("position", 0),
    }


def _parse_fixture(f: dict, rnd: int) -> list[dict]:
    """
    Parse a fixture entry using the confirmed structure:
      {fixture, round, user_team1, user_team2}
      user_team.stats[0].points = match score
      user_team.user.first_name = coach name
    """
    t1 = f.get("user_team1") or {}
    t2 = f.get("user_team2") or {}
    coach1 = (t1.get("user") or {}).get("first_name", "")
    coach2 = (t2.get("user") or {}).get("first_name", "")
    pts1   = ((t1.get("stats") or [{}])[0]).get("points", 0) or 0
    pts2   = ((t2.get("stats") or [{}])[0]).get("points", 0) or 0
    match_name = f"{coach1} vs {coach2}"

    rows = []
    if coach1:
        rows.append({
            "round_number": rnd, "match_name": match_name,
            "team_id": t1.get("id"), "team_name": t1.get("teamname", ""),
            "team_coach": coach1, "team_points": pts1,
            "opposition_team_id": t2.get("id"), "opposition_team_name": t2.get("teamname", ""),
            "opposition_team_coach": coach2, "opposition_team_points": pts2, "Rank": 0,
        })
    if coach2:
        rows.append({
            "round_number": rnd, "match_name": match_name,
            "team_id": t2.get("id"), "team_name": t2.get("teamname", ""),
            "team_coach": coach2, "team_points": pts2,
            "opposition_team_id": t1.get("id"), "opposition_team_name": t1.get("teamname", ""),
            "opposition_team_coach": coach1, "opposition_team_points": pts1, "Rank": 0,
        })
    return rows


def _parse_players(raw: dict, rnd: int) -> tuple[list[dict], list[dict]]:
    """
    Returns (player_match_rows, player_stats_rows) for player_match_results.csv
    and player_stats_current.csv respectively.

    Uses p["player"]["player_stats"][0]["points"] for actual individual SC score
    (both on-field and bench). The top-level p["points"] for nonscoring is always 0.
    """
    pm_rows: list[dict] = []
    ps_rows: list[dict] = []

    for entry in (raw.get("ladder") or []):
        team    = entry.get("userTeam") or {}
        user    = team.get("user") or {}
        coach   = user.get("first_name", "")
        team_id = team.get("id")
        scores  = team.get("scores") or {}

        def _add(p: dict, on_field: bool):
            sc_id = p.get("player_id")
            pl    = p.get("player") or {}
            ps    = (pl.get("player_stats") or [{}])[0]

            # Individual SC score: scoring uses p['points'], nonscoring needs player_stats
            points = p.get("points", 0) if on_field else ps.get("points", 0)

            feed_id    = pl.get("feed_id")
            first_name = pl.get("first_name", "")
            last_name  = pl.get("last_name", "")
            team_abbrev = (pl.get("team") or {}).get("abbrev", "")
            position   = str(p.get("position") or ps.get("position") or "").strip().upper()

            pm_rows.append({
                "round_x":          rnd,
                "coach_id":         user.get("id"),
                "coach_team_id":    team_id,
                "coach_first_name": coach,
                "coach_team_name":  team.get("teamname", ""),
                "player_id":        sc_id,
                "played_position":  position,
                "points_x":         points,
                "on_field":         on_field,
                "picked":           p.get("picked", ""),
                "feed_id":          feed_id,
                "first_name":       first_name,
                "last_name":        last_name,
                "team":             team_abbrev,
                "position":         position,
            })

            # player_stats_current.csv row — one row per player per round
            if feed_id:
                ps_rows.append({
                    "feed_id":      feed_id,
                    "player_id":    sc_id,
                    "first_name":   first_name,
                    "last_name":    last_name,
                    "team_abbrev":  team_abbrev,
                    "pos_1":        position,
                    "pos_2":        "",
                    "round":        rnd,
                    "points":       points,
                    "played":       1 if on_field else 0,
                    "avg":          ps.get("avg", 0.0),
                    "avg3":         ps.get("avg3", 0.0),
                    "avg5":         ps.get("avg5", 0.0),
                    "price":        ps.get("price", 0),
                })

        for p in (scores.get("scoring") or []):
            _add(p, on_field=True)
        for p in (scores.get("nonscoring") or []):
            _add(p, on_field=False)

    return pm_rows, ps_rows


# ── current_teams builder ─────────────────────────────────────────────────────

def _build_current_teams(
    player_match_df: pd.DataFrame,
    status_dict: dict,
    coach_list_csv: Path,
) -> pd.DataFrame:
    """Build current_teams.csv from playersStatus + player match data."""
    if not coach_list_csv.exists():
        return pd.DataFrame()

    coach_df = pd.read_csv(coach_list_csv)
    coach_lookup = dict(zip(
        coach_df["coach_team_id"].astype(int),
        coach_df["coach_first_name"]
    ))

    # Build player info from player_match_df (has feed_id, name, team, position)
    if player_match_df.empty:
        return pd.DataFrame()

    # One row per player (latest round's data)
    player_info = (
        player_match_df.sort_values("round_x")
        .drop_duplicates(subset=["player_id"], keep="last")
        [["player_id", "feed_id", "first_name", "last_name", "team", "position"]]
    )

    # On-field avg and avg3
    on_field = player_match_df[player_match_df["on_field"] == True]
    avg_df = (
        on_field.groupby("player_id")["points_x"]
        .mean().round(1).reset_index()
        .rename(columns={"points_x": "avg"})
    )
    last3_df = (
        on_field.sort_values("round_x")
        .groupby("player_id")["points_x"]
        .apply(lambda s: round(s.tail(3).mean(), 1))
        .reset_index()
        .rename(columns={"points_x": "avg3"})
    )

    owned_rows = []
    for sc_id_str, info in status_dict.items():
        if info.get("trade_status") != "owner":
            continue
        sc_id   = int(sc_id_str)
        team_id = info.get("user_team_id")
        coach   = coach_lookup.get(team_id, "")
        if not coach:
            continue
        owned_rows.append({"player_id": sc_id, "coach_first_name": coach})

    if not owned_rows:
        return pd.DataFrame()

    owned_df = pd.DataFrame(owned_rows)
    owned_df["player_id"] = owned_df["player_id"].astype(int)
    player_info["player_id"] = player_info["player_id"].astype(int)
    avg_df["player_id"]   = avg_df["player_id"].astype(int)
    last3_df["player_id"] = last3_df["player_id"].astype(int)

    df = owned_df.merge(player_info, on="player_id", how="left")
    df = df.merge(avg_df,   on="player_id", how="left")
    df = df.merge(last3_df, on="player_id", how="left")

    # Price from latest player_match row's player_stats (not stored in pm_rows)
    # Use 0 as placeholder — price can be sourced from player list separately
    df["price"] = 0
    df["avg"]   = df["avg"].fillna(0.0)
    df["avg3"]  = df["avg3"].fillna(0.0)

    return df[["coach_first_name", "first_name", "last_name", "team", "position",
               "avg", "avg3", "price", "feed_id"]]


# ── Transactions parser ───────────────────────────────────────────────────────

def _fetch_and_save_transactions(
    year: int, league_id: str, token: str, transactions_csv: Path
) -> int:
    """
    Fetches all three transaction endpoints and writes data/live/transactions.csv.
    Returns the number of transactions written, or 0 on failure.

    Endpoints and source labels:
      /processedWaivers → source = "Waiver"
      /trades           → source = "Free Agent"  (FA pickups: drop one, pick up from pool)
      /teamtrades       → source = "Trade"        (player-for-player between teams, status_id=7)
    """
    auth = {**_HEADERS, "Authorization": f"Bearer {token}"}
    base = f"https://www.supercoach.com.au/{year}/api/afl/draft/v1/leagues/{league_id}"
    rows: list[dict] = []

    # ── 1. Processed Waivers ──────────────────────────────────────────────────
    try:
        r = requests.get(f"{base}/processedWaivers", headers=auth, verify=False, timeout=30)
        r.raise_for_status()
        for i, t in enumerate(r.json() or []):
            pl   = t.get("player")         or {}
            drop = t.get("dropped_player") or {}
            rows.append({
                "round":                      t.get("round"),
                "user_team_id":               t.get("user_team_id"),
                "processed":                  t.get("processed"),
                "waiver_order":               i,
                "player_id":                  t.get("player_id"),
                "player.feed_id":             pl.get("feed_id"),
                "player.first_name":          pl.get("first_name"),
                "player.last_name":           pl.get("last_name"),
                "player.team.abbrev":         (pl.get("team") or {}).get("abbrev"),
                "dropped_player_id":          t.get("dropped_player_id"),
                "dropped_player.feed_id":     drop.get("feed_id"),
                "dropped_player.first_name":  drop.get("first_name"),
                "dropped_player.last_name":   drop.get("last_name"),
                "dropped_player.team.abbrev": (drop.get("team") or {}).get("abbrev"),
                "source":                     "Waiver",
            })
        log.info("processedWaivers: %d rows", sum(1 for r2 in rows if r2["source"] == "Waiver"))
    except Exception as exc:
        log.warning("processedWaivers fetch failed: %s", exc)

    # ── 2. Free Agent pickups (/trades endpoint) ──────────────────────────────
    fa_start = len(rows)
    try:
        r = requests.get(f"{base}/trades", headers=auth, verify=False, timeout=30)
        r.raise_for_status()
        for t in (r.json() or []):
            buy  = t.get("buy_player")  or {}
            sell = t.get("sell_player") or {}
            rows.append({
                "round":                      t.get("round"),
                "user_team_id":               t.get("user_id"),
                "processed":                  t.get("action_time"),
                "player_id":                  t.get("buy_player_id"),
                "player.feed_id":             buy.get("feed_id"),
                "player.first_name":          buy.get("first_name"),
                "player.last_name":           buy.get("last_name"),
                "player.team.abbrev":         (buy.get("team") or {}).get("abbrev"),
                "dropped_player_id":          t.get("sell_player_id"),
                "dropped_player.feed_id":     sell.get("feed_id"),
                "dropped_player.first_name":  sell.get("first_name"),
                "dropped_player.last_name":   sell.get("last_name"),
                "dropped_player.team.abbrev": (sell.get("team") or {}).get("abbrev"),
                "source":                     "Free Agent",
            })
        log.info("Free Agent pickups (/trades): %d rows", len(rows) - fa_start)
    except Exception as exc:
        log.warning("Free Agent trades fetch failed: %s", exc)

    # ── 3. Team-to-team trades (/teamtrades endpoint) ─────────────────────────
    #
    # API returns ONE record per trade (not both sides). Structure:
    #   user_team_id       = team that initiated the trade
    #   trade_user_team_id = the other team
    #   player_id          = player LEAVING user_team_id → going to trade_user_team_id
    #   trade_player_id    = player LEAVING trade_user_team_id → going to user_team_id
    #
    # We create two rows per trade:
    #   Row 1: trade_user_team_id receives player_id
    #   Row 2: user_team_id receives trade_player_id
    #
    # No player name/feed_id in this response — looked up in acquisition.py via player_id.
    trade_start = len(rows)
    try:
        r = requests.get(f"{base}/teamtrades", headers=auth, verify=False, timeout=30)
        r.raise_for_status()
        seen_trade_ids: set[int] = set()
        for trade in (r.json() or []):
            if trade.get("status_id") != 7:   # 7 = approved only
                continue
            trade_id = trade.get("id")
            if trade_id in seen_trade_ids:     # deduplicate in case API echoes both sides
                continue
            seen_trade_ids.add(trade_id)

            rnd             = trade.get("round")
            team_a          = trade.get("user_team_id")        # initiating team
            team_b          = trade.get("trade_user_team_id")  # other team
            processed       = trade.get("action_time")

            for tp in (trade.get("team_trade_players") or []):
                pl_a  = tp.get("player_id")        # leaves team_a, goes to team_b
                pl_b  = tp.get("trade_player_id")  # leaves team_b, goes to team_a

                # team_b receives pl_a
                if pl_a is not None and team_b is not None:
                    rows.append({
                        "round":                      rnd,
                        "user_team_id":               team_b,
                        "processed":                  processed,
                        "player_id":                  pl_a,
                        "player.feed_id":             None,
                        "player.first_name":          None,
                        "player.last_name":           None,
                        "player.team.abbrev":         None,
                        "dropped_player_id":          pl_b,
                        "dropped_player.feed_id":     None,
                        "dropped_player.first_name":  None,
                        "dropped_player.last_name":   None,
                        "dropped_player.team.abbrev": None,
                        "source":                     "Trade",
                    })

                # team_a receives pl_b
                if pl_b is not None and team_a is not None:
                    rows.append({
                        "round":                      rnd,
                        "user_team_id":               team_a,
                        "processed":                  processed,
                        "player_id":                  pl_b,
                        "player.feed_id":             None,
                        "player.first_name":          None,
                        "player.last_name":           None,
                        "player.team.abbrev":         None,
                        "dropped_player_id":          pl_a,
                        "dropped_player.feed_id":     None,
                        "dropped_player.first_name":  None,
                        "dropped_player.last_name":   None,
                        "dropped_player.team.abbrev": None,
                        "source":                     "Trade",
                    })

        log.info("Team trades (/teamtrades): %d rows from %d approved trades",
                 len(rows) - trade_start, len(seen_trade_ids))
    except Exception as exc:
        log.warning("teamtrades fetch failed: %s", exc)

    if not rows:
        log.info("No transactions returned from any endpoint.")
        return 0

    df = pd.DataFrame(rows).sort_values(["processed", "waiver_order"], na_position="last")
    transactions_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(transactions_csv, index=False)
    log.info("transactions.csv: %d rows → %s", len(df), transactions_csv)
    return len(df)


# ── Fixture schedule extractor ────────────────────────────────────────────────

def _fetch_full_fixture(
    year: int, league_id: str, my_team_id: int, token: str, json_dir: Path
) -> list:
    """
    Fetches /leagues/{league_id}/fixtures/{my_team_id} and saves the raw
    response to data/live/json/league_fixtures.json.
    Returns the parsed list (one item per round) or [] on failure.
    """
    import json as _json
    url = (
        f"https://www.supercoach.com.au/{year}/api/afl/draft/v1"
        f"/leagues/{league_id}/fixtures/{my_team_id}"
    )
    try:
        r = requests.get(
            url,
            headers={**_HEADERS, "Authorization": f"Bearer {token}"},
            verify=False,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("Full fixture fetch failed: %s", exc)
        return []
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / "league_fixtures.json").write_text(_json.dumps(data))
    log.info("Saved league_fixtures.json (%d rounds)", len(data) if isinstance(data, list) else 0)
    return data if isinstance(data, list) else []


def _fetch_future_fixture_rounds(
    year: int, league_id: str, token: str,
    from_round: int, to_round: int,
    json_dir: Path, raw_dir: Path | None = None,
) -> None:
    """
    Fetches ladderAndFixtures for rounds from_round..to_round and saves each
    to data/live/json/match_json_rdN.json so _extract_fixture_schedule can
    read the fixture pairings for future rounds.  Skips rounds whose file
    already exists.
    """
    for rnd in range(from_round, to_round + 1):
        dest = _json_path(json_dir, rnd)
        if dest.exists():
            continue
        raw = _fetch_round(year, league_id, token, rnd, raw_dir)
        if raw and raw.get("fixtures"):
            import json as _json
            json_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(_json.dumps(raw))
            log.debug("Saved future fixture round %d → %s", rnd, dest.name)


def _extract_fixture_schedule(json_dir: Path, total_rounds: int) -> "pd.DataFrame":
    """
    Reads match_json_rdN.json files for rounds 1..total_rounds and returns a
    DataFrame with columns: round_number, home_coach, away_coach.

    Works for both played and unplayed rounds — only the fixtures[] array is
    used (no scoring data required).  Rounds whose JSON file is missing are
    silently skipped.
    """
    rows: list[dict] = []
    for rnd in range(1, total_rounds + 1):
        raw = _load_round_json(json_dir, rnd)
        if not raw:
            continue
        for fix in (raw.get("fixtures") or []):
            t1 = fix.get("user_team1") or {}
            t2 = fix.get("user_team2") or {}
            c1 = (t1.get("user") or {}).get("first_name", "")
            c2 = (t2.get("user") or {}).get("first_name", "")
            if c1 and c2:
                rows.append({"round_number": rnd, "home_coach": c1, "away_coach": c2})
    return pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def refresh_2026_data() -> bool:
    """
    Fetches all available 2026 round data from the SC API and writes:
      data/live/ladder.csv
      data/live/fixture_results_by_team.csv
      data/live/player_match_results.csv
      data/live/current_teams.csv
      data/live/player_stats_current.csv

    Returns True on success, False if the API is unreachable.
    """
    (get_valid_token, fetch_player_status, SEASON_YEAR, LADDER_LEAGUE_ID, LEAGUE_ID,
     LADDER_CSV, FIXTURE_TEAM_CSV, PLAYER_MATCH_CSV, CURRENT_TEAMS_CSV,
     TOTAL_REGULAR_ROUNDS, player_stats_csv, coach_list_csv, json_dir, raw_dir) = _get_config()
    from website.config import FIXTURE_SCHEDULE_CSV

    token = get_valid_token(year=SEASON_YEAR)
    if not token:
        log.error("No SC token — cannot refresh outputs.")
        return False

    fetched = _detect_and_fetch_all(
        SEASON_YEAR, LADDER_LEAGUE_ID, token, TOTAL_REGULAR_ROUNDS, json_dir, raw_dir
    )
    if not fetched:
        log.error("No scored rounds returned from API — outputs left unchanged.")
        return False

    current_round = max(fetched.keys())
    log.info("Processing 2026 data for rounds 1..%d", current_round)

    ladder_rows:  list[dict] = []
    fixture_rows: list[dict] = []
    pm_rows:      list[dict] = []
    ps_rows:      list[dict] = []

    for rnd in sorted(fetched.keys()):
        raw = fetched[rnd]
        if not raw or not _round_was_played(raw, rnd):
            continue
        for entry in (raw.get("ladder") or []):
            ladder_rows.append(_parse_ladder_entry(entry, rnd))
        for f in (raw.get("fixtures") or []):
            fixture_rows.extend(_parse_fixture(f, rnd))
        new_pm, new_ps = _parse_players(raw, rnd)
        pm_rows.extend(new_pm)
        ps_rows.extend(new_ps)

    if not ladder_rows:
        log.error("No ladder data returned — outputs left unchanged.")
        return False

    pd.DataFrame(ladder_rows).to_csv(LADDER_CSV, index=False)
    log.info("ladder.csv: %d rows", len(ladder_rows))

    if fixture_rows:
        pd.DataFrame(fixture_rows).to_csv(FIXTURE_TEAM_CSV, index=False)
        log.info("fixture_results_by_team.csv: %d rows", len(fixture_rows))
    else:
        log.warning("No fixture data — fixture_results_by_team.csv not updated.")

    player_df = pd.DataFrame(pm_rows)
    if not player_df.empty:
        player_df.to_csv(PLAYER_MATCH_CSV, index=False)
        log.info("player_match_results.csv: %d rows", len(pm_rows))
    else:
        log.warning("No player match data — player_match_results.csv not updated.")

    if ps_rows:
        # Deduplicate: one row per (feed_id, round) — keep last occurrence
        ps_df = pd.DataFrame(ps_rows).drop_duplicates(subset=["feed_id", "round"], keep="last")
        ps_df.to_csv(player_stats_csv, index=False)
        log.info("player_stats_current.csv: %d rows", len(ps_df))

    # Build current_teams.csv from playersStatus
    status_dict = fetch_player_status(year=SEASON_YEAR, token=token)
    if status_dict and not player_df.empty:
        ct_df = _build_current_teams(player_df, status_dict, coach_list_csv)
        if not ct_df.empty:
            ct_df.to_csv(CURRENT_TEAMS_CSV, index=False)
            log.info("current_teams.csv: %d players", len(ct_df))

    # Fetch raw fixture data for user's team + all future rounds, then build CSV
    from waiver.config import MY_TEAM_ID
    _fetch_full_fixture(SEASON_YEAR, LADDER_LEAGUE_ID, MY_TEAM_ID, token, json_dir)
    _fetch_future_fixture_rounds(
        SEASON_YEAR, LADDER_LEAGUE_ID, token,
        current_round + 1, TOTAL_REGULAR_ROUNDS,
        json_dir, raw_dir,
    )
    sched_df = _extract_fixture_schedule(json_dir, TOTAL_REGULAR_ROUNDS)
    if not sched_df.empty:
        sched_df.to_csv(FIXTURE_SCHEDULE_CSV, index=False)
        log.info("fixture_schedule_2026.csv: %d matchups", len(sched_df))

    # Fetch and save transactions (trades) → data/live/transactions.csv
    transactions_csv = Path(__file__).resolve().parent.parent.parent / "data" / "live" / "transactions.csv"
    _fetch_and_save_transactions(SEASON_YEAR, LADDER_LEAGUE_ID, token, transactions_csv)

    # Archive additional league endpoints to data/raw/supercoach/{year}/
    # using the R naming convention (leagues_{id}_{endpoint}.json)
    for ep in ("processedWaivers", "teamtrades", "trades"):
        _fetch_and_save_endpoint(SEASON_YEAR, LADDER_LEAGUE_ID, ep, token, raw_dir)
    # playersStatus is under the draft league (LEAGUE_ID=4199), not the scoring league
    _fetch_and_save_endpoint(SEASON_YEAR, LEAGUE_ID, "playersStatus", token, raw_dir)
    # Draft recap (pick-by-pick)
    _fetch_and_save_endpoint(SEASON_YEAR, LADDER_LEAGUE_ID, "recap", token, raw_dir)
    # Top-level (non-league) endpoints
    for tl_path in ("me", "real_fixture"):
        _fetch_top_level_endpoint(SEASON_YEAR, tl_path, token, raw_dir)
    # Per-round player data (notes, odds, stats, positions) for each played round
    for rnd in range(1, current_round + 1):
        _fetch_top_level_endpoint(
            SEASON_YEAR,
            f"players?cf_embed=notes,odds,player_stats,positions,player_match_stats&round={rnd}",
            token,
            raw_dir,
        )

    log.info(
        "Refresh complete: %d ladder, %d fixture, %d player-match rows (rounds 1–%d).",
        len(ladder_rows), len(fixture_rows), len(pm_rows), current_round,
    )
    return True
