"""
field_view.py — Fetches live current-team data from the SC API and parses it
into a structured format for the field view widget. (inline SVG badges)

Uses ladderAndFixtures (same endpoint as api_ingest) fetched live so the
lineup reflects real-time changes (trades, emergency toggles), not the
stale pre-round snapshots stored in data/live/json/.
"""
import logging
import os
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Position display order
_POS_ORDER = {"DEF": 0, "MID": 1, "RUC": 2, "FWD": 3}

# Team abbreviation → inline SVG shapes (viewBox 0 0 24 24), rendered directly in each card
SVG_TEAM_CONTENT: dict[str, str] = {
    "ADE": '<rect fill="#174068" x="0" y="0" width="24" height="24"/><rect fill="#EB0707" x="0" y="6.316" width="24" height="5.053"/><rect fill="#FFD204" x="0" y="11.368" width="24" height="5.053"/>',
    "BRL": '<rect fill="#A0224B" x="0" y="0" width="24" height="24"/><rect fill="#0760AD" x="0" y="0" width="24" height="8.842"/><rect fill="#FEBF3E" x="0" y="8.842" width="24" height="5.053"/>',
    "CAR": '<rect fill="#0B3E6A" x="0" y="0" width="24" height="24"/>',
    "COL": '<rect fill="#000000" x="0" y="0" width="24" height="24"/><rect fill="#FFFFFF" x="4.364" y="0" width="4.364" height="24"/><rect fill="#FFFFFF" x="15.273" y="0" width="4.364" height="24"/>',
    "ESS": '<rect fill="#000000" x="0" y="0" width="24" height="24"/><polygon fill="#D51400" points="14.696,0 0,19.07 0.009,24 9.257,24 24,4.869 24,0"/>',
    "FRE": '<rect fill="#411383" x="0" y="0" width="24" height="24"/><polygon fill="#FFFFFF" points="0,5.057 12.019,16.403 23.986,4.994 23.986,10.15 12.019,21.538 0,10.15"/><path fill="#FFFFFF" d="M0.008,0.009 L4.826,0.009 L12.019,6.864 L19.218,0 L23.986,0 L23.986,0.611 L12.019,11.998 L0,0.611 Z"/>',
    "GEE": '<rect fill="#000A61" x="0" y="0" width="24" height="24"/><rect fill="#FFFFFF" x="0" y="5.053" width="24" height="5.053"/><rect fill="#FFFFFF" x="0" y="13.895" width="24" height="5.053"/>',
    "GCS": '<rect fill="#A10F0F" x="0" y="0" width="24" height="24"/><rect fill="#FF9200" x="0" y="0" width="6.575" height="24"/><rect fill="#FF9200" x="17.425" y="0" width="6.575" height="24"/>',
    "GWS": '<rect fill="#F1F1F1" x="0" y="0" width="24" height="24"/><rect fill="#E37211" x="0" y="0" width="24" height="12"/><polygon fill="#787878" points="15.123,12 11.507,24 24,24 24,12"/>',
    "HAW": '<rect fill="#633102" x="0" y="0" width="24" height="24"/><rect fill="#F1C813" x="4.364" y="0" width="4.364" height="24"/><rect fill="#F1C813" x="15.273" y="0" width="4.364" height="24"/>',
    "MEL": '<rect fill="#010C69" x="0" y="0" width="24" height="24"/><polygon fill="#E01E00" points="0,0 12,16.421 24,0"/>',
    "NTH": '<rect fill="#1412F6" x="0" y="0" width="24" height="24"/><rect fill="#FFFFFF" x="4.364" y="0" width="4.364" height="24"/><rect fill="#FFFFFF" x="15.273" y="0" width="4.364" height="24"/>',
    "PTA": '<rect fill="#000000" x="0" y="0" width="24" height="24"/><path fill="#FFFFFF" d="M2.931,0 L0,0 L0,3.981 L12.003,20.211 L24,3.492 L24,0 Z"/><polygon fill="#4AB0B3" points="0,0 12.012,16.495 24,0"/>',
    "RIC": '<rect fill="#000000" x="0" y="0" width="24" height="24"/><polygon fill="#F3C516" points="14.696,0 0,19.07 0.009,24 9.257,24 24,4.869 24,0"/>',
    "STK": '<rect fill="#000000" x="0" y="0" width="24" height="24"/><rect fill="#C71100" x="0" y="0" width="8.727" height="24"/><rect fill="#FFFFFF" x="8.727" y="0" width="6.545" height="24"/>',
    "SYD": '<rect fill="#FFFFFF" x="0" y="0" width="24" height="24"/><polygon fill="#EF2109" points="0,0 0,11.52 5.795,14.88 11.75,11.52 18.213,14.88 24,11.52 24,0"/>',
    "WBD": '<rect fill="#062BDB" x="0" y="0" width="24" height="24"/><rect fill="#D82018" x="0" y="6.579" width="24" height="5.263"/><rect fill="#FFFFFF" x="0" y="11.842" width="24" height="5.263"/>',
    "WCE": '<rect fill="#F6B910" x="0" y="0" width="24" height="24"/><rect fill="#011068" x="0" y="0" width="8.727" height="24"/><rect fill="#FFFFFF" x="8.727" y="0" width="6.545" height="24"/>',
}

# AFL team primary / secondary colours
AFL_TEAM_COLOURS: dict[str, tuple[str, str]] = {
    "ADE": ("#002B5C", "#E21D1D"),
    "BRL": ("#83002E", "#EFC04A"),
    "CAR": ("#021B3C", "#FFFFFF"),
    "COL": ("#000000", "#FFFFFF"),
    "ESS": ("#CC0000", "#000000"),
    "FRE": ("#7B3F98", "#FFFFFF"),
    "GEE": ("#001F4E", "#FFFFFF"),
    "GCS": ("#D4282C", "#FFD200"),
    "GWS": ("#E77426", "#8E8E8E"),
    "HAW": ("#4D2004", "#FBB722"),
    "MEL": ("#CC2031", "#003058"),
    "NTH": ("#003FA1", "#FFFFFF"),
    "PTA": ("#008080", "#7E001E"),
    "RIC": ("#FFE100", "#000000"),
    "STK": ("#ED1C24", "#FFFFFF"),
    "SYD": ("#E11E25", "#FFFFFF"),
    "WBD": ("#0039A6", "#D21036"),
    "WCE": ("#062CE2", "#F2A903"),
}


def _get_token() -> str:
    try:
        from waiver.league_api import get_valid_token
        return get_valid_token()
    except Exception as exc:
        log.warning("get_valid_token failed: %s — falling back to SC_API_TOKEN", exc)
        return os.getenv("SC_API_TOKEN", "")


def _get_current_round() -> int:
    """Derive current round from ladder.csv (max round with data)."""
    from website.config import LADDER_CSV
    try:
        import pandas as pd
        df = pd.read_csv(LADDER_CSV, usecols=["round"], low_memory=False)
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        return int(df["round"].dropna().max())
    except Exception:
        return 1


def _load_coach_team_id(coach_name: str) -> int | None:
    """Return the SC user_team_id for this coach from coach_list.csv."""
    project_root = Path(__file__).resolve().parent.parent.parent
    path = project_root / "data" / "live" / "coach_list.csv"
    try:
        import pandas as pd
        df = pd.read_csv(path)
        row = df[df["coach_first_name"] == coach_name]
        if not row.empty:
            return int(row.iloc[0]["coach_team_id"])
    except Exception as exc:
        log.warning("Could not load coach_list.csv: %s", exc)
    return None


def _fetch_ladder_round(year: int, league_id: str, token: str, round_num: int) -> dict:
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
        return r.json()
    except Exception as exc:
        log.warning("ladderAndFixtures round=%s fetch failed: %s", round_num, exc)
        return {}


def _parse_player(entry: dict, on_field: bool) -> dict:
    player = entry.get("player", {})
    stats_list = player.get("player_stats", [{}])
    stats = stats_list[0] if stats_list else {}
    team_info = player.get("team", {})
    team = team_info.get("abbrev", "")
    colours = AFL_TEAM_COLOURS.get(team, ("#555555", "#888888"))

    first = player.get("first_name", "")
    last  = player.get("last_name", "")
    name  = f"{first[0]}. {last}" if first else last

    played_status = player.get("played_status", {}) or {}
    display = (played_status.get("display") or "").lower()
    is_dnp  = display in ("did not play", "dnp")

    points       = entry.get("points", 0) or 0
    avg          = round(float(stats.get("avg") or 0), 1)
    total_points = int(stats.get("total_points") or 0)
    picked       = entry.get("picked", "")

    return {
        "name":         name,
        "first_name":   first,
        "last_name":    last,
        "team":         team,
        "colour1":      colours[0],
        "colour2":      colours[1],
        "svg_content":  SVG_TEAM_CONTENT.get(team, ""),
        "position":     entry.get("position", ""),
        "points":       points,
        "avg":          avg,
        "total_points": total_points,
        "on_field":     on_field,
        "is_dnp":       is_dnp,
        "is_emergency": picked == "emerg",
        "display":      display,
    }


def fetch_current_team(coach_name: str) -> dict:
    """
    Fetches the coach's current team from the live SC API.

    Returns a dict:
      {
        "on_field": [player_dict, ...],       # all 19 on-field players
        "bench":    [player_dict, ...],       # all bench players
        "positions": {"DEF": [...], ...},     # on-field grouped by position
        "round":    int,
      }
    Returns empty dict on any failure.
    """
    from waiver.config import LADDER_LEAGUE_ID, SEASON_YEAR

    team_id = _load_coach_team_id(coach_name)
    if team_id is None:
        log.warning("No team_id found for coach %s", coach_name)
        return {}

    # ladder.csv max round = last completed round.
    # The "current team" is configured for the upcoming round (current + 1).
    current_round = _get_current_round() + 1
    token = _get_token()

    if not token:
        log.warning("No SC API token available for field view")
        return {}

    data = _fetch_ladder_round(SEASON_YEAR, LADDER_LEAGUE_ID, token, current_round)
    if not data:
        return {}

    # Find this coach's ladder entry
    team_entry = None
    for entry in data.get("ladder", []):
        ut = entry.get("userTeam") or {}
        if ut.get("id") == team_id:
            team_entry = ut
            break

    if team_entry is None:
        log.warning("team_id %s not found in round %s ladder", team_id, current_round)
        return {}

    scores = team_entry.get("scores") or {}
    all_raw = (scores.get("scoring", []) or []) + (scores.get("nonscoring", []) or [])

    # Use the "picked" field to determine lineup, not scoring/nonscoring list membership.
    # "scoring" = players who actually contributed points (played); DNP starters land in
    # "nonscoring" alongside real bench players. "picked" is the source of truth:
    #   "true"  → on field,  "false" / "emerg" → bench
    on_field = [_parse_player(e, True)  for e in all_raw if e.get("picked") == "true"]
    bench    = [_parse_player(e, False) for e in all_raw if e.get("picked") != "true"]

    # Group on-field by position, sorted by season total points DESC (matches SC display order)
    positions: dict[str, list] = {"DEF": [], "MID": [], "RUC": [], "FWD": []}
    for p in on_field:
        pos = p["position"]
        if pos in positions:
            positions[pos].append(p)
        else:
            positions.setdefault(pos, []).append(p)
    for pos in positions:
        positions[pos].sort(key=lambda p: p["avg"], reverse=True)

    return {
        "on_field":  on_field,
        "bench":     bench,
        "positions": positions,
        "round":     current_round,
    }


def load_weekly_changes(coach_name: str, current_team: dict) -> dict:
    """
    Diffs the current team against the roster that played last completed round.
    Returns {"round": int, "ins": [{"name", "team", "position"}, ...], "outs": [...]}
    or {} on any failure / no data.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    pmr_path = project_root / "data" / "live" / "player_match_results.csv"

    try:
        import pandas as pd

        last_round = _get_current_round()

        df = pd.read_csv(pmr_path, low_memory=False)
        df["round_x"] = pd.to_numeric(df["round_x"], errors="coerce")
        prev = df[
            (df["coach_first_name"] == coach_name) &
            (df["round_x"] == last_round)
        ].copy()
        if prev.empty:
            return {}

        # Keyed by (first_name, last_name)
        prev_map = {
            (str(r["first_name"]), str(r["last_name"])): r
            for _, r in prev.iterrows()
        }

        all_current = current_team.get("on_field", []) + current_team.get("bench", [])
        curr_map = {
            (p["first_name"], p["last_name"]): p
            for p in all_current
        }

        ins = []
        for (first, last), p in curr_map.items():
            if (first, last) not in prev_map:
                name = f"{first[0]}. {last}" if first else last
                ins.append({"name": name, "team": p["team"], "position": p["position"]})

        outs = []
        for (first, last), r in prev_map.items():
            if (first, last) not in curr_map:
                name = f"{first[0]}. {last}" if first else last
                outs.append({
                    "name":     name,
                    "team":     str(r.get("team", "")),
                    "position": str(r.get("played_position", "")),
                })

        return {"round": last_round, "ins": ins, "outs": outs}

    except Exception as exc:
        log.warning("load_weekly_changes failed: %s", exc)
        return {}
