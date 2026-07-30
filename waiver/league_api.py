"""
league_api.py — Fetches player ownership/availability and injury data from the SC Draft API.

Endpoints used
--------------
  /leagues/{LADDER_LEAGUE_ID}/playersStatus  → trade_status per player (owner / waiver)
      Uses the IN-SEASON league (14889), NOT the draft league (4199).
      The draft league playersStatus is stale; only the in-season league reflects
      current ownership after trades and waiver claims.
  /players?embed=player_stats,odds → full player list including live injury/suspension fields

Adapted from ingest_supercoach.py; parameterised for year and league.
"""

import base64
import datetime
import json
import logging
import requests
import urllib3

from pathlib import Path

from waiver.config import (
    API_URL_TEMPLATE,
    LADDER_LEAGUE_ID,
    LEAGUE_ID,
    MY_TEAM_ID,
    SC_API_TOKEN,
    SC_CLIENT_ID,
    SC_REFRESH_TOKEN,
    SC_SOCIAL_TOKEN,
    SEASON_YEAR,
)

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# In-process token cache: set after first successful refresh so subsequent calls
# within the same pipeline run reuse the fresh token instead of consuming the
# rotating refresh_token a second time (which would get a 401).
_active_token: str = ""

# Suppress SSL warnings (mirrors existing ingest_supercoach.py behaviour)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger(__name__)

# ── Token refresh ───────────────────────────────────────────────────────────
_TOKEN_REFRESH_URL = (
    "https://supercoach.heraldsun.com.au/{year}/api/afl/classic/v1/access_token"
)


def refresh_sc_token(
    client_id: str = SC_CLIENT_ID,
    social_token: str = SC_SOCIAL_TOKEN,
    year: int = SEASON_YEAR,
) -> str:
    """
    Obtain a fresh SC bearer token via the Auth0 social login endpoint.

    This replicates the R `sc_auth(cid, tkn)` function found in the Fantasy-Banter
    project's `supercoach_functions_V2.R`:

        POST https://supercoach.heraldsun.com.au/{year}/api/afl/classic/v1/access_token
        Body: {grant_type:"social", client_id:cid, client_secret:"", service:"auth0", token:tkn}
        Returns: {"access_token": "<bearer_token>", ...}

    Parameters
    ----------
    client_id    : Auth0 client_id  — the `cid` value from secrets.R.
                   Set SC_CLIENT_ID in .env.
    social_token : Auth0 social token — the `tkn` value from secrets.R.
                   Set SC_SOCIAL_TOKEN in .env.
    year         : AFL season year (used in the URL path).

    Returns
    -------
    str
        A fresh bearer token string, or "" on failure.
    """
    if not client_id or not social_token:
        log.debug("SC_CLIENT_ID or SC_SOCIAL_TOKEN not set; skipping token refresh.")
        return ""

    url = _TOKEN_REFRESH_URL.format(year=year)
    payload = {
        "grant_type":    "social",
        "client_id":     client_id,
        "client_secret": "",
        "service":       "auth0",
        "token":         social_token,
    }

    log.info("Refreshing SC bearer token via Auth0...")
    try:
        resp = requests.post(url, json=payload, headers=_HEADERS, verify=False, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token", "")
        if token:
            log.info("SC token refreshed successfully.")
        else:
            log.warning("Token refresh response missing 'access_token'. Response: %s", data)
        return token
    except requests.exceptions.RequestException as exc:
        log.error("Token refresh request failed: %s", exc)
        return ""
    except ValueError:
        log.error("Token refresh returned non-JSON response.")
        return ""


def check_token_health() -> list[str]:
    """
    Returns a list of warning strings about SC token expiry/validity.
    Call at pipeline startup so warnings appear in the newsletter.
    """
    warnings = []

    # Check SC_SOCIAL_TOKEN JWT expiry
    if SC_SOCIAL_TOKEN:
        try:
            parts = SC_SOCIAL_TOKEN.split(".")
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
            exp = datetime.datetime.fromtimestamp(payload["exp"])
            days_left = (exp - datetime.datetime.now()).days
            if days_left < 0:
                warnings.append(
                    f"SC_SOCIAL_TOKEN expired {abs(days_left)} days ago ({exp.date()}) — "
                    "auto-refresh is broken. Update SC_SOCIAL_TOKEN in .env from browser localStorage."
                )
            elif days_left < 14:
                warnings.append(
                    f"SC_SOCIAL_TOKEN expires in {days_left} days ({exp.date()}) — "
                    "update SC_SOCIAL_TOKEN in .env soon to keep auto-refresh working."
                )
        except Exception:
            pass

    # Check SC_API_TOKEN validity with an auth-required endpoint
    if SC_API_TOKEN:
        try:
            from waiver.config import LADDER_LEAGUE_ID
            url = f"https://www.supercoach.com.au/{SEASON_YEAR}/api/afl/draft/v1/leagues/{LADDER_LEAGUE_ID}/playersStatus"
            resp = requests.get(url, headers={**_HEADERS, "Authorization": f"Bearer {SC_API_TOKEN}"},
                                verify=False, timeout=8)
            if resp.status_code == 401:
                warnings.append(
                    "SC_API_TOKEN is expired (401) — update SC_API_TOKEN in .env from browser "
                    "DevTools → Network → any supercoach.com.au request → Authorization header."
                )
        except Exception:
            pass

    for w in warnings:
        log.warning("TOKEN HEALTH: %s", w)
    return warnings


def _persist_tokens(access_token: str, refresh_token: str) -> None:
    """Write updated access + refresh tokens back to .env."""
    try:
        text = _ENV_PATH.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        found_access = found_refresh = False
        updated = []
        for line in lines:
            if line.startswith("SC_API_TOKEN="):
                updated.append(f"SC_API_TOKEN={access_token}\n")
                found_access = True
            elif line.startswith("SC_REFRESH_TOKEN="):
                updated.append(f"SC_REFRESH_TOKEN={refresh_token}\n")
                found_refresh = True
            else:
                updated.append(line)
        content = "".join(updated)
        if not found_access:
            if not content.endswith("\n"):
                content += "\n"
            content += f"SC_API_TOKEN={access_token}\n"
        if not found_refresh:
            if not content.endswith("\n"):
                content += "\n"
            content += f"SC_REFRESH_TOKEN={refresh_token}\n"
        _ENV_PATH.write_text(content, encoding="utf-8")
        log.info("SC tokens persisted to .env (access + refresh).")
    except Exception as exc:
        log.warning("Could not persist tokens to .env: %s", exc)


def refresh_via_refresh_token(
    refresh_token: str = SC_REFRESH_TOKEN,
    client_id: str = SC_CLIENT_ID,
    year: int = SEASON_YEAR,
) -> str:
    """
    Exchange a refresh_token for a new access_token (and rotating refresh_token).
    Persists both back to .env so the next run uses the latest refresh_token.
    """
    if not refresh_token:
        return ""
    url = _TOKEN_REFRESH_URL.format(year=year)
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": "",
    }
    log.info("Refreshing SC token via refresh_token grant...")
    try:
        resp = requests.post(url, json=payload, headers=_HEADERS, verify=False, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        access = data.get("access_token", "")
        new_refresh = data.get("refresh_token", "")
        if access:
            global _active_token
            _active_token = access
            log.info("SC token refreshed via refresh_token (expires in %ds).", data.get("expires_in", 0))
            if new_refresh:
                _persist_tokens(access, new_refresh)
            return access
        log.warning("refresh_token grant returned no access_token: %s", data)
        return ""
    except requests.exceptions.RequestException as exc:
        log.error("refresh_token grant failed: %s", exc)
        return ""


def get_valid_token(year: int = SEASON_YEAR) -> str:
    """
    Returns a valid SC bearer token.

    Priority: in-process cache → refresh_token grant → social token grant → static SC_API_TOKEN.
    The refresh_token rotates on each use and is persisted back to .env automatically.
    The in-process cache prevents consuming the rotating refresh_token more than once per run.
    """
    # 0. Return cached token from earlier in this process (avoids double-consuming refresh_token)
    if _active_token:
        return _active_token

    # 1. Try rotating refresh_token (best — fully automatic, no expiry issues)
    if SC_REFRESH_TOKEN:
        token = refresh_via_refresh_token(year=year)
        if token:
            return token
        log.warning("refresh_token grant failed; trying social token...")

    # 2. Try Auth0 social token (lasts ~85 days)
    if SC_CLIENT_ID and SC_SOCIAL_TOKEN:
        token = refresh_sc_token(year=year)
        if token:
            return token
        log.warning("Social token refresh failed; falling back to static SC_API_TOKEN.")

    # 3. Static token (manual update required every ~7 days)
    if SC_API_TOKEN:
        return SC_API_TOKEN

    log.error("No SC bearer token available. Set SC_REFRESH_TOKEN in .env.")
    return ""


_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}

_PLAYERS_URL_TEMPLATE = (
    "https://www.supercoach.com.au/{year}/api/afl/draft/v1/players"
    "?embed=player_stats,odds"
)


def fetch_player_status(
    year: int = SEASON_YEAR,
    league_id: str = LADDER_LEAGUE_ID,
    token: str = SC_API_TOKEN,
) -> dict[str, dict]:
    """
    Calls the SC in-season league playersStatus endpoint.

    Uses LADDER_LEAGUE_ID (14889) by default — this is the authoritative source
    for current ownership after trades and waiver claims. The draft league (4199)
    playersStatus is stale and should not be used for ownership checks.

    Returns
    -------
    dict[str, dict]
        Mapping of SC player_id (str) → {"trade_status": str, "user_team_id": int | None}.
        trade_status is one of: "owner", "waiver".
        Returns an empty dict on any failure.
    """
    if not token:
        log.error("SC_API_TOKEN is not set. Cannot fetch player status.")
        return {}

    url = API_URL_TEMPLATE.format(year=year, league_id=league_id)
    headers = {**_HEADERS, "Authorization": f"Bearer {token}"}

    log.info("Fetching player status from %s", url)
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        raw: dict = response.json()
    except requests.exceptions.RequestException as exc:
        log.error("SC API request failed: %s", exc)
        return {}
    except ValueError:
        log.error("SC API returned non-JSON response.")
        return {}

    status_map: dict[str, dict] = {}
    for player_id, details in raw.items():
        if isinstance(details, dict):
            status_map[str(player_id)] = {
                "trade_status": details.get("trade_status", ""),
                "user_team_id": details.get("user_team_id"),
            }
        else:
            status_map[str(player_id)] = {"trade_status": str(details), "user_team_id": None}

    # In SC Draft leagues: "waiver" = undrafted/available; "owner" = drafted.
    free_agents = sum(1 for v in status_map.values() if v["trade_status"] == "waiver")
    log.info(
        "Fetched status for %d players; %d are available (waiver).", len(status_map), free_agents
    )
    return status_map


def fetch_league_teams(
    year: int = SEASON_YEAR,
    ladder_league_id: str = LADDER_LEAGUE_ID,
    token: str = SC_API_TOKEN,
    round_num: int = 1,
) -> list[dict]:
    """
    Fetches the list of teams from the SC in-season scoring league (ladderAndFixtures).

    Uses /leagues/{ladder_league_id}/ladderAndFixtures?round={round_num}&scores=true.
    Note: LADDER_LEAGUE_ID (14889) is the in-season scoring league; LEAGUE_ID (4199)
    is the draft league used for playersStatus.

    Returns
    -------
    list of dicts with keys: coach_team_id, coach_id, coach_first_name, coach_team_name.
    Returns an empty list on failure.
    """
    if not token:
        log.error("SC_API_TOKEN is not set. Cannot fetch league teams.")
        return []

    url = (
        f"https://www.supercoach.com.au/{year}/api/afl/draft/v1"
        f"/leagues/{ladder_league_id}/ladderAndFixtures?round={round_num}&scores=true"
    )
    headers = {**_HEADERS, "Authorization": f"Bearer {token}"}

    log.info("Fetching league teams from %s", url)
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        raw: dict = response.json()
    except requests.exceptions.RequestException as exc:
        log.error("SC league teams API request failed: %s", exc)
        return []
    except ValueError:
        log.error("SC league teams API returned non-JSON response.")
        return []

    ladder = raw.get("ladder", []) if isinstance(raw, dict) else raw
    if not isinstance(ladder, list):
        log.error("SC ladderAndFixtures API returned unexpected format.")
        return []

    teams = []
    for entry in ladder:
        team_obj = entry.get("userTeam") or entry
        user_obj = team_obj.get("user") or {}

        team_id = team_obj.get("id") or entry.get("user_team_id")
        coach_id = user_obj.get("id") or team_obj.get("user_id")
        coach_name = user_obj.get("first_name") or ""
        team_name = team_obj.get("teamname") or ""

        if team_id:
            teams.append({
                "coach_team_id":    team_id,
                "coach_id":         coach_id,
                "coach_first_name": coach_name,
                "coach_team_name":  team_name,
            })

    log.info("Fetched %d teams from ladder league %s (round %d).", len(teams), ladder_league_id, round_num)
    return teams


def fetch_my_lineup(
    year: int = SEASON_YEAR,
    user_team_id: int = MY_TEAM_ID,
    round_num: int = 1,
    token: str = SC_API_TOKEN,
) -> dict[str, str]:
    """
    Fetches the on-field/bench status for each player on my team.

    Uses /userteams/{user_team_id}/statsPlayers?round={round_num}.

    Returns
    -------
    dict[str, str]
        Mapping of SC player_id (str) → picked status:
            "true"  = on field
            "false" = bench
            "emerg" = emergency (bench, nominated as emergency)
        Returns an empty dict on failure (caller should fall back gracefully).
    """
    if not token:
        log.error("SC_API_TOKEN is not set. Cannot fetch lineup.")
        return {}
    if not user_team_id:
        log.error("MY_TEAM_ID is not set. Cannot fetch lineup.")
        return {}

    url = (
        f"https://www.supercoach.com.au/{year}/api/afl/draft/v1"
        f"/userteams/{user_team_id}/statsPlayers?round={round_num}"
    )
    headers = {**_HEADERS, "Authorization": f"Bearer {token}"}

    log.info("Fetching lineup from %s", url)
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        raw: dict = response.json()
    except requests.exceptions.RequestException as exc:
        log.error("statsPlayers API request failed: %s", exc)
        return {}
    except ValueError:
        log.error("statsPlayers API returned non-JSON response.")
        return {}

    players = raw.get("players", []) if isinstance(raw, dict) else []
    lineup: dict[str, str] = {}
    for p in players:
        pid = str(p.get("player_id", ""))
        picked = str(p.get("picked", ""))
        if pid and picked:
            lineup[pid] = picked

    on_field = sum(1 for v in lineup.values() if v == "true")
    bench = sum(1 for v in lineup.values() if v in ("false", "emerg"))
    log.info("Lineup fetched: %d on field, %d bench (%d total).", on_field, bench, len(lineup))
    return lineup


def fetch_player_injuries(
    year: int = SEASON_YEAR,
    token: str = SC_API_TOKEN,
) -> dict[str, dict]:
    """
    Fetches live injury and suspension data for all players from the SC players API.

    Returns
    -------
    dict[str, dict]
        Mapping of SC player_id (str) → injury info dict with keys:
            status  : "Injury" | "Suspended" | None
            text    : human-readable injury/suspension description, e.g.
                      "Knee soreness. Expected recovery time is to be confirmed."
        Returns an empty dict on any failure (caller treats all players as healthy).
    """
    if not token:
        log.error("SC_API_TOKEN is not set. Cannot fetch player injuries.")
        return {}

    url = _PLAYERS_URL_TEMPLATE.format(year=year)
    headers = {**_HEADERS, "Authorization": f"Bearer {token}"}

    log.info("Fetching live player injury data from SC API...")
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=60)
        response.raise_for_status()
        players: list[dict] = response.json()
    except requests.exceptions.RequestException as exc:
        log.error("SC players API request failed: %s", exc)
        return {}
    except ValueError:
        log.error("SC players API returned non-JSON response.")
        return {}

    injury_map: dict[str, dict] = {}
    for player in players:
        sc_id = str(player.get("id", ""))
        if not sc_id:
            continue
        injury_map[sc_id] = {
            "status": player.get("injury_suspension_status"),   # "Injury" | "Suspended" | None
            "text":   player.get("injury_suspension_status_text") or "",
        }

    injured_count = sum(1 for v in injury_map.values() if v["status"])
    log.info(
        "Fetched injury data for %d players; %d have active injury/suspension.",
        len(injury_map), injured_count,
    )
    return injury_map
