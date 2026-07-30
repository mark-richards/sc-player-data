"""
fixture_schedule.py — Fetches the official AFL fixture and generates
Windows Task Scheduler registration for newsletter runs.

Each round's newsletter fires at 07:00 AEST the morning after the last
game of that round completes.  The generated waiver_schedule.bat file
must be re-run as Administrator whenever it changes.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

FIXTURE_JSON_URL = "https://fixturedownload.com/feed/json/afl-2026"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAT_PATH = _PROJECT_ROOT / "waiver_schedule.bat"
LAUNCHER_BAT = _PROJECT_ROOT / "run_waiver_newsletter.bat"

# Round 0 = Opening Round (pre-season exhibition — skip)
_SKIP_ROUND_NUMBERS = {0}

# Round number (int) → zero-padded task-name tag
def _round_tag(round_num: int) -> str:
    return f"{round_num:02d}"


def fetch_fixture() -> pd.DataFrame:
    """
    Downloads the AFL 2026 fixture from fixturedownload.com/feed/json/afl-2026.

    Returns a DataFrame with columns:
        round_num     — int AFL round number (0 = Opening Round)
        game_dt_utc   — tz-aware UTC datetime
        game_dt_aest  — tz-aware Melbourne (AEST/AEDT) datetime

    Returns an empty DataFrame on failure.
    """
    try:
        resp = requests.get(FIXTURE_JSON_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Could not fetch AFL fixture: %s", exc)
        return pd.DataFrame()

    if not isinstance(data, list) or not data:
        log.warning("AFL fixture JSON was empty or unexpected format.")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(data)
        df = df.rename(columns={"RoundNumber": "round_num", "DateUtc": "date_str"})
        df["round_num"] = df["round_num"].astype(int)

        # DateUtc format: "YYYY-MM-DD HH:MM:00Z"
        df["game_dt_utc"] = pd.to_datetime(df["date_str"], utc=True)
    except Exception as exc:
        log.warning("Could not parse AFL fixture data: %s", exc)
        return pd.DataFrame()

    # Convert to Melbourne time (handles AEDT/AEST DST automatically)
    try:
        df["game_dt_aest"] = df["game_dt_utc"].dt.tz_convert("Australia/Melbourne")
    except Exception:
        # Fallback: fixed UTC+10
        df["game_dt_aest"] = df["game_dt_utc"].dt.tz_convert("Etc/GMT-10")

    log.info("AFL fixture loaded: %d games across rounds %d–%d.",
             len(df), df["round_num"].min(), df["round_num"].max())
    return df[["round_num", "game_dt_utc", "game_dt_aest"]]


def get_round_schedule(df: pd.DataFrame) -> list[dict]:
    """
    For each AFL round, finds the last game's AEST datetime and computes
    the newsletter run time: 07:00 AEST the following morning.

    Returns list of dicts (sorted by run_dt):
        round_num    — int round number
        round_tag    — zero-padded string, e.g. "03"
        last_game_dt — last game AEST datetime
        run_dt       — 07:00 AEST next morning (tz-aware)
    """
    if df.empty:
        return []

    schedule = []
    for round_num, group in df.groupby("round_num"):
        if round_num in _SKIP_ROUND_NUMBERS:
            continue
        last_game = group["game_dt_aest"].max()

        # Next day at 07:00 in the same timezone
        next_day = (last_game + timedelta(days=1)).replace(
            hour=7, minute=0, second=0, microsecond=0
        )

        schedule.append(
            {
                "round_num": round_num,
                "round_tag": _round_tag(round_num),
                "last_game_dt": last_game,
                "run_dt": next_day,
            }
        )

    schedule.sort(key=lambda x: x["run_dt"])
    return schedule


def round_is_complete(round_num: int) -> bool:
    """Returns True if all AFL matches in round_num have finished (or fixture unavailable)."""
    from datetime import timezone
    df = fetch_fixture()
    if df.empty:
        return True  # fail-open: don't block refresh if fixture fetch fails
    games = df[df["round_num"] == round_num]
    if games.empty:
        return True  # unknown round — don't block
    last_game = games["game_dt_aest"].max()
    now = datetime.now(tz=timezone.utc).astimezone(last_game.tzinfo)
    return now > last_game + timedelta(hours=3)


def generate_bat_content(schedule: list[dict]) -> str:
    """
    Generates the content of waiver_schedule.bat.

    Creates one /sc ONCE schtasks entry per future round, plus a block
    that deletes all existing WaiverWire_Rd* tasks first.
    """
    now_utc = datetime.utcnow()
    launcher = str(LAUNCHER_BAT).replace("/", "\\")

    future = [s for s in schedule if s["run_dt"].utctimetuple() > now_utc.timetuple()]

    lines = [
        "@echo off",
        "REM ================================================================",
        "REM  Waiver Wire Newsletter — Round Schedule",
        f"REM  Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by waiver/fixture_schedule.py",
        "REM  Source: fixturedownload.com/results/afl-2026",
        "REM",
        "REM  Run as Administrator to register all scheduled tasks.",
        "REM  Re-run this file whenever the newsletter flags it has changed.",
        "REM",
        "REM  Rounds scheduled:",
    ]
    for s in future:
        lines.append(
            f"REM    Rd {s['round_tag']} — last game "
            f"{s['last_game_dt'].strftime('%a %d %b %Y %H:%M')} AEST"
            f" → newsletter {s['run_dt'].strftime('%a %d %b %Y @ %H:%M')} AEST"
        )
    lines += [
        "REM ================================================================",
        "",
        "echo Registering Waiver Wire Newsletter tasks...",
        "",
        "REM Remove all existing WaiverWire_Rd* tasks (silently)",
    ]

    all_tags = [_round_tag(s["round_num"]) for s in schedule]
    for tag in all_tags:
        lines.append(f'schtasks /delete /tn "WaiverWire_Rd{tag}" /f 2>nul')

    lines += ["", "REM Create tasks for future rounds only"]

    if not future:
        lines.append("echo No future rounds to schedule — season complete.")
    else:
        for s in future:
            tag = s["round_tag"]
            # Windows schtasks /sd format matches system locale — DD/MM/YYYY for Australian locale
            run_date = s["run_dt"].strftime("%d/%m/%Y")
            run_time = s["run_dt"].strftime("%H:%M")
            task_name = f"WaiverWire_Rd{tag}"
            lines += [
                f"REM Round {s['round_num']} — "
                f"{s['last_game_dt'].strftime('%a %d %b')} last game → "
                f"newsletter {s['run_dt'].strftime('%a %d %b @ %H:%M')}",
                (
                    f'schtasks /create /tn "{task_name}"'
                    f' /tr "\\"{launcher}\\""'
                    f" /sc ONCE /sd {run_date} /st {run_time}"
                    f' /ru "%USERNAME%" /rl HIGHEST /f'
                ),
                "",
            ]

    lines += [
        "",
        "if %ERRORLEVEL% EQU 0 (",
        "    echo Tasks registered successfully.",
        ") else (",
        "    echo Some tasks may have failed — check output above.",
        ")",
        "pause",
    ]

    return "\r\n".join(lines) + "\r\n"


def update_schedule() -> bool:
    """
    Fetches the fixture, generates the bat file, writes it if changed.

    Returns True if the bat file content changed (user needs to re-run it).
    Returns False if unchanged or if the fetch failed (no-op on failure).
    """
    df = fetch_fixture()
    if df.empty:
        log.warning("Fixture fetch failed — waiver_schedule.bat not updated.")
        return False

    schedule = get_round_schedule(df)
    if not schedule:
        log.warning("No rounds found in fixture — waiver_schedule.bat not updated.")
        return False

    new_content = generate_bat_content(schedule)
    new_hash = hashlib.md5(new_content.encode("utf-8")).hexdigest()

    # Compare against existing file
    old_hash = ""
    if BAT_PATH.exists():
        try:
            old_content = BAT_PATH.read_bytes()
            old_hash = hashlib.md5(old_content).hexdigest()
        except Exception:
            pass

    if new_hash == old_hash:
        log.info("waiver_schedule.bat is up to date (no changes).")
        return False

    BAT_PATH.write_text(new_content, encoding="utf-8")
    log.info(
        "waiver_schedule.bat updated — %d future rounds scheduled. "
        "Run as Administrator: %s",
        sum(1 for s in schedule if s["run_dt"].utctimetuple() > datetime.utcnow().timetuple()),
        BAT_PATH,
    )
    return True
