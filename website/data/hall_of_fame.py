"""
hall_of_fame.py — All-time, cross-season stats for the Hall of Fame page
(Trophy Cabinet, Notable Records, Championship Drought, Bridesmaid Tally,
Head-to-Head). Built on top of per-season data this site already computes
(website.data.awards, loader, finals) rather than a separate pipeline.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)


def completed_years() -> list[int]:
    """Every season (2016+) whose outcome is actually decided — the same
    season_phase() check season.py uses to gate the per-season Awards card,
    applied across the whole history so an in-progress current season isn't
    counted toward all-time tallies."""
    from website.data.loader import list_available_seasons, load_ladder
    from website.data.standings import get_current_round
    from website.data.finals import build_finals_bracket, season_phase

    years = []
    for year in list_available_seasons():
        ladder = load_ladder(year)
        current_round = get_current_round(ladder)
        try:
            finals_bracket = build_finals_bracket(ladder, year=year)
        except Exception:
            finals_bracket = None
        if season_phase(year, current_round, finals_bracket) == "completed":
            years.append(year)
    return sorted(years)


def build_trophy_cabinet(years: list[int]) -> list[dict]:
    """
    One entry per coach: every award key they've won at least once across
    `years`, with a count, sorted by count descending (their biggest
    specialty first). Returns [{coach, entries: [{key, title, icon, count}],
    total}, ...] in config.COACH_ORDER.
    """
    from website.config import COACH_ORDER
    from website.data.awards import build_season_awards, AWARD_ICONS

    counts: dict[str, dict[str, int]] = {c: {} for c in COACH_ORDER}
    titles: dict[str, str] = {}
    for year in years:
        try:
            awards = build_season_awards(year)
        except Exception:
            log.warning("build_season_awards failed for %s in trophy cabinet", year, exc_info=True)
            continue
        for a in awards:
            coach = a["coach"]
            if coach not in counts:
                continue
            counts[coach][a["key"]] = counts[coach].get(a["key"], 0) + 1
            titles[a["key"]] = a["title"]

    cabinet = []
    for coach in COACH_ORDER:
        entries = [
            {"key": k, "title": titles[k], "icon": AWARD_ICONS.get(k), "count": n}
            for k, n in sorted(counts[coach].items(), key=lambda kv: -kv[1])
        ]
        cabinet.append({"coach": coach, "entries": entries, "total": sum(counts[coach].values())})
    return cabinet


def build_notable_records(years: list[int]) -> dict:
    """
    All-time extremes from regular-season fixtures (same scope as every
    other per-season stat on the site — ladder.csv/fixtures.csv are
    deliberately regular-season-only). Returns None for any record with no
    usable data.
    """
    from website.data.loader import load_fixtures

    frames = []
    for year in years:
        fx = load_fixtures(year)
        if not fx.empty:
            fx = fx.copy()
            fx["year"] = year
            frames.append(fx)
    if not frames:
        return {}

    all_fx = pd.concat(frames, ignore_index=True)
    all_fx["margin"] = (all_fx["team_points"] - all_fx["opposition_team_points"]).abs()

    def _fmt(row, with_margin=False) -> dict:
        d = {
            "coach":    row["coach_first_name"],
            "opponent": row.get("opposition_team_coach"),
            "points":   row["team_points"],
            "year":     int(row["year"]),
            "round":    int(row["round_number"]),
        }
        if with_margin:
            d["margin"] = row["margin"]
        return d

    highest = all_fx.loc[all_fx["team_points"].idxmax()]
    lowest  = all_fx.loc[all_fx["team_points"].idxmin()]
    blowout = all_fx.loc[all_fx["margin"].idxmax()]
    nonzero = all_fx[all_fx["margin"] > 0]
    closest = nonzero.loc[nonzero["margin"].idxmin()] if not nonzero.empty else None
    draws   = int((all_fx["margin"] == 0).sum() // 2)

    return {
        "highest": _fmt(highest),
        "lowest":  _fmt(lowest),
        "blowout": _fmt(blowout, with_margin=True),
        "closest": _fmt(closest, with_margin=True) if closest is not None else None,
        "draws":   draws,
    }


def build_championship_drought(season_list: list[dict]) -> list[dict]:
    """
    Years since each coach's last title, sorted longest-waiting first.
    Coaches who've never won lead the list (an infinite drought beats any
    finite one) and are flagged never_won=True instead of a numeric drought.
    """
    from website.config import COACH_ORDER

    if not season_list:
        return []
    current_year = max(s["year"] for s in season_list)
    last_title: dict[str, int] = {}
    for s in season_list:
        champ = s.get("champion")
        if champ and (champ not in last_title or s["year"] > last_title[champ]):
            last_title[champ] = s["year"]

    rows = []
    for coach in COACH_ORDER:
        if coach in last_title:
            rows.append({"coach": coach, "last_title": last_title[coach],
                         "drought": current_year - last_title[coach], "never_won": False})
        else:
            rows.append({"coach": coach, "last_title": None, "drought": None, "never_won": True})
    rows.sort(key=lambda r: (not r["never_won"], -(r["drought"] or 0)))
    return rows


def build_bridesmaid_tally(season_list: list[dict]) -> list[dict]:
    """Runner-up finish counts per coach, most first. Coaches with zero are omitted."""
    from website.config import COACH_ORDER

    counts = {c: 0 for c in COACH_ORDER}
    for s in season_list:
        ru = s.get("runner_up")
        if ru in counts:
            counts[ru] += 1
    rows = [{"coach": c, "count": n} for c, n in counts.items() if n > 0]
    rows.sort(key=lambda r: -r["count"])
    return rows


def build_head_to_head(years: list[int]) -> dict:
    """
    Every coach's all-time W-L-D record against every other coach, across
    regular season AND finals (unlike the rest of the site's stats, this is
    about full rivalry history, not ladder-affecting results).

    Returns {"coaches": [...], "cells": {coach: {opponent: {"w","l","d"}}}}.
    """
    from website.config import COACH_ORDER
    from website.data.loader import load_fixtures
    from website.data.finals import load_finals_fixtures

    record: dict[str, dict[str, dict[str, int]]] = {
        c: {o: {"w": 0, "l": 0, "d": 0} for o in COACH_ORDER if o != c} for c in COACH_ORDER
    }

    def _tally(coach_a, pts_a, coach_b, pts_b):
        if coach_a not in record or coach_b not in record[coach_a]:
            return
        if pts_a > pts_b:
            record[coach_a][coach_b]["w"] += 1
            record[coach_b][coach_a]["l"] += 1
        elif pts_a < pts_b:
            record[coach_a][coach_b]["l"] += 1
            record[coach_b][coach_a]["w"] += 1
        else:
            record[coach_a][coach_b]["d"] += 1
            record[coach_b][coach_a]["d"] += 1

    for year in years:
        fx = load_fixtures(year)
        if not fx.empty:
            # Each regular-season game appears as 2 rows (one per team) —
            # only tally from one side per fixture to avoid double-counting.
            seen = set()
            for _, r in fx.iterrows():
                a, b = r["coach_first_name"], r.get("opposition_team_coach")
                key = tuple(sorted([str(a), str(b), str(r["round_number"])]))
                if key in seen:
                    continue
                seen.add(key)
                _tally(a, r["team_points"], b, r["opposition_team_points"])

        for f in load_finals_fixtures(year):
            _tally(f["coach1"], f["score1"], f["coach2"], f["score2"])

    return {"coaches": COACH_ORDER, "cells": record}
