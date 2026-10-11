"""
finals.py — Builds the McIntyre finals bracket (3 rounds after the regular
season) from saved match JSON.
"""
import json

from website.config import SEASON_YEAR, finals_json_dir, season_meta
from website.data.standings import compute_standings


def _load_round_fixtures(json_dir, rnd: int) -> list[dict]:
    path = json_dir / f"match_json_rd{rnd}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for f in data.get("fixtures") or []:
        t1 = f.get("user_team1") or {}
        t2 = f.get("user_team2") or {}
        c1 = (t1.get("user") or {}).get("first_name", "")
        c2 = (t2.get("user") or {}).get("first_name", "")
        p1 = ((t1.get("stats") or [{}])[0]).get("points", 0) or 0
        p2 = ((t2.get("stats") or [{}])[0]).get("points", 0) or 0
        if not c1 or not c2:
            continue
        matches.append({
            "coach1": c1, "score1": p1,
            "coach2": c2, "score2": p2,
            "winner": c1 if p1 > p2 else c2,
        })
    return matches


def build_finals_bracket(ladder_df, year: int | None = None) -> dict | None:
    """
    Returns the finals bracket built from that season's saved
    json/match_json_rd{N+1..}.json (N = regular rounds), or None if finals
    haven't been played / saved yet.

    Most seasons use a 3-week McIntyre double-chance format (QF+EF in week 1,
    PF in week 2, GF in week 3). Some seasons (confirmed by hand — see
    config.SEASON_META_OVERRIDES) only ran a 2-week straight-knockout format
    (two semis in week 1, GF directly in week 2, no preliminary final).
    """
    year = year or SEASON_YEAR
    meta = season_meta(year)
    total_regular_rounds = meta["total_regular_rounds"]
    finals_weeks = meta["finals_weeks"]
    json_dir = finals_json_dir(year)

    reg_season = ladder_df[ladder_df["round"] == total_regular_rounds]
    standings = compute_standings(reg_season, total_regular_rounds)
    seed = dict(zip(standings["coach_first_name"], standings["ladder_pos"]))

    if finals_weeks == 2:
        return _build_2week_bracket(json_dir, total_regular_rounds, seed)
    return _build_3week_bracket(json_dir, total_regular_rounds, seed)


def _build_3week_bracket(json_dir, total_regular_rounds: int, seed: dict) -> dict | None:
    rd_qf, rd_pf, rd_gf = total_regular_rounds + 1, total_regular_rounds + 2, total_regular_rounds + 3

    r22 = _load_round_fixtures(json_dir, rd_qf)
    r23 = _load_round_fixtures(json_dir, rd_pf)
    r24 = _load_round_fixtures(json_dir, rd_gf)
    if not r22 or not r23 or not r24:
        return None

    qf = next((m for m in r22 if seed.get(m["coach1"], 99) <= 2 and seed.get(m["coach2"], 99) <= 2), r22[0])
    ef = next((m for m in r22 if m is not qf), r22[-1])
    pf = r23[0]
    gf = r24[0]

    for m in (qf, ef, pf, gf):
        m["seed1"] = seed.get(m["coach1"])
        m["seed2"] = seed.get(m["coach2"])

    runner_up = gf["coach1"] if gf["winner"] == gf["coach2"] else gf["coach2"]

    return {
        "format": "3week",
        "qf": {**qf, "label": "Qualifying Final"},
        "ef": {**ef, "label": "Elimination Final"},
        "pf": {**pf, "label": "Preliminary Final"},
        "gf": {**gf, "label": "Grand Final"},
        "premier": gf["winner"],
        "runner_up": runner_up,
    }


def _build_2week_bracket(json_dir, total_regular_rounds: int, seed: dict) -> dict | None:
    rd_semis, rd_gf = total_regular_rounds + 1, total_regular_rounds + 2

    r_semis = _load_round_fixtures(json_dir, rd_semis)
    r_gf = _load_round_fixtures(json_dir, rd_gf)
    if len(r_semis) < 2 or not r_gf:
        return None

    sf1 = next((m for m in r_semis if seed.get(m["coach1"], 99) <= 2 or seed.get(m["coach2"], 99) <= 2), r_semis[0])
    sf2 = next((m for m in r_semis if m is not sf1), r_semis[-1])
    gf = r_gf[0]

    for m in (sf1, sf2, gf):
        m["seed1"] = seed.get(m["coach1"])
        m["seed2"] = seed.get(m["coach2"])

    runner_up = gf["coach1"] if gf["winner"] == gf["coach2"] else gf["coach2"]

    return {
        "format": "2week",
        "sf1": {**sf1, "label": "Semi Final"},
        "sf2": {**sf2, "label": "Semi Final"},
        "gf": {**gf, "label": "Grand Final"},
        "premier": gf["winner"],
        "runner_up": runner_up,
    }
