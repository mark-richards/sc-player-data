"""
finals.py — Builds the McIntyre finals bracket (Rd22-24) from saved match JSON.
"""
import json

from website.config import DATA_LIVE_DIR, TOTAL_REGULAR_ROUNDS
from website.data.standings import compute_standings

FINALS_ROUNDS = {22: "Week 1", 23: "Preliminary Final", 24: "Grand Final"}


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


def build_finals_bracket(ladder_df) -> dict | None:
    """
    Returns the McIntyre double-chance bracket (top 4, 3 rounds) built from
    data/live/json/match_json_rd{22,23,24}.json, or None if finals haven't
    been played / saved yet.
    """
    json_dir = DATA_LIVE_DIR / "json"
    r22 = _load_round_fixtures(json_dir, 22)
    r23 = _load_round_fixtures(json_dir, 23)
    r24 = _load_round_fixtures(json_dir, 24)
    if not r22 or not r23 or not r24:
        return None

    reg_season = ladder_df[ladder_df["round"] == TOTAL_REGULAR_ROUNDS]
    standings = compute_standings(reg_season, TOTAL_REGULAR_ROUNDS)
    seed = dict(zip(standings["coach_first_name"], standings["ladder_pos"]))

    qf = next((m for m in r22 if seed.get(m["coach1"], 99) <= 2 and seed.get(m["coach2"], 99) <= 2), r22[0])
    ef = next((m for m in r22 if m is not qf), r22[-1])
    pf = r23[0]
    gf = r24[0]

    for m in (qf, ef, pf, gf):
        m["seed1"] = seed.get(m["coach1"])
        m["seed2"] = seed.get(m["coach2"])

    runner_up = gf["coach1"] if gf["winner"] == gf["coach2"] else gf["coach2"]

    return {
        "qf": {**qf, "label": "Qualifying Final"},
        "ef": {**ef, "label": "Elimination Final"},
        "pf": {**pf, "label": "Preliminary Final"},
        "gf": {**gf, "label": "Grand Final"},
        "premier": gf["winner"],
        "runner_up": runner_up,
    }
