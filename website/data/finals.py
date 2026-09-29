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
    Returns the McIntyre double-chance bracket (top 4, 3 rounds) built from
    that season's json/match_json_rd{N,N+1,N+2}.json (N = regular rounds + 1),
    or None if finals haven't been played / saved yet.
    """
    year = year or SEASON_YEAR
    total_regular_rounds = season_meta(year)["total_regular_rounds"]
    rd_qf, rd_pf, rd_gf = total_regular_rounds + 1, total_regular_rounds + 2, total_regular_rounds + 3

    json_dir = finals_json_dir(year)
    r_qf = _load_round_fixtures(json_dir, rd_qf)
    r_pf = _load_round_fixtures(json_dir, rd_pf)
    r_gf = _load_round_fixtures(json_dir, rd_gf)
    if not r_qf or not r_pf or not r_gf:
        return None
    r22, r23, r24 = r_qf, r_pf, r_gf

    reg_season = ladder_df[ladder_df["round"] == total_regular_rounds]
    standings = compute_standings(reg_season, total_regular_rounds)
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
