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


def build_trophy_cabinet(years: list[int]) -> dict:
    """
    Two views of the same award-win tallies:
      by_coach: one entry per coach, {coach, entries: [...], total}, each
        entry carrying group ("highlight"/"lowlight") and prominent (the
        Dawnbreaker/Spoon headline treatment), same fields the per-season
        Awards card uses — so a coach's card can give their championships
        real visual weight instead of listing 18 categories as one flat,
        equally-weighted list.
      by_award: one row per award category actually won by someone,
        {key, title, icon, group, prominent, coach_counts: {coach: n},
        total}, for the "who's dominated each category" cross-coach view
        the per-coach cards can't answer well.
    """
    from website.config import COACH_ORDER
    from website.data.awards import build_season_awards, AWARD_ICONS, LOWLIGHT_KEYS, PROMINENT_KEYS

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

    def _entry(key: str, count: int) -> dict:
        return {
            "key": key, "title": titles[key], "icon": AWARD_ICONS.get(key), "count": count,
            "group": "lowlight" if key in LOWLIGHT_KEYS else "highlight",
            "prominent": key in PROMINENT_KEYS,
        }

    by_coach = []
    for coach in COACH_ORDER:
        entries = [_entry(k, n) for k, n in sorted(counts[coach].items(), key=lambda kv: -kv[1])]
        by_coach.append({"coach": coach, "entries": entries, "total": sum(counts[coach].values())})

    by_award = []
    for key in titles:
        coach_counts = {c: counts[c].get(key, 0) for c in COACH_ORDER}
        entry = _entry(key, sum(coach_counts.values()))
        entry["coach_counts"] = coach_counts
        entry["total"] = entry.pop("count")
        by_award.append(entry)
    by_award.sort(key=lambda r: (not r["prominent"], r["group"] != "highlight", -r["total"]))

    return {"by_coach": by_coach, "by_award": by_award, "coaches": COACH_ORDER}


def build_notable_records(years: list[int], top_n: int = 5) -> dict:
    """
    All-time top-`top_n` leaderboards from regular-season fixtures (same
    scope as every other per-season stat on the site — ladder.csv/
    fixtures.csv are deliberately regular-season-only). A single record per
    category left each card mostly empty and hid whether one coach shows up
    repeatedly — a ranked list surfaces that. Returns {} if there's no
    usable data at all.

    "draws" lists the actual games (deduped — each draw appears twice, once
    per team, in the source data) rather than just a count, since there are
    few enough in league history that listing them is more interesting than
    a bare number.
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

    def _rows(df: pd.DataFrame, with_margin: bool = False) -> list[dict]:
        rows = []
        for _, row in df.iterrows():
            d = {
                "coach":           row["coach_first_name"],
                "opponent":        row.get("opposition_team_coach"),
                "points":          row["team_points"],
                "opponent_points": row["opposition_team_points"],
                "year":            int(row["year"]),
                "round":           int(row["round_number"]),
            }
            if with_margin:
                d["margin"] = row["margin"]
            rows.append(d)
        return rows

    # Every fixture appears twice in all_fx (once per team's perspective), so
    # margin-based leaderboards (blowout/closest/draws) need deduping to one
    # row per actual game first, or the same handful of games would fill
    # most of the top 5 twice over. highest/lowest don't have this problem —
    # each row there is legitimately one team's own score for that round.
    games = all_fx.sort_values("win", ascending=False)
    games["_pair_key"] = games.apply(
        lambda r: (r["year"], r["round_number"],
                   tuple(sorted([str(r["coach_first_name"]), str(r.get("opposition_team_coach"))]))),
        axis=1,
    )
    games = games.drop_duplicates(subset="_pair_key", keep="first")

    highest = all_fx.nlargest(top_n, "team_points")
    lowest  = all_fx.nsmallest(top_n, "team_points")
    blowout = games.nlargest(top_n, "margin")
    nonzero = games[games["margin"] > 0]
    closest = nonzero.nsmallest(top_n, "margin")
    draws   = games[games["margin"] == 0].sort_values(["year", "round_number"])

    return {
        "highest": _rows(highest),
        "lowest":  _rows(lowest),
        "blowout": _rows(blowout, with_margin=True),
        "closest": _rows(closest, with_margin=True),
        "draws":   _rows(draws),
    }


def build_championship_drought(season_list: list[dict]) -> list[dict]:
    """
    Years since each coach's last title, sorted longest-waiting first.
    Coaches who've never won lead the list (an infinite drought beats any
    finite one) and are flagged never_won=True — but still carry a
    seasons_played count instead of just a bare "-", since "never won in 14
    seasons" is the actual story, not just a boolean.

    bar_value/bar_pct are included for a horizontal-bar leaderboard: for a
    never-won coach, bar_value is their seasons played (so the bar reflects
    how long they've been waiting, not an arbitrary max); for everyone else
    it's their numeric drought. bar_pct is relative to the largest bar_value
    in the list, so droughts are directly comparable at a glance.
    """
    from website.config import COACH_ORDER

    if not season_list:
        return []
    current_year = max(s["year"] for s in season_list)
    seasons_played = len(season_list)
    last_title: dict[str, int] = {}
    for s in season_list:
        champ = s.get("champion")
        if champ and (champ not in last_title or s["year"] > last_title[champ]):
            last_title[champ] = s["year"]

    rows = []
    for coach in COACH_ORDER:
        if coach in last_title:
            rows.append({"coach": coach, "last_title": last_title[coach],
                         "drought": current_year - last_title[coach], "never_won": False,
                         "bar_value": current_year - last_title[coach]})
        else:
            rows.append({"coach": coach, "last_title": None, "drought": None, "never_won": True,
                         "seasons_played": seasons_played, "bar_value": seasons_played})
    rows.sort(key=lambda r: (not r["never_won"], -(r["drought"] or 0)))

    max_bar = max((r["bar_value"] for r in rows), default=1) or 1
    for r in rows:
        r["bar_pct"] = round(r["bar_value"] / max_bar * 100)
    return rows


def build_bridesmaid_tally(season_list: list[dict]) -> list[dict]:
    """
    Runner-up finish counts per coach, most first. Coaches with zero are
    omitted. bar_pct is relative to the highest count, for a horizontal-bar
    leaderboard (same visual language as Championship Drought).
    """
    from website.config import COACH_ORDER

    counts = {c: 0 for c in COACH_ORDER}
    for s in season_list:
        ru = s.get("runner_up")
        if ru in counts:
            counts[ru] += 1
    rows = [{"coach": c, "count": n} for c, n in counts.items() if n > 0]
    rows.sort(key=lambda r: -r["count"])

    max_count = max((r["count"] for r in rows), default=1) or 1
    for r in rows:
        r["bar_pct"] = round(r["count"] / max_count * 100)
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


def build_favourite_players(years: list[int], top_n: int = 3) -> list[dict]:
    """
    Each coach's most-rostered players across every completed season (any
    round they were on the coach's roster, bench or on-field) — "who has
    this coach relied on the most, ever." Grouped by feed_id rather than
    name, since 2016-2019 (workbook-sourced) rows store only a first-name
    initial ("P" Dangerfield) while 2020+ rows have the full first name —
    grouping by name string would silently split one player's history into
    two. Years are processed most-recent-first so the display name prefers
    the fuller spelling when a player appears in both eras.
    """
    from website.config import COACH_ORDER
    from website.data.loader import load_player_matches

    frames = []
    for year in sorted(years, reverse=True):
        pm = load_player_matches(year)
        if not pm.empty:
            frames.append(pm)
    if not frames:
        return []

    all_pm = pd.concat(frames, ignore_index=True)
    # A handful of rows can still be missing feed_id (see ingestion notes) —
    # fall back to a name-based key for just those so the player isn't
    # silently dropped from the tally.
    all_pm["player_key"] = all_pm["feed_id"].astype("string").fillna(
        all_pm["first_name"].astype(str) + "|" + all_pm["last_name"].astype(str)
    )

    grouped = (
        all_pm.groupby(["coach_first_name", "player_key"])
        .agg(rounds=("round_x", "count"),
             first_name=("first_name", "first"),
             last_name=("last_name", "first"))
        .reset_index()
    )

    favourites = []
    for coach in COACH_ORDER:
        cdf = grouped[grouped["coach_first_name"] == coach].sort_values("rounds", ascending=False).head(top_n)
        players = [
            {"name": f"{r.first_name} {r.last_name}".strip(), "rounds": int(r.rounds)}
            for r in cdf.itertuples()
        ]
        favourites.append({"coach": coach, "players": players})
    return favourites
