"""
hall_of_fame.py — All-time, cross-season stats for the Hall of Fame page
(Trophy Cabinet, Notable Records, Championship Drought, Bridesmaid Tally,
Head-to-Head). Built on top of per-season data this site already computes
(website.data.awards, loader, finals) rather than a separate pipeline.
"""
import json
import logging
from datetime import datetime

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Injected from app.py after Flask-Caching is set up. build_notable_season_records()
# runs a Monte Carlo sim per year, so it's memoized the same way awards.py
# memoizes build_season_awards().
cache = None


def _memoize(fn):
    def wrapper(*args, **kwargs):
        if cache is not None:
            return cache.memoize()(fn)(*args, **kwargs)
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


_PHOTO_CACHE_PATH = None  # set lazily from website.config to avoid a circular import at module load


def _photo_cache_path():
    global _PHOTO_CACHE_PATH
    if _PHOTO_CACHE_PATH is None:
        from website.config import PROJECT_ROOT
        _PHOTO_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "afl_photo_cache.json"
    return _PHOTO_CACHE_PATH


def _load_photo_cache() -> dict:
    path = _photo_cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_photo_cache(cache: dict) -> None:
    path = _photo_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def resolve_player_photo_url(feed_id: int, cache: dict, years_back: int = 6) -> str | None:
    """
    Resolves a player headshot URL from AFL.com.au's own Champion-Data-keyed
    photo CDN, trying the current season's photo batch first and falling
    back through earlier seasons for anyone not in this year's batch
    (retired, delisted, or simply not yet photographed) — AFL.com.au
    confirmed to still serve older seasons' headshots at the same path.

    Pattern: https://s.afl.com.au/.../ChampIDImages/AFL/{year}014/{feed_id}.png

    Result (including a miss) is cached by feed_id in `cache` so repeat
    builds don't re-hit the CDN for players already resolved.
    """
    key = str(feed_id)
    if key in cache:
        return cache[key]

    base = "https://s.afl.com.au/staticfile/AFL%20Tenant/AFL/Players/ChampIDImages/AFL"
    current_year = datetime.now().year
    url = None
    for year in range(current_year, current_year - years_back, -1):
        candidate = f"{base}/{year}014/{feed_id}.png"
        try:
            resp = requests.head(candidate, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                url = candidate
                break
        except requests.RequestException:
            continue

    cache[key] = url
    return url


def completed_years() -> list[int]:
    """Every season whose outcome is actually decided — the same
    season_phase() check season.py uses to gate the per-season Awards card,
    applied across the whole history so an in-progress current season isn't
    counted toward all-time tallies.

    Years before 2016 (2013-2015) have no ingested fixtures/ladder data at
    all — list_available_seasons() never sees them — but honour_board.py's
    _HARDCODED has their champion/runner-up/spoon on record regardless, so
    they're included too. build_season_awards() already degrades gracefully
    for a year with no fixtures (it falls back to get_season_result() for
    just the Dawnbreaker award and skips every other, fixture-dependent
    award) — the earlier fixtures-only version of this
    function silently dropped those years from the Trophy Cabinet and
    similar all-time tallies entirely, undercounting real championships
    (e.g. Simon's 2013 and 2014 titles)."""
    from website.data.loader import list_available_seasons, load_ladder
    from website.data.standings import get_current_round
    from website.data.finals import build_finals_bracket, season_phase
    from website.data.honour_board import hardcoded_season_years

    years = hardcoded_season_years()
    for year in list_available_seasons():
        ladder = load_ladder(year)
        current_round = get_current_round(ladder)
        try:
            finals_bracket = build_finals_bracket(ladder, year=year)
        except Exception:
            finals_bracket = None
        if season_phase(year, current_round, finals_bracket) == "completed":
            years.add(year)
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

    "closest_finals" is the one exception to the regular-season scope —
    finals games live in a separate source (build_finals_bracket) entirely,
    since ladder.csv/fixtures.csv never include them. Its "round" field is
    the stage abbreviation (QF/EF/PF/SF/GF), not a round number.
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
    # margin-based leaderboards (closest/draws) need deduping to one row per
    # actual game first, or the same handful of games would fill most of the
    # top 5 twice over. "highest" doesn't have this problem — each row there
    # is legitimately one team's own score for that round.
    games = all_fx.sort_values("win", ascending=False)
    games["_pair_key"] = games.apply(
        lambda r: (r["year"], r["round_number"],
                   tuple(sorted([str(r["coach_first_name"]), str(r.get("opposition_team_coach"))]))),
        axis=1,
    )
    games = games.drop_duplicates(subset="_pair_key", keep="first")

    highest = all_fx.nlargest(top_n, "team_points")
    nonzero = games[games["margin"] > 0]
    closest = nonzero.nsmallest(top_n, "margin")
    draws   = games[games["margin"] == 0].sort_values(["year", "round_number"])

    # Finals games live outside ladder.csv/fixtures.csv entirely. Built from
    # build_finals_bracket (not load_finals_fixtures) specifically so each
    # game can be labelled with its actual stage (QF/EF/PF/GF, or SF/GF for
    # a 2-week finals format) — a 3-week format's week 1 has two games
    # (QF and EF) in the SAME round number, distinguished only by which two
    # coaches are playing (the higher seeds get the QF), so the round number
    # alone can't tell them apart the way load_finals_fixtures returns it.
    from website.data.finals import build_finals_bracket
    from website.data.loader import load_ladder

    _STAGE_ABBREV = {
        "Qualifying Final":   "QF",
        "Elimination Final":  "EF",
        "Preliminary Final":  "PF",
        "Semi Final":         "SF",
        "Grand Final":        "GF",
    }
    finals_rows = []
    for year in years:
        ladder = load_ladder(year)
        if ladder.empty:
            continue
        try:
            bracket = build_finals_bracket(ladder, year=year)
        except Exception:
            bracket = None
        if not bracket:
            continue
        for key, m in bracket.items():
            if key in ("format", "premier", "runner_up") or not isinstance(m, dict):
                continue
            margin = abs(m["score1"] - m["score2"])
            if margin <= 0:
                continue
            won_by_1 = m["winner"] == m["coach1"]
            finals_rows.append({
                "coach":           m["winner"],
                "opponent":        m["coach2"] if won_by_1 else m["coach1"],
                "points":          m["score1"] if won_by_1 else m["score2"],
                "opponent_points": m["score2"] if won_by_1 else m["score1"],
                "year":            year,
                "round":           _STAGE_ABBREV.get(m["label"], m["label"]),
                "margin":          margin,
            })
    closest_finals = sorted(finals_rows, key=lambda r: r["margin"])[:top_n]

    return {
        "highest":        _rows(highest),
        "closest":        _rows(closest, with_margin=True),
        "draws":          _rows(draws),
        "closest_finals": closest_finals,
    }


@_memoize
def build_notable_season_records(years: list[int], top_n: int = 5) -> dict:
    """
    All-time top-`top_n` leaderboards of per-coach-SEASON stats (as opposed
    to build_notable_records()'s per-GAME stats) — reuses the same
    computations behind several season awards (website.data.awards) so a
    coach's card there and their entry here tell the same story at a
    different scope. Returns {} if there's no usable data at all.

    dominant: lowest avg_score_rank (mean weekly score rank, 1-8, from
      build_fixture_strength — the Fixture Difficulty & Luck table's own
      "Avg Score Rank" column) in a season, best-first. Measures weekly
      scoring dominance, independent of ladder luck.

    unluckiest_spoon: lowest spoon_pct (Monte Carlo modelled chance of
      finishing last, from build_monte_carlo_schedule_sim) for the coach who
      actually got the spoon that season — paired with "rival", whichever
      other coach had the highest spoon_pct that year without getting it:
      the model's own pick for who "should've" got it instead.

    finals_dodger: most negative finals-probability surprise (actual finals
      outcome minus modelled top4_pct chance, same metric as the Unlucky:
      Finals Dodger season award) — missed finals despite the model rating
      them a strong chance, worst-first. Only included when it clears
      UNLUCKY_LUCK_THRESHOLD, same as the season award.

    lucky_wins / unlucky_losses / sunday_scaries: highest single-season
      count of each (see awards.py's _lucky_unlucky_results /
      _close_calls_sunday_scaries, which use the same underlying counts).

    win_streak: longest single-season run of consecutive wins (see awards.py's
      _best_run / win_streaks, which use the same underlying streak count).
    """
    from website.config import season_meta, FINALS_TOP_N
    from website.data.loader import load_fixtures
    from website.data.fixture_strength import build_fixture_strength, build_monte_carlo_schedule_sim
    from website.data.awards import close_call_counts, win_streaks, UNLUCKY_LUCK_THRESHOLD

    dominant, spoon_rows, dodger_rows, lucky_rows, unlucky_rows, scaries_rows, streak_rows = \
        [], [], [], [], [], [], []

    for year in years:
        fixtures = load_fixtures(year)
        if fixtures.empty:
            continue
        reg = fixtures[fixtures["round_number"] <= season_meta(year)["total_regular_rounds"]]
        if reg.empty:
            continue

        try:
            fs = build_fixture_strength(reg, year)
        except Exception:
            log.warning("build_fixture_strength failed for %s notable records", year, exc_info=True)
            fs = []
        for row in fs:
            dominant.append({"coach": row["coach"], "year": year, "value": row["avg_score_rank"],
                              "label": f"{row['avg_score_rank']:.2f} avg weekly rank"})
            if row["lucky_wins"] > 0:
                n = row["lucky_wins"]
                lucky_rows.append({"coach": row["coach"], "year": year, "value": n,
                                    "label": f"{n} lucky win{'s' if n != 1 else ''}"})
            if row["unlucky_losses"] > 0:
                n = row["unlucky_losses"]
                unlucky_rows.append({"coach": row["coach"], "year": year, "value": n,
                                      "label": f"{n} unlucky loss{'es' if n != 1 else ''}"})

        try:
            mc = build_monte_carlo_schedule_sim(reg, year=year)
        except Exception:
            log.warning("build_monte_carlo_schedule_sim failed for %s notable records", year, exc_info=True)
            mc = []
        if mc:
            spoon_rank = max(r["actual_rank"] for r in mc)
            holder = next((r for r in mc if r["actual_rank"] == spoon_rank), None)
            rivals = [r for r in mc if r["actual_rank"] != spoon_rank]
            if holder and rivals:
                rival = max(rivals, key=lambda r: r["spoon_pct"])
                spoon_rows.append({
                    "coach": holder["coach"], "year": year, "value": holder["spoon_pct"],
                    "rival": rival["coach"], "rival_value": rival["spoon_pct"],
                    "label": f"only a {holder['spoon_pct']:.0f}% modelled chance "
                             f"({rival['coach']} had {rival['spoon_pct']:.0f}%)",
                })

            for r in mc:
                made_finals = 1 if r["actual_rank"] <= FINALS_TOP_N else 0
                surprise = made_finals - r["top4_pct"] / 100
                if surprise <= -UNLUCKY_LUCK_THRESHOLD:
                    dodger_rows.append({
                        "coach": r["coach"], "year": year, "value": surprise,
                        "label": f"missed finals at rank {r['actual_rank']} despite a {r['top4_pct']:.0f}% modelled chance",
                    })

        _, close_losses = close_call_counts(reg)
        for coach, n in close_losses.items():
            n = int(n)
            scaries_rows.append({"coach": coach, "year": year, "value": n,
                                  "label": f"{n} close loss{'es' if n != 1 else ''}"})

        for coach, n in win_streaks(reg).items():
            if n > 0:
                streak_rows.append({"coach": coach, "year": year, "value": n,
                                     "label": f"{n} straight win{'s' if n != 1 else ''}"})

    if not (dominant or spoon_rows or dodger_rows or lucky_rows or unlucky_rows or scaries_rows or streak_rows):
        return {}

    return {
        "dominant":         sorted(dominant, key=lambda r: r["value"])[:top_n],
        "unluckiest_spoon": sorted(spoon_rows, key=lambda r: r["value"])[:top_n],
        "finals_dodger":    sorted(dodger_rows, key=lambda r: r["value"])[:top_n],
        "lucky_wins":       sorted(lucky_rows, key=lambda r: -r["value"])[:top_n],
        "unlucky_losses":   sorted(unlucky_rows, key=lambda r: -r["value"])[:top_n],
        "sunday_scaries":   sorted(scaries_rows, key=lambda r: -r["value"])[:top_n],
        "win_streak":       sorted(streak_rows, key=lambda r: -r["value"])[:top_n],
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


def build_h2h_games(years: list[int], coach_a: str, coach_b: str) -> list[dict]:
    """
    Every individual game between two specific coaches, across regular
    season and finals, newest first. Powers the Head-to-Head drill-down
    page (records.html's grid links each cell here).

    Returns [{"year", "round", "stage" ("Regular Season"|"Finals"),
              "points_a", "points_b", "winner"}, ...]
    """
    from website.data.loader import load_fixtures
    from website.data.finals import load_finals_fixtures

    games: list[dict] = []

    for year in years:
        fx = load_fixtures(year)
        if not fx.empty:
            seen = set()
            for _, r in fx.iterrows():
                a, b = str(r["coach_first_name"]), str(r.get("opposition_team_coach"))
                if {a, b} != {coach_a, coach_b}:
                    continue
                key = (year, int(r["round_number"]))
                if key in seen:
                    continue
                seen.add(key)
                pts_a = r["team_points"] if a == coach_a else r["opposition_team_points"]
                pts_b = r["opposition_team_points"] if a == coach_a else r["team_points"]
                games.append({
                    "year": year, "round": int(r["round_number"]), "stage": "Regular Season",
                    "points_a": int(pts_a), "points_b": int(pts_b),
                    "winner": coach_a if pts_a > pts_b else (coach_b if pts_b > pts_a else None),
                })

        for f in load_finals_fixtures(year):
            if {f["coach1"], f["coach2"]} != {coach_a, coach_b}:
                continue
            pts_a = f["score1"] if f["coach1"] == coach_a else f["score2"]
            pts_b = f["score2"] if f["coach1"] == coach_a else f["score1"]
            games.append({
                "year": year, "round": f["round"], "stage": "Finals",
                "points_a": int(pts_a), "points_b": int(pts_b),
                "winner": coach_a if pts_a > pts_b else (coach_b if pts_b > pts_a else None),
            })

    games.sort(key=lambda g: (g["year"], g["round"]), reverse=True)
    return games


def build_h2h_droughts(years: list[int], top_n: int = 5) -> list[dict]:
    """
    Displayed on records.html as "Biggest Hoodoos" — AFL commentary's own
    term for a specific opponent a coach just can't beat regardless of form.
    Current, ongoing head-to-head droughts: for every ordered coach pair
    (coach, opponent), how many of their most recent consecutive meetings
    coach has gone without beating opponent — a draw keeps the drought
    alive, same as a loss, since this is specifically "hasn't beaten", not
    "has lost every time". The streak resets the moment coach wins, so only
    an active (still-unbroken, counting back from their most recent game)
    run counts. Returns the top_n longest, worst-first; "all_time": True
    flags a pair coach has never once beaten across every recorded meeting.
    """
    from website.config import COACH_ORDER
    from website.data.loader import load_fixtures
    from website.data.finals import load_finals_fixtures

    # One chronological game log per UNORDERED pair, built in a single pass
    # over the data (one row per actual game — winner or None for a draw).
    games_by_pair: dict[frozenset, list[tuple]] = {}

    def _add(year, rnd, a, pts_a, b, pts_b):
        if a not in COACH_ORDER or b not in COACH_ORDER or a == b:
            return
        winner = a if pts_a > pts_b else (b if pts_b > pts_a else None)
        games_by_pair.setdefault(frozenset((a, b)), []).append((year, rnd, winner))

    for year in years:
        fx = load_fixtures(year)
        if not fx.empty:
            seen = set()
            for _, r in fx.iterrows():
                a, b = str(r["coach_first_name"]), str(r.get("opposition_team_coach"))
                key = (year, int(r["round_number"]), tuple(sorted([a, b])))
                if key in seen:
                    continue
                seen.add(key)
                _add(year, int(r["round_number"]), a, r["team_points"], b, r["opposition_team_points"])
        for f in load_finals_fixtures(year):
            _add(year, f["round"], f["coach1"], f["score1"], f["coach2"], f["score2"])

    droughts = []
    for pair, glist in games_by_pair.items():
        glist.sort(key=lambda g: (g[0], g[1]))
        a, b = tuple(pair)
        for coach, opp in ((a, b), (b, a)):
            streak = 0
            since_year = None
            for year, rnd, winner in reversed(glist):
                if winner == coach:
                    break
                streak += 1
                since_year = year
            if streak == 0:
                continue
            all_time = streak == len(glist)
            label = f"hasn't beaten {opp} in {streak} meeting{'s' if streak != 1 else ''}"
            if all_time:
                label += " (ever)"
            droughts.append({
                "coach": coach, "opponent": opp, "value": streak,
                "all_time": all_time, "year": since_year, "label": label,
            })

    droughts.sort(key=lambda r: -r["value"])
    return droughts[:top_n]


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
             last_name=("last_name", "first"),
             feed_id=("feed_id", "first"))
        .reset_index()
    )

    photo_cache = _load_photo_cache()
    favourites = []
    for coach in COACH_ORDER:
        cdf = grouped[grouped["coach_first_name"] == coach].sort_values("rounds", ascending=False).head(top_n)
        players = []
        for r in cdf.itertuples():
            photo_url = None
            if pd.notna(r.feed_id):
                try:
                    photo_url = resolve_player_photo_url(int(r.feed_id), photo_cache)
                except Exception:
                    photo_url = None  # a flaky CDN/network call should never break the page
            players.append({
                "name":      f"{r.first_name} {r.last_name}".strip(),
                "rounds":    int(r.rounds),
                "photo_url": photo_url,
            })
        favourites.append({"coach": coach, "players": players})
    _save_photo_cache(photo_cache)
    return favourites
