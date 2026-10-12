"""
awards.py — Per-season "Awards" computed from data this site already builds
(ladder, fixtures, the Draft Heat Map, Acquisition analysis, and the Monte
Carlo schedule simulation). Only meaningful for a completed season.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

CLOSE_GAME_MARGIN = 55          # bottom quartile of every fixture margin in league history
UNLUCKY_LUCK_THRESHOLD = 1.0    # luck_mc_index must clear this (either direction) to award
SPOON_DODGE_THRESHOLD = 15.0    # spoon_pct (%) must clear this, among coaches who didn't get the spoon
MIN_WIN_STREAK = 2
WORST_PICK_MAX_ROUND = 8        # Worst Draft Pick only considers early/premium draft capital
IRON_MAN_MAX_RANK = 7           # Iron Man excludes the last-place finisher (consistently bad isn't "iron man")

# The 4 acquisition Kings reuse the exact Bootstrap Icons + colour already
# used for their cards on the Acquisition page, for visual consistency.
# Everything else is an emoji — several of these ideas (rollercoaster,
# diamond, dread-face) have no good Bootstrap Icons equivalent anyway.
AWARD_ICONS: dict[str, dict] = {
    "dawnbreaker":         {"type": "emoji", "value": "🏆"},
    "spoon":               {"type": "emoji", "value": "🥄"},
    "lineup_tinkerer":     {"type": "emoji", "value": "🔄"},
    "set_and_forget":      {"type": "emoji", "value": "⚓"},
    "top_scorer":          {"type": "emoji", "value": "🔥"},
    "lowest_scorer":       {"type": "emoji", "value": "📉"},
    "best_run":            {"type": "emoji", "value": "⚡"},
    "iron_man":            {"type": "emoji", "value": "🛡️"},
    "boom_or_bust":        {"type": "emoji", "value": "🎢"},
    "strong_finisher":     {"type": "emoji", "value": "🏁"},
    "close_calls":         {"type": "emoji", "value": "🎯"},
    "sunday_scaries":      {"type": "emoji", "value": "😱"},
    "unluckiest":          {"type": "emoji", "value": "🌧️"},
    "luckiest":            {"type": "emoji", "value": "🍀"},
    "spoon_dodge":         {"type": "emoji", "value": "🫣"},
    "draft_heat_map_king": {"type": "emoji", "value": "🧠"},
    "best_draft_pick":     {"type": "emoji", "value": "💎"},
    "worst_draft_pick":    {"type": "emoji", "value": "🗑️"},
    "draft_king":          {"type": "bi", "icon": "bi-trophy-fill",       "color": "primary"},
    "waiver_king":         {"type": "bi", "icon": "bi-person-plus-fill",  "color": "success"},
    "fa_king":             {"type": "bi", "icon": "bi-person-check-fill", "color": "info"},
    "trade_king":          {"type": "bi", "icon": "bi-arrow-left-right",  "color": "warning"},
}

# "highlight" = a flattering/celebratory result, "lowlight" = an unflattering
# one. Boom or Bust sits with the lowlights: paired opposite Iron Man, and
# "volatile/unreliable" reads as a knock rather than a compliment. Spoon
# Dodge is a highlight even though it's about nearly getting the spoon - the
# story is "they escaped it", a good outcome.
LOWLIGHT_KEYS = {"spoon", "lowest_scorer", "boom_or_bust", "sunday_scaries", "unluckiest", "worst_draft_pick",
                  "lineup_tinkerer"}

# Dawnbreaker (the league's own nickname for the premiership) and the Spoon
# always lead a coach's card and get special "hero" styling — the premier
# and the wooden-spooner are the headline story of a season, not just one
# category among many.
PROMINENT_KEYS = {"dawnbreaker", "spoon"}


def _award(key: str, title: str, coach: str, subtitle: str, detail: str | None = None) -> dict:
    return {
        "key": key, "title": title, "coach": coach, "subtitle": subtitle, "detail": detail,
        "icon": AWARD_ICONS.get(key),
        "group": "lowlight" if key in LOWLIGHT_KEYS else "highlight",
        "prominent": key in PROMINENT_KEYS,
    }


def _dawnbreaker_and_spoon(year: int, ladder_df: pd.DataFrame, final_ranks: dict) -> list[dict]:
    from website.data.finals import build_finals_bracket
    from website.data.honour_board import get_season_result

    awards = []

    try:
        finals_bracket = build_finals_bracket(ladder_df, year=year)
    except Exception:
        finals_bracket = None
    premier = finals_bracket["premier"] if finals_bracket else None
    if not premier:
        result = get_season_result(year)
        premier = result.get("champion") if result else None
    if premier:
        awards.append(_award("dawnbreaker", "Dawnbreaker", premier, f"{year} Premier"))

    if final_ranks:
        spoon_rank = max(final_ranks.values())
        spoon_coach = next((c for c, r in final_ranks.items() if r == spoon_rank), None)
        if spoon_coach:
            awards.append(_award("spoon", "Spoon", spoon_coach, f"{year} Wooden Spoon"))

    return awards


def _top_lowest_scorer(ladder_df: pd.DataFrame) -> list[dict]:
    if ladder_df.empty or "points_for" not in ladder_df.columns:
        return []
    final_round = ladder_df["round"].max()
    final = ladder_df[ladder_df["round"] == final_round].sort_values("points_for", ascending=False)
    if final.empty:
        return []
    top, low = final.iloc[0], final.iloc[-1]
    return [
        _award("top_scorer", "Top Scorer", top["coach_first_name"], f"{top['points_for']:,.0f} pts"),
        _award("lowest_scorer", "Lowest Scorer", low["coach_first_name"], f"{low['points_for']:,.0f} pts"),
    ]


def _best_run(fixtures_df: pd.DataFrame) -> list[dict]:
    if fixtures_df.empty:
        return []
    best_coach, best_streak = None, 0
    for coach, grp in fixtures_df.groupby("coach_first_name"):
        cur = streak = 0
        for win in grp.sort_values("round_number")["win"]:
            cur = cur + 1 if win else 0
            streak = max(streak, cur)
        if streak > best_streak:
            best_coach, best_streak = coach, streak
    if best_coach is None or best_streak < MIN_WIN_STREAK:
        return []
    return [_award("best_run", "Best Run", best_coach, f"{best_streak} straight wins")]


def _iron_man_boom_bust(fixtures_df: pd.DataFrame, final_ranks: dict) -> list[dict]:
    if fixtures_df.empty:
        return []
    std_by_coach = fixtures_df.groupby("coach_first_name")["team_points"].std()
    if std_by_coach.empty or std_by_coach.isna().all():
        return []

    awards = []
    # Iron Man excludes the last-place finisher: scoring consistently badly
    # isn't the "rock-solid, reliable" story the award is meant to capture.
    iron_candidates = std_by_coach[std_by_coach.index.map(lambda c: final_ranks.get(c, 99) <= IRON_MAN_MAX_RANK)]
    if not iron_candidates.empty:
        iron_coach = iron_candidates.idxmin()
        awards.append(_award("iron_man", "Iron Man", iron_coach,
                              f"{iron_candidates[iron_coach]:.0f} pt std-dev (most consistent)"))

    boom_coach = std_by_coach.idxmax()
    awards.append(_award("boom_or_bust", "Boom or Bust", boom_coach,
                          f"{std_by_coach[boom_coach]:.0f} pt std-dev (most volatile)"))
    return awards


def _strong_finisher(fixtures_df: pd.DataFrame) -> list[dict]:
    if fixtures_df.empty:
        return []
    last3 = (
        fixtures_df.sort_values("round_number")
        .groupby("coach_first_name")["team_points"]
        .apply(lambda s: s.tail(3).mean())
    )
    if last3.empty:
        return []
    coach = last3.idxmax()
    return [_award("strong_finisher", "Strong Finisher", coach, f"{last3[coach]:.0f} avg over the last 3 rounds")]


def _close_calls_sunday_scaries(fixtures_df: pd.DataFrame) -> list[dict]:
    if fixtures_df.empty:
        return []
    df = fixtures_df.copy()
    df["margin"] = (df["team_points"] - df["opposition_team_points"]).abs()
    close = df[df["margin"] <= CLOSE_GAME_MARGIN]
    close_wins = close[close["win"]].groupby("coach_first_name").size()
    close_losses = close[(~close["win"]) & (~close["draw"])].groupby("coach_first_name").size()

    awards = []
    if not close_wins.empty:
        coach = close_wins.idxmax()
        awards.append(_award("close_calls", "Close Calls", coach,
                              f"{int(close_wins[coach])} wins by {CLOSE_GAME_MARGIN} pts or less"))
    if not close_losses.empty:
        coach = close_losses.idxmax()
        awards.append(_award("sunday_scaries", "Sunday Scaries", coach,
                              f"{int(close_losses[coach])} losses by {CLOSE_GAME_MARGIN} pts or less"))
    return awards


def _lineup_turnover(players_df: pd.DataFrame) -> list[dict]:
    """
    Week-to-week turnover of each coach's on-field 19: for every round after
    the first, how many players are on-field who weren't on-field the round
    before. Since the on-field count is fixed, "new faces this round" and
    "players dropped this round" are always equal, so counting one side is
    enough. Summed across the season.
    """
    if players_df.empty:
        return []
    onfield = players_df[players_df["on_field"] == True]
    if onfield.empty:
        return []

    turnover: dict[str, int] = {}
    for coach, grp in onfield.groupby("coach_first_name"):
        prev_set: set | None = None
        total = 0
        for rnd in sorted(grp["round_x"].unique()):
            cur_set = set(grp[grp["round_x"] == rnd]["feed_id"])
            if prev_set is not None:
                total += len(cur_set - prev_set)
            prev_set = cur_set
        turnover[coach] = total

    if not turnover:
        return []

    most_coach = max(turnover, key=turnover.get)
    least_coach = min(turnover, key=turnover.get)
    return [
        _award("lineup_tinkerer", "Lineup Tinkerer", most_coach,
               f"{turnover[most_coach]} on-field changes across the season"),
        _award("set_and_forget", "Set and Forget", least_coach,
               f"{turnover[least_coach]} on-field changes across the season"),
    ]


def _luck_awards(mc_results: list[dict]) -> list[dict]:
    if not mc_results:
        return []

    spoon_rank = max(r["actual_rank"] for r in mc_results)
    awards = []

    # Raw luck_mc_index only (actual_rank - avg_mc_rank across randomised
    # schedules). An earlier version added a flat bonus for narrowly missing/
    # sneaking into finals, but a razor-thin margin (e.g. avg_mc_rank 4.07,
    # barely over the cutoff) got the exact same flat bonus as a decisive one
    # (e.g. 6.5) - strong enough to wrongly flip the winner over a coach with
    # a much bigger genuine luck swing. Removed rather than patched.
    unlucky = max(mc_results, key=lambda r: r["luck_mc_index"])
    if unlucky["luck_mc_index"] >= UNLUCKY_LUCK_THRESHOLD:
        awards.append(_award("unluckiest", "Unluckiest Coach", unlucky["coach"],
                              f"Finished rank {unlucky['actual_rank']}, averaged rank {unlucky['avg_mc_rank']:.2f} across simulated schedules"))

    lucky = min(mc_results, key=lambda r: r["luck_mc_index"])
    if lucky["luck_mc_index"] <= -UNLUCKY_LUCK_THRESHOLD:
        awards.append(_award("luckiest", "Luckiest Coach", lucky["coach"],
                              f"Finished rank {lucky['actual_rank']}, averaged rank {lucky['avg_mc_rank']:.2f} across simulated schedules"))

    dodgers = [r for r in mc_results if r["actual_rank"] != spoon_rank]
    if dodgers:
        dodger = max(dodgers, key=lambda r: r["spoon_pct"])
        if dodger["spoon_pct"] >= SPOON_DODGE_THRESHOLD:
            awards.append(_award("spoon_dodge", "Spoon Dodge", dodger["coach"],
                                  f"{dodger['spoon_pct']:.0f}% simulated chance of the spoon, finished rank {dodger['actual_rank']} instead"))

    return awards


def _draft_heat_map_king(draft_summary: list[dict]) -> list[dict]:
    if not draft_summary or draft_summary[0].get("avg_rank") is None:
        return []
    king = draft_summary[0]
    return [_award("draft_heat_map_king", "Draft Heat Map King", king["coach"],
                    f"{king['avg_rank']:.2f} avg pick rank")]


def _draft_pick_awards(extremes: dict | None) -> list[dict]:
    if not extremes:
        return []
    best, worst = extremes.get("best"), extremes.get("worst")
    awards = []
    if best:
        awards.append(_award("best_draft_pick", "Best Draft Pick", best["coach"],
                              f"{best['player']}, Rd {best['draft_round']} Pick {best['pick']}",
                              detail=f"{best['pick_value']:+.2f} value vs. the picks around it"))
    if worst:
        awards.append(_award("worst_draft_pick", "Worst Draft Pick", worst["coach"],
                              f"{worst['player']}, Rd {worst['draft_round']} Pick {worst['pick']}",
                              detail=f"{worst['pick_value']:+.2f} value vs. the picks around it"))
    return awards


def _acquisition_kings(kings: dict) -> list[dict]:
    titles = {"Draft": "Draft King", "Waiver": "Waiver King", "FA": "Free Agent King", "Trade": "Trade King"}
    awards = []
    for cat, title in titles.items():
        data = kings.get(cat)
        if not data or not data.get("coach"):
            continue
        awards.append(_award(f"{cat.lower()}_king", title, data["coach"],
                              f"{data['avg']:.1f} avg, {data['count']} players"))
    return awards


def build_season_awards(year: int) -> list[dict]:
    """
    Returns a list of award dicts: {key, title, coach, subtitle, detail}.
    Some awards are conditionally omitted for a given season (see each
    helper's threshold) rather than always forcing a marginal winner.
    """
    from website.config import (
        DRAFT_ALIAS_MAP, draft_csv_path, transactions_csv_path, coach_list_csv_path,
        player_match_csv_path, processed_waivers_json_path, player_list_csv_path, season_meta,
    )
    from website.data.loader import (
        load_ladder, load_fixtures, load_player_matches, load_sc_current, load_sc_round_files,
        list_season_coaches,
    )
    from website.data.standings import compute_standings
    from website.data.fixture_strength import build_monte_carlo_schedule_sim
    from website.data.draft_data import load_draft_board, get_pick_value_extremes
    from website.data.acquisition import load_player_sources, build_acquisition_stats, get_acquisition_kings

    total_regular_rounds = season_meta(year)["total_regular_rounds"]

    ladder = load_ladder(year)
    fixtures = load_fixtures(year)
    players = load_player_matches(year)
    reg_ladder = ladder[ladder["round"] <= total_regular_rounds] if not ladder.empty else ladder
    reg_fixtures = fixtures[fixtures["round_number"] <= total_regular_rounds] if not fixtures.empty else fixtures
    reg_players = players[players["round_x"] <= total_regular_rounds] if not players.empty else players

    standings = compute_standings(reg_ladder, total_regular_rounds)
    final_ranks = dict(zip(standings["coach_first_name"], standings["ladder_pos"])) if not standings.empty else {}

    awards: list[dict] = []
    awards += _dawnbreaker_and_spoon(year, ladder, final_ranks)
    awards += _top_lowest_scorer(reg_ladder)
    awards += _best_run(reg_fixtures)
    awards += _iron_man_boom_bust(reg_fixtures, final_ranks)
    awards += _strong_finisher(reg_fixtures)
    awards += _close_calls_sunday_scaries(reg_fixtures)
    awards += _lineup_turnover(reg_players)

    try:
        mc_results = build_monte_carlo_schedule_sim(reg_fixtures)
    except Exception:
        log.warning("build_monte_carlo_schedule_sim failed for %s awards", year, exc_info=True)
        mc_results = []
    awards += _luck_awards(mc_results)

    # Draft/acquisition-dependent awards — 2017/2018 have no draft CSV at all.
    draft_csv = draft_csv_path(year)
    if draft_csv.exists():
        sc_df = load_sc_current(year)
        sc_round_df = load_sc_round_files(year)
        player_list_csv = player_list_csv_path(year)
        coaches = list_season_coaches(year)

        summary, _, _, _ = load_draft_board(
            draft_csv, player_list_csv, DRAFT_ALIAS_MAP, coaches, sc_df, sc_round_df,
        )
        awards += _draft_heat_map_king(summary)

        extremes = get_pick_value_extremes(draft_csv, player_list_csv, DRAFT_ALIAS_MAP, sc_df, sc_round_df,
                                            worst_max_round=WORST_PICK_MAX_ROUND)
        awards += _draft_pick_awards(extremes)

        player_sources = load_player_sources(
            draft_csv=draft_csv,
            transactions_csv=transactions_csv_path(year),
            coach_list_csv=coach_list_csv_path(year),
            player_match_csv=player_match_csv_path(year),
            draft_alias_map=DRAFT_ALIAS_MAP,
            processed_waivers_json=processed_waivers_json_path(year),
        )
        stats = build_acquisition_stats(player_sources, player_match_csv_path(year))
        awards += _acquisition_kings(get_acquisition_kings(stats))

    return awards


def group_awards_by_coach(awards: list[dict], coach_order: list[str]) -> list[dict]:
    """
    For displaying Awards grouped by coach instead of by category. Returns
    one entry per coach in coach_order (final ladder position, so it reads
    top-to-bottom the same way the Standings table below it does) - even a
    coach who won nothing gets an entry, so the section reads as a complete
    roster rather than looking cut off.

    Each entry: {coach, awards: [...]}.
    """
    by_coach: dict[str, list[dict]] = {c: [] for c in coach_order}
    for a in awards:
        if a["coach"] in by_coach:
            by_coach[a["coach"]].append(a)
    return [{"coach": c, "awards": by_coach[c]} for c in coach_order]
