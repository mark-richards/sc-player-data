"""
fixture_strength.py — Fixture-adjusted strength and luck analysis.

Head-to-head fantasy football has several independent luck factors:

1. OPPONENT SELECTION  — who you drew each round
   • opp_avg          : mean score of opponents actually faced
   • avg_opp_rank     : mean weekly rank (1–8) of opponents faced
                        (1 = always faced the top scorer; 8 = always faced the bottom scorer)

2. EXPECTED RECORD    — how many wins your SCORES deserved
   • exp_wins         : for each round, fraction of the other 7 teams your score beat,
                        summed over all rounds  (e.g. 2.57 expected wins from 4 rounds)
   • exp_win_pct      : exp_wins / rounds played × 100
   • luck_wins        : actual_wins − exp_wins  (positive = won MORE than scores warranted)

3. OUTCOME LUCK       — individual game results that defied the field
   • lucky_wins       : wins where your score was BELOW the round average
                        (you won despite under-performing the field)
   • unlucky_losses   : losses where your score was ABOVE the round average
                        (you lost despite out-performing the field)

4. SCHEDULE SIMULATION — counterfactual ranking across all possible fixtures
   • avg_sim_rank     : mean ladder rank across 8 alternate universes, one per
                        real coach's opponent-score sequence used as a shared
                        benchmark for every coach that round (see note below)
   • best / worst     : best and worst rank across those 8 simulated universes
   • luck_index       : actual_rank − avg_sim_rank
                        negative = lucky (ranked better than strength suggests)
                        positive = unlucky (ranked worse than strength suggests)

   Each simulated universe must rank ALL coaches against the SAME benchmark
   opponent-score sequence, not compare one coach's simulated record against
   everyone else's real record — mixing simulated vs actual records is
   internally inconsistent and can let a weaker team's "best case" exceed a
   stronger team's actual dominance (e.g. two coaches both showing a best
   possible rank of 1, which can't both be true against a shared field).

5. MONTE CARLO SCHEDULE SIMULATION — the same idea, at scale
   build_simulation_ladders() only samples 8 alternate universes — one per
   real coach's actual opponent-score sequence, because that's the only
   fixture history that exists. build_monte_carlo_schedule_sim() instead
   draws thousands of randomised, STRUCTURALLY VALID round-robin schedules
   (circle/polygon method: fix one coach, rotate the rest — see
   _generate_round_robin_cycle()) rather than independently-random
   round-by-round pairings. This respects the real league's actual
   constraint — a fixed m-round single round-robin cycle (m = n_coaches-1)
   repeated identically for the whole season (e.g. the real 2026 fixture is
   the same 7-round cycle repeated 3x for 21 rounds) — so within any one
   trial every pair of coaches meets exactly as many times as they really
   would, never more, never less. Each trial randomises WHICH valid cycle is
   drawn, not whether the structure is respected. Every coach's own real
   weekly score is judged against whichever real opponent that trial's
   schedule assigns them each round — never a fabricated score. Reports the
   resulting rank *distribution* per coach: how often each coach lands at
   each ladder position across the sample, plus avg/best/worst.
"""
import logging
import random

import pandas as pd

log = logging.getLogger(__name__)


def _build_all_records(fixtures_df: pd.DataFrame):
    """
    Shared groundwork for both build_fixture_strength() and
    build_simulation_ladders(): per-round lookups and the 8×8 simulated
    W/D/L record matrix (all_records[a][b] = coach a's record if every round
    they'd faced the opponent-score that coach b actually faced).

    Returns (coaches, rounds, round_data, round_scores, all_records) or
    (None, None, None, None, None) if there's not enough data to simulate.
    """
    if fixtures_df.empty:
        return None, None, None, None, None

    rounds  = sorted(fixtures_df["round_number"].unique())
    coaches = sorted(fixtures_df["coach_first_name"].unique())
    if len(coaches) < 2 or len(rounds) < 1:
        return None, None, None, None, None

    # round_data[r][coach] = {"score": float, "opp": str, "opp_score": float}
    round_data: dict[int, dict] = {}
    for _, row in fixtures_df.iterrows():
        r = int(row["round_number"])
        c = row["coach_first_name"]
        if r not in round_data:
            round_data[r] = {}
        round_data[r][c] = {
            "score":     float(row["team_points"]),
            "opp":       row["opposition_team_coach"],
            "opp_score": float(row["opposition_team_points"]),
        }

    def _record_a_under_b(coach_a: str, coach_b: str) -> dict:
        """Coach A's W/D/L if they had faced coach B's opponents every round."""
        wins = draws = losses = 0
        pts_for = pts_against = 0.0
        for r in rounds:
            rd = round_data.get(r, {})
            a_data = rd.get(coach_a)
            b_data = rd.get(coach_b)
            if not a_data or not b_data:
                continue
            a_score = a_data["score"]
            b_opp   = b_data["opp"]
            # Self-match guard: if B's opponent was A, use A's actual opponent score
            if b_opp == coach_a:
                opp_score = a_data["opp_score"]
            else:
                opp_data  = rd.get(b_opp)
                opp_score = opp_data["score"] if opp_data else a_data["opp_score"]
            pts_for     += a_score
            pts_against += opp_score
            if   a_score > opp_score: wins   += 1
            elif a_score == opp_score: draws  += 1
            else:                      losses += 1
        return {
            "wins": wins, "draws": draws, "losses": losses,
            "pts_for": pts_for, "pts_against": pts_against,
            "league_points": wins * 4 + draws * 2,
        }

    # Pre-compute all 8×8 simulated records
    all_records = {
        a: {b: _record_a_under_b(a, b) for b in coaches}
        for a in coaches
    }

    # round_scores[r] = {coach: score}
    round_scores: dict[int, dict] = {
        r: {c: d["score"] for c, d in rd.items()}
        for r, rd in round_data.items()
    }

    return coaches, rounds, round_data, round_scores, all_records


def build_simulation_ladders(fixtures_df: pd.DataFrame) -> list[dict]:
    """
    Returns one dict per schedule template: {"template", "ladder", "detail"}
    where "ladder" is that template's full 8-coach simulated standings (all
    coaches judged against the same benchmark opponent-score sequence — the
    one the template coach actually faced), ordered by rank, and "detail" is
    the round-by-round breakdown that produced it (for drilling into exactly
    which fixture swaps changed the outcome).

    Ladder entry: {"rank", "coach", "wins", "draws", "losses", "pts_for",
                   "actual_rank", "delta"}  (delta = actual_rank − sim_rank;
                   positive = this simulation ranked them better than reality)

    Detail row: {"round", "opponent", "benchmark_score",
                 "scores": {coach: {"score", "bench", "result"}}}
    """
    coaches, rounds, round_data, round_scores, all_records = _build_all_records(fixtures_df)
    if coaches is None:
        return []

    actual_ladder = sorted(
        coaches,
        key=lambda c: (
            -all_records[c][c]["league_points"],
            -all_records[c][c]["pts_for"],
        ),
    )
    actual_ladder_rank = {c: actual_ladder.index(c) + 1 for c in coaches}

    sims = []
    for b in coaches:
        ladder_b = sorted(
            coaches,
            key=lambda c: (
                -all_records[c][b]["league_points"],
                -all_records[c][b]["pts_for"],
            ),
        )
        ladder = []
        for i, c in enumerate(ladder_b):
            sim_rank = i + 1
            ladder.append({
                "rank":        sim_rank,
                "coach":       c,
                "wins":        all_records[c][b]["wins"],
                "draws":       all_records[c][b]["draws"],
                "losses":      all_records[c][b]["losses"],
                "pts_for":     round(all_records[c][b]["pts_for"], 1),
                "actual_rank": actual_ladder_rank[c],
                "delta":       actual_ladder_rank[c] - sim_rank,
            })

        # Round-by-round detail: every coach's real score judged against the
        # same benchmark opponent-score that template coach b actually faced
        # that round (self-match guard mirrors _record_a_under_b exactly).
        detail = []
        for r in rounds:
            rd = round_data.get(r, {})
            b_data = rd.get(b)
            if not b_data:
                continue
            b_opp = b_data["opp"]
            benchmark_score = b_data["opp_score"]
            row_scores = {}
            for c in coaches:
                cd = rd.get(c)
                if not cd:
                    continue
                bench = cd["opp_score"] if b_opp == c else benchmark_score
                score = cd["score"]
                result = "W" if score > bench else ("D" if score == bench else "L")
                row_scores[c] = {"score": score, "bench": bench, "result": result}
            detail.append({
                "round":           r,
                "opponent":        b_opp,
                "benchmark_score": benchmark_score,
                "scores":          row_scores,
            })

        sims.append({"template": b, "ladder": ladder, "detail": detail})
    return sims


def build_fixture_strength(fixtures_df: pd.DataFrame) -> list[dict]:
    """
    Returns one dict per coach with all luck/difficulty metrics.
    Sorted by avg_opp_rank ascending (hardest schedule first).
    Returns [] if fewer than 2 coaches or 1 round.
    """
    coaches, rounds, round_data, round_scores, all_records = _build_all_records(fixtures_df)
    if coaches is None:
        return []

    # ── Actual ladder rank (each coach plays their own schedule) ─────────────
    actual_ladder = sorted(
        coaches,
        key=lambda c: (
            -all_records[c][c]["league_points"],
            -all_records[c][c]["pts_for"],
        ),
    )
    actual_ladder_rank = {c: actual_ladder.index(c) + 1 for c in coaches}

    # ── Simulated ladder per schedule template ────────────────────────────────
    # For each template b, rank every coach using all_records[c][b] — i.e. every
    # coach's own real weekly scores judged against the SAME benchmark opponent-
    # score sequence (the one coach b actually faced). This keeps each simulated
    # universe internally consistent: all 8 coaches are compared on equal footing
    # within that universe, rather than one coach's simulation vs everyone else's
    # real record.
    sim_ladder_rank: dict[str, dict[str, int]] = {}
    for b in coaches:
        ladder_b = sorted(
            coaches,
            key=lambda c: (
                -all_records[c][b]["league_points"],
                -all_records[c][b]["pts_for"],
            ),
        )
        sim_ladder_rank[b] = {c: ladder_b.index(c) + 1 for c in coaches}

    # ── Per-coach metrics ─────────────────────────────────────────────────────
    results = []

    for coach_a in coaches:
        n_rounds_played   = 0
        exp_wins          = 0.0
        actual_wins       = 0
        lucky_wins_count  = 0
        unlucky_loss_count = 0
        opp_avg_scores    = []
        opp_weekly_ranks  = []
        own_weekly_ranks  = []

        for r in rounds:
            rd = round_data.get(r, {})
            if coach_a not in rd:
                continue
            a_data   = rd[coach_a]
            a_score  = a_data["score"]
            opp      = a_data["opp"]
            opp_score = a_data["opp_score"]
            n_rounds_played += 1

            # ── Expected wins: fraction of the OTHER 7 teams beaten ──────────
            others = [s for c, s in round_scores[r].items() if c != coach_a]
            n_others = len(others)
            if n_others > 0:
                exp_wins += sum(1 for s in others if a_score > s) / n_others

            # ── Opponent's and own rank within the full 8-team field this round ──
            all_scores_sorted = sorted(round_scores[r].values(), reverse=True)
            opp_rank = all_scores_sorted.index(opp_score) + 1  # 1 = top scorer
            opp_weekly_ranks.append(opp_rank)
            opp_avg_scores.append(opp_score)
            own_rank = all_scores_sorted.index(a_score) + 1  # 1 = top scorer
            own_weekly_ranks.append(own_rank)

            # ── Actual outcome vs round average ──────────────────────────────
            round_avg = sum(round_scores[r].values()) / len(round_scores[r])
            won  = a_score > opp_score
            drew = a_score == opp_score
            if won:
                actual_wins += 1
                if a_score < round_avg:   # won despite scoring below average
                    lucky_wins_count += 1
            elif not drew:
                if a_score > round_avg:   # lost despite scoring above average
                    unlucky_loss_count += 1

        opp_avg      = round(sum(opp_avg_scores)   / len(opp_avg_scores), 1)   if opp_avg_scores   else 0.0
        avg_opp_rank = round(sum(opp_weekly_ranks) / len(opp_weekly_ranks), 2) if opp_weekly_ranks else 0.0
        avg_score_rank = round(sum(own_weekly_ranks) / len(own_weekly_ranks), 2) if own_weekly_ranks else 0.0
        luck_wins    = round(actual_wins - exp_wins, 2)
        exp_win_pct  = round(exp_wins / n_rounds_played * 100, 1) if n_rounds_played else 0.0

        # ── Schedule simulation ranking ───────────────────────────────────────
        # Rank coach_a within each of the 8 self-consistent simulated ladders
        # computed above (sim_ladder_rank).
        sim_ranks    = [sim_ladder_rank[b][coach_a] for b in coaches]
        actual_rank  = actual_ladder_rank[coach_a]
        avg_sim_rank = round(sum(sim_ranks) / len(sim_ranks), 2)
        best_rank    = min(sim_ranks)
        worst_rank   = max(sim_ranks)
        luck_index   = round(actual_rank - avg_sim_rank, 2)

        results.append({
            "coach":               coach_a,
            # Opponent difficulty
            "opp_avg":             opp_avg,
            "avg_opp_rank":        avg_opp_rank,
            # Expected record
            "exp_wins":            round(exp_wins, 2),
            "exp_win_pct":         exp_win_pct,
            "actual_wins":         actual_wins,
            "luck_wins":           luck_wins,
            # Outcome luck
            "lucky_wins":          lucky_wins_count,
            "unlucky_losses":      unlucky_loss_count,
            # Schedule simulation
            "actual_rank":         actual_rank,
            "avg_score_rank":      avg_score_rank,
            "avg_sim_rank":        avg_sim_rank,
            "best_sim_rank":       best_rank,
            "worst_sim_rank":      worst_rank,
            "luck_index":          luck_index,
        })

    # Sort fixture difficulty by avg_opp_rank ascending (hardest schedule first)
    results.sort(key=lambda x: x["avg_opp_rank"])
    return results


def _generate_round_robin_cycle(coaches: list, rng: random.Random) -> list:
    """
    Circle/polygon-method round-robin generator: fixes one coach and rotates
    the rest around it to produce a randomised, STRUCTURALLY VALID single
    round-robin cycle (every pair meets exactly once across the cycle).

    Given n coaches, returns m = n-1 rounds (after padding to even length
    with a None bye if n is odd), each a list of (coach_a, coach_b) pairs.

    Randomising `order` (who's fixed, and the rotation order of the rest via
    a single rng.shuffle()) is what makes each call sample a different valid
    schedule — this is the Monte Carlo "draw" for one trial.
    """
    teams = list(coaches)
    if len(teams) % 2 == 1:
        teams.append(None)  # bye placeholder

    order = teams[:]
    rng.shuffle(order)
    fixed    = order[0]
    rotating = order[1:]
    m = len(rotating)  # n-1; always odd since n (after padding) is even

    cycle = []
    for r in range(m):
        round_pairs = [(fixed, rotating[r])]
        for i in range(1, (m - 1) // 2 + 1):
            a = rotating[(r + i) % m]
            b = rotating[(r - i) % m]
            round_pairs.append((a, b))
        cycle.append(round_pairs)
    return cycle


def build_monte_carlo_schedule_sim(
    fixtures_df: pd.DataFrame, n_trials: int = 20000, seed: int = 42
) -> list[dict]:
    """
    Draws n_trials randomised, structurally valid round-robin schedules (see
    _generate_round_robin_cycle) and, in each trial, judges every coach's own
    real weekly score against whichever real opponent that trial's schedule
    assigns them each round (scores are never fabricated — only the schedule
    is randomised). Each trial respects the real league's actual constraint:
    a fixed m-round single round-robin cycle (m = n_coaches-1) repeated for
    the whole season, so within any one trial every pair of coaches meets
    exactly as many times as they really would — never more, never less,
    never zero. Aggregating across all trials gives a genuine "if you'd
    played any possible valid schedule" rank distribution per coach.

    For every (coach, rank) combination that occurs in at least one trial,
    the first trial to produce it is kept as a concrete illustrative example
    — a full round-by-round breakdown (opponent, both scores, result) plus
    the resulting W/D/L record — so a user can drill into "show me one way
    this coach could have finished 3rd".

    Returns one dict per coach, sorted by avg_mc_rank ascending:
        coach, actual_rank, avg_mc_rank, best_mc_rank, worst_mc_rank,
        top2_pct, top4_pct, spoon_pct (% of trials finishing rank 1-2,
        rank 1-4, and last place respectively),
        luck_mc_index (actual_rank − avg_mc_rank),
        rank_dist: list of 8 {rank, pct, cls, example} dicts — "example" is
        None if that rank never occurred for this coach across all trials,
        otherwise {rounds: [...], wins, draws, losses, pts_for, league_points,
        ladder: [...]} where "ladder" is the full 8-coach standings for that
        same trial (see _rank_tier_class for "cls")
    """
    coaches, rounds, _round_data, round_scores, all_records = _build_all_records(fixtures_df)
    if coaches is None:
        return []

    n = len(coaches)
    rng = random.Random(seed)

    actual_ladder = sorted(
        coaches,
        key=lambda c: (
            -all_records[c][c]["league_points"],
            -all_records[c][c]["pts_for"],
        ),
    )
    actual_ladder_rank = {c: actual_ladder.index(c) + 1 for c in coaches}

    def _rank_tier_class(rank: int) -> str:
        """Same 5-tier rank colour scale used by the Draft Heat Map (custom.css)."""
        if rank == 1:
            return "rank-1"
        elif rank <= 3:
            return "rank-2-3"
        elif rank <= 5:
            return "rank-4-5"
        elif rank <= 7:
            return "rank-6-7"
        return "rank-8"

    rank_counts = {c: [0] * n for c in coaches}
    sum_rank    = {c: 0 for c in coaches}
    best_rank   = {c: n for c in coaches}
    worst_rank  = {c: 1 for c in coaches}
    examples: dict = {}  # (coach, rank) -> example dict, first trial to hit it

    for _ in range(n_trials):
        wins          = {c: 0 for c in coaches}
        draws         = {c: 0 for c in coaches}
        pts_for       = {c: 0.0 for c in coaches}
        rounds_played = {c: 0 for c in coaches}

        cycle = _generate_round_robin_cycle(coaches, rng)
        m = len(cycle)

        # Track each coach's opponent this trial, per round, so an example
        # can be reconstructed cheaply if this trial turns out to be the
        # first to hit a not-yet-seen (coach, rank) combination.
        opponent_by_round: dict = {c: {} for c in coaches}

        for k, r in enumerate(rounds):
            scores_r = round_scores.get(r, {})
            for a, b in cycle[k % m]:
                if a is None or b is None:
                    continue
                if a not in scores_r or b not in scores_r:
                    continue
                opponent_by_round[a][r] = b
                opponent_by_round[b][r] = a
                rounds_played[a] += 1
                rounds_played[b] += 1
                sa, sb = scores_r[a], scores_r[b]
                pts_for[a] += sa
                pts_for[b] += sb
                if sa > sb:
                    wins[a] += 1
                elif sb > sa:
                    wins[b] += 1
                else:
                    draws[a] += 1
                    draws[b] += 1

        league_points = {c: wins[c] * 4 + draws[c] * 2 for c in coaches}
        ladder = sorted(coaches, key=lambda c: (-league_points[c], -pts_for[c]))

        # Lazily built once per trial, only if this trial turns out to
        # contribute at least one new (coach, rank) example — the full
        # simulated ladder for that trial, so a drilldown can show not just
        # the clicked coach's own record but the whole resulting table.
        trial_ladder = None

        for i, c in enumerate(ladder):
            rk = i + 1
            rank_counts[c][rk - 1] += 1
            sum_rank[c] += rk
            best_rank[c] = min(best_rank[c], rk)
            worst_rank[c] = max(worst_rank[c], rk)

            key = (c, rk)
            if key not in examples:
                round_detail = []
                for r in rounds:
                    opp = opponent_by_round[c].get(r)
                    if opp is None:
                        continue
                    own_score = round_scores[r][c]
                    opp_score = round_scores[r][opp]
                    result = "W" if own_score > opp_score else ("D" if own_score == opp_score else "L")
                    round_detail.append({
                        "round": r, "opponent": opp,
                        "own_score": own_score, "opp_score": opp_score,
                        "result": result,
                    })
                if trial_ladder is None:
                    trial_ladder = [
                        {
                            "rank": j + 1, "coach": lc, "cls": _rank_tier_class(j + 1),
                            "wins": wins[lc], "draws": draws[lc],
                            "losses": rounds_played[lc] - wins[lc] - draws[lc],
                            "pts_for": round(pts_for[lc], 1),
                            "league_points": league_points[lc],
                        }
                        for j, lc in enumerate(ladder)
                    ]
                examples[key] = {
                    "rounds":        round_detail,
                    "wins":          wins[c],
                    "draws":         draws[c],
                    "losses":        rounds_played[c] - wins[c] - draws[c],
                    "pts_for":       round(pts_for[c], 1),
                    "league_points": league_points[c],
                    "ladder":        trial_ladder,
                }

    results = []
    for c in coaches:
        avg_rank = sum_rank[c] / n_trials
        pct = [round(cnt / n_trials * 100, 1) for cnt in rank_counts[c]]
        results.append({
            "coach":          c,
            "actual_rank":    actual_ladder_rank[c],
            "avg_mc_rank":    round(avg_rank, 2),
            "best_mc_rank":   best_rank[c],
            "worst_mc_rank":  worst_rank[c],
            "top2_pct":       round(sum(rank_counts[c][0:2]) / n_trials * 100, 1),
            "top4_pct":       round(sum(rank_counts[c][0:4]) / n_trials * 100, 1),
            "spoon_pct":      round(rank_counts[c][n - 1] / n_trials * 100, 1),
            "luck_mc_index":  round(actual_ladder_rank[c] - avg_rank, 2),
            "rank_dist":      [
                {
                    "rank": i + 1, "pct": pct[i], "cls": _rank_tier_class(i + 1),
                    "example": examples.get((c, i + 1)),
                }
                for i in range(n)
            ],
            "n_trials":       n_trials,
        })

    results.sort(key=lambda x: x["avg_mc_rank"])
    return results
