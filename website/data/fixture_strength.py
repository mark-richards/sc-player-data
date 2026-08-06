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
"""
import logging

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
