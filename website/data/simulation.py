"""
simulation.py — Monte Carlo season simulation.

10,000 NumPy-vectorised simulations of the remaining regular season
plus finals series. Returns championship/finals/GF/spoon probabilities.

Design principles:
- All teams draw from the same league-wide score distribution for future
  games — no per-coach Bayesian means.  This makes each remaining matchup
  an ~50/50 coin flip, so finals probability is driven by current standings
  (already-earned wins) rather than projected scoring ability.
- Finals use the McIntyre double-chance format (top 4, 3 rounds):
    Week 1: QF1 (1st vs 2nd), EF (3rd vs 4th)
    Week 2: PF  (QF1-loser vs EF-winner)
    Week 3: GF  (QF1-winner vs PF-winner)
  Top 2 teams get a second chance if they lose their qualifying final.
"""
import json as _json
import logging

import numpy as np
import pandas as pd

from website.config import FINALS_TOP_N, N_TEAMS, SCORE_FLOOR, SIMULATION_RUNS, TOTAL_REGULAR_ROUNDS

log = logging.getLogger(__name__)


def _build_remaining_fixture(
    fixture_df: pd.DataFrame,
    current_round: int,
    coaches: list[str],
    schedule_df: "pd.DataFrame | None" = None,
) -> list[tuple[str, str]]:
    """
    Returns list of (home_coach, away_coach) matchups for rounds
    current_round+1 through TOTAL_REGULAR_ROUNDS.

    Priority:
    1. schedule_df — full 2026 fixture from fixture_schedule_2026.csv
    2. fixture_df  — played-rounds CSV (fallback if no schedule file)
    3. Synthetic round-robin (last resort)
    """
    coaches_set = set(coaches)

    # 1. Use the official schedule file
    if schedule_df is not None and not schedule_df.empty:
        future = schedule_df[
            (schedule_df["round_number"] > current_round) &
            (schedule_df["round_number"] <= TOTAL_REGULAR_ROUNDS)
        ]
        matchups = [
            (row["home_coach"], row["away_coach"])
            for _, row in future.iterrows()
            if row["home_coach"] in coaches_set and row["away_coach"] in coaches_set
        ]
        if matchups:
            log.debug(
                "_build_remaining_fixture: %d matchups from schedule file (rounds %d-%d)",
                len(matchups), current_round + 1, TOTAL_REGULAR_ROUNDS,
            )
            return matchups

    # 2. Fall back to played-rounds fixture CSV
    future = fixture_df[
        (fixture_df["round_number"] > current_round) &
        (fixture_df["round_number"] <= TOTAL_REGULAR_ROUNDS)
    ][["round_number", "coach_first_name", "opposition_team_coach"]].copy()
    future = future.dropna(subset=["coach_first_name", "opposition_team_coach"])

    matchups = []
    for _, row in future.iterrows():
        h = row["coach_first_name"]
        a = row["opposition_team_coach"]
        if h in coaches_set and a in coaches_set and (a, h) not in matchups:
            matchups.append((h, a))

    if matchups:
        log.debug(
            "_build_remaining_fixture: %d matchups from fixture CSV (fallback)", len(matchups)
        )
        return matchups

    # 3. Synthetic round-robin
    log.warning(
        "_build_remaining_fixture: no schedule data — using synthetic round-robin. "
        "Run refresh_2026_data() to generate fixture_schedule_2026.csv."
    )
    n_remaining = TOTAL_REGULAR_ROUNDS - current_round
    pairs = [
        (coaches[i], coaches[j])
        for i in range(len(coaches))
        for j in range(i + 1, len(coaches))
    ]
    for r in range(n_remaining):
        matchups.extend(pairs[r % len(pairs): r % len(pairs) + N_TEAMS // 2])
    return matchups


def run_monte_carlo(
    fixture_df: pd.DataFrame,
    ladder_df: pd.DataFrame,
    current_round: int,
    n_sims: int = SIMULATION_RUNS,
    rng_seed: int = 42,
    schedule_df: "pd.DataFrame | None" = None,
    historical_scores: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """
    Returns DataFrame: coach_first_name | finals_pct | gf_pct | champ_pct | spoon_pct

    Parameters
    ----------
    fixture_df        : played-rounds fixture data (from fixture_results_by_team.csv)
    ladder_df         : cumulative standings per round (from ladder.csv)
    current_round     : last completed round number
    n_sims            : number of Monte Carlo iterations
    rng_seed          : NumPy RNG seed for reproducibility
    schedule_df       : full 2026 fixture schedule (from fixture_schedule_2026.csv),
                        columns: round_number, home_coach, away_coach
    historical_scores : array of all historical game scores (2016-2021) used to
                        anchor the league-wide score distribution.
                        If None, falls back to current-season scores only.
    """
    rng = np.random.default_rng(rng_seed)

    # Current standings
    curr = ladder_df[ladder_df["round"] == current_round].copy()
    coaches = sorted(curr["coach_first_name"].unique())
    n_coaches = len(coaches)
    coach_idx = {c: i for i, c in enumerate(coaches)}

    # ── League-wide score distribution ───────────────────────────────────────
    # All teams draw from the same distribution for every remaining game.
    # This makes each matchup ~50/50, so finals probability is determined
    # by current standings (earned wins) rather than scoring projections.
    current_season_scores = fixture_df["team_points"].dropna().values.astype(float)

    if historical_scores is not None and len(historical_scores) >= 50:
        combined = np.concatenate([
            historical_scores,
            np.repeat(current_season_scores, 2),  # double-weight current season
        ])
        league_mean = float(combined.mean())
        league_std  = max(float(combined.std(ddof=1)), 100.0)
        log.debug(
            "League distribution (historical+current): mean=%.0f std=%.0f",
            league_mean, league_std,
        )
    else:
        league_mean = float(current_season_scores.mean()) if len(current_season_scores) else 1590.0
        league_std  = max(float(current_season_scores.std(ddof=1)), 100.0) if len(current_season_scores) > 1 else 175.0
        log.debug("League distribution (current only): mean=%.0f std=%.0f", league_mean, league_std)

    # ── Starting league points and points-for per coach ───────────────────────
    current_pts = np.zeros((n_coaches, n_sims))
    current_for = np.zeros((n_coaches, n_sims))
    for _, row in curr.iterrows():
        i = coach_idx[row["coach_first_name"]]
        current_pts[i] = row["league_points"]
        current_for[i] = row["points_for"]

    # ── Simulate remaining regular-season matchups ────────────────────────────
    matchups = _build_remaining_fixture(fixture_df, current_round, coaches, schedule_df)
    log.debug("Simulating %d remaining matchups (rounds %d-%d)",
              len(matchups), current_round + 1, TOTAL_REGULAR_ROUNDS)

    for home, away in matchups:
        hi = coach_idx[home]
        ai = coach_idx[away]
        h_scores = rng.normal(league_mean, league_std, n_sims).clip(min=SCORE_FLOOR)
        a_scores = rng.normal(league_mean, league_std, n_sims).clip(min=SCORE_FLOOR)

        h_wins = h_scores > a_scores
        draws  = h_scores == a_scores

        current_pts[hi] += np.where(h_wins, 4, np.where(draws, 2, 0))
        current_pts[ai] += np.where(~h_wins & ~draws, 4, np.where(draws, 2, 0))
        current_for[hi] += h_scores
        current_for[ai] += a_scores

    # ── Rank teams: primary = league_points DESC, secondary = points_for DESC ─
    sorted_idx = np.lexsort((-current_for, -current_pts), axis=0)  # (n_coaches, n_sims)
    ranks = np.empty_like(sorted_idx)
    for s in range(n_sims):
        ranks[sorted_idx[:, s], s] = np.arange(n_coaches)  # 0-indexed rank

    # ── Finals appearances: top FINALS_TOP_N ─────────────────────────────────
    finals_count = np.zeros(n_coaches, dtype=np.int64)
    spoon_count  = np.zeros(n_coaches, dtype=np.int64)
    for i in range(n_coaches):
        finals_count[i] = int((ranks[i] < FINALS_TOP_N).sum())
        spoon_count[i]  = int((ranks[i] == n_coaches - 1).sum())

    # ── McIntyre double-chance finals (3 rounds, top 4) ───────────────────────
    # finalists: shape (4, n_sims) — coach indices ranked 1st through 4th
    finalists = sorted_idx[:FINALS_TOP_N, :]
    f0, f1, f2, f3 = finalists[0], finalists[1], finalists[2], finalists[3]

    # Week 1: QF1 (1st vs 2nd) and EF (3rd vs 4th)
    qf1_h = rng.normal(league_mean, league_std, n_sims)
    qf1_a = rng.normal(league_mean, league_std, n_sims)
    ef_h  = rng.normal(league_mean, league_std, n_sims)
    ef_a  = rng.normal(league_mean, league_std, n_sims)

    qf1_winner = np.where(qf1_h >= qf1_a, f0, f1)   # straight to GF
    qf1_loser  = np.where(qf1_h >= qf1_a, f1, f0)   # second chance
    ef_winner  = np.where(ef_h  >= ef_a,  f2, f3)   # to PF
    # ef_loser is eliminated

    # Week 2: Preliminary Final (QF1-loser vs EF-winner)
    pf_h = rng.normal(league_mean, league_std, n_sims)
    pf_a = rng.normal(league_mean, league_std, n_sims)
    pf_winner = np.where(pf_h >= pf_a, qf1_loser, ef_winner)  # to GF

    # Week 3: Grand Final (QF1-winner vs PF-winner)
    gf_h = rng.normal(league_mean, league_std, n_sims)
    gf_a = rng.normal(league_mean, league_std, n_sims)
    champ = np.where(gf_h >= gf_a, qf1_winner, pf_winner)

    gf_count = np.zeros(n_coaches, dtype=np.int64)
    gf_count += np.bincount(qf1_winner, minlength=n_coaches)
    gf_count += np.bincount(pf_winner,  minlength=n_coaches)
    champ_count = np.bincount(champ, minlength=n_coaches)

    results = pd.DataFrame({
        "coach_first_name": coaches,
        "finals_pct": (finals_count / n_sims * 100).round(1),
        "gf_pct":     (gf_count    / n_sims * 100).round(1),
        "champ_pct":  (champ_count / n_sims * 100).round(1),
        "spoon_pct":  (spoon_count / n_sims * 100).round(1),
    })

    log.info(
        "Simulation complete (round %d, %d matchups). Championships: %s",
        current_round,
        len(matchups),
        dict(zip(results["coach_first_name"], results["champ_pct"])),
    )
    return results
