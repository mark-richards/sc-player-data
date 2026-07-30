"""
metrics.py — Computes TPOR, Ceiling Index, CV, and Flex-Adjusted Average.

Scoring philosophy
------------------
League rule: Best 19 of 20 starters score (lowest is dropped).
This means:
  • High-ceiling / high-variance players are MORE valuable — their floor games
    get dropped, keeping only their monster scores.
  • "Flex Bonus" = adjusted_avg - raw_avg quantifies this upside.

TPOR (Total Points Over Replacement)
  = player's effective average − replacement-level average at their position.
Replacement level = the best available free agent at the same position
(or at ANY position for the FLEX slot).

Historical data weighting
  All available seasons (2011–2025) contribute, with recent seasons weighted
  more heavily. If the current (2026) season has ≥ CURRENT_SEASON_MIN_ROUNDS
  data, it becomes the primary source and historical data fills gaps only.
"""

import logging
import statistics
from typing import Optional

import pandas as pd

from waiver.config import (
    CEILING_THRESHOLD,
    CURRENT_SEASON_MIN_ROUNDS,
    DROP_N,
    MIN_GAMES_FOR_METRICS,
)

log = logging.getLogger(__name__)


# ── Pure score calculators ──────────────────────────────────────────────────

def calc_ceiling_index(scores: list[int], threshold: int = CEILING_THRESHOLD) -> float:
    """Percentage of games where player scored above `threshold`."""
    if not scores:
        return 0.0
    return round(sum(1 for s in scores if s > threshold) / len(scores) * 100, 1)


def calc_cv(scores: list[int]) -> float:
    """
    Coefficient of Variation = std_dev / mean.
    Higher CV = more volatile = MORE valuable in a drop-lowest league.
    Returns 0.0 when there are fewer than 2 scores.
    """
    if len(scores) < 2:
        return 0.0
    mean = statistics.mean(scores)
    if mean == 0:
        return 0.0
    return round(statistics.stdev(scores) / mean, 3)


def calc_flex_adjusted_avg(scores: list[int], drop_n: int = DROP_N) -> float:
    """
    Simulates the 'drop lowest N' rule on a player's score history.
    Returns the adjusted average that the rule produces over their game sample.
    """
    if len(scores) <= drop_n:
        return float(sum(scores)) / len(scores) if scores else 0.0
    kept = sorted(scores)[drop_n:]   # drop the drop_n lowest scores
    return round(sum(kept) / len(kept), 1)


def _weighted_scores(
    hist_df: pd.DataFrame,
    feed_id: int,
) -> list[int]:
    """
    Builds a weighted score list for a player from historical fanfooty data.
    Each game score is repeated `weight` times (rounded to int) so that
    recent seasons naturally dominate the distribution.
    """
    player_rows = hist_df[hist_df["feed_id"] == feed_id]
    if player_rows.empty:
        return []
    weighted: list[int] = []
    for _, row in player_rows.iterrows():
        repeats = max(1, round(row["weight"]))
        weighted.extend([int(row["sc_score"])] * repeats)
    return weighted


def _effective_scores(
    row: pd.Series,
    hist_df: pd.DataFrame,
) -> list[int]:
    """
    Choose the best available score sample for a player.

    Priority:
      1. Weighted historical scores from all available fanfooty seasons (2011–2025).
         These have proper year labels so recent seasons can be up-weighted.
      2. Current-season scores from player_stats_current.csv ONLY if hist_df has
         no data for this player at all. This file contains career-wide data of
         unknown year mix, so we only use it as a last resort.
      3. Empty list → player flagged as 'No data'.

    NOTE: Once 2026 fanfooty processed files appear in data/processed/, they will
    automatically receive weight=3 and become the dominant source.
    """
    hist_scores = _weighted_scores(hist_df, int(row["feed_id"]))
    if hist_scores:
        return hist_scores

    # Fallback: use player_stats_current.csv career data (unweighted, year-unknown)
    scores_career: list[int] = row.get("scores_2026") or []
    return scores_career


# ── Replacement level ────────────────────────────────────────────────────────

def _position_bucket(position: str) -> str:
    """Normalise dual-position tags (e.g. 'MID FWD') to a single primary bucket."""
    pos = position.upper().strip()
    for bucket in ("DEF", "RUC", "MID", "FWD"):
        if bucket in pos:
            return bucket
    return "UNK"


def calc_replacement_levels(
    player_table: pd.DataFrame,
    score_col: str = "effective_avg",
) -> dict[str, float]:
    """
    For each position, find the highest average among free agents.
    That player IS the replacement — the next best thing you could pick up.

    The FLEX position uses ALL free agents regardless of position.

    Returns dict: position_bucket → replacement_avg
    """
    fa = player_table[player_table["is_free_agent"]].copy()
    if fa.empty:
        log.warning("No free agents found; replacement levels set to 0.")
        return {"DEF": 0.0, "MID": 0.0, "RUC": 0.0, "FWD": 0.0, "FLEX": 0.0}

    # Only use players active in 2026 (≥2 games, played in last 3 rounds) as the
    # replacement baseline. Without this, an injured historical star sets an
    # unreachable replacement level and every active player shows negative TPOR.
    if "games_played_2026" in fa.columns and "last_round_2026" in fa.columns:
        current_round = int(fa["last_round_2026"].max()) if fa["last_round_2026"].max() > 0 else 0
        recency_cutoff = max(1, current_round - 2)
        active = fa[(fa["games_played_2026"] >= 2) & (fa["last_round_2026"] >= recency_cutoff)]
        if not active.empty:
            fa = active

    fa["pos_bucket"] = fa["position"].apply(_position_bucket)
    levels: dict[str, float] = {}
    for pos in ("DEF", "MID", "RUC", "FWD"):
        pos_fa = fa[fa["pos_bucket"] == pos]
        levels[pos] = float(pos_fa[score_col].max()) if not pos_fa.empty else 0.0

    # FLEX replacement = best FA of any position
    levels["FLEX"] = float(fa[score_col].max()) if not fa.empty else 0.0

    for pos, lvl in levels.items():
        log.debug("Replacement level [%s]: %.1f", pos, lvl)

    return levels


# ── Main scoring function ────────────────────────────────────────────────────

def calc_bayesian_avg(
    scores_2026: list[int],
    effective_avg: float,
    prior_weight: int = 20,
) -> float:
    """
    Bayesian posterior mean blending current-season games with the historical prior.

    Inspired by Ian Graham (Liverpool analytics): "Your team's strength tomorrow
    is about 98% of your team's strength yesterday, plus 2% of how well you did
    today." Rare events (goals, SC scores) provide weak evidence per observation —
    small samples should barely move the prior.

    Formula: posterior = (n * observed + k * prior) / (n + k)
    where k = effective_k, which shrinks as more 2026 games accumulate.

    At R2  (n=2):  k=19 → weight on 2026 =  2/21  =  9%  — nearly all prior
    At R8  (n=8):  k=16 → weight on 2026 =  8/24  = 33%
    At R14 (n=14): k=13 → weight on 2026 = 14/27  = 52%
    At R18 (n=18): k=11 → weight on 2026 = 18/29  = 62%
    """
    n = len(scores_2026)
    if n == 0 or effective_avg == 0:
        return effective_avg
    effective_k = max(8, prior_weight - n // 2)
    observed = sum(scores_2026) / n
    return round((n * observed + effective_k * effective_avg) / (n + effective_k), 1)


def calc_durability_score(scores_by_year: dict, seasons: int = 2) -> float:
    """
    Fraction of available rounds played across the last `seasons` seasons.

    Inspired by Ian Graham: "Availability is almost as important as quality.
    Messi and Ronaldo's career dominance was substantially enabled by playing
    3,800-3,900 top-level minutes per season for nearly 20 years."

    1.0 = perfect availability; 0.5 = missed half the rounds.
    Missing data years default to 0.8 (unknown = assume typical).
    """
    if not scores_by_year:
        return 0.8
    current_year = max(scores_by_year.keys()) if scores_by_year else 2026
    total_played = 0
    total_available = 0
    for yr in range(current_year - seasons + 1, current_year + 1):
        games = scores_by_year.get(yr, None)
        if games is None:
            # Unknown year — count as 0.8 contribution
            total_played += 18
            total_available += 22
        else:
            total_played += len([s for s in games if s > 0])
            total_available += 22
    if total_available == 0:
        return 0.8
    return round(total_played / total_available, 3)


def score_all_players(
    player_table: pd.DataFrame,
    hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enriches the player table with all metrics:
        effective_scores  — the score list used for calculations
        effective_avg     — mean of effective_scores (or avg_2026 if no history)
        bayesian_avg      — Bayesian posterior blending 2026 data with historical prior
        durability_score  — fraction of rounds played in last 2 seasons (0–1)
        ceiling_index     — % of games above CEILING_THRESHOLD
        cv                — Coefficient of Variation (higher = more volatile)
        flex_adj_avg      — adjusted avg after dropping DROP_N lowest scores
        flex_bonus        — flex_adj_avg − effective_avg (upside from drop rule)
        pos_bucket        — normalised position (DEF/MID/RUC/FWD)
        data_source       — "2026", "historical", or "none"
        tpor              — effective_avg − replacement_level[pos_bucket]
        sufficient_data   — True if enough games to trust CI/CV
        hot_streak        — True if recent3_avg > effective_avg * 1.35 (regression risk)
    """
    if player_table.empty:
        return player_table

    df = player_table.copy()
    df["pos_bucket"] = df["position"].apply(_position_bucket)

    # ── Build scores_by_year lookup from hist_df ───────────────────────────
    scores_by_year_map: dict[int, dict[int, list[int]]] = {}
    if not hist_df.empty and "year" in hist_df.columns:
        for fid, grp in hist_df.groupby("feed_id"):
            by_year: dict[int, list[int]] = {}
            for yr, ygrp in grp.groupby("year"):
                if "round_num" in ygrp.columns:
                    ygrp = ygrp.sort_values("round_num")
                by_year[int(yr)] = [int(s) for s in ygrp["sc_score"].tolist()]
            scores_by_year_map[int(fid)] = by_year

    # ── Per-player metric computation ──────────────────────────────────────
    records: list[dict] = []
    for _, row in df.iterrows():
        scores_2026: list[int] = row.get("scores_2026") or []
        eff_scores = _effective_scores(row, hist_df)

        # Determine data source label
        hist_has_data = not hist_df.empty and (hist_df["feed_id"] == int(row["feed_id"])).any()
        if eff_scores:
            if hist_has_data:
                source = "historical"
            elif scores_2026:
                source = "career (unweighted)"
            else:
                source = "historical"
        else:
            source = "none"

        eff_avg = round(sum(eff_scores) / len(eff_scores), 1) if eff_scores else 0.0
        sufficient = len(eff_scores) >= MIN_GAMES_FOR_METRICS

        # Bayesian average: blends 2026 data with historical prior (Graham principle)
        bayes_avg = calc_bayesian_avg(scores_2026, eff_avg)

        # Durability: fraction of rounds played in last 2 seasons
        fid = int(row["feed_id"])
        scores_by_year = dict(scores_by_year_map.get(fid, {}))
        if scores_2026:
            scores_by_year[2026] = scores_2026
        durability = calc_durability_score(scores_by_year)

        # Last-20-games average across seasons — stability input to the
        # positional targets ranking (0.6*avg5 + 0.4*avg20 blend)
        ordered_scores = [s for yr in sorted(scores_by_year) for s in scores_by_year[yr]]
        last20 = ordered_scores[-20:]
        avg20 = round(sum(last20) / len(last20), 1) if last20 else 0.0

        # Hot streak flag: recent form significantly above long-term avg (regression risk)
        recent3 = scores_2026[-3:] if len(scores_2026) >= 3 else scores_2026
        recent3_avg = sum(recent3) / len(recent3) if recent3 else 0.0
        hot_streak = bool(recent3_avg > 0 and eff_avg > 0 and recent3_avg > eff_avg * 1.35)

        records.append(
            {
                "feed_id": fid,
                "effective_scores": eff_scores,
                "effective_avg": eff_avg,
                "bayesian_avg": bayes_avg,
                "avg20": avg20,
                "durability_score": durability,
                "hot_streak": hot_streak,
                "ceiling_index": calc_ceiling_index(eff_scores) if sufficient else 0.0,
                "cv": calc_cv(eff_scores) if sufficient else 0.0,
                "flex_adj_avg": calc_flex_adjusted_avg(eff_scores) if eff_scores else 0.0,
                "data_source": source,
                "sufficient_data": sufficient,
            }
        )

    metrics_df = pd.DataFrame(records)
    df = df.merge(metrics_df, on="feed_id", how="left")

    df["flex_bonus"] = (df["flex_adj_avg"] - df["effective_avg"]).round(1)

    # ── 2026 fanfooty stats for all players ────────────────────────────────
    # Source of truth for current-season activity regardless of ownership status.
    # Used to filter out injured/inactive players from recommendations and to
    # show accurate season averages for free agents who were never in the league.
    hist_2026 = hist_df[hist_df["year"] == 2026] if not hist_df.empty and "year" in hist_df.columns else pd.DataFrame()
    if not hist_2026.empty:
        g2026 = (
            hist_2026.groupby("feed_id")["sc_score"]
            .agg(games_played_2026="count", avg_fanfooty_2026="mean")
            .reset_index()
        )
        g2026["avg_fanfooty_2026"] = g2026["avg_fanfooty_2026"].round(1)
        # Last round played in 2026 — used to detect players who haven't featured recently
        last_round = hist_2026.groupby("feed_id")["round_num"].max().reset_index()
        last_round.columns = ["feed_id", "last_round_2026"]
        g2026 = g2026.merge(last_round, on="feed_id", how="left")
        df = df.merge(g2026, on="feed_id", how="left")
    df["games_played_2026"] = df["games_played_2026"].fillna(0).astype(int) if "games_played_2026" in df.columns else 0
    df["avg_fanfooty_2026"] = df["avg_fanfooty_2026"].fillna(0.0) if "avg_fanfooty_2026" in df.columns else 0.0
    df["last_round_2026"] = df["last_round_2026"].fillna(0).astype(int) if "last_round_2026" in df.columns else 0

    # ── Replacement levels (computed after all averages are populated) ─────
    # Restrict to players with ≥1 game in 2026 so injured/inactive FAs don't
    # inflate the replacement level and make every active player look below-replacement.
    repl = calc_replacement_levels(df, score_col="effective_avg")

    def _tpor(row: pd.Series) -> float:
        repl_level = repl.get(row["pos_bucket"], 0.0)
        return round(float(row["effective_avg"]) - repl_level, 1)

    df["tpor"] = df.apply(_tpor, axis=1)
    df["replacement_level"] = df.apply(
        lambda r: repl.get(r["pos_bucket"], 0.0), axis=1
    )

    log.info(
        "Metrics computed for %d players. Replacement levels: %s",
        len(df),
        {k: f"{v:.1f}" for k, v in repl.items()},
    )
    return df


def calc_positional_scarcity(
    metrics_df: pd.DataFrame,
    tiers: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Count how many free agents are available at each position by quality tier.

    Returns dict: position → {"Elite": N, "Solid": N, "Depth": N}

    Tiers (SC avg thresholds — adjustable):
        Elite:  avg >= 90
        Solid:  75 <= avg < 90
        Depth:  60 <= avg < 75
    """
    if tiers is None:
        tiers = {"Elite": (90.0, 9999.0), "Solid": (75.0, 90.0), "Depth": (60.0, 75.0)}

    avg_col = "avg_fanfooty_2026" if "avg_fanfooty_2026" in metrics_df.columns else "effective_avg"

    fa = metrics_df[
        metrics_df.get("is_free_agent", pd.Series(True, index=metrics_df.index))
        & ~metrics_df.get("is_injured", pd.Series(False, index=metrics_df.index))
    ].copy()

    scarcity: dict[str, dict[str, int]] = {}
    for pos in ("DEF", "MID", "RUC", "FWD"):
        pos_fa = fa[fa["pos_bucket"] == pos]
        counts: dict[str, int] = {}
        for tier_name, (lo, hi) in tiers.items():
            n = int(((pos_fa[avg_col] >= lo) & (pos_fa[avg_col] < hi)).sum())
            counts[tier_name] = n
        scarcity[pos] = counts

    return scarcity
