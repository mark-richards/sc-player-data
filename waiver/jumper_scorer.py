"""
jumper_scorer.py — Scores 2026 free agents as "mid-year jumpers" or "streamers".

Mid-year jumper: FA currently averaging < 80 who is structurally likely to
                 average 85+ for the rest of the season.

Streamer:        FA worth picking up for 1-3 weeks due to a sudden spike in
                 role or score makeup (hot tag, CBA% jump, TOG% spike).

Feature weights derived from analyze_jumpers.py historical analysis (2021-2025).
"""

import glob
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from waiver.config import FANFOOTY_PROCESSED_DIR, SEASON_YEAR

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DFS_DIR = PROJECT_ROOT / "data" / "raw" / "dfsaustralia"

# Feature weights — trajectory-first design.
# Primary signals: what is the player doing RIGHT NOW (CBA, TOG, EMA trend)?
# Career history used only as a ceiling gate (must have shown 80+ is achievable),
# not as a primary ranking signal. This prevents role-decay players (e.g. historically
# great but permanently diminished role) from dominating the recommendations.
_JUMPER_WEIGHTS = {
    "cba_trend_mid":      0.30,   # CBA gaining midfield time (strongest signal — validated +3.0 vs non-breakout)
    "SC_ema_trend":       0.20,   # Scoring momentum (weak but positive +0.41 in historical data)
    "SC_form_delta_3v6":  0.15,   # Recent 3 better than last 6 (+0.86 in historical data)
    "career_avg_delta":   0.10,   # Career ceiling evidence (gate only — must have shown 80+ before)
    "SC_trend_slope_8":   0.10,   # Improving trajectory (+0.66 in historical data)
    "tog_avg_low":        0.08,   # inverted: lower initial TOG = more room to grow
    "SC_skewness_inv":    0.07,   # inverted: lower skewness = more consistent
    # Age-curve proxy: career games as experience indicator. Players with 50-200
    # career games are in their prime for breakouts. Heavy veterans (250+) have
    # limited structural upside. Exact DOB not in our data; games count is used.
    "career_games_prime": 0.05,
}

# AFL age-curve career-games windows (based on Champion Data / PubMed research).
# These are approximate — peak performance in AFL aligns with 50–200 career games
# for most positions, with ruckmen peaking later (often 100–250 games).
_CAREER_GAMES_SCORE = {
    # (min_games, max_games_exclusive): multiplier [0, 1]
    (0,   20):  0.50,  # very early career — high variance, not breakout candidates
    (20,  50):  0.70,  # developing
    (50,  200): 1.00,  # prime breakout window
    (200, 260): 0.75,  # experienced but declining structural upside
    (260, 9999): 0.50, # veteran — very limited upside
}


def _career_games_factor(career_games: int, pos_bucket: str = "") -> float:
    """
    Returns a [0, 1] multiplier capturing the breakout likelihood by career stage.
    Ruckmen peak later so we give them an extra 30-game buffer.
    """
    adjusted = career_games - (30 if pos_bucket == "RUC" else 0)
    adjusted = max(0, adjusted)
    for (lo, hi), score in _CAREER_GAMES_SCORE.items():
        if lo <= adjusted < hi:
            return score
    return 0.50


# ── Data loading ──────────────────────────────────────────────────────────────

def load_2026_fanfooty(
    base_dir: Path = FANFOOTY_PROCESSED_DIR,
    season_year: int = SEASON_YEAR,
) -> pd.DataFrame:
    """
    Loads and concatenates all 2026 fanfooty processed CSV rounds.
    Returns long-form DataFrame with one row per player per round.
    """
    pattern = str(base_dir / f"{season_year}_round_*_fanfooty_data.csv")
    files = sorted(glob.glob(pattern))
    year_round_re = re.compile(r"(\d{4})_round_(\d+)_fanfooty_data\.csv$")

    wanted_rename = {
        "Player ID": "feed_id",
        "SC": "SC",
        "Time on ground": "tog",
        "Clearances": "clearances",
        "Tag": "tag",
        "Tag Notes": "tag_notes",
        "Tag 2": "tag2",
        "Tag 2 Notes": "tag2_notes",
        "First Name": "first_name",
        "Surname": "surname",
    }

    frames = []
    for fpath in files:
        m = year_round_re.search(fpath)
        if not m or int(m.group(1)) != season_year:
            continue
        round_num = int(m.group(2))
        try:
            chunk = pd.read_csv(fpath, low_memory=False)
        except Exception as exc:
            log.warning("Could not read %s: %s", fpath, exc)
            continue

        chunk = chunk.rename(columns={k: v for k, v in wanted_rename.items() if k in chunk.columns})
        keep = [c for c in wanted_rename.values() if c in chunk.columns]
        chunk = chunk[keep].copy()

        chunk["feed_id"] = pd.to_numeric(chunk.get("feed_id", pd.Series(dtype=float)), errors="coerce")
        chunk["SC"] = pd.to_numeric(chunk.get("SC", pd.Series(dtype=float)), errors="coerce")
        chunk["tog"] = pd.to_numeric(chunk.get("tog", pd.Series(dtype=float)), errors="coerce")
        chunk["clearances"] = pd.to_numeric(chunk.get("clearances", pd.Series(dtype=float)), errors="coerce")
        chunk = chunk.dropna(subset=["feed_id", "SC"])
        chunk["feed_id"] = chunk["feed_id"].astype(int)
        chunk["round_num"] = round_num
        frames.append(chunk)

    if not frames:
        log.warning("No 2026 fanfooty data found in %s", base_dir)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d 2026 fanfooty rows across %d rounds.", len(out), out["round_num"].nunique())
    return out


def load_2026_cbas(
    dfs_dir: Path = DFS_DIR,
    season_year: int = SEASON_YEAR,
) -> pd.DataFrame:
    """
    Loads CBA% data for the current season.
    Returns empty DataFrame if unavailable (scorer degrades gracefully).
    """
    path = dfs_dir / str(season_year) / f"cbas_{season_year}.csv"
    if not path.exists():
        log.info("No CBA data found at %s — CBA signals will be skipped.", path)
        return pd.DataFrame(columns=["feed_id", "round_num", "cba_pct"])
    try:
        df = pd.read_csv(path)
        df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce").astype("Int64")
        df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce").astype("Int64")
        df["cba_pct"] = pd.to_numeric(df["cba_pct"], errors="coerce")
        log.info("Loaded 2026 CBA data: %d rows, max round %d.", len(df), df["round_num"].max())
        return df.dropna(subset=["feed_id", "cba_pct"])
    except Exception as exc:
        log.warning("Could not load 2026 CBA data: %s", exc)
        return pd.DataFrame(columns=["feed_id", "round_num", "cba_pct"])


# ── Feature computation ────────────────────────────────────────────────────────

def _slope(arr: list[float]) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def _ewm_trend(arr: list[float]) -> float:
    if len(arr) < 3:
        return 0.0
    s = pd.Series(arr, dtype=float)
    return float(s.ewm(span=3, adjust=False).mean().iloc[-1] - s.ewm(span=6, adjust=False).mean().iloc[-1])


def _safe_mean(arr: list) -> float | None:
    clean = [x for x in arr if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(clean)) if clean else None


def compute_features(
    fid: int,
    scores_2026: list[int],
    ff_player: pd.DataFrame,
    cbas_player: pd.DataFrame,
    career_avg_prior: float | None,
    pos_bucket: str,
    career_games: int = 0,
) -> dict:
    """
    Compute the jumper/streamer feature vector for a single player using 2026 data.
    All features are derived from data already available (no leakage).
    """
    feats: dict = {
        "feed_id": fid,
        "pos_bucket": pos_bucket,
        "n_games": len(scores_2026),
        "season_avg": float(np.mean(scores_2026)) if scores_2026 else 0.0,
    }

    # SC trajectory features
    scores = [float(s) for s in scores_2026]
    last3 = scores[-3:] if len(scores) >= 3 else scores
    last6 = scores[-6:] if len(scores) >= 6 else scores
    avg3 = float(np.mean(last3)) if last3 else 0.0
    avg6 = float(np.mean(last6)) if last6 else 0.0

    feats["SC_ema_trend"] = _ewm_trend(scores)
    feats["SC_trend_slope_8"] = _slope(scores)
    feats["SC_form_delta_3v6"] = avg3 - avg6
    feats["SC_skewness"] = float(scipy_stats.skew(scores)) if len(scores) >= 3 else 0.0
    feats["SC_ceiling_rate"] = sum(1 for s in scores if s >= 100) / len(scores) if scores else 0.0
    feats["avg3"] = avg3

    # Career avg delta (regression-to-mean signal)
    feats["career_avg_prior"] = career_avg_prior
    feats["career_avg_delta"] = (career_avg_prior - feats["season_avg"]) if career_avg_prior else None

    # Career-stage proxy: maps career games count to a [0, 1] breakout-readiness score.
    feats["career_games"] = career_games
    feats["career_games_prime"] = _career_games_factor(career_games, pos_bucket)

    # TOG features
    togs = ff_player["tog"].dropna().tolist() if not ff_player.empty and "tog" in ff_player.columns else []
    tog_avg = _safe_mean(togs)
    tog_last2 = _safe_mean(togs[-2:]) if len(togs) >= 2 else tog_avg
    feats["tog_avg"] = tog_avg
    feats["tog_trend"] = (tog_last2 - tog_avg) if (tog_avg is not None and tog_last2 is not None) else None

    # Clearances features
    clears = ff_player["clearances"].dropna().tolist() if not ff_player.empty and "clearances" in ff_player.columns else []
    clears_avg = _safe_mean(clears)
    clears_last2 = _safe_mean(clears[-2:]) if len(clears) >= 2 else clears_avg
    feats["clearances_avg"] = clears_avg
    feats["clearances_trend"] = (clears_last2 - clears_avg) if (clears_avg is not None and clears_last2 is not None) else None

    # CBA features
    cbas = cbas_player["cba_pct"].dropna().tolist() if not cbas_player.empty else []
    cba_avg = _safe_mean(cbas)
    cba_last2 = _safe_mean(cbas[-2:]) if len(cbas) >= 2 else cba_avg
    feats["cba_avg"] = cba_avg
    feats["cba_trend"] = (cba_last2 - cba_avg) if (cba_avg is not None and cba_last2 is not None) else None

    # Tag signals (scan last 2 rounds in fanfooty data)
    if not ff_player.empty and "tag" in ff_player.columns:
        all_tags = ff_player.sort_values("round_num")["tag"].str.lower().fillna("").tolist()
        last2_tags = all_tags[-2:] if len(all_tags) >= 2 else all_tags
        feats["tag_is_hot_r78"] = int(any("hot" in t for t in last2_tags))
        # Injured early then clean
        early3 = all_tags[:3]
        late_tags = all_tags[3:]
        feats["tag_injured_early_clean_later"] = int(
            any(t in ("injured", "sore") for t in early3)
            and all(t not in ("injured", "sore") for t in late_tags)
            if late_tags else False
        )
    else:
        feats["tag_is_hot_r78"] = 0
        feats["tag_injured_early_clean_later"] = 0

    return feats


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Rank values to [0, 1] using percentile. NaN → 0.5 (neutral)."""
    ranked = series.rank(pct=True, na_option="keep")
    return ranked.fillna(0.5)


# ── Jumper scoring ─────────────────────────────────────────────────────────────

def _build_jumper_score(feats_df: pd.DataFrame) -> pd.Series:
    """
    Applies weighted composite jumper scoring.
    Each feature is percentile-ranked to [0,1], then weighted and summed.
    """
    score = pd.Series(0.0, index=feats_df.index)

    def _add(col: pd.Series, weight: float, higher_is_better: bool = True):
        ranked = _percentile_rank(col)
        if not higher_is_better:
            ranked = 1.0 - ranked
        score.add(ranked * weight, fill_value=0.0)
        return score.__iadd__(ranked * weight) or score

    # career_avg_delta: higher = better
    score += _percentile_rank(feats_df["career_avg_delta"]) * _JUMPER_WEIGHTS["career_avg_delta"]

    # SC_skewness: lower = more consistent → invert
    score += (1.0 - _percentile_rank(feats_df["SC_skewness"])) * _JUMPER_WEIGHTS["SC_skewness_inv"]

    # tog_avg: lower initial TOG = more room to grow → invert
    score += (1.0 - _percentile_rank(feats_df["tog_avg"])) * _JUMPER_WEIGHTS["tog_avg_low"]

    # SC_ema_trend: higher = better
    score += _percentile_rank(feats_df["SC_ema_trend"]) * _JUMPER_WEIGHTS["SC_ema_trend"]

    # SC_form_delta_3v6: higher = better
    score += _percentile_rank(feats_df["SC_form_delta_3v6"]) * _JUMPER_WEIGHTS["SC_form_delta_3v6"]

    # CBA trend: only meaningful for MID/FWD
    cba_col = feats_df["cba_trend"].where(feats_df["pos_bucket"].isin(["MID", "FWD"]))
    score += _percentile_rank(cba_col) * _JUMPER_WEIGHTS["cba_trend_mid"]

    # SC_trend_slope_8: positive slope = improving
    score += _percentile_rank(feats_df["SC_trend_slope_8"]) * _JUMPER_WEIGHTS["SC_trend_slope_8"]

    # career_games_prime: already in [0, 1] — no need to percentile-rank; use directly
    if "career_games_prime" in feats_df.columns:
        score += feats_df["career_games_prime"].fillna(0.5) * _JUMPER_WEIGHTS["career_games_prime"]

    # DVP opponent bonus: flat +0.03 for players facing a top-4 weakest defence
    if "next_dvp_rank" in feats_df.columns:
        soft_matchup = feats_df["next_dvp_rank"].fillna(99) <= 4
        score += soft_matchup.astype(float) * 0.03

    return score.round(4)


def _build_key_reason(row: pd.Series) -> str:
    """
    Generates a human-readable 1-2 sentence explanation of the top signals
    driving a player's jumper or streamer score.
    """
    signals = []

    delta = row.get("career_avg_delta")
    if isinstance(delta, float) and delta > 8:
        signals.append(f"career avg {delta:.0f}pts above current")

    tog_t = row.get("tog_trend")
    if isinstance(tog_t, float) and tog_t > 3:
        signals.append(f"TOG rising +{tog_t:.0f}%")

    cba_t = row.get("cba_trend")
    if isinstance(cba_t, float) and cba_t > 5 and row.get("pos_bucket") in ("MID", "FWD"):
        signals.append(f"CBA% up +{cba_t:.0f}pts")

    ema = row.get("SC_ema_trend")
    if isinstance(ema, float) and ema > 5:
        signals.append(f"short-term momentum +{ema:.0f}")

    fd = row.get("SC_form_delta_3v6")
    if isinstance(fd, float) and fd > 8:
        signals.append(f"last-3 avg +{fd:.0f} vs last-6")

    if row.get("tag_injured_early_clean_later"):
        signals.append("injured early, now cleared")

    if row.get("tag_is_hot_r78"):
        signals.append("hot tag R7/8")

    if not signals:
        slope = row.get("SC_trend_slope_8", 0)
        if isinstance(slope, float) and slope > 0:
            signals.append(f"improving trend (slope +{slope:.1f})")
        else:
            signals.append("regression candidate")

    return " + ".join(signals[:3])


# ── Streamer scoring ────────────────────────────────────────────────────────────

def _build_streamer_score(feats_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Counts triggered short-horizon signals for each player.
    Returns (score series [0–7], trigger_text series).
    """
    scores = pd.Series(0, index=feats_df.index)
    triggers: list[list[str]] = [[] for _ in feats_df.index]

    for i, (_, row) in enumerate(feats_df.iterrows()):
        t = triggers[i]

        # TOG spike: last-2 avg >= 8pts above season avg
        tog_t = row.get("tog_trend")
        if isinstance(tog_t, float) and tog_t >= 8:
            scores.iloc[i] += 2
            t.append(f"TOG spike +{tog_t:.0f}%")

        # Hot tag in last 1-2 rounds
        if row.get("tag_is_hot_r78"):
            scores.iloc[i] += 2
            t.append("hot tag R7/8")

        # CBA spike (MID/FWD): cba_trend >= 20 from near-zero
        cba_t = row.get("cba_trend")
        cba_avg = row.get("cba_avg") or 0
        if (row.get("pos_bucket") in ("MID", "FWD")
                and isinstance(cba_t, float) and cba_t >= 20
                and cba_avg < 30):
            scores.iloc[i] += 2
            t.append(f"CBA spike +{cba_t:.0f}pts")

        # avg3 >> season avg (recent 3-game avg 15+ pts above season)
        fd = row.get("SC_form_delta_3v6")
        if isinstance(fd, float) and fd >= 15:
            scores.iloc[i] += 1
            t.append(f"last-3 avg +{fd:.0f} vs season")

        # Clearances spike: last-2 clearances >= 1.5x season avg
        clears_t = row.get("clearances_trend")
        clears_a = row.get("clearances_avg") or 0
        if isinstance(clears_t, float) and clears_a > 0 and clears_t >= clears_a * 0.5:
            scores.iloc[i] += 1
            t.append(f"clearances up +{clears_t:.1f}")

        # Injury return: cleaned up from early injury
        if row.get("tag_injured_early_clean_later"):
            scores.iloc[i] += 1
            t.append("return from injury")

    trigger_series = pd.Series(
        [" + ".join(t) if t else "no clear trigger" for t in triggers],
        index=feats_df.index,
    )
    return scores, trigger_series


# ── Main entry point ─────────────────────────────────────────────────────────

def score_free_agents(
    metrics_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    fanfooty_2026_df: pd.DataFrame,
    cbas_2026_df: pd.DataFrame,
    min_games: int = 5,
    season_avg_max: float = 80.0,
    current_round: int | None = None,
    jumper_threshold: float = 0.55,
    streamer_threshold: int = 2,
    min_career_avg: float = 75.0,
    min_avg3_jumper: float = 75.0,
    min_avg3_streamer: float = 80.0,
    dvp_ranks: "dict[int, int] | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scores all free agents as jumper candidates and streamer candidates.

    Parameters
    ----------
    metrics_df        : enriched player table from metrics.score_all_players()
    hist_df           : historical fanfooty scores (from data_loader.load_historical_scores())
    fanfooty_2026_df  : all 2026 fanfooty rounds concatenated
    cbas_2026_df      : 2026 CBA% data
    min_games         : minimum 2026 games before scoring
    season_avg_max    : only target players averaging below this (typically 80)
    current_round     : last completed round (for logging)
    jumper_threshold  : minimum jumper_score to include in output (0-1 scale)
    streamer_threshold: minimum streamer signal count to include
    min_career_avg    : jumper must have career avg >= this OR recent avg3 >= min_avg3_jumper
                        (ensures the player has a realistic ceiling to reach 85+)
    min_avg3_jumper   : fallback ceiling check for jumpers without enough career history
    min_avg3_streamer : streamers must have last-3 avg >= this (already near 85)

    Returns
    -------
    (jumpers_df, streamers_df) — ranked DataFrames, or (empty, empty) on error.
    """
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Build career avg lookup (pre-2026 seasons) ────────────────────────────
    career_avgs: dict[int, float] = {}
    if not hist_df.empty:
        pre2026 = hist_df[hist_df["year"] < 2026] if "year" in hist_df.columns else hist_df
        for fid, grp in pre2026.groupby("feed_id"):
            scores = grp["sc_score"].dropna().tolist()
            if len(scores) >= 5:
                career_avgs[int(fid)] = float(np.mean(scores))

    # ── Filter to FA candidates ───────────────────────────────────────────────
    fa = metrics_df[
        metrics_df.get("is_free_agent", pd.Series(True, index=metrics_df.index))
    ].copy()

    # Use avg_fanfooty_2026 as primary avg (fanfooty-sourced), fall back to avg_2026
    if "avg_fanfooty_2026" in fa.columns:
        avg_col = "avg_fanfooty_2026"
        games_col = "games_played_2026" if "games_played_2026" in fa.columns else None
    else:
        avg_col = "avg_2026"
        games_col = "rounds_played_2026" if "rounds_played_2026" in fa.columns else None

    fa["_avg"] = pd.to_numeric(fa.get(avg_col, 0), errors="coerce").fillna(0.0)
    if games_col:
        fa["_games"] = pd.to_numeric(fa.get(games_col, 0), errors="coerce").fillna(0).astype(int)
    elif "scores_2026" in fa.columns:
        fa["_games"] = fa["scores_2026"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        fa["_games"] = 0

    candidates = fa[
        (fa["_avg"] < season_avg_max) & (fa["_games"] >= min_games)
    ].copy()

    if candidates.empty:
        log.info(
            "No FA candidates (avg < %.0f, games >= %d). "
            "No jumper/streamer sections will be generated.",
            season_avg_max, min_games,
        )
        return pd.DataFrame(), pd.DataFrame()

    log.info(
        "Scoring %d FA candidates (avg < %.0f, games >= %d) for R%s jumper/streamer signals.",
        len(candidates), season_avg_max, min_games, current_round or "?",
    )

    # ── Compute features for each candidate ───────────────────────────────────
    all_feats = []
    for _, row in candidates.iterrows():
        fid = int(row["feed_id"])
        sc_scores = row.get("scores_2026") or []
        if not isinstance(sc_scores, list):
            sc_scores = []

        ff_player = (
            fanfooty_2026_df[fanfooty_2026_df["feed_id"] == fid].sort_values("round_num")
            if not fanfooty_2026_df.empty else pd.DataFrame()
        )
        cba_player = (
            cbas_2026_df[cbas_2026_df["feed_id"] == fid].sort_values("round_num")
            if not cbas_2026_df.empty else pd.DataFrame()
        )

        # Use fanfooty SC scores as the primary source for trajectory features.
        # Some players appear in fanfooty data but not in the SC API scores list
        # (e.g. players who joined late / not in the SC player list). In these
        # cases ff_sc_scores provides the round-by-round signal.
        if sc_scores:
            scores_for_features = sc_scores
        elif not ff_player.empty and "SC" in ff_player.columns:
            scores_for_features = ff_player["SC"].dropna().astype(int).tolist()
        else:
            scores_for_features = []

        feats = compute_features(
            fid=fid,
            scores_2026=scores_for_features,
            ff_player=ff_player,
            cbas_player=cba_player,
            career_avg_prior=career_avgs.get(fid),
            pos_bucket=str(row.get("pos_bucket", "UNK")),
            career_games=int(row.get("previous_games") or 0),
        )

        # Always use avg_fanfooty_2026 as season_avg (excludes DNP rounds → more accurate)
        fanfooty_avg = float(row.get("avg_fanfooty_2026") or feats["season_avg"])
        feats["season_avg"] = round(fanfooty_avg, 1)
        # Recompute career_avg_delta against the corrected season_avg
        if feats.get("career_avg_prior") is not None:
            feats["career_avg_delta"] = round(feats["career_avg_prior"] - feats["season_avg"], 1)

        # Carry forward display columns
        feats["name"] = row.get("name", "Unknown")
        feats["team"] = row.get("team", "")
        feats["bayesian_avg"] = row.get("bayesian_avg", feats["season_avg"])
        feats["price_change_sum3"] = (
            sum(sc_scores[-3:]) if len(sc_scores) >= 3 else None
        )
        feats["next_dvp_rank"] = dvp_ranks.get(fid) if dvp_ranks else None

        all_feats.append(feats)

    feats_df = pd.DataFrame(all_feats).set_index("feed_id")

    # ── Score jumpers ─────────────────────────────────────────────────────────
    feats_df["jumper_score"] = _build_jumper_score(feats_df)
    feats_df["jumper_reason"] = feats_df.apply(_build_key_reason, axis=1)

    # ── Score streamers ───────────────────────────────────────────────────────
    feats_df["streamer_score"], feats_df["streamer_trigger"] = _build_streamer_score(feats_df)

    # ── Build output DataFrames ───────────────────────────────────────────────
    display_cols = [
        "name", "pos_bucket", "team", "season_avg", "career_avg_prior",
        "avg3", "tog_avg", "tog_trend", "cba_avg", "cba_trend",
        "SC_ema_trend", "SC_form_delta_3v6", "SC_trend_slope_8",
        "tag_is_hot_r78", "tag_injured_early_clean_later",
        "career_avg_delta", "n_games",
    ]

    # ── Ceiling filters: only keep players who can realistically average 85+ ──
    # Jumpers need a career avg >= min_career_avg (proven ceiling) OR a recent
    # 3-game avg >= min_avg3_jumper (current form already trending there).
    career_ok = feats_df["career_avg_prior"].fillna(0) >= min_career_avg
    avg3_ok   = feats_df["avg3"].fillna(0) >= min_avg3_jumper
    jumper_ceiling_mask = career_ok | avg3_ok

    # Streamers need avg3 >= min_avg3_streamer — they must already be near 85+.
    streamer_ceiling_mask = feats_df["avg3"].fillna(0) >= min_avg3_streamer

    # ── Role-decay filter: exclude persistent underperformers with no recovery signals ──
    # Players averaging <65 for 6+ games with no positive trajectory are role-decay
    # cases (permanent position change, reduced role), not regression-to-mean candidates.
    role_decay_mask = (
        (feats_df["season_avg"] < 65)
        & (feats_df["n_games"] >= 6)
        & (feats_df["SC_ema_trend"] < 0)
        & (feats_df["cba_trend"].fillna(0) < 5)
        & (feats_df["tog_trend"].fillna(0) < 3)
    )

    # ── Positive trajectory filter: must show at least one improving signal ──
    # Career history alone (career_avg_delta) is not enough — the player must show
    # active evidence of improvement in the current season.
    positive_trajectory_mask = (
        (feats_df["SC_ema_trend"] > 2)
        | (feats_df["cba_trend"].fillna(0) > 8)
        | (feats_df["tog_trend"].fillna(0) > 4)
        | (feats_df["tag_is_hot_r78"] == 1)
    )

    jumpers_df = (
        feats_df[
            (feats_df["jumper_score"] >= jumper_threshold)
            & jumper_ceiling_mask
            & ~role_decay_mask
            & positive_trajectory_mask
        ]
        [display_cols + ["jumper_score", "jumper_reason"]]
        .sort_values("jumper_score", ascending=False)
        .reset_index()
    )

    streamers_df = (
        feats_df[
            (feats_df["streamer_score"] >= streamer_threshold)
            & streamer_ceiling_mask
        ]
        [display_cols + ["streamer_score", "streamer_trigger"]]
        .sort_values("streamer_score", ascending=False)
        .reset_index()
    )

    log.info(
        "Jumper candidates: %d (score >= %.2f, career/avg3 ceiling) | "
        "Streamer candidates: %d (signals >= %d, avg3 >= %.0f)",
        len(jumpers_df), jumper_threshold,
        len(streamers_df), streamer_threshold, min_avg3_streamer,
    )

    return jumpers_df, streamers_df
