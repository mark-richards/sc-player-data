"""
analyze_jumpers.py — Historical discovery script for mid-year SC jumpers.

Scans 2021-2025 seasons to identify players who averaged < 80 SC through
round 8 but averaged 85+ for the rest of the season ("jumpers"), then
computes feature snapshots at round 8 and ranks them by predictive importance.

Usage:
    py analyze_jumpers.py
    py analyze_jumpers.py --years 2022 2023 2024 2025
    py analyze_jumpers.py --early-threshold 80 --late-threshold 85
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_player_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "jumper_analysis"

NEEDED_COLS = [
    "Player ID", "First Name", "Last Name", "sc_position", "Year", "Round_Num",
    "is_pre_season", "played", "SC", "price", "price_change",
    "Time on ground", "Clearances", "cba_pct", "Tag", "Tag Notes",
    "minutes_played",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def _normalise_position(pos: str) -> str:
    """Map multi-position strings to a single primary position bucket."""
    if not isinstance(pos, str):
        return "UNK"
    p = pos.strip().upper()
    if p.startswith("RUC"):
        return "RUC"
    if p.startswith("MID"):
        return "MID"
    if p.startswith("DEF"):
        return "DEF"
    if p.startswith("FWD"):
        return "FWD"
    return "UNK"


def load_data(years: list[int]) -> pd.DataFrame:
    log.info("Loading master_player_data.csv ...")
    df = pd.read_csv(MASTER_PATH, usecols=NEEDED_COLS, low_memory=False)

    df = df[
        df["Year"].isin(years)
        & (df["is_pre_season"] == 0)
        & (df["played"] == 1)
        & (df["Round_Num"] >= 1)
        & df["SC"].notna()
    ].copy()

    df["pos"] = df["sc_position"].apply(_normalise_position)
    df["feed_id"] = df["Player ID"].astype(int)
    df["SC"] = pd.to_numeric(df["SC"], errors="coerce")
    df["price_change"] = pd.to_numeric(df["price_change"], errors="coerce").fillna(0)
    df["Time on ground"] = pd.to_numeric(df["Time on ground"], errors="coerce")
    df["Clearances"] = pd.to_numeric(df["Clearances"], errors="coerce")
    df["cba_pct"] = pd.to_numeric(df["cba_pct"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    log.info("Loaded %d played rows across years %s.", len(df), years)
    return df


# ── Career average (prior seasons) ───────────────────────────────────────────

def compute_career_avgs(df: pd.DataFrame) -> pd.Series:
    """For each (feed_id, year) pair, compute career SC avg in all PRIOR years."""
    records = []
    for (fid, yr), grp in df.groupby(["feed_id", "Year"]):
        prior = df[(df["feed_id"] == fid) & (df["Year"] < yr)]
        if len(prior) >= 5:
            records.append({"feed_id": fid, "Year": yr, "career_avg_prior": prior["SC"].mean()})
        else:
            records.append({"feed_id": fid, "Year": yr, "career_avg_prior": np.nan})
    return pd.DataFrame(records).set_index(["feed_id", "Year"])["career_avg_prior"]


# ── Label jumpers ─────────────────────────────────────────────────────────────

def label_jumpers(
    df: pd.DataFrame,
    early_threshold: float,
    late_threshold: float,
    min_early: int,
    min_late: int,
) -> pd.DataFrame:
    """
    Returns one row per qualifying player-season with:
        early_avg, late_avg, n_early, n_late, is_jumper
    Qualifying = min_early games in R1-8 AND min_late games in R9+.
    """
    early = df[df["Round_Num"] <= 8].groupby(["feed_id", "Year"])["SC"].agg(["mean", "count"]).rename(columns={"mean": "early_avg", "count": "n_early"})
    late = df[df["Round_Num"] >= 9].groupby(["feed_id", "Year"])["SC"].agg(["mean", "count"]).rename(columns={"mean": "late_avg", "count": "n_late"})

    merged = early.join(late, how="inner").reset_index()
    merged = merged[
        (merged["n_early"] >= min_early)
        & (merged["n_late"] >= min_late)
    ].copy()

    merged["is_jumper"] = (merged["early_avg"] < early_threshold) & (merged["late_avg"] >= late_threshold)
    merged["improvement"] = merged["late_avg"] - merged["early_avg"]

    log.info(
        "Qualifying player-seasons: %d | Jumpers (early<%.0f, late>=%.0f): %d",
        len(merged), early_threshold, late_threshold, merged["is_jumper"].sum(),
    )
    return merged


# ── R8 feature snapshot ───────────────────────────────────────────────────────

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
    ewm3 = s.ewm(span=3, adjust=False).mean().iloc[-1]
    ewm6 = s.ewm(span=6, adjust=False).mean().iloc[-1]
    return ewm3 - ewm6


def _skewness(arr: list[float]) -> float:
    if len(arr) < 3:
        return 0.0
    try:
        return float(stats.skew(arr))
    except Exception:
        return 0.0


def compute_r8_snapshot(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    career_avgs: pd.Series,
) -> pd.DataFrame:
    """
    For each qualifying player-season, compute features at the END of round 8
    using only data from rounds 1 through 8 (no leakage).
    """
    early_df = df[df["Round_Num"] <= 8].copy()
    rows = []

    for _, label_row in labels.iterrows():
        fid = label_row["feed_id"]
        yr = label_row["Year"]

        player_early = early_df[
            (early_df["feed_id"] == fid) & (early_df["Year"] == yr)
        ].sort_values("Round_Num")

        if player_early.empty:
            continue

        scores = player_early["SC"].tolist()
        togs = player_early["Time on ground"].dropna().tolist()
        clears = player_early["Clearances"].dropna().tolist()
        cbas = player_early["cba_pct"].dropna().tolist()
        price_changes = player_early["price_change"].fillna(0).tolist()

        # Name / position (from last played row)
        last = player_early.iloc[-1]
        first_name = last.get("First Name", "")
        last_name = last.get("Last Name", "")
        pos = last.get("pos", "UNK")

        # Score features
        avg_r8 = np.mean(scores)
        last3 = scores[-3:] if len(scores) >= 3 else scores
        last6 = scores[-6:] if len(scores) >= 6 else scores
        avg3 = np.mean(last3)
        avg6 = np.mean(last6)
        form_delta_3v6 = avg3 - avg6
        ema_trend = _ewm_trend(scores)
        slope = _slope(scores)
        skew = _skewness(scores)
        ceiling_rate = sum(1 for s in scores if s >= 100) / len(scores)

        # Career avg delta
        career_avg = career_avgs.get((fid, yr), np.nan)
        career_delta = (career_avg - avg_r8) if not np.isnan(career_avg) else np.nan

        # TOG features
        tog_avg = np.mean(togs) if togs else np.nan
        tog_last2 = np.mean(togs[-2:]) if len(togs) >= 2 else (togs[0] if togs else np.nan)
        tog_trend = (tog_last2 - tog_avg) if not np.isnan(tog_avg) else np.nan

        # Clearances features
        clears_avg = np.mean(clears) if clears else np.nan
        clears_last2 = np.mean(clears[-2:]) if len(clears) >= 2 else (clears[0] if clears else np.nan)
        clears_trend = (clears_last2 - clears_avg) if not np.isnan(clears_avg) else np.nan

        # CBA features
        cba_avg = np.mean(cbas) if cbas else np.nan
        cba_last2 = np.mean(cbas[-2:]) if len(cbas) >= 2 else (cbas[0] if cbas else np.nan)
        cba_trend = (cba_last2 - cba_avg) if not np.isnan(cba_avg) else np.nan

        # Price momentum (last 3 rounds cumulative change)
        pc_last3 = price_changes[-3:]
        price_change_sum3 = sum(pc_last3)
        last_price = player_early["price"].iloc[-1] if "price" in player_early.columns else np.nan
        price_is_floor = (
            1 if (not np.isnan(last_price) and last_price < 300_000 and price_change_sum3 >= 0)
            else 0
        )

        # Tag signals — scan last 2 rounds
        tags_all = player_early["Tag"].str.lower().fillna("").tolist()
        tags_last2 = tags_all[-2:]
        tag_is_hot_r78 = int(any("hot" in t for t in tags_last2))
        tag_is_hot_any = int(any("hot" in t for t in tags_all))
        # Injured early (R1-3) then clean later
        tags_early3 = tags_all[:3]
        tags_late = tags_all[3:] if len(tags_all) > 3 else []
        tag_injured_early = int(any(t in ("injured", "sore") for t in tags_early3))
        tag_clean_later = int(all(t not in ("injured", "sore") for t in tags_late) if tags_late else False)
        tag_is_injured_returning = int(tag_injured_early and tag_clean_later)

        # Position flags
        pos_is_mid = int(pos == "MID")
        pos_is_def = int(pos == "DEF")
        pos_is_fwd = int(pos == "FWD")
        pos_is_ruc = int(pos == "RUC")

        rows.append({
            "feed_id": fid,
            "Year": yr,
            "name": f"{first_name} {last_name}".strip(),
            "pos": pos,
            "early_avg": round(avg_r8, 1),
            "late_avg": round(label_row["late_avg"], 1),
            "improvement": round(label_row["improvement"], 1),
            "n_early": label_row["n_early"],
            "n_late": label_row["n_late"],
            "is_jumper": int(label_row["is_jumper"]),
            "career_avg_prior": round(career_avg, 1) if not np.isnan(career_avg) else np.nan,
            "career_avg_delta": round(career_delta, 1) if not np.isnan(career_delta) else np.nan,
            "SC_trend_slope_8": round(slope, 3),
            "SC_form_delta_3v6": round(form_delta_3v6, 1),
            "SC_ema_trend": round(ema_trend, 1),
            "SC_skewness": round(skew, 3),
            "SC_ceiling_rate": round(ceiling_rate, 3),
            "tog_avg_r8": round(tog_avg, 1) if not np.isnan(tog_avg) else np.nan,
            "tog_trend": round(tog_trend, 1) if not np.isnan(tog_trend) else np.nan,
            "clearances_avg_r8": round(clears_avg, 2) if not np.isnan(clears_avg) else np.nan,
            "clearances_trend": round(clears_trend, 2) if not np.isnan(clears_trend) else np.nan,
            "cba_avg_r8": round(cba_avg, 1) if not np.isnan(cba_avg) else np.nan,
            "cba_trend": round(cba_trend, 1) if not np.isnan(cba_trend) else np.nan,
            "price_change_sum3": int(price_change_sum3),
            "price_is_floor": price_is_floor,
            "tag_is_hot_r78": tag_is_hot_r78,
            "tag_is_hot_any": tag_is_hot_any,
            "tag_is_injured_returning": tag_is_injured_returning,
            "pos_is_mid": pos_is_mid,
            "pos_is_def": pos_is_def,
            "pos_is_fwd": pos_is_fwd,
            "pos_is_ruc": pos_is_ruc,
        })

    return pd.DataFrame(rows)


# ── Feature importance ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "career_avg_delta",
    "SC_trend_slope_8",
    "SC_form_delta_3v6",
    "SC_ema_trend",
    "SC_skewness",
    "SC_ceiling_rate",
    "tog_avg_r8",
    "tog_trend",
    "clearances_avg_r8",
    "clearances_trend",
    "cba_avg_r8",
    "cba_trend",
    "price_change_sum3",
    "price_is_floor",
    "tag_is_hot_r78",
    "tag_is_hot_any",
    "tag_is_injured_returning",
    "pos_is_mid",
    "pos_is_def",
    "pos_is_fwd",
]


def run_feature_importance(snapshot: pd.DataFrame) -> pd.DataFrame:
    valid = snapshot.copy()
    # Fill NaN with column median for modelling (not for display)
    for col in FEATURE_COLS:
        if col in valid.columns:
            valid[col] = valid[col].fillna(valid[col].median())

    y = valid["is_jumper"].values
    results = []

    for col in FEATURE_COLS:
        if col not in valid.columns:
            continue
        x = valid[col].values
        pb_corr, pb_pval = stats.pointbiserialr(y, x)
        results.append({
            "feature": col,
            "point_biserial_corr": round(pb_corr, 4),
            "abs_corr": round(abs(pb_corr), 4),
            "p_value": round(pb_pval, 4),
        })

    importance_df = pd.DataFrame(results).sort_values("abs_corr", ascending=False)

    # LightGBM feature importance if available
    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold

        X = valid[FEATURE_COLS].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        importances = np.zeros(len(FEATURE_COLS))

        for tr_idx, val_idx in skf.split(X, y):
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=5,
                class_weight="balanced",
                verbose=-1,
                random_state=42,
            )
            clf.fit(X[tr_idx], y[tr_idx])
            importances += clf.feature_importances_

        importances /= 5
        lgb_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "lgb_importance": importances,
        })
        importance_df = importance_df.merge(lgb_df, on="feature", how="left")
        importance_df = importance_df.sort_values("lgb_importance", ascending=False)
        log.info("LightGBM feature importance computed.")
    except ImportError:
        log.warning("lightgbm not installed — only point-biserial correlations computed.")

    return importance_df


# ── Printing helpers ──────────────────────────────────────────────────────────

def print_top_jumpers(snapshot: pd.DataFrame, n: int = 30) -> None:
    jumpers = (
        snapshot[snapshot["is_jumper"] == 1]
        .sort_values("improvement", ascending=False)
        .head(n)
    )
    cols = ["name", "pos", "Year", "early_avg", "late_avg", "improvement",
            "career_avg_prior", "career_avg_delta", "SC_trend_slope_8",
            "tog_trend", "cba_trend", "tag_is_hot_r78", "tag_is_injured_returning"]
    cols = [c for c in cols if c in jumpers.columns]

    print(f"\n{'='*100}")
    print(f"TOP {n} HISTORICAL MID-YEAR JUMPERS (R1-8 avg < threshold, R9+ avg >= threshold)")
    print(f"{'='*100}")
    pd.set_option("display.max_rows", n + 5)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    print(jumpers[cols].to_string(index=False))

    print(f"\n{'='*100}")
    print("POSITION BREAKDOWN")
    print(jumpers["pos"].value_counts().to_string())

    print(f"\n{'='*100}")
    print("YEAR BREAKDOWN")
    print(jumpers["Year"].value_counts().sort_index().to_string())


def print_feature_importance(importance_df: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE FOR JUMPER PREDICTION")
    print(f"{'='*80}")
    print(importance_df.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--early-threshold", type=float, default=80.0)
    parser.add_argument("--late-threshold", type=float, default=85.0)
    parser.add_argument("--min-early", type=int, default=5, help="Min games in R1-8")
    parser.add_argument("--min-late", type=int, default=5, help="Min games in R9+")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.years)
    career_avgs = compute_career_avgs(df)
    labels = label_jumpers(df, args.early_threshold, args.late_threshold, args.min_early, args.min_late)
    snapshot = compute_r8_snapshot(df, labels, career_avgs)

    if snapshot.empty:
        log.error("No qualifying player-seasons found. Check thresholds and data.")
        return

    importance_df = run_feature_importance(snapshot)

    # Write outputs
    jumpers_only = snapshot[snapshot["is_jumper"] == 1].sort_values("improvement", ascending=False)
    jumpers_path = OUTPUT_DIR / "historical_jumpers.csv"
    all_path = OUTPUT_DIR / "all_qualifying_seasons.csv"
    imp_path = OUTPUT_DIR / "feature_importance.csv"

    jumpers_only.to_csv(jumpers_path, index=False)
    snapshot.to_csv(all_path, index=False)
    importance_df.to_csv(imp_path, index=False)

    log.info("Wrote %d jumper rows → %s", len(jumpers_only), jumpers_path)
    log.info("Wrote %d total qualifying rows → %s", len(snapshot), all_path)
    log.info("Wrote feature importance → %s", imp_path)

    print_top_jumpers(snapshot, n=args.top_n)
    print_feature_importance(importance_df)

    # Summary stats for jumpers vs non-jumpers
    print(f"\n{'='*80}")
    print("JUMPER vs NON-JUMPER FEATURE MEANS")
    print(f"{'='*80}")
    compare_cols = [c for c in FEATURE_COLS if c in snapshot.columns]
    summary = snapshot.groupby("is_jumper")[compare_cols].mean().round(2)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
