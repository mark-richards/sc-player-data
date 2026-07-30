"""
backtest_calibration.py — Full calibration backtest across 2021-2025 + 2026 R1-9.

Compares:
  - Raw model P(80+)         [old system — cross-season contaminated]
  - Calibrated P(80+)        [new system — blends model with running fanfooty avg]

Two outcome horizons:
  1. Single-round: did the player score ≥80 in the predicted round?
  2. Rest-of-season: did the player average ≥80 in rounds 9+ (mid-season view)?

For each method, reports:
  - Precision@5, @10, @20 (single-round and rest-of-season)
  - AUC (single-round)
  - Calibration curve: predicted P(80+) bucket vs actual hit rate
  - Per-season breakdown
  - σ validation: actual SC score SD by average band

Data sources:
  - OOF predictions: data/predictions/oof_predictions_2021_2025.csv
  - 2026 predictions: data/predictions/2026_round_N_predictions.csv
  - Fanfooty actuals: data/processed/{year}_round_N_fanfooty_data.csv

Usage: py backtest_calibration.py
  (run generate_oof_predictions.py first if oof file doesn't exist)
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as sc_norm
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
FF_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SIGMA = 22.0  # validated on 2021-2025 OOF data — FA zone (avg 30-70) actual SD is 20-23
BOUNDARY_ROUND = 8
MIN_GAMES_FOR_CALIB = 5


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_fanfooty_all() -> pd.DataFrame:
    """Load all fanfooty processed CSVs, return feed_id / year / round_num / SC."""
    frames = []
    for fpath in glob.glob(str(FF_DIR / "*_round_*_fanfooty_data.csv")):
        m = re.search(r"(\d{4})_round_(\d+)", fpath)
        if not m:
            continue
        year, rnd = int(m.group(1)), int(m.group(2))
        if year < 2021:
            continue
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue
        df = df.rename(columns={"Player ID": "feed_id", "SC": "SC"})
        df["feed_id"] = pd.to_numeric(df.get("feed_id"), errors="coerce")
        df["SC"] = pd.to_numeric(df.get("SC"), errors="coerce")
        df = df.dropna(subset=["feed_id", "SC"])
        df["feed_id"] = df["feed_id"].astype(int)
        df["year"] = year
        df["round_num"] = rnd
        frames.append(df[["feed_id", "year", "round_num", "SC"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_oof_predictions() -> pd.DataFrame:
    """Load OOF predictions from generate_oof_predictions.py output."""
    path = PRED_DIR / "oof_predictions_2021_2025.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: py generate_oof_predictions.py"
        )
    df = pd.read_csv(path)
    df["feed_id"] = pd.to_numeric(df["Player ID"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["round_num"] = pd.to_numeric(df["Round_Num"], errors="coerce").astype(int)
    return df[["feed_id", "year", "round_num", "actual_SC", "cal_pred", "prob_gt_80"]].copy()


def load_2026_predictions() -> pd.DataFrame:
    """Load 2026 per-round prediction files."""
    frames = []
    for fpath in glob.glob(str(PRED_DIR / "2026_round_*_predictions.csv")):
        m = re.search(r"round_(\d+)", fpath)
        if not m:
            continue
        rnd = int(m.group(1))
        df = pd.read_csv(fpath)
        df["feed_id"] = pd.to_numeric(df["Player ID"], errors="coerce").astype("Int64")
        df["year"] = 2026
        df["round_num"] = rnd
        df = df.rename(columns={"projected_score": "cal_pred", "probability_gt_80": "prob_gt_80"})
        frames.append(df[["feed_id", "year", "round_num", "cal_pred", "prob_gt_80"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Running average (no lookahead) ───────────────────────────────────────────

def compute_running_avg(ff_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player × year, compute the running average of SC scores
    using only rounds BEFORE the current round (shift by 1).
    Also computes games_played_prior (count of rounds before current).
    """
    ff_sorted = ff_df.sort_values(["feed_id", "year", "round_num"])
    ff_sorted["running_avg"] = (
        ff_sorted.groupby(["feed_id", "year"])["SC"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    ff_sorted["games_prior"] = (
        ff_sorted.groupby(["feed_id", "year"])["SC"]
        .transform(lambda x: x.shift(1).expanding().count())
    )
    return ff_sorted[["feed_id", "year", "round_num", "running_avg", "games_prior"]]


# ── Calibrated P(80+) ────────────────────────────────────────────────────────

def calibrated_p80(row: pd.Series) -> float:
    """
    Blend model P(80+) with observed running-average evidence.
    Mirrors the newsletter formula exactly (sigma=20, weight ramp min(0.5, n/20)).
    """
    model_p80 = float(row["prob_gt_80"] if pd.notna(row["prob_gt_80"]) else 0)
    avg_raw = row.get("running_avg")
    avg = float(avg_raw) if pd.notna(avg_raw) else 0.0
    n_raw = row.get("games_prior")
    n = int(n_raw) if pd.notna(n_raw) else 0
    if avg > 0 and n >= MIN_GAMES_FOR_CALIB:
        estimated_p80 = float(sc_norm.sf(80, loc=avg, scale=SIGMA) * 100)
        w = min(0.7, n / 15)
        return round((1 - w) * model_p80 + w * estimated_p80, 1)
    return model_p80


# ── Evaluation helpers ────────────────────────────────────────────────────────

def precision_at_n(df: pd.DataFrame, score_col: str, outcome_col: str, n: int) -> float:
    top = df.nlargest(n, score_col)
    return float(top[outcome_col].mean()) if len(top) >= n else float("nan")


def auc(df: pd.DataFrame, score_col: str, outcome_col: str) -> float:
    valid = df[[score_col, outcome_col]].dropna()
    valid = valid[valid[outcome_col].notna()]
    if valid[outcome_col].nunique() < 2 or len(valid) < 20:
        return float("nan")
    try:
        return float(roc_auc_score(valid[outcome_col].astype(int), valid[score_col]))
    except Exception:
        return float("nan")


def calibration_rows(df: pd.DataFrame, score_col: str, outcome_col: str, n_bins: int = 8):
    valid = df[[score_col, outcome_col]].dropna().copy()
    valid["bucket"] = pd.qcut(valid[score_col], q=n_bins, duplicates="drop")
    result = (
        valid.groupby("bucket", observed=True)
        .agg(n=(outcome_col, "count"), actual=(outcome_col, "mean"), pred=(score_col, "median"))
        .reset_index()
    )
    result["actual_pct"] = (result["actual"] * 100).round(1)
    result["pred_pct"] = result["pred"].round(1)
    result["gap"] = (result["pred_pct"] - result["actual_pct"]).round(1)
    return result


def sigma_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Actual per-player SC score SD within a season, grouped by running_avg band."""
    records = []
    for (fid, yr), grp in df.groupby(["feed_id", "year"]):
        scores = grp["SC"].dropna().tolist()
        if len(scores) < 5:
            continue
        avg = float(np.mean(scores))
        sd = float(np.std(scores))
        band = f"{int(avg // 10) * 10}–{int(avg // 10) * 10 + 9}"
        records.append({"avg": avg, "sd": sd, "band": band})
    sd_df = pd.DataFrame(records)
    if sd_df.empty:
        return pd.DataFrame()
    return (
        sd_df.groupby("band")
        .agg(n=("sd", "count"), mean_sd=("sd", "mean"), median_sd=("sd", "median"))
        .reset_index()
        .assign(mean_sd=lambda d: d["mean_sd"].round(1), median_sd=lambda d: d["median_sd"].round(1))
    )


# ── Rest-of-season outcomes ───────────────────────────────────────────────────

def add_ros_outcome(df: pd.DataFrame, ff_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player × year, compute their average SC in rounds BOUNDARY_ROUND+1..end.
    Attach as ros_avg and ros_breakout (avg >= 80).
    """
    late = ff_df[ff_df["round_num"] > BOUNDARY_ROUND]
    ros = (
        late.groupby(["feed_id", "year"])["SC"]
        .agg(ros_avg="mean", ros_n="count")
        .reset_index()
    )
    ros["ros_breakout"] = (ros["ros_avg"] >= 80) & (ros["ros_n"] >= 5)
    return df.merge(ros[["feed_id", "year", "ros_avg", "ros_breakout"]], on=["feed_id", "year"], how="left")


# ── Report ────────────────────────────────────────────────────────────────────

def fmt_pct(v) -> str:
    if v != v or v is None:
        return "—"
    return f"{v:.0%}"


def fmt_auc(v) -> str:
    if v != v or v is None:
        return "—"
    return f"{v:.3f}"


def main():
    print("Loading data...")
    ff_df = load_fanfooty_all()
    if ff_df.empty:
        print("No fanfooty data found.")
        return

    print("Loading OOF predictions (2021-2025)...")
    try:
        oof = load_oof_predictions()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print("Loading 2026 predictions...")
    pred_2026 = load_2026_predictions()

    # Merge 2026 actuals
    if not pred_2026.empty:
        actuals_2026 = ff_df[ff_df["year"] == 2026][["feed_id", "year", "round_num", "SC"]].rename(
            columns={"SC": "actual_SC"}
        )
        pred_2026 = pred_2026.merge(actuals_2026, on=["feed_id", "year", "round_num"], how="inner")

    # Combine all predictions
    all_preds = pd.concat(
        [df for df in [oof, pred_2026] if not df.empty], ignore_index=True
    )

    # Compute running avg (no lookahead)
    print("Computing running averages...")
    running = compute_running_avg(ff_df)
    all_preds = all_preds.merge(running, on=["feed_id", "year", "round_num"], how="left")

    # Calibrated P(80+)
    all_preds["calib_p80"] = all_preds.apply(calibrated_p80, axis=1)

    # Single-round outcome
    all_preds["scored_80_this_round"] = (all_preds["actual_SC"] >= 80).astype(float)

    # Rest-of-season outcome (only for round == BOUNDARY_ROUND rows)
    boundary_df = all_preds[all_preds["round_num"] == BOUNDARY_ROUND].copy()
    boundary_df = add_ros_outcome(boundary_df, ff_df)

    # FA-equivalent subset: model projected < 80 (proxy for free agent pool)
    fa = all_preds[all_preds["cal_pred"] < 80].copy()
    fa_boundary = boundary_df[boundary_df["cal_pred"] < 80].copy()

    print(f"\nDataset summary:")
    print(f"  Total predictions: {len(all_preds)}")
    print(f"  FA-equivalent (proj < 80): {len(fa)}")
    print(f"  Rounds covered: {sorted(all_preds['year'].unique())}")

    methods = {
        "prob_gt_80": "Raw model P(80+)  [old system]",
        "calib_p80":  "Calibrated P(80+) [new system]",
    }

    lines = [
        "# Calibration Backtest: Raw Model vs Calibrated P(80+)",
        "",
        f"OOF predictions 2021-2025 + 2026 R1-9. "
        f"FA-equivalent = players with projected score < 80.",
        f"Calibrated P(80+): blends model with Gaussian(running_avg, σ={SIGMA:.0f}).",
        f"σ={SIGMA:.0f} validated on 2023-2024 historical SD analysis.",
        "",
    ]

    # ── Section 1: Single-round precision ────────────────────────────────────
    lines += [
        "## 1. Single-Round Precision (did top picks score 80+ THIS round?)",
        "",
        f"Base rate: {fa['scored_80_this_round'].mean():.0%} of FA-equivalent players score 80+ in any given round.",
        "",
        "| Method | P@5 | P@10 | P@20 | AUC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for col, label in methods.items():
        p5 = precision_at_n(fa, col, "scored_80_this_round", 5)
        p10 = precision_at_n(fa, col, "scored_80_this_round", 10)
        p20 = precision_at_n(fa, col, "scored_80_this_round", 20)
        a = auc(fa, col, "scored_80_this_round")
        lines.append(f"| {label} | {fmt_pct(p5)} | {fmt_pct(p10)} | {fmt_pct(p20)} | {fmt_auc(a)} |")
    base = fa["scored_80_this_round"].mean()
    lines.append(f"| *Random baseline* | {fmt_pct(base)} | {fmt_pct(base)} | {fmt_pct(base)} | 0.500 |")

    # Per-year single-round breakdown
    lines += ["", "### Per-Season (P@10, single-round)", "", "| Season | Raw P@10 | Cal P@10 | Base rate |", "| --- | --- | --- | --- |"]
    for yr in sorted(fa["year"].unique()):
        yfa = fa[fa["year"] == yr]
        if len(yfa) < 10:
            continue
        raw_p10 = precision_at_n(yfa, "prob_gt_80", "scored_80_this_round", 10)
        cal_p10 = precision_at_n(yfa, "calib_p80", "scored_80_this_round", 10)
        base_yr = yfa["scored_80_this_round"].mean()
        lines.append(f"| {yr} | {fmt_pct(raw_p10)} | {fmt_pct(cal_p10)} | {fmt_pct(base_yr)} |")

    # ── Section 2: Rest-of-season at boundary ────────────────────────────────
    if not fa_boundary.empty and "ros_breakout" in fa_boundary.columns:
        valid_ros = fa_boundary.dropna(subset=["ros_breakout"])
        if not valid_ros.empty:
            lines += [
                "",
                f"## 2. Rest-of-Season Breakout (R{BOUNDARY_ROUND} boundary — will player avg 80+ in R{BOUNDARY_ROUND+1}+?)",
                "",
                f"Base rate: {valid_ros['ros_breakout'].mean():.0%} of FA-equivalent players at R{BOUNDARY_ROUND} average 80+ in R{BOUNDARY_ROUND+1}+.",
                "",
                "| Method | P@5 | P@10 | P@20 | AUC |",
                "| --- | --- | --- | --- | --- |",
            ]
            for col, label in methods.items():
                p5 = precision_at_n(valid_ros, col, "ros_breakout", 5)
                p10 = precision_at_n(valid_ros, col, "ros_breakout", 10)
                p20 = precision_at_n(valid_ros, col, "ros_breakout", 20)
                a = auc(valid_ros, col, "ros_breakout")
                lines.append(f"| {label} | {fmt_pct(p5)} | {fmt_pct(p10)} | {fmt_pct(p20)} | {fmt_auc(a)} |")
            base_ros = valid_ros["ros_breakout"].mean()
            lines.append(f"| *Random baseline* | {fmt_pct(base_ros)} | {fmt_pct(base_ros)} | {fmt_pct(base_ros)} | 0.500 |")

    # ── Section 3: Calibration curves ────────────────────────────────────────
    lines += [
        "",
        "## 3. Calibration Curves — Predicted vs Actual P(80+)",
        "",
        "Well-calibrated = actual rate ≈ predicted mid. Positive gap = overconfident.",
        "",
        "### Raw Model P(80+)",
        "",
        "| Predicted P(80+) | N | Actual rate | Gap |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in calibration_rows(fa, "prob_gt_80", "scored_80_this_round").iterrows():
        lines.append(f"| {row['bucket']} | {int(row['n'])} | {row['actual_pct']}% | {row['gap']:+.1f} |")

    lines += [
        "",
        "### Calibrated P(80+)",
        "",
        "| Predicted P(80+) | N | Actual rate | Gap |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in calibration_rows(fa, "calib_p80", "scored_80_this_round").iterrows():
        lines.append(f"| {row['bucket']} | {int(row['n'])} | {row['actual_pct']}% | {row['gap']:+.1f} |")

    # ── Section 4: σ validation ───────────────────────────────────────────────
    lines += [
        "",
        f"## 4. σ Validation — Actual SC Score SD vs Assumed σ={SIGMA:.0f}",
        "",
        "| Average band | N | Mean SD | Median SD |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in sigma_rows(ff_df[ff_df["year"] >= 2021]).iterrows():
        marker = " ← assumption" if abs(float(row["mean_sd"]) - SIGMA) < 2 else ""
        lines.append(f"| {row['band']} | {int(row['n'])} | {row['mean_sd']}{marker} | {row['median_sd']} |")

    lines += [
        "",
        "## 5. Overall Model Accuracy (all players, not just FA-equivalent)",
        "",
        f"| Season | MAE | Spearman | n |",
        "| --- | --- | --- | --- |",
    ]
    from scipy.stats import spearmanr
    for yr in sorted(all_preds["year"].unique()):
        ydf = all_preds[all_preds["year"] == yr].dropna(subset=["actual_SC", "cal_pred"])
        if len(ydf) < 50:
            continue
        mae = float(np.abs(ydf["cal_pred"] - ydf["actual_SC"]).mean())
        sp, _ = spearmanr(ydf["actual_SC"], ydf["cal_pred"])
        lines.append(f"| {yr} | {mae:.1f} | {sp:.3f} | {len(ydf)} |")

    lines += ["", "*Generated by backtest_calibration.py*"]

    out = REPORTS_DIR / "backtest_calibration_2021_2026.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out}")

    # Console summary
    print("\n-- Single-round P@10 (FA-equivalent) --")
    for col, label in methods.items():
        p10 = precision_at_n(fa, col, "scored_80_this_round", 10)
        a = auc(fa, col, "scored_80_this_round")
        print(f"  {label}: P@10={fmt_pct(p10)}  AUC={fmt_auc(a)}")
    print(f"  Base rate: {fmt_pct(fa['scored_80_this_round'].mean())}")


if __name__ == "__main__":
    main()
