"""
backtest_newsletter.py — Calibration backtest for newsletter ranking system.

For each season (2023, 2024), at the round-8 boundary, identifies "free agent
equivalent" players (R1-8 avg < 80, ≥5 games) and compares how well different
ranking methods predict who will average 80+ in R9+.

Ranking methods compared:
  simple_avg        — R1-8 average (proxy for bayesian-dominated old system)
  calib_p80_gauss   — Gaussian(R1-8 avg, σ=25) [new system Path 2]
  calib_p80_obs     — Actual R1-8 hit rate (games ≥80 / n) [new system Path 1]
  cba_enhanced      — Weighted combo: 0.6×calib_p80_gauss + 0.4×cba_avg_norm

Outputs:
  1. Precision@N table per method
  2. Calibration curve: predicted P(80+) buckets vs actual hit rate
  3. σ validation: actual SC score SD by average band
  4. AUC comparison

Usage: py backtest_newsletter.py
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as sc_norm

PROJECT_ROOT = Path(__file__).resolve().parent
FANFOOTY_DIR = PROJECT_ROOT / "data" / "processed"
DFS_DIR = PROJECT_ROOT / "data" / "raw" / "dfsaustralia"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

BACKTEST_SEASONS = [2023, 2024]
BOUNDARY_ROUND = 8
MIN_EARLY_GAMES = 5
MIN_LATE_GAMES = 5
EARLY_AVG_MAX = 80.0
BREAKOUT_AVG_MIN = 80.0
SIGMA = 20.0  # SC score standard deviation for Gaussian P(80+) — validated on 2023-2024 data


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_fanfooty_season(year: int) -> pd.DataFrame:
    pattern = str(FANFOOTY_DIR / f"{year}_round_*_fanfooty_data.csv")
    files = sorted(glob.glob(pattern))
    frames = []
    for fpath in files:
        m = re.search(r"round_(\d+)", fpath)
        if not m:
            continue
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue
        rename = {"Player ID": "feed_id", "SC": "SC", "Time on ground": "tog"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        keep = [c for c in ["feed_id", "SC", "tog"] if c in df.columns]
        df = df[keep].copy()
        df["round_num"] = int(m.group(1))
        df["year"] = year
        df["feed_id"] = pd.to_numeric(df.get("feed_id"), errors="coerce")
        df["SC"] = pd.to_numeric(df.get("SC"), errors="coerce")
        df = df.dropna(subset=["feed_id", "SC"])
        df["feed_id"] = df["feed_id"].astype(int)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_cba_season(year: int) -> pd.DataFrame:
    path = DFS_DIR / str(year) / f"cbas_{year}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feed_id", "round_num", "cba_pct"])
    df = pd.read_csv(path)
    df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce").astype("Int64")
    df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce").astype("Int64")
    df["cba_pct"] = pd.to_numeric(df["cba_pct"], errors="coerce")
    return df.dropna(subset=["feed_id", "cba_pct"])


# ── Signal computation ────────────────────────────────────────────────────────

def compute_player_features(ff_df: pd.DataFrame, cba_df: pd.DataFrame, feed_id: int) -> dict:
    """Compute all ranking signals at the round-8 boundary."""
    player = ff_df[ff_df["feed_id"] == feed_id].sort_values("round_num")
    early = player[player["round_num"] <= BOUNDARY_ROUND]
    sc_scores = early["SC"].tolist()
    n = len(sc_scores)

    early_avg = float(np.mean(sc_scores)) if sc_scores else 0.0
    obs_p80 = sum(1 for s in sc_scores if s >= 80) / n * 100 if n > 0 else 0.0
    gauss_p80 = float(sc_norm.sf(80, loc=early_avg, scale=SIGMA) * 100)

    # CBA signals
    player_cba = (
        cba_df[(cba_df["feed_id"] == feed_id) & (cba_df["round_num"] <= BOUNDARY_ROUND)]
        .sort_values("round_num")
        if not cba_df.empty else pd.DataFrame()
    )
    cbas = player_cba["cba_pct"].dropna().tolist() if not player_cba.empty else []
    cba_avg = float(np.mean(cbas)) if cbas else None
    cba_last2 = float(np.mean(cbas[-2:])) if len(cbas) >= 2 else cba_avg
    cba_trend = (cba_last2 - cba_avg) if (cba_avg is not None and cba_last2 is not None) else None

    return {
        "feed_id": feed_id,
        "early_avg": early_avg,
        "n_early": n,
        "obs_p80": obs_p80,
        "gauss_p80": gauss_p80,
        "cba_avg": cba_avg,
        "cba_trend": cba_trend,
        "sc_scores_early": sc_scores,
    }


def build_season_dataset(year: int) -> pd.DataFrame:
    ff_df = load_fanfooty_season(year)
    cba_df = load_cba_season(year)
    if ff_df.empty:
        print(f"  No data for {year}", file=sys.stderr)
        return pd.DataFrame()

    early_stats = (
        ff_df[ff_df["round_num"] <= BOUNDARY_ROUND]
        .groupby("feed_id")["SC"]
        .agg(early_avg="mean", early_n="count")
        .reset_index()
    )
    late_stats = (
        ff_df[ff_df["round_num"] > BOUNDARY_ROUND]
        .groupby("feed_id")["SC"]
        .agg(late_avg="mean", late_n="count", late_p80=lambda x: (x >= 80).mean() * 100)
        .reset_index()
    )

    combined = early_stats.merge(late_stats, on="feed_id", how="inner")
    candidates = combined[
        (combined["early_avg"] < EARLY_AVG_MAX)
        & (combined["early_n"] >= MIN_EARLY_GAMES)
        & (combined["late_n"] >= MIN_LATE_GAMES)
    ].copy()
    candidates["breakout"] = candidates["late_avg"] >= BREAKOUT_AVG_MIN
    candidates["year"] = year

    features = [compute_player_features(ff_df, cba_df, int(fid)) for fid in candidates["feed_id"]]
    feat_df = pd.DataFrame(features)
    candidates = candidates.merge(feat_df.drop(columns=["early_avg", "n_early"]), on="feed_id", how="left")

    # CBA-enhanced ranking: blend P(80+) with normalised CBA level
    cba_vals = candidates["cba_avg"].dropna()
    cba_max = cba_vals.max() if not cba_vals.empty else 1.0
    candidates["cba_norm"] = (candidates["cba_avg"].fillna(0) / max(cba_max, 1)) * 100
    candidates["cba_enhanced"] = (
        0.6 * candidates["gauss_p80"] + 0.4 * candidates["cba_norm"]
    )

    return candidates


# ── Evaluation ────────────────────────────────────────────────────────────────

def precision_at_n(df: pd.DataFrame, rank_col: str, n: int) -> float:
    top = df.nlargest(n, rank_col)
    return float(top["breakout"].mean()) if len(top) > 0 else 0.0


def auc_roc(df: pd.DataFrame, score_col: str) -> float:
    from sklearn.metrics import roc_auc_score
    valid = df[[score_col, "breakout"]].dropna()
    if valid["breakout"].nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(valid["breakout"].astype(int), valid[score_col]))
    except Exception:
        return float("nan")


def calibration_table(df: pd.DataFrame, score_col: str, n_bins: int = 5) -> pd.DataFrame:
    """Bucket players by predicted P(80+), measure actual hit rate in each bucket."""
    valid = df[[score_col, "breakout"]].dropna().copy()
    valid["bucket"] = pd.qcut(valid[score_col], q=n_bins, duplicates="drop")
    result = (
        valid.groupby("bucket", observed=True)
        .agg(
            n=("breakout", "count"),
            actual_rate=("breakout", "mean"),
            pred_mid=(score_col, "median"),
        )
        .reset_index()
    )
    result["actual_rate_pct"] = (result["actual_rate"] * 100).round(1)
    result["pred_mid"] = result["pred_mid"].round(1)
    return result


def sigma_validation(all_df: pd.DataFrame) -> pd.DataFrame:
    """Check actual per-player SC score SD vs σ=25 assumption, by average band."""
    records = []
    for _, row in all_df.iterrows():
        scores = row.get("sc_scores_early") or []
        if len(scores) < 3:
            continue
        avg = float(np.mean(scores))
        sd = float(np.std(scores))
        band = f"{int(avg // 10) * 10}–{int(avg // 10) * 10 + 9}"
        records.append({"avg": avg, "sd": sd, "band": band})
    sd_df = pd.DataFrame(records)
    if sd_df.empty:
        return pd.DataFrame()
    summary = (
        sd_df.groupby("band")
        .agg(n=("sd", "count"), mean_sd=("sd", "mean"), median_sd=("sd", "median"))
        .reset_index()
    )
    summary["mean_sd"] = summary["mean_sd"].round(1)
    summary["median_sd"] = summary["median_sd"].round(1)
    return summary


# ── Report builder ────────────────────────────────────────────────────────────

def main():
    print("Building backtest dataset...")
    all_dfs = []
    for year in BACKTEST_SEASONS:
        print(f"  Processing {year}...", end=" ")
        df = build_season_dataset(year)
        if not df.empty:
            n_bo = int(df["breakout"].sum())
            print(f"{len(df)} candidates, {n_bo} breakouts ({100*n_bo/len(df):.0f}%)")
            all_dfs.append(df)
        else:
            print("skipped")

    if not all_dfs:
        print("No data. Exiting.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    methods = {
        "simple_avg": "R1-8 average (old bayesian proxy)",
        "obs_p80": "Observed P(80+) in R1-8 [new Path 1]",
        "gauss_p80": "Gaussian P(80+) from avg [new Path 2]",
        "cba_enhanced": "CBA-enhanced Gaussian P(80+) [new Path 2 + CBA]",
    }

    lines = [
        "# Newsletter Ranking Backtest (2023–2024)",
        "",
        f"Free-agent equivalent players: R1-{BOUNDARY_ROUND} avg < 80, ≥{MIN_EARLY_GAMES} games.",
        f"Outcome: averaged ≥80 in rounds {BOUNDARY_ROUND+1}+ (≥{MIN_LATE_GAMES} games).",
        f"Seasons: {', '.join(str(y) for y in BACKTEST_SEASONS)}",
        f"**Total candidates**: {len(combined)} | **Breakouts**: {int(combined['breakout'].sum())} "
        f"({100*combined['breakout'].mean():.0f}%)",
        "",
        "## 1. Precision@N by Ranking Method",
        "",
        "Fraction of top-N picks who actually averaged 80+ in R9+.",
        "",
        "| Method | P@5 | P@10 | P@20 | AUC |",
        "| --- | --- | --- | --- | --- |",
    ]

    for col, label in methods.items():
        if col not in combined.columns:
            continue
        p5 = precision_at_n(combined, col, 5)
        p10 = precision_at_n(combined, col, 10)
        p20 = precision_at_n(combined, col, 20)
        auc = auc_roc(combined, col)
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "—"
        lines.append(f"| {label} | {p5:.0%} | {p10:.0%} | {p20:.0%} | {auc_str} |")

    # Baseline: random
    base = float(combined["breakout"].mean())
    lines.append(f"| *Random baseline* | {base:.0%} | {base:.0%} | {base:.0%} | 0.500 |")

    lines += [
        "",
        "## 2. Calibration Curve — Gaussian P(80+)",
        "",
        "Does the predicted P(80+) match actual breakout rates? "
        "Well-calibrated = actual rate ≈ predicted mid.",
        "",
        "| Predicted P(80+) bucket | N players | Actual breakout rate |",
        "| --- | --- | --- |",
    ]
    calib = calibration_table(combined, "gauss_p80", n_bins=5)
    for _, row in calib.iterrows():
        lines.append(f"| {row['bucket']} | {int(row['n'])} | {row['actual_rate_pct']}% |")

    lines += [
        "",
        "## 3. σ Validation — Is σ=25 a Good Assumption?",
        "",
        f"Actual per-player SC score standard deviation (R1-{BOUNDARY_ROUND} games) "
        f"vs assumed σ={SIGMA:.0f}.",
        "",
        "| Average band | N players | Mean SD | Median SD |",
        "| --- | --- | --- | --- |",
    ]
    sigma_df = sigma_validation(combined)
    for _, row in sigma_df.iterrows():
        lines.append(f"| {row['band']} | {int(row['n'])} | {row['mean_sd']} | {row['median_sd']} |")

    lines += [
        "",
        "## 4. CBA Coverage",
        "",
    ]
    has_cba = combined["cba_avg"].notna().sum()
    lines.append(f"Players with CBA data at R{BOUNDARY_ROUND}: **{has_cba}** / {len(combined)} "
                 f"({100*has_cba/len(combined):.0f}%)")
    if has_cba > 0:
        bo_cba = combined[combined["breakout"] & combined["cba_avg"].notna()]
        non_cba = combined[~combined["breakout"] & combined["cba_avg"].notna()]
        lines += [
            "",
            f"Mean CBA% — breakouts: **{bo_cba['cba_avg'].mean():.1f}%** | "
            f"non-breakouts: {non_cba['cba_avg'].mean():.1f}%",
            f"Mean CBA trend — breakouts: **{bo_cba['cba_trend'].mean():.1f}** | "
            f"non-breakouts: {non_cba['cba_trend'].mean():.1f}",
        ]

    lines += [
        "",
        "## 5. Per-Season Breakdown",
        "",
        "| Season | Candidates | Breakouts | Gauss P@10 | CBA-Enhanced P@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for year in BACKTEST_SEASONS:
        ydf = combined[combined["year"] == year]
        if ydf.empty:
            continue
        g_p10 = precision_at_n(ydf, "gauss_p80", 10)
        c_p10 = precision_at_n(ydf, "cba_enhanced", 10)
        lines.append(
            f"| {year} | {len(ydf)} | {int(ydf['breakout'].sum())} | {g_p10:.0%} | {c_p10:.0%} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **AUC > 0.6**: ranking method adds meaningful signal over random",
        "- **Calibration**: if predicted 30% bucket has 30% actual breakout rate, Gaussian σ=25 is correct",
        "- **σ validation**: if actual SD is consistently higher/lower than 25, adjust σ in newsletter.py",
        "- **CBA enhancement**: if CBA-enhanced P@N > Gaussian P@N, CBA data is worth including in ranking",
        "",
        "*Generated by backtest_newsletter.py*",
    ]

    output_path = REPORTS_DIR / "backtest_newsletter_2023_2024.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {output_path}")

    # Print key findings to console
    print("\nKey findings:")
    print(f"  Breakout base rate: {100*combined['breakout'].mean():.1f}%")
    for col, label in methods.items():
        if col in combined.columns:
            p10 = precision_at_n(combined, col, 10)
            auc = auc_roc(combined, col)
            auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
            print(f"  {col:20s}: P@10={p10:.0%}  AUC={auc_str}")
    if not sigma_df.empty:
        print(f"\n  Sigma validation (assumed sigma=25):")
        for _, row in sigma_df.iterrows():
            print(f"    avg {row['band']:6s}: mean SD={row['mean_sd']}, median SD={row['median_sd']}")


if __name__ == "__main__":
    main()
