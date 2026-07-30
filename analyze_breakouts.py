"""
analyze_breakouts.py — Historical validation of mid-season breakout signals.

For each season (2023, 2024, 2025), identifies players who:
  - Averaged < 80 SC in rounds 1–8 (free agent / sub-replacement territory)
  - Averaged >= 80 SC in rounds 9+ (broke out for the rest of the season)

Then measures their trajectory signals at the round-8 boundary:
  - CBA trend (last 2 rounds vs full-season avg) — gaining midfield time?
  - SC_ema_trend (short EWM - long EWM) — scoring momentum?
  - TOG trend (last 2 vs full avg) — more game time?
  - SC_form_delta_3v6 — last-3 avg better than last-6?

Compares breakout players vs non-breakout players on each signal.
Outputs: reports/breakout_analysis_2023_2025.md

Usage: py analyze_breakouts.py
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FANFOOTY_DIR = PROJECT_ROOT / "data" / "processed"
DFS_DIR = PROJECT_ROOT / "data" / "raw" / "dfsaustralia"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

BREAKOUT_SEASONS = [2023, 2024, 2025]
BOUNDARY_ROUND = 8
MIN_EARLY_GAMES = 5   # need this many games in rounds 1-8 to qualify
MIN_LATE_GAMES = 5    # need this many games in rounds 9+ to evaluate
EARLY_AVG_MAX = 80.0  # must be below this in rounds 1-8
BREAKOUT_AVG_MIN = 80.0  # must reach this in rounds 9+


def _slope(arr: list) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def _ewm_trend(arr: list) -> float:
    if len(arr) < 3:
        return 0.0
    s = pd.Series(arr, dtype=float)
    return float(s.ewm(span=3, adjust=False).mean().iloc[-1] - s.ewm(span=6, adjust=False).mean().iloc[-1])


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
        df["tog"] = pd.to_numeric(df.get("tog"), errors="coerce")
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
    try:
        df = pd.read_csv(path)
        df["feed_id"] = pd.to_numeric(df["feed_id"], errors="coerce").astype("Int64")
        df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce").astype("Int64")
        df["cba_pct"] = pd.to_numeric(df["cba_pct"], errors="coerce")
        return df.dropna(subset=["feed_id", "cba_pct"])
    except Exception:
        return pd.DataFrame(columns=["feed_id", "round_num", "cba_pct"])


def compute_signals_at_boundary(
    ff_df: pd.DataFrame,
    cba_df: pd.DataFrame,
    feed_id: int,
    boundary: int = BOUNDARY_ROUND,
) -> dict:
    """Compute trajectory signals for a player at the round-8 boundary."""
    player_ff = ff_df[ff_df["feed_id"] == feed_id].sort_values("round_num")
    early = player_ff[player_ff["round_num"] <= boundary]

    sc_scores = early["SC"].tolist()
    togs = early["tog"].dropna().tolist()

    avg3 = float(np.mean(sc_scores[-3:])) if len(sc_scores) >= 3 else (float(np.mean(sc_scores)) if sc_scores else 0)
    avg6 = float(np.mean(sc_scores[-6:])) if len(sc_scores) >= 6 else (float(np.mean(sc_scores)) if sc_scores else 0)
    tog_avg = float(np.mean(togs)) if togs else None
    tog_last2 = float(np.mean(togs[-2:])) if len(togs) >= 2 else tog_avg

    # CBA signals
    player_cba = cba_df[
        (cba_df["feed_id"] == feed_id) & (cba_df["round_num"] <= boundary)
    ].sort_values("round_num") if not cba_df.empty else pd.DataFrame()
    cbas = player_cba["cba_pct"].dropna().tolist() if not player_cba.empty else []
    cba_avg = float(np.mean(cbas)) if cbas else None
    cba_last2 = float(np.mean(cbas[-2:])) if len(cbas) >= 2 else cba_avg

    return {
        "feed_id": feed_id,
        "SC_ema_trend": _ewm_trend(sc_scores),
        "SC_form_delta_3v6": avg3 - avg6,
        "SC_trend_slope_8": _slope(sc_scores),
        "tog_trend": (tog_last2 - tog_avg) if (tog_avg is not None and tog_last2 is not None) else None,
        "cba_trend": (cba_last2 - cba_avg) if (cba_avg is not None and cba_last2 is not None) else None,
        "cba_avg": cba_avg,
        "early_avg": float(np.mean(sc_scores)) if sc_scores else 0,
        "n_early": len(sc_scores),
    }


def analyse_season(year: int) -> pd.DataFrame:
    ff_df = load_fanfooty_season(year)
    cba_df = load_cba_season(year)
    if ff_df.empty:
        print(f"  No fanfooty data for {year}", file=sys.stderr)
        return pd.DataFrame()

    early_df = ff_df[ff_df["round_num"] <= BOUNDARY_ROUND]
    late_df = ff_df[ff_df["round_num"] > BOUNDARY_ROUND]

    early_stats = early_df.groupby("feed_id")["SC"].agg(["mean", "count"]).reset_index()
    early_stats.columns = ["feed_id", "early_avg", "early_n"]

    late_stats = late_df.groupby("feed_id")["SC"].agg(["mean", "count"]).reset_index()
    late_stats.columns = ["feed_id", "late_avg", "late_n"]

    combined = early_stats.merge(late_stats, on="feed_id", how="inner")
    candidates = combined[
        (combined["early_avg"] < EARLY_AVG_MAX)
        & (combined["early_n"] >= MIN_EARLY_GAMES)
        & (combined["late_n"] >= MIN_LATE_GAMES)
    ].copy()

    candidates["breakout"] = candidates["late_avg"] >= BREAKOUT_AVG_MIN
    candidates["year"] = year

    signals = [
        compute_signals_at_boundary(ff_df, cba_df, int(fid))
        for fid in candidates["feed_id"]
    ]
    signals_df = pd.DataFrame(signals).set_index("feed_id")
    candidates = candidates.set_index("feed_id").join(signals_df.drop(columns=["early_avg", "n_early"]), how="left")
    candidates = candidates.reset_index()
    return candidates


def summarise_signal(df: pd.DataFrame, col: str) -> dict:
    """Compare signal values between breakout vs non-breakout players."""
    data = df[col].dropna()
    if data.empty:
        return {"col": col, "n_breakout": 0, "n_non": 0, "mean_breakout": None, "mean_non": None, "diff": None}
    breakout = df.loc[df["breakout"] & df[col].notna(), col]
    non = df.loc[~df["breakout"] & df[col].notna(), col]
    mean_b = float(breakout.mean()) if len(breakout) > 0 else None
    mean_n = float(non.mean()) if len(non) > 0 else None
    diff = (mean_b - mean_n) if (mean_b is not None and mean_n is not None) else None
    return {
        "col": col,
        "n_breakout": len(breakout),
        "n_non": len(non),
        "mean_breakout": mean_b,
        "mean_non": mean_n,
        "diff": diff,
    }


def main():
    print("Analysing mid-season breakouts (2023–2025)...")
    all_dfs = []
    for year in BREAKOUT_SEASONS:
        print(f"  Processing {year}...", end=" ")
        df = analyse_season(year)
        if not df.empty:
            n_bo = df["breakout"].sum()
            print(f"{len(df)} candidates, {n_bo} breakouts")
            all_dfs.append(df)
        else:
            print("skipped (no data)")

    if not all_dfs:
        print("No data found. Exiting.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    signals = ["SC_ema_trend", "SC_form_delta_3v6", "SC_trend_slope_8", "tog_trend", "cba_trend", "cba_avg"]
    summaries = [summarise_signal(combined, s) for s in signals]

    # Build report
    lines = [
        "# Breakout Analysis: Mid-Season Signals (2023–2025)",
        "",
        f"Analysis of players averaging <80 in rounds 1–{BOUNDARY_ROUND}, then ≥80 in rounds 9+.",
        f"Seasons: {', '.join(str(y) for y in BREAKOUT_SEASONS)}",
        "",
        f"**Total candidates**: {len(combined)} | **Breakouts**: {combined['breakout'].sum()} "
        f"({100*combined['breakout'].mean():.0f}%)",
        "",
        "## Signal Comparison: Breakout vs Non-Breakout at Round 8 Boundary",
        "",
        "| Signal | Breakout avg | Non-breakout avg | Difference | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]

    ranked = sorted(
        [s for s in summaries if s["diff"] is not None],
        key=lambda x: abs(x["diff"]),
        reverse=True,
    )
    for s in ranked:
        diff_str = f"+{s['diff']:.2f}" if s["diff"] > 0 else f"{s['diff']:.2f}"
        verdict = "✅ Strong" if abs(s["diff"]) > 5 else ("⚠️ Moderate" if abs(s["diff"]) > 2 else "❌ Weak")
        bo_str = f"{s['mean_breakout']:.2f}" if s["mean_breakout"] is not None else "—"
        non_str = f"{s['mean_non']:.2f}" if s["mean_non"] is not None else "—"
        lines.append(f"| {s['col']} | {bo_str} | {non_str} | {diff_str} | {verdict} |")
    for s in summaries:
        if s["diff"] is None:
            lines.append(f"| {s['col']} | — | — | — | ❌ No data |")

    lines += [
        "",
        "## Breakout Players by Season",
        "",
    ]
    for year in BREAKOUT_SEASONS:
        year_df = combined[(combined["year"] == year) & combined["breakout"]].sort_values("late_avg", ascending=False)
        if year_df.empty:
            lines.append(f"### {year}: none")
            continue
        lines.append(f"### {year} ({len(year_df)} breakouts)")
        lines.append("")
        lines.append("| feed_id | Early avg | Late avg | SC_ema | CBA trend | TOG trend |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, r in year_df.iterrows():
            ema = f"{r['SC_ema_trend']:.1f}" if pd.notna(r.get("SC_ema_trend")) else "—"
            cba = f"+{r['cba_trend']:.1f}" if pd.notna(r.get("cba_trend")) and r["cba_trend"] > 0 else (
                f"{r['cba_trend']:.1f}" if pd.notna(r.get("cba_trend")) else "—")
            tog = f"+{r['tog_trend']:.1f}%" if pd.notna(r.get("tog_trend")) and r["tog_trend"] > 0 else (
                f"{r['tog_trend']:.1f}%" if pd.notna(r.get("tog_trend")) else "—")
            lines.append(f"| {int(r['feed_id'])} | {r['early_avg']:.1f} | {r['late_avg']:.1f} | {ema} | {cba} | {tog} |")
        lines.append("")

    lines += [
        "## Interpretation for Jumper Score Weights",
        "",
        "Signals with the largest positive difference (breakout > non-breakout) are the",
        "most predictive features. These should have the highest weights in `_JUMPER_WEIGHTS`.",
        "",
        "Current weights after redesign:",
        "- cba_trend_mid: 0.25 (CBA gaining midfield access)",
        "- SC_ema_trend: 0.20 (scoring momentum)",
        "- tog_trend: 0.15 (getting more game time)",
        "- SC_form_delta_3v6: 0.15 (recent acceleration)",
        "- career_avg_delta: 0.10 (ceiling gate only)",
        "",
        f"*Generated by analyze_breakouts.py*",
    ]

    output_path = REPORTS_DIR / "breakout_analysis_2023_2025.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {output_path}")

    # Print key findings
    print("\nKey signal differences (breakout vs non-breakout):")
    for s in ranked[:5]:
        if s["diff"] is not None:
            print(f"  {s['col']:25s}: {s['diff']:+.2f} (breakout={s['mean_breakout']:.2f}, non={s['mean_non']:.2f})")


if __name__ == "__main__":
    main()
