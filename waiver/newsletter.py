"""
newsletter.py — Generates the weekly Waiver Wire Markdown newsletter.

Sections
--------
1. Positional Targets   — Top 6 free agents per position (DEF / MID / FWD)
2. The Ceiling Chasers  — Top 5 free agents by Ceiling Index (Flex-Bonus tiebreak)
3. The Ruck Radar       — Best available Rucks by TPOR
4. Injury & Suspension  — My roster players with active injuries/suspensions
5. The Drop Zone        — My 3 most droppable on-field starters
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import norm as _sc_norm

from waiver import llm_narrator
from waiver.config import BYE_ROUNDS, CEILING_THRESHOLD, REPORTS_DIR, SEASON_YEAR, SLOTS
from waiver.dvp import _LONG_TO_ABBR

TRANSACTIONS_PATH = Path(__file__).parent.parent / "data" / "live" / "transactions.csv"
_DFS_DIR = Path(__file__).parent.parent / "data" / "raw" / "dfsaustralia"

log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 1, suffix: str = "") -> str:
    """Format a numeric value, falling back gracefully."""
    try:
        f = float(val)
        if f != f:  # NaN check (NaN != NaN)
            return "—"
        return f"{f:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _pct(val) -> str:
    return _fmt(val, 1, "%")


def _opp(val) -> str:
    """Format an opponent team code, falling back to dash for NaN/None."""
    if val is None:
        return "—"
    try:
        if float(val) != float(val):  # NaN
            return "—"
    except (TypeError, ValueError):
        pass
    return str(val)


def _team(val) -> str:
    """Short 2-letter team code (falls back to the raw value)."""
    if not val:
        return ""
    return _LONG_TO_ABBR.get(str(val), str(val))


def _last5_scores(row: pd.Series, current_round: int) -> str:
    """Comma-delimited scores for the last 5 season rounds, DNP where absent."""
    rs = row.get("round_scores_2026")
    if not isinstance(rs, dict) or not rs or current_round < 1:
        return "—"
    cells = []
    for rnd in range(max(1, current_round - 4), current_round + 1):
        score = rs.get(rnd)
        cells.append(str(int(score)) if score is not None else "DNP")
    return ", ".join(cells)


def _load_role_signals(recent_rounds: int = 3) -> dict[int, str]:
    """
    Return {feed_id: note} for players with notably high recent CBA% or
    kick-in involvement (last recent_rounds rounds of DFS Australia data).
    """
    notes: dict[int, str] = {}
    try:
        cba = pd.read_csv(_DFS_DIR / str(SEASON_YEAR) / f"cbas_{SEASON_YEAR}.csv")
        cutoff = cba["round_num"].max() - recent_rounds
        for fid, grp in cba[cba["round_num"] > cutoff].groupby("feed_id"):
            avg = grp["cba_pct"].mean()
            if avg >= 40:
                notes[int(fid)] = f"CBA {avg:.0f}%"
    except Exception:
        pass
    try:
        ki = pd.read_csv(_DFS_DIR / str(SEASON_YEAR) / f"kickins_{SEASON_YEAR}.csv")
        cutoff = ki["round_num"].max() - recent_rounds
        for fid, grp in ki[ki["round_num"] > cutoff].groupby("feed_id"):
            avg_ki = grp["kickin_count"].mean()
            if avg_ki >= 3:
                note = f"KI {avg_ki:.1f}/gm"
                prev = notes.get(int(fid))
                notes[int(fid)] = f"{prev} · {note}" if prev else note
    except Exception:
        pass
    return notes


def _load_fanfooty_tags(recent_rounds: int = 3) -> dict[int, str]:
    """
    Most recent fanfooty tags (Tag / Tag 2 columns) per player from the last
    recent_rounds rounds, e.g. 'hot, shovel (R17)'. Later rounds win.
    """
    import glob
    import re
    pattern = str(
        Path(__file__).parent.parent / "data" / "processed"
        / f"{SEASON_YEAR}_round_*_fanfooty_data.csv"
    )
    rre = re.compile(r"_round_(\d+)_")
    frames = []
    for f in sorted(glob.glob(pattern)):
        m = rre.search(f)
        if not m:
            continue
        try:
            df = pd.read_csv(f, usecols=["Player ID", "Tag", "Tag 2"], low_memory=False)
        except Exception:
            continue
        df["round_num"] = int(m.group(1))
        frames.append(df)
    if not frames:
        return {}
    d = pd.concat(frames)
    d["Player ID"] = pd.to_numeric(d["Player ID"], errors="coerce")
    d = d.dropna(subset=["Player ID"])
    d = d[d["round_num"] > d["round_num"].max() - recent_rounds]
    tags: dict[int, str] = {}
    for _, r in d.sort_values("round_num").iterrows():
        vals = [
            str(v).strip() for v in (r.get("Tag"), r.get("Tag 2"))
            if isinstance(v, str) and str(v).strip()
        ]
        if vals:
            tags[int(r["Player ID"])] = f"{', '.join(vals)} (R{int(r['round_num'])})"
    return tags


def _data_note(source: str, sufficient: bool) -> str:
    if source == "none":
        return "*(no data)*"
    if not sufficient:
        return "*(small sample)*"
    if source == "2026":
        return "*(2026 data)*"
    return "*(hist.)*"


def _injury_flag(row: pd.Series) -> str:
    """Return an emoji suffix for injured/suspended players."""
    status = str(row.get("injury_status", "") or "")
    if status == "Injury":
        return " 🚑"
    if status == "Suspended":
        return " ⚠️"
    return ""


def _md_table(headers: list[str], rows: list[list]) -> str:
    """Render a simple Markdown table."""
    sep = " | "
    header_row = sep.join(f"**{h}**" for h in headers)
    divider = " | ".join("---" for _ in headers)
    lines = [f"| {header_row} |", f"| {divider} |"]
    for row in rows:
        lines.append("| " + sep.join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _drop_zone_composite(row: pd.Series) -> float:
    """Lower composite = more droppable."""
    return float(row.get("tpor", 0.0)) + float(row.get("ceiling_index", 0.0)) * 0.5


def _load_recently_targeted_feed_ids() -> set[int]:
    """
    Return set of feed_ids picked up in recent transactions (last 4 rounds).
    Used to flag 'undiscovered' gems — players who rank highly but no coach
    has targeted. Inspired by Ian Graham's 'Robertson was invisible' insight.
    """
    try:
        tx = pd.read_csv(TRANSACTIONS_PATH)
        if "player.feed_id" in tx.columns:
            recent = tx.sort_values("round", ascending=False).head(200)
            return set(recent["player.feed_id"].dropna().astype(int).tolist())
    except Exception:
        pass
    return set()


def _hot_streak_note(row: pd.Series) -> str:
    """
    Flag players on hot streaks that may regress to mean.
    Graham: hot-streak players are like Markovic — scouted on their best games.
    34/69 Bundesliga overachievers fell below expectation the following season.
    """
    if row.get("hot_streak"):
        recent = row.get("scores_2026") or []
        recent3 = recent[-3:] if len(recent) >= 3 else recent
        r_avg = sum(recent3) / len(recent3) if recent3 else 0
        hist = row.get("effective_avg", 0)
        return f" ⚠️ hot (+{r_avg - hist:.0f})"
    return ""


def _news_badge(feed_id: "int | None", sentiment_map: "dict[int, float]") -> str:
    """
    Return a news badge if the player has significant recent news sentiment.
    sentiment_map: {feed_id: avg_sentiment_score} pre-computed from load_news_sentiment().
    """
    if feed_id is None or not sentiment_map:
        return ""
    score = sentiment_map.get(int(feed_id))
    if score is None:
        return ""
    if score <= -0.5:
        return " 📰⚠"   # strong negative (ruled out / injury confirmed)
    if score <= -0.2:
        return " 📰"     # mild negative (managed / under cloud)
    return ""


def _trajectory_badge(row: pd.Series) -> str:
    """
    Flag players whose last-3 avg is rising significantly vs their last-6 avg.
    Catches the 'dead zone' players (avg 75-85) who are trending to 90+ but
    are above the Mid-Year Movers cutoff. Delta >= 10 = rising, >= 18 = surging.
    """
    scores = row.get("scores_2026") or []
    if not isinstance(scores, list) or len(scores) < 5:
        return ""
    avg3 = sum(scores[-3:]) / 3
    avg_all = sum(scores) / len(scores)
    delta = avg3 - avg_all
    if delta >= 18:
        return f" *(↑ +{delta:.0f})*"
    if delta >= 10:
        return f" *(↑ +{delta:.0f})*"
    return ""


def _durability_pct(row: pd.Series) -> str:
    d = row.get("durability_score")
    if d is None or (isinstance(d, float) and d != d):
        return "—"
    return f"{d * 100:.0f}%"


def _drop_confidence(n_games: int) -> str:
    """How confident is the Drop Zone recommendation? (Graham: Suárez lesson)"""
    if n_games <= 3:
        return "⚠️ low confidence"
    if n_games <= 8:
        return "moderate"
    return "confident"


# ── Section builders ────────────────────────────────────────────────────────

def _active_2026_mask(df: pd.DataFrame, min_games: int = 2, recency_rounds: int = 3) -> pd.Series:
    """
    True for players who have played ≥ min_games in 2026 AND appeared in at least
    one of the last recency_rounds rounds. Filters out injured/inactive players
    whose historical metrics would otherwise inflate their ranking.
    """
    if "games_played_2026" not in df.columns or "last_round_2026" not in df.columns:
        return pd.Series(True, index=df.index)
    current_round = int(df["last_round_2026"].max()) if df["last_round_2026"].max() > 0 else 0
    recency_cutoff = max(1, current_round - recency_rounds + 1)
    return (df["games_played_2026"] >= min_games) & (df["last_round_2026"] >= recency_cutoff)


def _section_ceiling_chasers(metrics_df: pd.DataFrame, top_n: int = 5) -> str:
    fa = metrics_df[
        metrics_df["is_free_agent"]
        & ~(metrics_df["is_injured"] | metrics_df["is_suspended"])
        & _active_2026_mask(metrics_df)
    ].copy()
    if fa.empty:
        return "*No free agents found.*\n"

    fa = fa.sort_values(
        ["ceiling_index", "flex_bonus"], ascending=[False, False]
    ).head(top_n)

    has_predictions = "projected_score" in fa.columns and fa["projected_score"].notna().any()
    has_p120 = "probability_gt_120" in fa.columns and fa["probability_gt_120"].notna().any()

    has_2026_avg = "avg_fanfooty_2026" in fa.columns

    if has_predictions:
        headers = ["Player", "Pos", "Team", "vs", "Proj.", "P(80+)", "P(100+)", "P(120+)",
                   f"CI% (>{CEILING_THRESHOLD})", "Avg'26", "Gms", "FlexBonus"]
    else:
        headers = ["Player", "Pos", "Team", f"CI% (>{CEILING_THRESHOLD})",
                   "Avg'26", "Gms", "FlexBonus", "CV", "TPOR", "Data"]

    rows = []
    for _, r in fa.iterrows():
        avg26 = _fmt(r["avg_fanfooty_2026"]) if has_2026_avg else _fmt(r["effective_avg"])
        gms = str(int(r["games_played_2026"])) if "games_played_2026" in r else "—"
        if has_predictions:
            rows.append([
                f"**{r['name']}**",
                r["pos_bucket"],
                _team(r.get("team", "")),
                _opp(r.get("opponent")),
                _fmt(r.get("projected_score"), 1) if pd.notna(r.get("projected_score")) else "—",
                _pct(r.get("probability_gt_80", 0)),
                _pct(r.get("probability_gt_100", 0)),
                _pct(r.get("probability_gt_120", 0)) if has_p120 else "—",
                _pct(r["ceiling_index"]),
                avg26,
                gms,
                f"+{_fmt(r['flex_bonus'])}" if r["flex_bonus"] > 0 else _fmt(r["flex_bonus"]),
            ])
        else:
            rows.append([
                f"**{r['name']}**",
                r["pos_bucket"],
                _team(r.get("team", "")),
                _pct(r["ceiling_index"]),
                avg26,
                gms,
                f"+{_fmt(r['flex_bonus'])}" if r["flex_bonus"] > 0 else _fmt(r["flex_bonus"]),
                _fmt(r["cv"], 3),
                _fmt(r["tpor"]),
                _data_note(r.get("data_source", ""), r.get("sufficient_data", False)),
            ])

    return _md_table(headers, rows) + "\n"


def _section_ruck_radar(metrics_df: pd.DataFrame, top_n: int = 10) -> str:
    fa_ruc = metrics_df[
        metrics_df["is_free_agent"]
        & (metrics_df["pos_bucket"] == "RUC")
        & _active_2026_mask(metrics_df)
        & ~(metrics_df["is_injured"] | metrics_df["is_suspended"])
    ].copy()

    if fa_ruc.empty:
        return "*No Ruck free agents currently available. Ruck market is dry.*\n"

    fa_ruc = fa_ruc.sort_values("tpor", ascending=False).head(top_n)

    has_predictions = "projected_score" in fa_ruc.columns and fa_ruc["projected_score"].notna().any()
    has_p120 = "probability_gt_120" in fa_ruc.columns and fa_ruc["probability_gt_120"].notna().any()
    has_2026_avg = "avg_fanfooty_2026" in fa_ruc.columns

    if has_predictions:
        headers = ["Player", "Team", "vs", "Proj.", "P(80+)", "P(100+)", "P(120+)", "Avg'26", "Gms", "TPOR"]
    else:
        headers = ["Player", "Team", "Avg'26", "Gms", "TPOR", f"CI% (>{CEILING_THRESHOLD})", "CV", "Data"]

    rows = []
    for _, r in fa_ruc.iterrows():
        avg26 = _fmt(r["avg_fanfooty_2026"]) if has_2026_avg else _fmt(r["effective_avg"])
        gms = str(int(r["games_played_2026"])) if "games_played_2026" in r else "—"
        if has_predictions:
            rows.append([
                f"**{r['name']}**",
                _team(r.get("team", "")),
                _opp(r.get("opponent")),
                _fmt(r.get("projected_score"), 1) if pd.notna(r.get("projected_score")) else "—",
                _pct(r.get("probability_gt_80", 0)),
                _pct(r.get("probability_gt_100", 0)),
                _pct(r.get("probability_gt_120", 0)) if has_p120 else "—",
                avg26,
                gms,
                _fmt(r["tpor"]),
            ])
        else:
            rows.append([
                f"**{r['name']}**",
                _team(r.get("team", "")),
                avg26,
                gms,
                _fmt(r["tpor"]),
                _pct(r["ceiling_index"]),
                _fmt(r["cv"], 3),
                _data_note(r.get("data_source", ""), r.get("sufficient_data", False)),
            ])

    repl = fa_ruc["replacement_level"].iloc[0] if "replacement_level" in fa_ruc.columns else 0.0
    note = f"> Ruck replacement level: **{_fmt(repl)}** avg (best available RUC FA)\n\n"
    return note + _md_table(headers, rows) + "\n"


def _recent3_avg(scores_2026) -> float | None:
    """Average of the last 3 played rounds from 2026 data, or None if unavailable."""
    if not isinstance(scores_2026, list) or len(scores_2026) == 0:
        return None
    recent = scores_2026[-3:]
    return round(sum(recent) / len(recent), 1)


def _current_round(metrics_df: pd.DataFrame) -> int:
    """Highest 2026 round seen in the data (0 = pre-season)."""
    if "last_round_2026" in metrics_df.columns and metrics_df["last_round_2026"].max() > 0:
        return int(metrics_df["last_round_2026"].max())
    return 0


def _calibrated_p80(row) -> float:
    """Gaussian P(80+) on a recency/stability blend:
    0.6 * last-5 average + 0.4 * last-20-games average (cross-season).

    2013-2026 backtest (285 rounds, reports/backtest_positional_targets_2013_2026.md):
      blend_p80:        44.0% hit / 17.8% bullseye  (best on all metrics)
      pure avg5 P(80+): 43.6% / 17.2%
      model projected_score ranking: 36.1% / 12.9% (2026, clean window)
    The avg20 term damps single-game noise in avg5 without losing the
    role-change signal; the model projection is display-only.

    Priority:
    1. Gaussian from blend of avg5 (last 5 played SC rounds) and avg20.
    2. Gaussian from blend of available rounds (< 5 games) and avg20.
    3. Gaussian from fanfooty season avg (no SC round files yet).
    4. Gaussian from career effective avg (pre-season / no 2026 data).
    """
    scores_2026 = row.get("scores_2026") or []
    n_rounds = len(scores_2026)
    avg20 = float(row.get("avg20") or 0)

    # Path 1: last-5 played rounds, stabilised by last-20 avg
    if n_rounds >= 5:
        avg5 = sum(scores_2026[-5:]) / 5
        mean = 0.6 * avg5 + 0.4 * avg20 if avg20 > 0 else avg5
        return round(float(_sc_norm.sf(80, loc=mean, scale=22) * 100), 1)

    # Path 2: fewer than 5 games — use all available, stabilised
    if n_rounds >= 1:
        avg_recent = sum(scores_2026) / n_rounds
        mean = 0.6 * avg_recent + 0.4 * avg20 if avg20 > 0 else avg_recent
        return round(float(_sc_norm.sf(80, loc=mean, scale=22) * 100), 1)

    # Path 3: fanfooty season aggregate
    avg_2026 = float(row.get("avg_fanfooty_2026") or 0)
    if avg_2026 > 0:
        return round(float(_sc_norm.sf(80, loc=avg_2026, scale=22) * 100), 1)

    # Path 4: career effective avg (pre-season or no 2026 data)
    eff_avg = float(row.get("bayesian_avg") or row.get("effective_avg") or 0)
    if eff_avg > 0:
        return round(float(_sc_norm.sf(80, loc=eff_avg, scale=22) * 100), 1)

    return 0.0


def _apply_positional_sort(fa: pd.DataFrame, current_round: int) -> pd.DataFrame:
    """Adds calibrated_p80 and sort_score (P(80+) x availability discount).

    Primary ranking for Positional Targets. 2013-2026 backtest: model
    projected_score as sort key scored 36.1% hits vs 43.6-44.0% for calibrated
    P(80+); the DVP (+/-3) and CBA (0-1.5) sort bonuses measured zero effect
    over 285 rounds — both removed (model/DVP stay as display context). The
    availability discount (x0.8 missed last round, x0.6 missed last two) cut
    listed-player DNP rate 15.1% -> 12.8% at no hit-rate cost.
    """
    fa = fa.copy()
    fa["calibrated_p80"] = fa.apply(_calibrated_p80, axis=1)
    if current_round > 0 and "last_round_2026" in fa.columns:
        gap = (current_round - fa["last_round_2026"]).clip(lower=0)
        fa["_avail_factor"] = gap.map({0: 1.0, 1: 0.8, 2: 0.6}).fillna(0.5)
    else:
        fa["_avail_factor"] = 1.0
    fa["sort_score"] = fa["calibrated_p80"] * fa["_avail_factor"]
    return fa


def _section_positional_targets(metrics_df: pd.DataFrame, top_n: int = 10, sentiment_map: dict | None = None, dvp_ranks: "dict | None" = None) -> str:
    """
    Top free agents by position (DEF, MID, FWD, RUC) for sustained pickup value.
    Dual-position players appear in every list they are eligible for.

    When model predictions are available (projected_score column present):
      - Ranked by calibrated P(80+) (Gaussian on 0.6*avg5 + 0.4*avg20 blend)
        with an availability discount for players who missed recent rounds
      - Shows Proj., P(80+), P(100+), P(120+), and opponent columns (model
        values are display context only — see 2013-2026 backtest report)
    Otherwise:
      - Ranked by pick_score: Bayesian avg blending 2026 form with career prior

    dvp_ranks is retained for the call signature but no longer affects the
    sort (backtest measured zero effect); DVP context lives in Matchup Intel.
    """
    fa = metrics_df[
        metrics_df["is_free_agent"]
        & ~(metrics_df["is_injured"] | metrics_df["is_suspended"])
        & _active_2026_mask(metrics_df)
    ].copy()

    if fa.empty:
        return "*No free agents available.*\n"

    has_predictions = (
        "projected_score" in fa.columns and fa["projected_score"].notna().any()
    )
    has_2026_data = (
        "scores_2026" in fa.columns
        and fa["scores_2026"].apply(lambda x: isinstance(x, list) and len(x) > 0).any()
    )

    recently_targeted = _load_recently_targeted_feed_ids()

    if has_predictions:
        bayes_col = "bayesian_avg" if "bayesian_avg" in fa.columns else "effective_avg"
        has_p120 = "probability_gt_120" in fa.columns and fa["probability_gt_120"].notna().any()

        fa = _apply_positional_sort(fa, _current_round(metrics_df))

        section_note = (
            "> Ranked by **calibrated P(80+)** — Gaussian on 0.6·last-5 avg + "
            "0.4·last-20-game avg — discounted ×0.8/×0.6 if the player missed the "
            "last 1/2 rounds (2013–2026 backtest: 44.0% hit rate vs 36.1% for "
            "model-projection ranking). Proj./P(100+)/P(120+) are model context, "
            "not the sort. Dual-position players appear in every eligible list. "
            "**Last 5** = scores in the last 5 season rounds (DNP = did not play). "
            "**Role** = recent CBA% / kick-ins per game (last 3 rounds, notable only). "
            "**Tags** = most recent fanfooty tags (last 3 rounds). "
            "*(↓ form)* = season avg < 70 (≥6 games). "
            "*(↑ +N)* = last-3 avg is N pts above season avg. "
            "📰⚠ = strong negative news. 📰 = mild negative news. "
            "⚠️ hot = recent form >35% above career avg (regression risk). "
            "💎 = highly ranked but untargeted by any coach this season.\n\n"
        )
        headers = ["Player", "Team", "vs", "Proj.", "Cal P(80+)", "P(100+)", "P(120+)", "2026 Avg", "Last 5", "Role", "Tags"]
    else:
        fa["recent3_avg"] = fa["scores_2026"].apply(_recent3_avg) if "scores_2026" in fa.columns else None
        bayes_col = "bayesian_avg" if "bayesian_avg" in fa.columns else "effective_avg"
        fa["pick_score"] = fa[bayes_col]
        if has_2026_data:
            section_note = (
                "> Ranked by **Bayesian avg** (blends 2026 form with career history — "
                "small 2026 samples barely move the prior). "
                "Avail% = games played last 2 seasons. ⚠️ hot = regression risk. 💎 = untargeted gem.\n\n"
            )
            form_header = "Form (last 3)"
        else:
            section_note = (
                "> Pre-season: ranked by weighted historical average (2011–2025). "
                "Run `py run_predictions.py` to enable fixture-aware projections.\n\n"
            )
            form_header = "Hist. avg"
        headers = ["Player", "Team", "Avg", "TPOR", f"CI% (>{CEILING_THRESHOLD})", form_header, "Avail%"]

    sections = [section_note]

    sort_col = "sort_score" if has_predictions else "pick_score"
    current_round = _current_round(metrics_df)
    role_notes = _load_role_signals()
    ff_tags = _load_fanfooty_tags()

    def _recent_avg(scores, n: int = 5) -> float | None:
        recent = (scores or [])[-n:]
        return sum(recent) / len(recent) if recent else None

    for pos in ("DEF", "MID", "FWD", "RUC"):
        # Dual-position players appear in EVERY list they're eligible for
        # (previously _position_bucket collapsed e.g. MID/FWD into MID only,
        # so the FWD list never saw FWD-eligible midfielders).
        if "position" in fa.columns:
            pos_mask = fa["position"].astype(str).str.upper().str.contains(pos, regex=False)
        else:
            pos_mask = fa["pos_bucket"] == pos
        pos_all = fa[pos_mask].copy()
        if pos_all.empty:
            sections.append(f"### {pos}\n*No {pos} free agents available.*\n")
            continue

        # Fix 4: suppress persistent underperformers — last-5 avg < 55 with ≥5 games.
        # These players keep appearing due to high career averages despite tanking form.
        # Apply a large penalty so they fall out of any plausible top-N.
        if "scores_2026" in pos_all.columns:
            pos_all["_recent5_avg"] = pos_all["scores_2026"].apply(
                lambda s: _recent_avg(s, 5)
            )
            underperformer = (
                pos_all["_recent5_avg"].notna()
                & (pos_all["_recent5_avg"] < 55)
                & (pos_all.get("games_played_2026", 0) >= 5)
            )
            pos_all[sort_col] = pos_all[sort_col].where(~underperformer, -1000.0)

        # Fix 5: demote players with season-long form decay (avg < 70, ≥6 games).
        # The (↓ form) badge already flags them visually; this moves them down the list.
        avg26_col = "avg_fanfooty_2026" if "avg_fanfooty_2026" in pos_all.columns else None
        if avg26_col:
            games_col_ok = pos_all.get("games_played_2026", pd.Series(0, index=pos_all.index))
            form_decay = (
                pos_all[avg26_col] > 0
                & (pos_all[avg26_col] < 70)
                & (games_col_ok >= 6)
            )
            pos_all[sort_col] = pos_all[sort_col] - form_decay.astype(float) * 10.0

        pos_fa = pos_all.sort_values(sort_col, ascending=False).head(top_n)
        if pos_fa.empty:
            sections.append(f"### {pos}\n*No {pos} free agents available.*\n")
            continue

        rows = []
        for i, (_, r) in enumerate(pos_fa.iterrows()):
            fid = int(r.get("feed_id", 0))
            gem_badge = " 💎" if (i < 5 and fid not in recently_targeted) else ""
            hot_note = _hot_streak_note(r)
            avail = _durability_pct(r)

            if has_predictions:
                avg26 = float(r.get("avg_fanfooty_2026") or 0)
                games26 = int(r.get("games_played_2026") or 0)
                form_decay_badge = " *(↓ form)*" if (avg26 > 0 and avg26 < 70 and games26 >= 6) else ""
                traj_badge = _trajectory_badge(r)
                news_badge = _news_badge(r.get("feed_id"), sentiment_map or {})
                name_cell = f"**{r['name']}**{gem_badge}{hot_note}{form_decay_badge}{traj_badge}{news_badge}"
                avg26_str = _fmt(avg26) if avg26 > 0 else "—"
                proj_str = _fmt(r.get("projected_score"), 1) if pd.notna(r.get("projected_score")) else "—"
                opponent = _opp(r.get("opponent"))
                rows.append([
                    name_cell,
                    _team(r.get("team", "")),
                    opponent,
                    proj_str,
                    _pct(r.get("calibrated_p80", r.get("probability_gt_80", 0))),
                    _pct(r.get("probability_gt_100", 0)),
                    _pct(r.get("probability_gt_120", 0)) if has_p120 else "—",
                    avg26_str,
                    _last5_scores(r, current_round),
                    role_notes.get(fid, "—"),
                    ff_tags.get(fid, "—"),
                ])
            else:
                name_cell = f"**{r['name']}**{gem_badge}{hot_note}"
                if has_2026_data and pd.notna(r.get("recent3_avg")):
                    recent_scores = r["scores_2026"][-3:] if isinstance(r.get("scores_2026"), list) else []
                    form_str = ", ".join(str(s) for s in recent_scores)
                else:
                    form_str = _fmt(r["effective_avg"])
                rows.append([
                    name_cell,
                    _team(r.get("team", "")),
                    _fmt(r[bayes_col]),
                    _fmt(r["tpor"]),
                    _pct(r["ceiling_index"]),
                    form_str,
                    avail,
                ])

        sections.append(f"### {pos}\n" + _md_table(headers, rows) + "\n")

    return "\n".join(sections)


def _section_previous_performance(round_num: int) -> str:
    """
    Shows how last round's newsletter recommendations scored in the following week.
    Loads the previous round's newsletter file and looks up each Positional Targets
    pick against the SC API round JSON for round_num.
    """
    import re, json, glob
    from pathlib import Path

    prev_round = round_num - 1
    if prev_round < 1:
        return "*No previous newsletter to compare against.*\n"

    # Find previous newsletter file
    reports_dir = REPORTS_DIR
    pattern = str(reports_dir / f"waiver_{SEASON_YEAR}_rd{prev_round:02d}_*.md")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return f"*No Round {prev_round} newsletter found to compare.*\n"
    prev_nl_path = matches[-1]

    # Load round_num SC JSON for actual scores
    sc_json_path = (
        Path(__file__).parent.parent
        / "data" / "raw" / "supercoach" / str(SEASON_YEAR)
        / f"players_round_{round_num}.json"
    )
    if not sc_json_path.exists():
        return f"*Round {round_num} SC data not yet available.*\n"

    with open(sc_json_path) as f:
        sc_data = json.load(f)
    id_to_score = {}
    id_to_name = {}
    for p in sc_data:
        pid = int(p["id"])
        id_to_name[pid] = f"{p['first_name']} {p['last_name']}"
        if p["player_stats"]:
            stat = p["player_stats"][0]
            if stat.get("games", 0) == 1:
                id_to_score[pid] = float(stat.get("points", 0))

    # Load SC player list for name -> id mapping
    pl = pd.read_csv(
        Path(__file__).parent.parent / "draft_prep" / f"SC {SEASON_YEAR}" / f"{SEASON_YEAR}_SC_Player_list.csv",
        low_memory=False,
    )
    pl["full_name"] = (pl["first_name"].str.strip() + " " + pl["last_name"].str.strip()).str.strip()
    name_to_id = {row["full_name"].lower(): int(row["id"]) for _, row in pl.iterrows() if not pd.isna(row["id"])}

    def resolve_name(raw: str):
        key = raw.lower().strip()
        pid = name_to_id.get(key)
        if pid is None:
            parts = key.split()
            if parts:
                last = parts[-1]
                cands = [(n, v) for n, v in name_to_id.items() if n.split()[-1] == last]
                if len(cands) == 1:
                    pid = cands[0][1]
                elif len(cands) > 1 and len(parts) > 1:
                    first = parts[0]
                    exact = [c for c in cands if c[0].startswith(first)]
                    if len(exact) == 1:
                        pid = exact[0][1]
        return pid

    # Parse positional targets from previous newsletter
    txt = Path(prev_nl_path).read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"\n## ", txt)
    pt_section = next((s for s in sections if "Positional Targets" in s.split("\n")[0]), "")

    picks = []
    pos = None
    pos_count = 0
    for line in pt_section.split("\n"):
        m_pos = re.match(r"^### (DEF|MID|FWD|RUC)\s*$", line.strip())
        if m_pos:
            pos = m_pos.group(1)
            pos_count = 0
            continue
        if pos and pos_count < 5 and line.startswith("|") and "**" in line:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 4:
                continue
            nc = cells[1]
            if "**Player**" in nc or "---" in nc:
                continue
            mn = re.search(r"\*\*(.+?)\*\*", nc)
            if not mn:
                continue
            raw = re.sub(r"[^\x00-\x7F]+", "", mn.group(1)).strip()
            raw = re.sub(r"\s+", " ", raw).strip()
            if len(raw) < 3:
                continue
            picks.append({"pos": pos, "name": raw, "pid": resolve_name(raw)})
            pos_count += 1

    if not picks:
        return f"*Could not parse Round {prev_round} newsletter picks.*\n"

    hits = misses = dnp = unmatched = 0
    rows = []
    for pick in picks:
        pid = pick["pid"]
        if pid is None:
            unmatched += 1
            rows.append((pick["pos"], pick["name"], "?", "—"))
            continue
        if pid in id_to_score:
            sc = id_to_score[pid]
            result = "HIT" if sc >= 80 else "miss"
            if sc >= 80:
                hits += 1
            else:
                misses += 1
            rows.append((pick["pos"], pick["name"], f"{sc:.0f}", result))
        else:
            dnp += 1
            rows.append((pick["pos"], pick["name"], "DNP", "—"))

    played = hits + misses
    hit_rate = f"{hits}/{played} ({hits/played*100:.0f}%)" if played else "0/0"

    lines = [
        f"**Round {prev_round} Positional Targets — Round {round_num} results:** "
        f"{hit_rate} scored 80+  |  {dnp} picks did not play",
        "",
        "| Pos | Player | Rd{} Score | Result |".format(round_num),
        "|-----|--------|------------|--------|",
    ]
    for pos, name, score, result in rows:
        marker = " **HIT**" if result == "HIT" else ""
        lines.append(f"| {pos} | {name} | {score} |{marker} |")

    return "\n".join(lines) + "\n"


def _section_injury_watchlist(afl_injury_df: "pd.DataFrame | None", metrics_df: pd.DataFrame) -> str:
    """
    Free agents currently on the AFL injury list — to monitor for return to play.
    Excludes players already on the user's roster.
    """
    import json
    if afl_injury_df is None or afl_injury_df.empty:
        return "*AFL injury list unavailable.*\n"

    # Only include matched free agents (not on my roster)
    matched = afl_injury_df[
        afl_injury_df["sc_id"].notna()
        & afl_injury_df["match_confidence"].isin(["exact", "exact+team", "fuzzy", "fuzzy+team"])
    ].copy()
    matched["sc_id"] = matched["sc_id"].astype(int)

    # Get set of my player SC ids
    my_ids = set()
    if "sc_id" in metrics_df.columns:
        my_ids = set(metrics_df[metrics_df["is_mine"]]["sc_id"].dropna().astype(int).tolist())
    elif "id" in metrics_df.columns:
        my_ids = set(metrics_df[metrics_df["is_mine"]]["id"].dropna().astype(int).tolist())

    fa_injured = matched[~matched["sc_id"].isin(my_ids)].copy()

    if fa_injured.empty:
        return "*No injured free agents of note this week.*\n"

    # Look up avg and avg3 from SC JSON for context
    sc_json_path = (
        Path(__file__).parent.parent
        / "data" / "raw" / "supercoach" / str(SEASON_YEAR)
        / f"players_round_{_latest_round()}.json"
    )
    sc_avg = {}
    sc_avg3 = {}
    if sc_json_path.exists():
        with open(sc_json_path) as f:
            sc_data = json.load(f)
        for p in sc_data:
            pid = int(p["id"])
            if p["player_stats"]:
                stat = p["player_stats"][0]
                sc_avg[pid] = float(stat.get("avg", 0) or 0)
                sc_avg3[pid] = float(stat.get("avg3", 0) or 0)

    # Sort by season avg descending — prioritise high-value return targets
    fa_injured["_avg"] = fa_injured["sc_id"].map(sc_avg).fillna(0)
    fa_injured = fa_injured[fa_injured["_avg"] > 0].sort_values("_avg", ascending=False)

    if fa_injured.empty:
        return "*No injured free agents with SC data this week.*\n"

    headers = ["Player", "Team", "Injury", "Est. Return", "Season Avg"]
    rows = []
    for _, r in fa_injured.iterrows():
        avg_val = _fmt(r["_avg"])
        rows.append([
            r.get("name") or r.get("afl_name"),
            r.get("team_abbrev", ""),
            r.get("injury", ""),
            r.get("est_return", ""),
            avg_val,
        ])
    return _md_table(headers, rows) + "\n"


def _latest_round() -> int:
    """Return the latest round number available in SC JSON files."""
    import glob
    files = glob.glob(str(
        Path(__file__).parent.parent / "data" / "raw" / "supercoach" / str(SEASON_YEAR)
        / "players_round_*.json"
    ))
    if not files:
        return 1
    rounds = []
    import re
    for f in files:
        m = re.search(r"players_round_(\d+)\.json", f)
        if m:
            rounds.append(int(m.group(1)))
    return max(rounds) if rounds else 1


def _section_injury_report(metrics_df: pd.DataFrame, sentiment_map: dict | None = None) -> str:
    """Shows my roster players with active injuries or suspensions."""
    injured = metrics_df[
        metrics_df["is_mine"]
        & (metrics_df["is_injured"] | metrics_df["is_suspended"])
    ].copy()

    if injured.empty:
        return "*No injuries or suspensions on your roster this week.*\n"

    # Load recent news for injured players
    news_context: dict = {}
    try:
        from waiver.data_loader import load_news_sentiment
        fids = [int(f) for f in injured["feed_id"].dropna().tolist()]
        if fids:
            news_df = load_news_sentiment(feed_ids=fids, lookback_days=7)
            if not news_df.empty:
                # Pick most recent negative-sentiment headline per player
                neg = news_df[news_df["sentiment_score"] < -0.1].sort_values("published_at", ascending=False)
                for fid, grp in neg.groupby("feed_id"):
                    row = grp.iloc[0]
                    news_context[int(fid)] = str(row.get("headline", ""))[:80]
    except Exception:
        pass

    # On-field starters first, then bench
    injured["_priority"] = (~injured["is_on_field"]).astype(int)
    injured = injured.sort_values("_priority")

    headers = ["Player", "Pos", "Status", "Details", "News", "On Field?"]
    rows = []
    for _, r in injured.iterrows():
        status_icon = "🚑 Injury" if r.get("is_injured") else "⚠️ Suspended"
        on_field = "**Yes — act now**" if r.get("is_on_field") else "Bench"
        details = str(r.get("injury_text", "") or "—")[:60]
        fid = int(r["feed_id"]) if pd.notna(r.get("feed_id")) else None
        headline = news_context.get(fid, "—")
        rows.append([
            f"**{r['name']}**",
            r["pos_bucket"],
            status_icon,
            details,
            headline,
            on_field,
        ])

    return _md_table(headers, rows) + "\n"


def _section_my_team(metrics_df: pd.DataFrame) -> str:
    mine = metrics_df[metrics_df["is_mine"]].copy()
    if mine.empty:
        return "*Could not find your roster in the player table.*\n"

    has_predictions = "projected_score" in mine.columns and mine["projected_score"].notna().any()
    has_p120 = "probability_gt_120" in mine.columns and mine["probability_gt_120"].notna().any()

    mine = mine.sort_values("tpor", ascending=False)

    if has_predictions:
        headers = ["Player", "Pos", "Status", "Avg", "Proj.", "vs", "P(80+)", "P(100+)", "P(120+)", "TPOR"]
    else:
        headers = ["Player", "Pos", "Status", "Avg", "TPOR", f"CI% (>{CEILING_THRESHOLD})"]

    rows = []
    for _, r in mine.iterrows():
        if r.get("is_on_field"):
            status = "On-field"
        else:
            status = "Bench"
        name_cell = f"**{r['name']}**{_injury_flag(r)}"
        if has_predictions:
            rows.append([
                name_cell,
                r["pos_bucket"],
                status,
                _fmt(r["effective_avg"]),
                _fmt(r.get("projected_score"), 1) if pd.notna(r.get("projected_score")) else "—",
                _opp(r.get("opponent")),
                _pct(r.get("probability_gt_80", 0)),
                _pct(r.get("probability_gt_100", 0)),
                _pct(r.get("probability_gt_120", 0)) if has_p120 else "—",
                _fmt(r["tpor"]),
            ])
        else:
            rows.append([
                name_cell,
                r["pos_bucket"],
                status,
                _fmt(r["effective_avg"]),
                _fmt(r["tpor"]),
                _pct(r["ceiling_index"]),
            ])

    return _md_table(headers, rows) + "\n"


def _drop_verdict(rank: int, row: pd.Series) -> str:
    """A short action-oriented verdict for each Drop Zone player."""
    if row.get("is_injured"):
        return "**Waiver priority** — Injured"
    if row.get("is_suspended"):
        return "**Waiver priority** — Suspended"
    tpor = float(row.get("tpor", 0.0))
    ci = float(row.get("ceiling_index", 0.0))
    if rank == 1 and tpor < -5:
        return "Drop candidate"
    if ci < 5 and tpor < 0:
        return "Monitor closely"
    if row.get("data_source") == "none":
        return "No data — watch"
    return "Hold for now"


# ── Main generator ───────────────────────────────────────────────────────────

def _section_mid_year_movers(jumpers_df: pd.DataFrame | None, top_n: int = 5) -> str:
    """
    Renders the Mid-Year Movers section: top FA candidates likely to
    average 85+ for the rest of the season from a < 80 current avg.
    """
    if jumpers_df is None or jumpers_df.empty:
        return "*No strong mid-year mover candidates identified this week.*\n"

    df = jumpers_df.head(top_n)
    headers = ["Player", "Pos", "Team", "Curr Avg", "Career Avg", "TOG Trend", "Key Signal"]
    rows = []
    for _, r in df.iterrows():
        tog_t = r.get("tog_trend")
        tog_str = f"+{tog_t:.0f}%" if isinstance(tog_t, float) and tog_t > 0 else (f"{tog_t:.0f}%" if isinstance(tog_t, float) else "—")
        ca = r.get("career_avg_prior")
        ca_str = _fmt(ca) if ca is not None and not (isinstance(ca, float) and ca != ca) else "—"
        rows.append([
            f"**{r.get('name', '?')}**",
            r.get("pos_bucket", ""),
            _team(r.get("team", "")),
            _fmt(r.get("season_avg", 0)),
            ca_str,
            tog_str,
            str(r.get("jumper_reason", "")),
        ])
    return _md_table(headers, rows) + "\n"


def _scarcity_line(scarcity: "dict | None") -> str:
    """
    Compact one-line scarcity summary for the newsletter header.
    e.g. "DEF: 2 Elite · 5 Solid  MID: 0 Elite · 3 Solid  ..."
    """
    if not scarcity:
        return ""
    parts = []
    for pos in ("DEF", "MID", "RUC", "FWD"):
        counts = scarcity.get(pos, {})
        elite = counts.get("Elite", 0)
        solid = counts.get("Solid", 0)
        depth = counts.get("Depth", 0)
        parts.append(f"**{pos}**: {elite}E / {solid}S / {depth}D")
    return "> **FA Pool** (Elite≥90 / Solid≥75 / Depth≥60): " + "  ·  ".join(parts) + "\n"


def _section_matchup_intel(dvp_targets_df: "pd.DataFrame | None") -> str:
    """
    Renders the Matchup Intel section: FAs facing weak defences this round.
    dvp_targets_df comes from waiver.dvp.get_dvp_targets().
    """
    if dvp_targets_df is None or dvp_targets_df.empty:
        return "*No matchup data available — DVP table requires 2+ rounds of data.*\n"

    avg_col = next((c for c in ["avg_fanfooty_2026", "effective_avg"] if c in dvp_targets_df.columns), None)
    has_proj = "projected_score" in dvp_targets_df.columns

    if has_proj and avg_col:
        headers = ["Player", "Pos", "Team", "vs", "Avg Conceded", "DVP Rank", "Season Avg", "Proj."]
    elif avg_col:
        headers = ["Player", "Pos", "Team", "vs", "Avg Conceded", "DVP Rank", "Season Avg"]
    else:
        headers = ["Player", "Pos", "Team", "vs", "Avg Conceded", "DVP Rank"]

    rows = []
    for _, r in dvp_targets_df.iterrows():
        row_data = [
            f"**{r.get('name', '?')}**",
            r.get("pos_bucket", ""),
            _team(r.get("team", "")),
            r.get("opponent", "—"),
            _fmt(r.get("avg_sc_conceded")),
            str(int(r["dvp_rank"])) if pd.notna(r.get("dvp_rank")) else "—",
        ]
        if avg_col:
            row_data.append(_fmt(r.get(avg_col)))
        if has_proj:
            row_data.append(_fmt(r.get("projected_score")) if pd.notna(r.get("projected_score")) else "—")
        rows.append(row_data)

    return _md_table(headers, rows) + "\n"


def _section_schedule_streamers(schedule_df: "pd.DataFrame | None") -> str:
    """
    Renders the Schedule Window section: FAs with best 3-round fixture by DVP.
    schedule_df comes from waiver.dvp.get_schedule_streamers().
    """
    if schedule_df is None or schedule_df.empty:
        return "*Schedule streamer data not available.*\n"

    avg_col = next((c for c in ["avg_fanfooty_2026", "effective_avg"] if c in schedule_df.columns), None)
    headers = ["Player", "Pos", "Team", "Avg DVP Rank (next 3)", "Season Avg"]
    if not avg_col:
        headers = headers[:-1]

    rows = []
    for _, r in schedule_df.iterrows():
        row_data = [
            f"**{r.get('name', '?')}**",
            r.get("pos_bucket", ""),
            _team(r.get("team", "")),
            _fmt(r.get("avg_dvp_rank"), 1),
        ]
        if avg_col:
            row_data.append(_fmt(r.get(avg_col)))
        rows.append(row_data)

    return _md_table(headers, rows) + "\n"


def _section_weekly_streamers(streamers_df: pd.DataFrame | None, top_n: int = 5) -> str:
    """
    Renders the Weekly Streamers section: top FA candidates for a
    1-3 week pickup based on sudden role or score makeup changes.
    """
    if streamers_df is None or streamers_df.empty:
        return "*No clear weekly streamer opportunities identified this week.*\n"

    df = streamers_df.head(top_n)
    headers = ["Player", "Pos", "Team", "Last 3 Avg", "Season Avg", "Trigger", "Action"]
    rows = []
    for _, r in df.iterrows():
        avg3 = r.get("avg3", 0)
        season = r.get("season_avg", 0)
        trigger = str(r.get("streamer_trigger", ""))
        score = int(r.get("streamer_score", 0))
        action = "Pickup now" if score >= 4 else "Monitor — confirm role"
        rows.append([
            f"**{r.get('name', '?')}**",
            r.get("pos_bucket", ""),
            _team(r.get("team", "")),
            _fmt(avg3),
            _fmt(season),
            trigger,
            action,
        ])
    return _md_table(headers, rows) + "\n"


def generate(
    metrics_df: pd.DataFrame,
    round_num: int | None,
    pred_round: int | None = None,
    warnings: list[str] | None = None,
    schedule_updated: bool = False,
    output_dir: Path = REPORTS_DIR,
    jumpers_df: "pd.DataFrame | None" = None,
    streamers_df: "pd.DataFrame | None" = None,
    dvp_targets_df: "pd.DataFrame | None" = None,
    schedule_streamers_df: "pd.DataFrame | None" = None,
    scarcity_dict: "dict | None" = None,
    afl_injury_df: "pd.DataFrame | None" = None,
    dvp_ranks: "dict | None" = None,
) -> Path:
    """
    Assembles the full Markdown newsletter and writes it to output_dir.

    Parameters
    ----------
    metrics_df   : enriched player table from metrics.score_all_players()
    round_num    : AFL round number (None = pre-season)
    pred_round   : round number predictions target (None if unavailable)
    warnings     : list of data issue strings to surface in the newsletter
    output_dir   : directory to write the .md file
    jumpers_df   : scored mid-year jumper candidates (from jumper_scorer.score_free_agents)
    streamers_df : scored streamer candidates (from jumper_scorer.score_free_agents)

    Returns
    -------
    Path to the generated Markdown file.
    """
    # Load recent news sentiment (keyed by feed_id → avg score last 7 days)
    sentiment_map: dict = {}
    try:
        from waiver.data_loader import load_news_sentiment
        _news_df = load_news_sentiment(lookback_days=7)
        if not _news_df.empty and "feed_id" in _news_df.columns and "sentiment_score" in _news_df.columns:
            sentiment_map = (
                _news_df.groupby("feed_id")["sentiment_score"].mean().to_dict()
            )
            log.info("News sentiment loaded: %d players with recent mentions.", len(sentiment_map))
    except Exception as _sent_exc:
        log.debug("News sentiment load failed (non-fatal): %s", _sent_exc)

    now = datetime.now()
    today = now.date()
    round_label = f"Round {round_num}" if round_num else "Pre-Season"
    filename = (
        f"waiver_{SEASON_YEAR}_rd{round_num:02d}_{today.isoformat()}.md"
        if round_num
        else f"waiver_{SEASON_YEAR}_preseason_{today.isoformat()}.md"
    )
    output_path = Path(output_dir) / filename

    # ── Apply AFL injury list to flag additional injured players ────────────
    import json as _json
    if afl_injury_df is not None and not afl_injury_df.empty:
        _afl_injured_ids = set(
            afl_injury_df[
                afl_injury_df["sc_id"].notna()
                & afl_injury_df["match_confidence"].isin(
                    ["exact", "exact+team", "fuzzy", "fuzzy+team"]
                )
            ]["sc_id"].astype(int).tolist()
        )
        id_col = "sc_id" if "sc_id" in metrics_df.columns else ("id" if "id" in metrics_df.columns else None)
        if id_col:
            _in_afl = metrics_df[id_col].apply(
                lambda x: int(x) in _afl_injured_ids if pd.notna(x) else False
            )
            metrics_df = metrics_df.copy()
            metrics_df["is_injured"] = metrics_df["is_injured"] | _in_afl
            n_extra = (_in_afl & ~metrics_df["is_injured"].shift(fill_value=False)).sum()
            log.info("AFL injury list: flagged %d additional injured players", int(_in_afl.sum()))

    n_starters = sum(SLOTS.values())  # 19
    is_bye_round = round_num in BYE_ROUNDS if round_num else False
    scoring_note = (
        f"> ⚠️ **Bye Round {round_num}** — only **16 of {n_starters}** starters score this week."
        if is_bye_round
        else f"> Scoring rule: all **{n_starters} starters** score each round — emergencies cover non-players."
    )

    # ── Pre-compute LLM blurbs (fails gracefully when API key not set) ────────
    _cc_fa = metrics_df[
        metrics_df["is_free_agent"]
        & ~(metrics_df["is_injured"] | metrics_df["is_suspended"])
        & _active_2026_mask(metrics_df)
    ].sort_values(["ceiling_index", "flex_bonus"], ascending=[False, False]).head(5)

    # Blurb context mirrors the section's pool and ranking (calibrated P(80+)
    # x availability) so the narrator talks about players actually listed.
    _pt_fa = metrics_df[
        metrics_df["is_free_agent"]
        & ~(metrics_df["is_injured"] | metrics_df["is_suspended"])
        & _active_2026_mask(metrics_df)
    ]
    _pt_df = (
        _apply_positional_sort(_pt_fa, _current_round(metrics_df))
        .sort_values("sort_score", ascending=False)
        .head(10)
        if not _pt_fa.empty
        else _pt_fa
    )

    blurb_targets = llm_narrator.generate_section_blurb(
        "positional_targets", _pt_df, round_num=round_num
    )
    blurb_ceiling = llm_narrator.generate_section_blurb(
        "ceiling_chasers", _cc_fa, round_num=round_num
    )
    blurb_movers = llm_narrator.generate_section_blurb(
        "mid_year_movers",
        jumpers_df if jumpers_df is not None and not jumpers_df.empty else pd.DataFrame(),
        round_num=round_num,
    )
    blurb_streamers = llm_narrator.generate_section_blurb(
        "weekly_streamers",
        streamers_df if streamers_df is not None and not streamers_df.empty else pd.DataFrame(),
        round_num=round_num,
    )

    no_2026_data = (
        metrics_df["rounds_played_2026"].max() < 1
        if "rounds_played_2026" in metrics_df.columns
        else True
    )
    data_banner = (
        "\n> 📅 **Pre-season mode** — 2026 rounds not yet played. "
        "All metrics based on weighted historical data (2011–2025).\n"
        if no_2026_data
        else ""
    )

    warnings_block = ""
    if warnings:
        warning_lines = "\n".join(f"> ⚠️ {w}" for w in warnings)
        warnings_block = f"\n{warning_lines}\n"

    # Prominent schedule-change notice — shown above the title so it's impossible to miss
    from waiver.fixture_schedule import BAT_PATH
    bat_notice = ""
    if schedule_updated:
        bat_path_str = str(BAT_PATH).replace("/", "\\")
        bat_notice = (
            "\n> ## 🔔 ACTION REQUIRED — Task Schedule Updated\n"
            "> The AFL fixture has changed. Re-run the bat file as **Administrator** "
            "to re-register the newsletter schedule:\n>\n"
            f"> **`{bat_path_str}`**\n>\n"
            "> *(Right-click → Run as administrator)*\n"
        )

    pred_label = f" · Projections: Round {pred_round}" if pred_round else ""
    run_time = now.strftime("%A, %d %B %Y at %I:%M %p AEST")
    lines = [
        bat_notice,
        f"# Waiver Wire Newsletter — {round_label}, {SEASON_YEAR}",
        f"*Generated: {run_time} | Next waiver deadline: **Tuesday 4:00 AM AEST**{pred_label}*",
        "",
        scoring_note,
        _scarcity_line(scarcity_dict),
        data_banner,
        warnings_block,
        "---",
        "",
        "## Last Round Results",
        f"*How did Round {round_num - 1 if round_num else '?'} newsletter picks score in the following week?*",
        "",
        _section_previous_performance(round_num) if round_num else "*Pre-season — no previous picks.*\n",
        "---",
        "",
        "## Positional Targets",
        "*Top 10 picks at each position (DEF / MID / FWD / RUC) — ordered by sustained "
        "value over the next 3–5 rounds, weighted by recent form once the season starts.*",
        "",
        _section_positional_targets(metrics_df, sentiment_map=sentiment_map, dvp_ranks=dvp_ranks),
        f"\n{blurb_targets}\n" if blurb_targets else "",
        "---",
        "",
        "## 🚀 The Ceiling Chasers",
        "*Top 5 available (healthy) free agents by Ceiling Index — the frequency of 120+ games. "
        f"High Ceiling Index + high Flex Bonus = best FLEX/speculative pickup.*",
        "",
        _section_ceiling_chasers(metrics_df),
        f"\n{blurb_ceiling}\n" if blurb_ceiling else "",
        "---",
        "",
        "## 🎯 Matchup Intel",
        "*Free agents facing the weakest defences this round — ranked by how many SC points "
        "their opponent concedes to that position (rolling last 6 rounds).*",
        "",
        _section_matchup_intel(dvp_targets_df),
        "---",
        "",
        "## 📅 Schedule Window",
        "*Free agents with the softest 3-round fixture ahead, by average DVP rank. "
        "Lower avg rank = easier run of opponents. Plan ahead of the market.*",
        "",
        _section_schedule_streamers(schedule_streamers_df),
        "---",
        "",
        "## 📈 Mid-Year Movers",
        "*Free agents currently averaging < 85 with active trajectory signals pointing to 85+ "
        "for the rest of the season. Key drivers: CBA% gaining, rising TOG, or positive SC momentum. "
        "Career history used only to confirm the player has a proven ceiling — not as a primary signal.*",
        "",
        _section_mid_year_movers(jumpers_df),
        f"\n{blurb_movers}\n" if blurb_movers else "",
        "---",
        "",
        "## ⚡ Weekly Streamers",
        "*Short-term pickups for 1-3 weeks based on sudden role or score makeup changes. "
        "Pick up before the waiver deadline — these signals may not sustain long-term.*",
        "",
        _section_weekly_streamers(streamers_df),
        f"\n{blurb_streamers}\n" if blurb_streamers else "",
        "---",
        "",
        "## Injury & Suspension Report",
        "*Your roster players with active injuries or suspensions. "
        "On-field starters listed first — these need immediate waiver action.*",
        "",
        _section_injury_report(metrics_df, sentiment_map=sentiment_map),
        "---",
        "",
        "## Injury Watchlist — Free Agents",
        "*Injured free agents to monitor for return to play. Not recommended this week "
        "but worth targeting once their return timeline clears. Sorted by season average.*",
        "",
        _section_injury_watchlist(afl_injury_df, metrics_df),
        "---",
        "",
        "## My Team",
        "*Full roster ranked best to worst by TPOR. TPOR < 0 means a free agent outscores "
        "them at their position.*",
        "",
        _section_my_team(metrics_df),
        "---",
        "",
        f"*Powered by sc-player-data · {SEASON_YEAR} Season*",
    ]

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    log.info("Newsletter written to %s", output_path)
    return output_path
