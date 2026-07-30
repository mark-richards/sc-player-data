"""
Historical backtest of newsletter Positional Targets, 2013-2026.

For every H&A round R with a successor R' in the same season, simulates the
top-10 DEF/MID/FWD free-agent lists that the newsletter logic would have
produced using only data through round R, then scores them against R' results.

FA pool simulation (per user spec): rank all recently-active players by their
average SC over their past 20 games (spanning seasons); ranks 1-200 = owned,
the rest are free agents.

Metrics per strategy (DNPs excluded from denominators):
  hit       = listed player plays R' and scores 80+
  bullseye  = listed player plays R' and scores 100+
  miss      = simulated FA who plays R', scores 100+, is DEF/MID/FWD-eligible,
              and appears on none of their eligible lists
  dnp_rate  = listed players who did not play R' (injury-pollution diagnostic)

Dual-position players appear on every list they are eligible for; headline
metrics dedupe them so a player never counts twice.

Usage:
  py experiments/backtest_positional_targets.py [--start-year 2013]
     [--top-n 10] [--fa-cutoff 200] [--exclude-2020] [--with-dvp]
"""
import argparse
import re
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as _sc_norm

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waiver.metrics import calc_bayesian_avg  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
SC_LIST_CSV = ROOT / "draft_prep" / "SC 2026" / "combined_SC_Player_lists.csv"
PRED_2026_GLOB = "2026_round_*_predictions.csv"
OUT_DIR = Path(__file__).parent / "results"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SIGMA = 22          # Gaussian score spread (matches waiver/newsletter.py)
EVAL_POSITIONS = ("DEF", "MID", "FWD")
FF_POS_MAP = {"Back": "DEF", "Centre": "MID", "Midfielder": "MID",
              "Forward": "FWD", "Ruck": "RUC"}
FF_POS_MIN_SHARE = 0.25  # bucket eligible if recorded in >=25% of season games

USECOLS = ["Fanfooty Match ID", "Round", "Year", "Player ID", "First Name",
           "Surname", "Team", "SC", "Position"]


# ── Data loading ────────────────────────────────────────────────────────────

def load_all_rounds() -> pd.DataFrame:
    """All H&A fanfooty rounds 2011+ as one long df sorted by (Year, round)."""
    frames = []
    for path in sorted(PROCESSED_DIR.glob("*_round_*_fanfooty_data.csv")):
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in USECOLS if c in header]
        df = pd.read_csv(path, usecols=cols, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Round"].astype(str).str.match(r"^R\d+$")].copy()
    df["round_num"] = df["Round"].str[1:].astype(int)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["SC"] = pd.to_numeric(df["SC"], errors="coerce")
    df["Player ID"] = pd.to_numeric(df["Player ID"], errors="coerce")
    df = df.dropna(subset=["Year", "SC", "Player ID"])
    df["Year"] = df["Year"].astype(int)
    df["Player ID"] = df["Player ID"].astype(int)
    df = df[df["SC"] > 0]  # rows with SC==0 are non-participants (subs/emergencies)
    return df.sort_values(["Year", "round_num"]).reset_index(drop=True)


def load_sc_positions() -> dict[int, dict[int, frozenset]]:
    """{year: {feed_id: frozenset of position buckets}} from SC lists (2022+)."""
    df = pd.read_csv(SC_LIST_CSV, low_memory=False)
    out: dict[int, dict[int, frozenset]] = {}
    for (year, fid), pos in zip(
        zip(df["season_year"], df["feed_id"]), df["position"]
    ):
        buckets = frozenset(str(pos).split()) & {"DEF", "MID", "RUC", "FWD"}
        if buckets:
            out.setdefault(int(year), {})[int(fid)] = buckets
    return out


def load_model_predictions() -> dict[tuple[int, int, int], float]:
    """{(year, round, feed_id): predicted score} — 2026 live CSVs only.

    The 2021-2025 OOF file was rejected: it only contains rows for players who
    played the target round (a play-next-week oracle) and produced an
    implausible 88.7% hit rate in validation — look-ahead contamination.
    The 2026 per-round CSVs were generated before each round, so they're clean.
    """
    preds: dict[tuple[int, int, int], float] = {}
    for path in sorted((ROOT / "data" / "predictions").glob(PRED_2026_GLOB)):
        df = pd.read_csv(path, usecols=["Player ID", "projected_score", "pred_round"])
        for pid, p, rnd in zip(df["Player ID"], df["projected_score"], df["pred_round"]):
            if pd.notna(pid) and pd.notna(p):
                preds[(2026, int(rnd), int(pid))] = float(p)
    return preds


# ── As-of player state ──────────────────────────────────────────────────────

class PlayerState:
    __slots__ = ("last20", "season_scores", "last_round", "year",
                 "career_sum", "career_n", "prior_avg", "pos_cur", "pos_prev",
                 "name", "team")

    def __init__(self):
        self.last20 = deque(maxlen=20)
        self.season_scores: list[float] = []
        self.last_round = 0
        self.year = 0
        self.career_sum = 0.0
        self.career_n = 0
        self.prior_avg = 0.0     # avg20 frozen at the start of the current season
        self.pos_cur: Counter = Counter()   # fanfooty positions this season
        self.pos_prev: Counter = Counter()  # previous season
        self.name = ""
        self.team = ""

    def roll_season(self, year: int):
        self.prior_avg = (sum(self.last20) / len(self.last20)) if self.last20 else 0.0
        self.season_scores = []
        self.last_round = 0
        self.pos_prev = self.pos_cur
        self.pos_cur = Counter()
        self.year = year

    def ingest(self, year: int, rnd: int, sc: float, position, name: str, team: str):
        if year != self.year:
            self.roll_season(year)
        self.last20.append(sc)
        self.season_scores.append(sc)
        self.last_round = rnd
        self.career_sum += sc
        self.career_n += 1
        self.name = name
        self.team = team
        if isinstance(position, str):
            for word in position.split():
                bucket = FF_POS_MAP.get(word)
                if bucket:
                    self.pos_cur[bucket] += 1

    def fanfooty_buckets(self) -> frozenset:
        """Eligibility from observed field positions (pre-2022 fallback)."""
        counts = self.pos_cur if sum(self.pos_cur.values()) >= 3 else self.pos_cur + self.pos_prev
        total = sum(counts.values())
        if not total:
            return frozenset()
        return frozenset(b for b, n in counts.items() if n / total >= FF_POS_MIN_SHARE or n == total)


def snapshot(states: dict[int, PlayerState], year: int) -> pd.DataFrame:
    """Materialise per-player as-of features for simulating one round.

    Limited to players seen within the last 2 seasons so retired players'
    stale averages don't linger in the ownership ranking or miss pool.
    """
    rows = []
    for pid, st in states.items():
        if not st.last20 or st.year < year - 2:
            continue
        in_season = st.year == year
        scores = st.season_scores if in_season else []
        n = len(scores)
        avg20 = sum(st.last20) / len(st.last20)
        rows.append({
            "player_id": pid,
            "name": st.name,
            "team": st.team,
            "avg20": avg20,
            "games_20": len(st.last20),
            "avg5": sum(scores[-5:]) / min(n, 5) if n else 0.0,
            "avg3": sum(scores[-3:]) / min(n, 3) if n else 0.0,
            "season_avg": sum(scores) / n if n else 0.0,
            "games_season": n,
            "last_round": st.last_round if in_season else 0,
            "p80_freq20": sum(1 for s in st.last20 if s >= 80) / len(st.last20),
            "p100_freq20": sum(1 for s in st.last20 if s >= 100) / len(st.last20),
            "career_avg": st.career_sum / st.career_n,
            "prior_avg": st.prior_avg if in_season else avg20,
            "state_year": st.year,
            "_season_scores": scores,
        })
    return pd.DataFrame(rows)


# ── Ranking strategies ──────────────────────────────────────────────────────

def _gauss_p80(mean: float) -> float:
    return float(_sc_norm.sf(80, loc=mean, scale=SIGMA) * 100)


def strat_newsletter_current(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Replica of the live newsletter's calibrated P(80+) + form penalties.

    Mean ladder mirrors waiver/newsletter.py _calibrated_p80: avg5 (>=5 games)
    -> partial-season avg (>=1 game) -> career avg. Penalties mirror the
    underperformer sink (-1000) and form-decay demotion (-10).
    """
    mean = np.where(df["games_season"] >= 5, df["avg5"],
                    np.where(df["games_season"] >= 1, df["season_avg"], df["career_avg"]))
    score = pd.Series([_gauss_p80(m) for m in mean], index=df.index)
    underperformer = (df["games_season"] >= 5) & (df["avg5"] < 55)
    score = score.where(~underperformer, -1000.0)
    form_decay = (df["games_season"] >= 6) & (df["season_avg"] < 70)
    score = score - form_decay.astype(float) * 10.0
    return score


def strat_avg20(df: pd.DataFrame, ctx: dict) -> pd.Series:
    return df["avg20"]


def strat_avg5(df: pd.DataFrame, ctx: dict) -> pd.Series:
    return df["avg5"].where(df["games_season"] >= 1, df["career_avg"])


def strat_avg3(df: pd.DataFrame, ctx: dict) -> pd.Series:
    return df["avg3"].where(df["games_season"] >= 1, df["career_avg"])


def strat_ceiling_p100(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Frequency of 100+ scores in last 20 games; avg20 breaks ties."""
    return df["p100_freq20"] * 1000 + df["avg20"]


def strat_bayes_p80(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Gaussian P(80+) on a Bayesian shrinkage mean (season blended with prior)."""
    means = [
        calc_bayesian_avg(r["_season_scores"], r["prior_avg"] or r["career_avg"])
        or r["career_avg"]
        for _, r in df.iterrows()
    ]
    return pd.Series([_gauss_p80(m) for m in means], index=df.index)


def strat_model(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """LightGBM predicted score for the target round (OOF 2021-25 / live 2026)."""
    preds, year, target = ctx["model_preds"], ctx["year"], ctx["target_round"]
    vals = df["player_id"].map(lambda pid: preds.get((year, target, pid), np.nan))
    return vals


def strat_blend_p80(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Gaussian P(80+) on a recency/stability blend: 0.6*avg5 + 0.4*avg20.

    Same form penalties as newsletter_current so only the mean changes.
    """
    blend = 0.6 * df["avg5"] + 0.4 * df["avg20"]
    mean = np.where(df["games_season"] >= 5, blend,
                    np.where(df["games_season"] >= 1,
                             0.6 * df["season_avg"] + 0.4 * df["avg20"],
                             df["career_avg"]))
    score = pd.Series([_gauss_p80(m) for m in mean], index=df.index)
    underperformer = (df["games_season"] >= 5) & (df["avg5"] < 55)
    score = score.where(~underperformer, -1000.0)
    form_decay = (df["games_season"] >= 6) & (df["season_avg"] < 70)
    score = score - form_decay.astype(float) * 10.0
    return score


def strat_avail_p80(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """newsletter_current discounted by recent availability.

    Players who missed the most recent round(s) get discounted — the cheap
    proxy for 'is this player actually going to be on the park next week'.
    """
    base = strat_newsletter_current(df, ctx)
    rnd = ctx["round"]
    gap = (rnd - df["last_round"]).clip(lower=0)
    factor = gap.map({0: 1.0, 1: 0.8, 2: 0.6}).fillna(0.5)
    return base.where(base <= 0, base * factor)


def strat_blend_avail_p80(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """blend_p80 mean + availability discount — the combined candidate."""
    base = strat_blend_p80(df, ctx)
    rnd = ctx["round"]
    gap = (rnd - df["last_round"]).clip(lower=0)
    factor = gap.map({0: 1.0, 1: 0.8, 2: 0.6}).fillna(0.5)
    return base.where(base <= 0, base * factor)


STRATEGIES = {
    "newsletter_current": strat_newsletter_current,
    "avg20": strat_avg20,
    "avg5": strat_avg5,
    "avg3": strat_avg3,
    "ceiling_p100": strat_ceiling_p100,
    "bayes_p80": strat_bayes_p80,
    "model": strat_model,
    "blend_p80": strat_blend_p80,
    "avail_p80": strat_avail_p80,
    "blend_avail_p80": strat_blend_avail_p80,
}


# ── DVP (optional) ──────────────────────────────────────────────────────────

class DvpTracker:
    """Season-to-date average SC conceded by each team to each position bucket."""

    def __init__(self):
        self.year = 0
        self.conceded: dict[tuple[str, str], list] = {}  # (team, bucket) -> [sum, n]

    def ingest_round(self, round_df: pd.DataFrame, eligibility: dict[int, frozenset]):
        year = int(round_df["Year"].iloc[0])
        if year != self.year:
            self.conceded = {}
            self.year = year
        # opponent = the other team sharing the Fanfooty Match ID
        match_teams = round_df.groupby("Fanfooty Match ID")["Team"].unique()
        opp_map = {}
        for mid, teams in match_teams.items():
            if len(teams) == 2:
                opp_map[(mid, teams[0])] = teams[1]
                opp_map[(mid, teams[1])] = teams[0]
        for _, r in round_df.iterrows():
            opp = opp_map.get((r["Fanfooty Match ID"], r["Team"]))
            if not opp:
                continue
            for bucket in eligibility.get(int(r["Player ID"]), frozenset()):
                key = (opp, bucket)
                acc = self.conceded.setdefault(key, [0.0, 0])
                acc[0] += r["SC"]
                acc[1] += 1

    def bonus(self, opp_team: str, bucket: str) -> float:
        """+3 / +1.5 / -1.5 / -3 by rank of opponent's concession to bucket."""
        rows = [(t, s / n) for (t, b), (s, n) in self.conceded.items()
                if b == bucket and n >= 10]
        if len(rows) < 12:
            return 0.0
        rows.sort(key=lambda x: -x[1])  # rank 1 = concedes most = softest
        ranks = {t: i + 1 for i, (t, _) in enumerate(rows)}
        rank = ranks.get(opp_team)
        if rank is None:
            return 0.0
        if rank <= 3:
            return 3.0
        if rank <= 6:
            return 1.5
        if rank >= len(rows) - 2:
            return -3.0
        if rank >= len(rows) - 5:
            return -1.5
        return 0.0


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_strategy(
    lists: dict[str, list[int]],
    outcome: dict[int, float],
    miss_pool: set[int],
    rankable_ids: set[int],
) -> dict:
    union = set()
    for ids in lists.values():
        union.update(ids)
    played = {pid: outcome[pid] for pid in union if pid in outcome}
    hits = sum(1 for s in played.values() if s >= 80)
    bulls = sum(1 for s in played.values() if s >= 100)
    missed_ids = [pid for pid in miss_pool
                  if pid not in union and outcome.get(pid, 0) >= 100]
    misses = len(missed_ids)
    # was the missed player rankable (in the strategy's candidate pool) or
    # excluded upstream by the activity mask / missing score?
    misses_rankable = sum(1 for pid in missed_ids if pid in rankable_ids)
    per_pos = {}
    for pos, ids in lists.items():
        p = [outcome[pid] for pid in ids if pid in outcome]
        per_pos[pos] = {
            "listed": len(ids),
            "played": len(p),
            "hits": sum(1 for s in p if s >= 80),
            "bulls": sum(1 for s in p if s >= 100),
        }
    return {
        "listed": len(union),
        "played": len(played),
        "dnp": len(union) - len(played),
        "hits": hits,
        "bullseyes": bulls,
        "misses": misses,
        "misses_rankable": misses_rankable,
        "misses_masked": misses - misses_rankable,
        "per_pos": per_pos,
    }


def build_lists(
    eligible: pd.DataFrame,
    scores: pd.Series,
    eligibility: dict[int, frozenset],
    top_n: int,
    dvp_bonus: dict[int, dict[str, float]] | None = None,
) -> dict[str, list[int]]:
    """Top-N ids per position; dual-position players enter every eligible list."""
    lists: dict[str, list[int]] = {}
    ranked = eligible.assign(_s=scores).dropna(subset=["_s"])
    for pos in EVAL_POSITIONS:
        mask = ranked["player_id"].map(lambda pid: pos in eligibility.get(pid, frozenset()))
        pos_df = ranked[mask]
        if dvp_bonus:
            adj = pos_df["_s"] + pos_df["player_id"].map(
                lambda pid: dvp_bonus.get(pid, {}).get(pos, 0.0)
            )
            pos_df = pos_df.assign(_s=adj)
        lists[pos] = pos_df.nlargest(top_n, "_s")["player_id"].tolist()
    return lists


# ── Main loop ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2013)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--fa-cutoff", type=int, default=200)
    ap.add_argument("--exclude-2020", action="store_true")
    ap.add_argument("--with-dvp", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Loading fanfooty rounds...")
    all_df = load_all_rounds()
    print(f"  {len(all_df):,} player-rounds, {all_df['Year'].min()}-{all_df['Year'].max()}")
    sc_positions = load_sc_positions()
    model_preds = load_model_predictions()
    print(f"  {len(model_preds):,} model predictions loaded")

    states: dict[int, PlayerState] = {}
    dvp = DvpTracker() if args.with_dvp else None
    results = []

    round_groups = list(all_df.groupby(["Year", "round_num"], sort=True))
    rounds_by_year: dict[int, list[int]] = {}
    for (year, rnd), _ in round_groups:
        rounds_by_year.setdefault(year, []).append(rnd)

    for (year, rnd), round_df in round_groups:
        # eligibility as of this round (used for both DVP accrual and lists)
        def get_eligibility() -> dict[int, frozenset]:
            elig = {}
            sc_year = sc_positions.get(year, {})
            for pid, st in states.items():
                if st.year < year - 2:
                    continue
                buckets = sc_year.get(pid) or st.fanfooty_buckets()
                if buckets:
                    elig[pid] = buckets
            return elig

        # 1. ingest round R into state
        for _, r in round_df.iterrows():
            st = states.setdefault(int(r["Player ID"]), PlayerState())
            st.ingest(year, rnd, float(r["SC"]),
                      r.get("Position"), f"{r['First Name']} {r['Surname']}", r["Team"])
        if dvp is not None:
            dvp.ingest_round(round_df, get_eligibility())

        # 2. simulate for R+1 if it exists in the same season
        year_rounds = rounds_by_year[year]
        idx = year_rounds.index(rnd)
        if idx + 1 >= len(year_rounds):
            continue
        target = year_rounds[idx + 1]
        if year < args.start_year or (args.exclude_2020 and year == 2020):
            continue

        snap = snapshot(states, year)

        # FA pool: top fa-cutoff by avg20 are owned. Ownership candidates are
        # players active this season, or last season with a full 20-game history
        # (injured stars stay "owned"; long-retired players do not).
        pool = snap[
            (snap["games_season"] > 0)
            | ((snap["state_year"] >= year - 1) & (snap["games_20"] >= 20))
        ]
        owned = set(pool.nlargest(args.fa_cutoff, "avg20")["player_id"])
        snap["is_fa"] = ~snap["player_id"].isin(owned)

        eligibility = get_eligibility()

        # baseline list-eligibility mask (mirrors live newsletter activity mask)
        eligible = snap[
            snap["is_fa"]
            & (snap["games_season"] >= 2)
            & (snap["last_round"] >= rnd - 2)
        ].copy()

        # outcome + miss pool
        target_df = all_df[(all_df["Year"] == year) & (all_df["round_num"] == target)]
        outcome = dict(zip(target_df["Player ID"].astype(int), target_df["SC"].astype(float)))
        miss_pool = {
            pid for pid in snap[snap["is_fa"]]["player_id"]
            if eligibility.get(pid, frozenset()) & set(EVAL_POSITIONS)
        }

        # DVP bonus per player per bucket for the target round
        dvp_bonus = None
        if dvp is not None:
            team_opp = {}
            match_teams = target_df.groupby("Fanfooty Match ID")["Team"].unique()
            for mid, teams in match_teams.items():
                if len(teams) == 2:
                    team_opp[teams[0]], team_opp[teams[1]] = teams[1], teams[0]
            dvp_bonus = {}
            for _, r in eligible.iterrows():
                opp = team_opp.get(r["team"])
                if opp:
                    dvp_bonus[r["player_id"]] = {
                        pos: dvp.bonus(opp, pos) for pos in EVAL_POSITIONS
                    }

        ctx = {"year": year, "round": rnd, "target_round": target,
               "model_preds": model_preds}

        for name, fn in STRATEGIES.items():
            scores = fn(eligible, ctx)
            if scores.dropna().empty:
                continue
            lists = build_lists(
                eligible, scores, eligibility, args.top_n,
                dvp_bonus=dvp_bonus if name == "newsletter_current" else None,
            )
            rankable = set(eligible.assign(_s=scores).dropna(subset=["_s"])["player_id"])
            res = evaluate_strategy(lists, outcome, miss_pool, rankable)
            row = {
                "year": year, "round": rnd, "target_round": target,
                "strategy": name,
                "listed": res["listed"], "played": res["played"], "dnp": res["dnp"],
                "hits": res["hits"], "bullseyes": res["bullseyes"],
                "misses": res["misses"],
                "misses_rankable": res["misses_rankable"],
                "misses_masked": res["misses_masked"],
            }
            for pos in EVAL_POSITIONS:
                pp = res["per_pos"][pos]
                row[f"{pos.lower()}_played"] = pp["played"]
                row[f"{pos.lower()}_hits"] = pp["hits"]
                row[f"{pos.lower()}_bulls"] = pp["bulls"]
            results.append(row)

        if rnd == year_rounds[len(year_rounds) // 2]:
            print(f"  {year} R{rnd} done ({len(eligible)} eligible FAs)")

    res_df = pd.DataFrame(results)
    csv_path = OUT_DIR / "positional_backtest_rounds.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nRow-level results: {csv_path} ({len(res_df)} rows)")

    write_report(res_df, args)


# ── Reporting ───────────────────────────────────────────────────────────────

def agg_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("strategy").agg(
        rounds=("round", "count"),
        listed=("listed", "sum"), played=("played", "sum"), dnp=("dnp", "sum"),
        hits=("hits", "sum"), bullseyes=("bullseyes", "sum"), misses=("misses", "sum"),
    )
    g["hit_rate"] = g["hits"] / g["played"] * 100
    g["bullseye_rate"] = g["bullseyes"] / g["played"] * 100
    g["miss_rate"] = g["misses"] / (g["misses"] + g["bullseyes"]) * 100
    g["dnp_rate"] = g["dnp"] / g["listed"] * 100
    return g.sort_values("hit_rate", ascending=False)


def write_report(res_df: pd.DataFrame, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = agg_table(res_df)
    print("\n=== OVERALL (all years) ===")
    print(overall[["rounds", "played", "hit_rate", "bullseye_rate", "miss_rate", "dnp_rate"]]
          .round(1).to_string())

    # figures
    years = sorted(res_df["year"].unique())
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for strat in res_df["strategy"].unique():
        sub = res_df[res_df["strategy"] == strat]
        by_year = sub.groupby("year").apply(
            lambda d: d["hits"].sum() / max(d["played"].sum(), 1) * 100,
            include_groups=False,
        )
        ax.plot(by_year.index, by_year.values, marker="o", ms=3, label=strat)
    ax.set_xlabel("Season")
    ax.set_ylabel("Hit rate % (80+ | played)")
    ax.set_title("Positional targets hit rate by season and strategy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "positional_backtest_hit_rate_by_year.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=False)
    for ax, (col, title) in zip(axes, [
        ("hit_rate", "Hit rate % (80+)"), ("bullseye_rate", "Bullseye rate % (100+)"),
        ("miss_rate", "Miss rate % (lower=better)"), ("dnp_rate", "DNP rate % (lower=better)"),
    ]):
        vals = overall[col].sort_values(ascending=(col in ("miss_rate", "dnp_rate")))
        ax.barh(vals.index, vals.values)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Strategy comparison, all seasons", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "positional_backtest_summary.png", dpi=130)
    plt.close(fig)

    # precision@N by position heatmap
    pos_rows = []
    for strat, sub in res_df.groupby("strategy"):
        for pos in EVAL_POSITIONS:
            played = sub[f"{pos.lower()}_played"].sum()
            hits = sub[f"{pos.lower()}_hits"].sum()
            pos_rows.append({"strategy": strat, "pos": pos,
                             "hit_rate": hits / played * 100 if played else 0})
    pos_df = pd.DataFrame(pos_rows).pivot(index="strategy", columns="pos", values="hit_rate")
    pos_df = pos_df[list(EVAL_POSITIONS)].loc[overall.index]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pos_df.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pos_df.columns)), pos_df.columns)
    ax.set_yticks(range(len(pos_df.index)), pos_df.index, fontsize=8)
    for i in range(pos_df.shape[0]):
        for j in range(pos_df.shape[1]):
            ax.text(j, i, f"{pos_df.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Hit rate % by position (top-{args.top_n} lists)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "positional_backtest_precision_by_pos.png", dpi=130)
    plt.close(fig)

    # markdown report
    lines = [
        f"# Positional Targets Backtest — {res_df['year'].min()}–{res_df['year'].max()}",
        "",
        f"Simulated top-{args.top_n} DEF/MID/FWD free-agent lists for every H&A round "
        f"with a same-season successor. FA pool = players outside the top "
        f"{args.fa_cutoff} ranked by past-20-game SC average. Dual-position players "
        "appear on every eligible list (deduped in headline metrics). DNPs excluded "
        "from hit/bullseye denominators.",
        "",
        "**Definitions**: hit = listed player scores 80+ next round; bullseye = 100+; "
        "miss = an available (simulated-FA) DEF/MID/FWD player scoring 100+ next round "
        "who was on none of their eligible lists. Miss rate = misses / (misses + bullseyes).",
        "",
        "## Overall results",
        "",
        "| Strategy | Rounds | Played | Hit rate | Bullseye rate | Miss rate | DNP rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for strat, r in overall.iterrows():
        lines.append(
            f"| {strat} | {int(r['rounds'])} | {int(r['played'])} | "
            f"{r['hit_rate']:.1f}% | {r['bullseye_rate']:.1f}% | "
            f"{r['miss_rate']:.1f}% | {r['dnp_rate']:.1f}% |"
        )

    lines += ["", "## Per-year hit rates", ""]
    year_pivot = res_df.groupby(["strategy", "year"]).apply(
        lambda d: d["hits"].sum() / max(d["played"].sum(), 1) * 100,
        include_groups=False,
    ).unstack()
    lines.append("| Strategy | " + " | ".join(str(y) for y in year_pivot.columns) + " |")
    lines.append("| --- |" + " --- |" * len(year_pivot.columns))
    for strat, row in year_pivot.loc[overall.index].iterrows():
        lines.append(f"| {strat} | " + " | ".join(
            f"{v:.0f}%" if pd.notna(v) else "—" for v in row) + " |")

    lines += [
        "",
        "## Figures",
        "",
        "![Hit rate by year](figures/positional_backtest_hit_rate_by_year.png)",
        "![Summary](figures/positional_backtest_summary.png)",
        "![Precision by position](figures/positional_backtest_precision_by_pos.png)",
        "",
        "## Caveats",
        "",
        "- FA pool is simulated (top-200 by past-20 avg), not real league ownership.",
        "- 2013–2021 position eligibility approximated from fanfooty field positions "
        "(>=25% of season games); 2022+ uses true SC dual positions (pre-season snapshot).",
        "- `model` covers 2026 only (live pre-round prediction CSVs). The 2021-2025 "
        "OOF file was rejected for look-ahead contamination (implausible 88.7% hit rate; "
        "it also only contains players who played the target round).",
        "- 2020 shortened quarters depress scores; see per-year table"
        + (" (excluded via --exclude-2020)." if args.exclude_2020 else "."),
    ]

    md_path = REPORTS_DIR / "backtest_positional_targets_2013_2026.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
