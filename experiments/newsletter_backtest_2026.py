"""
Walk-forward backtest of newsletter positional ranking, Rounds 4–13 (2026).

For each round N, simulates what the newsletter would have picked using only
data available up to round N, then scores those picks against round N+1 actual
scores from the SC API JSON.

Approaches compared:
  - FA base rate: random FA selection
  - Avg3 top-20: top 20 FAs across all positions by SC API avg3
  - New newsletter: top 5 per position by avg3 (current live logic)
  - Old newsletter: top 5 per position by observed season P(80+) rate
  - Avg5 variant: top 5 per position by last-5 avg
  - Model newsletter: top 5 per position by LightGBM projected_score

FA pool: current_teams.csv used as proxy for all rounds (best available
approximation — some early-season picks will be slightly off).

Outputs per-round markdown files to experiments/results/ and a summary CSV.
"""
import json
import re
import csv
import sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import norm as _sc_norm
import pandas as pd

# Ensure project root is on sys.path for feature_engineering / train_model_v2 imports
_ROOT_STR = str(Path(__file__).parent.parent)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
JSON_DIR = ROOT / "data/raw/supercoach/2026"
PLAYER_LIST = ROOT / "draft_prep/SC 2026/2026_SC_Player_list.csv"
TEAMS_CSV = ROOT / "data/live/current_teams.csv"
NL_DIR = ROOT / "reports"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

ROUNDS = list(range(4, 14))   # simulate Rd4..Rd13, score against Rd5..Rd14
TOP_N_PER_POS = 5
AVG3_BASELINE_N = 20
SIGMA = 22  # Gaussian score spread for P(80+) calculation

# ── Load player metadata ────────────────────────────────────────────────────
pl = pd.read_csv(PLAYER_LIST, low_memory=False)
pl["full_name"] = (
    pl["first_name"].str.strip() + " " + pl["last_name"].str.strip()
).str.strip()

# Build feed_id → SC draft player_id mapping from round-14 JSON (most complete)
with open(JSON_DIR / "players_round_14.json") as f:
    _all_players = json.load(f)

feed_to_pid: dict[str, int] = {
    str(p["feed_id"]): int(p["id"])
    for p in _all_players
    if p.get("feed_id")
}
pid_to_feed: dict[int, str] = {v: k for k, v in feed_to_pid.items()}

# Position lookup: SC player id → position bucket
pid_to_pos: dict[int, str] = {}
for _, row in pl.iterrows():
    if pd.isna(row.get("feed_id")) or pd.isna(row.get("position")):
        continue
    fid = str(int(float(row["feed_id"])))
    pid = feed_to_pid.get(fid)
    if pid:
        pos = str(row["position"])
        pid_to_pos[pid] = "FWD" if pos == "FLEX" else pos

# Name lookup: SC player id → full name
pid_to_name: dict[int, str] = {}
for _, row in pl.iterrows():
    if pd.isna(row.get("feed_id")):
        continue
    fid = str(int(float(row["feed_id"])))
    pid = feed_to_pid.get(fid)
    if pid:
        pid_to_name[pid] = row["full_name"]

# ── FA pool (current ownership as proxy) ───────────────────────────────────
teams_df = pd.read_csv(TEAMS_CSV)
owned_pids: set[int] = set()
for _, row in teams_df.iterrows():
    if pd.isna(row.get("feed_id")):
        continue
    fid = str(int(float(row["feed_id"])))
    pid = feed_to_pid.get(fid)
    if pid:
        owned_pids.add(pid)

all_pids = {int(p["id"]) for p in _all_players}
fa_ids = all_pids - owned_pids


# ── Data loading helpers ────────────────────────────────────────────────────
def load_round_json(rnd: int) -> pd.DataFrame:
    """Load players_round_N.json → DataFrame with player_id, points, games, avg3."""
    path = JSON_DIR / f"players_round_{rnd}.json"
    with open(path) as f:
        data = json.load(f)
    rows = []
    for p in data:
        if not p.get("player_stats"):
            continue
        stat = p["player_stats"][0]
        rows.append({
            "player_id": int(p["id"]),
            "name": f"{p['first_name']} {p['last_name']}",
            "points": float(stat.get("points", 0) or 0),
            "games": int(stat.get("games", 0) or 0),
            "avg3": float(stat.get("avg3", 0) or 0),
            "avg5": float(stat.get("avg5", 0) or 0),
            "avg": float(stat.get("avg", 0) or 0),
            "total_games": int(stat.get("total_games", 0) or 0),
        })
    return pd.DataFrame(rows)


def build_season_scores_up_to(max_rnd: int) -> dict[int, list[int]]:
    """
    For each player, collect all played SC scores from rounds 1..max_rnd.
    Returns {player_id: [score, score, ...]} ordered by round.
    """
    scores: dict[int, dict[int, int]] = {}
    for rnd in range(1, max_rnd + 1):
        path = JSON_DIR / f"players_round_{rnd}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for p in data:
            if not p.get("player_stats"):
                continue
            stat = p["player_stats"][0]
            pid = int(p["id"])
            if int(stat.get("games", 0)) == 1:
                pts = int(stat.get("points", 0) or 0)
                if pts > 0:
                    scores.setdefault(pid, {})[rnd] = pts
    return {
        pid: [round_scores[r] for r in sorted(round_scores)]
        for pid, round_scores in scores.items()
    }


# ── Ranking helpers ─────────────────────────────────────────────────────────
def avg3_p80(scores: list[int]) -> float:
    """New newsletter: P(80+) based on last-3 played rounds (avg3)."""
    n = len(scores)
    if n >= 3:
        a = sum(scores[-3:]) / 3
    elif n >= 1:
        a = sum(scores) / n
    else:
        return 0.0
    return float(_sc_norm.sf(80, loc=a, scale=SIGMA) * 100)


def observed_p80(scores: list[int]) -> float:
    """CURRENT newsletter: observed season P(80+) hit rate (primary from >=3 rounds)."""
    if not scores:
        return 0.0
    n = len(scores)
    if n >= 3:
        return sum(1 for s in scores if s >= 80) / n * 100
    avg_recent = sum(scores) / n
    return float(_sc_norm.sf(80, loc=avg_recent, scale=SIGMA) * 100)


def avg5_p80(scores: list[int]) -> float:
    """Avg5 variant: P(80+) from last-5 games (more stable than avg3)."""
    n = len(scores)
    if n >= 5:
        a = sum(scores[-5:]) / 5
    elif n >= 1:
        a = sum(scores) / n
    else:
        return 0.0
    return float(_sc_norm.sf(80, loc=a, scale=SIGMA) * 100)


def top_n_per_pos(
    fa_pids: list[int],
    ranking_fn,  # callable(scores) → float, higher = better
    season_scores: dict[int, list[int]],
    n: int = TOP_N_PER_POS,
) -> list[int]:
    """Return top-n FAs per position ranked by ranking_fn, filtered to played_next."""
    by_pos: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for pid in fa_pids:
        pos = pid_to_pos.get(pid)
        if not pos:
            continue
        scores = season_scores.get(pid, [])
        score = ranking_fn(scores)
        by_pos[pos].append((score, pid))

    picks = []
    for pos in ("DEF", "MID", "FWD", "RUC"):
        ranked = sorted(by_pos[pos], reverse=True)[:n]
        picks.extend(pid for _, pid in ranked)
    return picks


def evaluate(
    pick_ids: list[int],
    next_df: pd.DataFrame,
    played_ids: set[int],
) -> dict:
    """Score pick_ids against played_ids from next_df."""
    played_picks = [pid for pid in pick_ids if pid in played_ids]
    hits = 0
    details = []
    for pid in played_picks:
        row = next_df[next_df["player_id"] == pid]
        if row.empty:
            continue
        score = float(row.iloc[0]["points"])
        hit = score >= 80
        if hit:
            hits += 1
        details.append({
            "player_id": pid,
            "name": pid_to_name.get(pid, str(pid)),
            "pos": pid_to_pos.get(pid, "?"),
            "score": score,
            "hit": hit,
        })
    return {
        "hits": hits,
        "played": len(played_picks),
        "dnp": len(pick_ids) - len(played_picks),
        "pct": hits / len(played_picks) * 100 if played_picks else 0.0,
        "details": details,
    }


# ── LightGBM model setup ────────────────────────────────────────────────────
import joblib
import numpy as np

_model_path = ROOT / "models" / "sc_lgb_model_v2.joblib"
_model_artifact = None
_model = None
_model_features = None
_cal_slope = 1.0
_cal_intercept = 0.0

try:
    _model_artifact = joblib.load(_model_path)
    _model = _model_artifact["model"]
    _model_features = _model_artifact["features"]
    _cal_slope = _model_artifact.get("cal_slope", 1.0)
    _cal_intercept = _model_artifact.get("cal_intercept", 0.0)
    print(f"Model loaded: {len(_model_features)} features, cal_slope={_cal_slope:.3f}")
except Exception as _e:
    print(f"WARNING: Could not load model — model newsletter skipped ({_e})")

# Load master dataset once (filtered per-round inside simulate fn)
_master_path = ROOT / "data" / "processed" / "master_player_data.csv"
_master_df = None
try:
    from feature_engineering import engineer_features, get_latest_player_features
    from train_model_v2 import add_extended_features
    _master_df = pd.read_csv(_master_path, low_memory=False)
    _master_df["Round_Num"] = pd.to_numeric(_master_df["Round_Num"], errors="coerce")
    _master_df = _master_df[_master_df["played"] > 0].copy()
    print(f"Master data loaded: {len(_master_df)} rows")
except Exception as _e:
    print(f"WARNING: Could not load master data or feature engineering ({_e})")

# feed_id (str) → Player ID (int) — needed to map model output back to SC draft pids
_feed_str_to_sc_pid: dict[str, int] = feed_to_pid  # already built above


def simulate_model_picks(predict_round: int) -> dict[int, float]:
    """
    Run model inference using only data available before predict_round.
    Returns {sc_draft_player_id: projected_score}.
    """
    if _model is None or _master_df is None:
        return {}
    try:
        # Filter to data strictly before the prediction round
        hist = _master_df[
            (_master_df["Year"] < 2026)
            | ((_master_df["Year"] == 2026) & (_master_df["Round_Num"] < predict_round))
        ].copy()
        if hist.empty:
            return {}

        df_feat = engineer_features(hist, for_training=False)
        df_feat = add_extended_features(df_feat, for_training=False)
        latest = get_latest_player_features(df_feat)

        # Keep only recent players
        recent = latest[latest["Year"].isin([2026, 2025])].copy()
        if recent.empty:
            return {}

        # Fill missing features with 0
        X = recent[_model_features].fillna(0)
        raw = _model.predict(X)
        proj = _cal_slope * raw + _cal_intercept

        # Map Player ID (feed_id) → SC draft player_id
        result: dict[int, float] = {}
        for feed_id_val, score in zip(recent["Player ID"].values, proj):
            fid_str = str(int(float(feed_id_val))) if not pd.isna(feed_id_val) else None
            if fid_str:
                sc_pid = _feed_str_to_sc_pid.get(fid_str)
                if sc_pid:
                    result[sc_pid] = float(score)
        return result
    except Exception as e:
        print(f"  [model sim error for predict_round={predict_round}]: {e}")
        return {}


def top_n_per_pos_model(
    fa_pids: list[int],
    model_scores: dict[int, float],
    n: int = TOP_N_PER_POS,
) -> list[int]:
    """Return top-n FAs per position ranked by model projected_score."""
    by_pos: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for pid in fa_pids:
        pos = pid_to_pos.get(pid)
        if not pos:
            continue
        score = model_scores.get(pid)
        if score is None:
            continue
        by_pos[pos].append((score, pid))

    picks = []
    for pos in ("DEF", "MID", "FWD", "RUC"):
        ranked = sorted(by_pos[pos], reverse=True)[:n]
        picks.extend(pid for _, pid in ranked)
    return picks


def parse_newsletter_picks(nl_path: Path, max_per_pos: int = 5) -> list[int]:
    """Extract top-N picks per position from an actual newsletter Positional Targets."""
    txt = nl_path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"\n## ", txt)
    pt = next((s for s in sections if "Positional Targets" in s.split("\n")[0]), "")

    pid_list = []
    pos = None
    pos_count = 0
    name_to_pid = {
        row["full_name"].lower(): feed_to_pid.get(
            str(int(float(row["feed_id"]))) if not pd.isna(row["feed_id"]) else "", -1
        )
        for _, row in pl.iterrows()
        if not pd.isna(row.get("feed_id"))
    }

    for line in pt.split("\n"):
        m = re.match(r"^### (DEF|MID|FWD|RUC)\s*$", line.strip())
        if m:
            pos = m.group(1)
            pos_count = 0
            continue
        if pos and pos_count < max_per_pos and line.startswith("|") and "**" in line:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 4:
                continue
            mn = re.search(r"\*\*(.+?)\*\*", cells[1])
            if not mn:
                continue
            raw = re.sub(r"[^\x00-\x7F]+", "", mn.group(1)).strip()
            raw = re.sub(r"\s+", " ", raw).strip()
            if len(raw) < 3:
                continue
            pid = name_to_pid.get(raw.lower())
            if pid is None:
                parts = raw.lower().split()
                cands = [v for k, v in name_to_pid.items() if k.split()[-1] == parts[-1]]
                if len(cands) == 1:
                    pid = cands[0]
            if pid and pid > 0:
                pid_list.append(pid)
                pos_count += 1

    return pid_list


# ── Per-round backtest ──────────────────────────────────────────────────────
results = []

print("=" * 72)
print("Newsletter Walk-Forward Backtest — 2026 Season (Rounds 4–13)")
print("=" * 72)

for rd in ROUNDS:
    nrd = rd + 1
    nxt_path = JSON_DIR / f"players_round_{nrd}.json"
    if not nxt_path.exists():
        print(f"\nRd{rd}: round {nrd} JSON not found — skipping.")
        continue

    # Build season scores available at the time of round N newsletter
    season_scores = build_season_scores_up_to(rd)

    # Current round FA data (for avg3 pre-computed by SC)
    cur_df = load_round_json(rd)
    nxt_df = load_round_json(nrd)

    # Which FAs actually played in round N+1?
    fa_nxt = nxt_df[nxt_df["player_id"].isin(fa_ids)]
    fa_nxt_played = fa_nxt[fa_nxt["games"] == 1]
    played_ids = set(fa_nxt_played["player_id"].tolist())

    # Base rate
    base_h = int((fa_nxt_played["points"] >= 80).sum())
    base_t = len(fa_nxt_played)

    # Avg3 top-20 (cross-position, played only, using SC API avg3)
    fa_cur = cur_df[cur_df["player_id"].isin(fa_ids)].copy()
    avg3_pool = fa_cur[fa_cur["player_id"].isin(played_ids) & (fa_cur["avg3"] > 0)]
    a3_top20 = avg3_pool.sort_values("avg3", ascending=False).head(AVG3_BASELINE_N)["player_id"].tolist()
    a3_res = evaluate(a3_top20, nxt_df, played_ids)

    # New newsletter: top-5 per position by avg3 (played only)
    fa_played_pool = [pid for pid in fa_ids if pid in played_ids]
    new_picks = top_n_per_pos(fa_played_pool, avg3_p80, season_scores)
    new_res = evaluate(new_picks, nxt_df, played_ids)

    # Old newsletter: top-5 per position by observed season P(80+) (played only)
    old_picks = top_n_per_pos(fa_played_pool, observed_p80, season_scores)
    old_res = evaluate(old_picks, nxt_df, played_ids)

    # Avg5 variant: top-5 per position by last-5 avg
    avg5_picks = top_n_per_pos(fa_played_pool, avg5_p80, season_scores)
    avg5_res = evaluate(avg5_picks, nxt_df, played_ids)

    # Model newsletter: top-5 per position by LightGBM projected_score
    model_scores = simulate_model_picks(predict_round=nrd)
    model_fa_pool = [pid for pid in fa_ids if pid in played_ids]
    model_picks = top_n_per_pos_model(model_fa_pool, model_scores)
    model_res = evaluate(model_picks, nxt_df, played_ids) if model_picks else None

    # Model top-20 cross-position (apples-to-apples vs avg3 top-20)
    model_top20 = sorted(
        [(model_scores[pid], pid) for pid in model_fa_pool if pid in model_scores],
        reverse=True,
    )[:AVG3_BASELINE_N]
    model_top20_picks = [pid for _, pid in model_top20]
    model_top20_res = evaluate(model_top20_picks, nxt_df, played_ids) if model_top20_picks else None

    # Actual newsletter (if file exists for this round)
    nl_files = sorted(NL_DIR.glob(f"waiver_2026_rd{rd:02d}_*.md"))
    actual_res = None
    if nl_files:
        actual_pids = parse_newsletter_picks(nl_files[-1])
        actual_played = [pid for pid in actual_pids if pid in played_ids]
        actual_res = evaluate(actual_played, nxt_df, played_ids)

    def f(h, t):
        return f"{h}/{t} ({h/t*100:.0f}%)" if t else "n/a"

    print(f"\n--- Rd{rd} -> Rd{nrd} ---")
    print(f"  FA base rate (played):          {f(base_h, base_t)}")
    print(f"  [cross-pos] Avg3 top-{AVG3_BASELINE_N}:    {f(a3_res['hits'], a3_res['played'])}")
    if model_top20_res:
        print(f"  [cross-pos] Model top-{AVG3_BASELINE_N}:   {f(model_top20_res['hits'], model_top20_res['played'])}")
    print(f"  [pos-constrd] Avg3/pos:         {f(new_res['hits'], new_res['played'])}")
    print(f"  [pos-constrd] Obs P(80+)/pos:   {f(old_res['hits'], old_res['played'])}")
    print(f"  [pos-constrd] Avg5/pos:         {f(avg5_res['hits'], avg5_res['played'])}")
    if model_res:
        print(f"  [pos-constrd] Model/pos:        {f(model_res['hits'], model_res['played'])}")
    if actual_res:
        print(f"  Actual newsletter (file):       {f(actual_res['hits'], actual_res['played'])}")

    row = {
        "round": rd,
        "next_round": nrd,
        "fa_base_hits": base_h,
        "fa_base_played": base_t,
        "fa_base_pct": round(base_h / base_t * 100, 1) if base_t else None,
        "avg3_top20_hits": a3_res["hits"],
        "avg3_top20_played": a3_res["played"],
        "avg3_top20_pct": round(a3_res["pct"], 1),
        "new_nl_hits": new_res["hits"],
        "new_nl_played": new_res["played"],
        "new_nl_pct": round(new_res["pct"], 1),
        "old_nl_hits": old_res["hits"],
        "old_nl_played": old_res["played"],
        "old_nl_pct": round(old_res["pct"], 1),
        "avg5_nl_hits": avg5_res["hits"],
        "avg5_nl_played": avg5_res["played"],
        "avg5_nl_pct": round(avg5_res["pct"], 1),
    }
    if model_res:
        row["model_nl_hits"] = model_res["hits"]
        row["model_nl_played"] = model_res["played"]
        row["model_nl_pct"] = round(model_res["pct"], 1)
    if model_top20_res:
        row["model_top20_hits"] = model_top20_res["hits"]
        row["model_top20_played"] = model_top20_res["played"]
        row["model_top20_pct"] = round(model_top20_res["pct"], 1)
    if actual_res:
        row["actual_nl_hits"] = actual_res["hits"]
        row["actual_nl_played"] = actual_res["played"]
        row["actual_nl_pct"] = round(actual_res["pct"], 1)

    results.append(row)

    # ── Per-round markdown report ─────────────────────────────────────────
    md_lines = [
        f"# Backtest Rd{rd} → Rd{nrd}",
        "",
        "## Summary",
        "",
        "| Approach | Hits/Played | Hit Rate |",
        "| --- | --- | --- |",
        f"| FA base rate | {base_h}/{base_t} | {base_h/base_t*100:.1f}% |" if base_t else "",
        f"| Avg3 top-{AVG3_BASELINE_N} (cross-pos) | {a3_res['hits']}/{a3_res['played']} | {a3_res['pct']:.1f}% |",
        f"| **New newsletter (avg3/pos)** | **{new_res['hits']}/{new_res['played']}** | **{new_res['pct']:.1f}%** |",
        f"| Old newsletter (obs P(80+)) | {old_res['hits']}/{old_res['played']} | {old_res['pct']:.1f}% |",
    ]
    if actual_res:
        md_lines.append(
            f"| Actual newsletter (file) | {actual_res['hits']}/{actual_res['played']} | {actual_res['pct']:.1f}% |"
        )
    md_lines += [""]

    for label, res, picks in [
        ("New newsletter picks", new_res, new_picks),
        ("Old newsletter picks", old_res, old_picks),
        ("Avg3 top-20", a3_res, a3_top20),
    ]:
        md_lines += [
            f"## {label}",
            "",
            "| Pos | Player | Avg3 (Rd{}) | Rd{} Score | Hit |".format(rd, nrd),
            "| --- | --- | --- | --- | --- |",
        ]
        for d in res["details"]:
            pid = d["player_id"]
            a3_row = cur_df[cur_df["player_id"] == pid]
            a3_val = f"{a3_row.iloc[0]['avg3']:.1f}" if not a3_row.empty else "—"
            hit_str = "**HIT**" if d["hit"] else ""
            md_lines.append(
                f"| {pid_to_pos.get(pid, '?')} | {d['name']} | {a3_val} | {d['score']:.0f} | {hit_str} |"
            )
        md_lines.append("")

    if actual_res:
        md_lines += [
            "## Actual newsletter picks (from file)",
            "",
            "| Pos | Player | Avg3 (Rd{}) | Rd{} Score | Hit |".format(rd, nrd),
            "| --- | --- | --- | --- | --- |",
        ]
        for d in actual_res["details"]:
            pid = d["player_id"]
            a3_row = cur_df[cur_df["player_id"] == pid]
            a3_val = f"{a3_row.iloc[0]['avg3']:.1f}" if not a3_row.empty else "—"
            hit_str = "**HIT**" if d["hit"] else ""
            md_lines.append(
                f"| {pid_to_pos.get(pid, '?')} | {d['name']} | {a3_val} | {d['score']:.0f} | {hit_str} |"
            )
        md_lines.append("")

    md_path = OUT_DIR / f"backtest_rd{rd:02d}_to_rd{nrd:02d}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


# ── Aggregate totals ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("OVERALL TOTALS (Rd4–Rd13)")
print("=" * 72)

def tot(key_h, key_t):
    h = sum(r[key_h] for r in results if key_h in r)
    t = sum(r[key_t] for r in results if key_t in r)
    return h, t, (h / t * 100 if t else 0.0)

bh, bt, bp = tot("fa_base_hits", "fa_base_played")
ah, at_, ap = tot("avg3_top20_hits", "avg3_top20_played")
m20h, m20t, m20p = tot("model_top20_hits", "model_top20_played")
nh, nt, np_ = tot("new_nl_hits", "new_nl_played")
oh, ot, op = tot("old_nl_hits", "old_nl_played")
a5h, a5t, a5p = tot("avg5_nl_hits", "avg5_nl_played")
mh, mt, mp = tot("model_nl_hits", "model_nl_played")

actual_rows = [r for r in results if "actual_nl_hits" in r]
axh = sum(r["actual_nl_hits"] for r in actual_rows)
axt = sum(r["actual_nl_played"] for r in actual_rows)
axp = axh / axt * 100 if axt else 0.0

def ff(h, t, p):
    return f"{h}/{t} ({p:.1f}%)"

print(f"  FA base rate:                       {ff(bh, bt, bp)}")
print()
print(f"  -- Cross-position (top-20, no pos constraint) --")
print(f"  Avg3 top-20:                        {ff(ah, at_, ap)}")
if m20t:
    print(f"  Model top-20:                       {ff(m20h, m20t, m20p)}")
    print(f"    Model top-20 vs avg3 top-20:      {m20p - ap:+.1f}pp")
print()
print(f"  -- Position-constrained (top-5 per pos = 20 picks) --")
print(f"  Avg3/pos:                           {ff(nh, nt, np_)}")
print(f"  Obs P(80+)/pos:                     {ff(oh, ot, op)}")
print(f"  Avg5/pos:                           {ff(a5h, a5t, a5p)}")
if mt:
    print(f"  Model/pos:                          {ff(mh, mt, mp)}")
    print(f"    Model/pos vs avg3/pos:            {mp - np_:+.1f}pp")
    print(f"    Model/pos vs avg5/pos:            {mp - a5p:+.1f}pp")
if actual_rows:
    print(f"  Actual newsletter (Rd8-13):         {ff(axh, axt, axp)}  [{len(actual_rows)} rounds]")
print()
print(f"  -- Positional constraint cost --")
print(f"  Avg3: cross-pos={ap:.1f}% vs constrd={np_:.1f}% -> cost={np_ - ap:+.1f}pp")
if mt and m20t:
    print(f"  Model: cross-pos={m20p:.1f}% vs constrd={mp:.1f}% -> cost={mp - m20p:+.1f}pp")
if actual_rows:
    base_for_actual = sum(r["fa_base_pct"] for r in actual_rows) / len(actual_rows)
    avg3_for_actual = sum(r["avg3_top20_pct"] for r in actual_rows) / len(actual_rows)
    print(f"  Actual NL vs base (Rd8-13 avg): {axp - base_for_actual:+.1f}pp")
    print(f"  Actual NL vs avg3 (Rd8-13 avg): {axp - avg3_for_actual:+.1f}pp")

# ── Save CSV summary ────────────────────────────────────────────────────────
csv_path = OUT_DIR / "backtest_summary_2026.csv"
if results:
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

print(f"\nPer-round reports: {OUT_DIR}/")
print(f"Summary CSV:       {csv_path}")
