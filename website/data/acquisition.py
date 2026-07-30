"""
acquisition.py — Player acquisition source analysis.

Assigns each currently-owned player a source (Draft / Waiver / Free Agent / Trade)
based on the draft CSV and the transactions history. The most recent transaction
for a player overrides their draft source.

Display categories: Draft, Waiver, FA, Trade.

Public API:
    load_player_sources(...)       → DataFrame per owned player with source
    build_acquisition_stats(...)   → nested dict {coach: {source: {avg, count, players}}}
    get_acquisition_kings(...)     → {category: {coach, avg, count}}
    build_acquisition_heatmap(...) → list of row dicts for the template table
"""
import json as _json
import logging

import pandas as pd

log = logging.getLogger(__name__)

_DISPLAY_CATS = ["Draft", "Waiver", "FA", "Trade"]


def _cell_color(val: float, lo: float, hi: float) -> str:
    """Red → yellow → green scale (same as power_rankings.py)."""
    if hi <= lo or pd.isna(val):
        return "rgb(230,230,230)"
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    if t < 0.5:
        t2 = t * 2
        r = int(215 + (255 - 215) * t2)
        g = int(48  + (255 - 48)  * t2)
        b = int(39  - 39 * t2)
    else:
        t2 = (t - 0.5) * 2
        r = int(255 - 255 * t2)
        g = int(255 - (255 - 104) * t2)
        b = int(191 - 191 * t2)
    return f"rgb({r},{g},{b})"


def _text_color(val: float, lo: float, hi: float) -> str:
    if hi <= lo or pd.isna(val):
        return "#000"
    t = (val - lo) / (hi - lo)
    return "#fff" if (t < 0.15 or t > 0.85) else "#000"


def load_player_sources(
    draft_csv,
    transactions_csv,
    coach_list_csv,
    player_match_csv,
    draft_alias_map: dict,
    processed_waivers_json=None,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        player_id, feed_id, first_name, last_name, coach, source, round_acquired

    One row per (player, coach) pair that appears in player_match_results —
    covering all historical ownership periods, not just the current roster.
    Source reflects how that specific coach acquired the player.
    """
    if not player_match_csv.exists():
        log.warning("player_match_results.csv not found")
        return pd.DataFrame()

    pm = pd.read_csv(player_match_csv, low_memory=False)
    pm["player_id"] = pd.to_numeric(pm["player_id"], errors="coerce").astype("Int64")
    pm["feed_id"]   = pd.to_numeric(pm["feed_id"],   errors="coerce").astype("Int64")

    # All unique (player_id, coach) pairs across every round
    pairs = (
        pm.sort_values("round_x")
        .drop_duplicates(subset=["player_id", "coach_first_name"], keep="last")
        [["player_id", "feed_id", "first_name", "last_name", "coach_first_name"]]
        .rename(columns={"coach_first_name": "coach"})
        .dropna(subset=["player_id"])
    )

    # Earliest round each (player, coach) pair appears — used to anchor acquisition source
    min_round_map: dict[tuple, int] = {}
    for _, row in pm.iterrows():
        pval = row.get("player_id")
        cval = row.get("coach_first_name")
        rval = row.get("round_x")
        if pd.notna(pval) and cval and pd.notna(rval):
            key = (int(pval), str(cval))
            min_round_map[key] = min(min_round_map.get(key, 9999), int(rval))

    # ── Draft source by feed_id ─────────────────────────────────────────────────
    draft_source: dict[int, dict] = {}  # feed_id → {coach, source, round_acquired}
    if draft_csv.exists():
        df_draft = pd.read_csv(draft_csv)
        df_draft.columns = [c.strip() for c in df_draft.columns]
        for _, row in df_draft.iterrows():
            coach = draft_alias_map.get(str(row.get("Coach", "")).strip())
            fid   = pd.to_numeric(row.get("feed_id"), errors="coerce")
            if pd.isna(fid) or not coach:
                continue
            draft_source[int(fid)] = {
                "coach":          coach,
                "source":         "Draft",
                "round_acquired": 0,
            }

    # ── Waiver pick order ───────────────────────────────────────────────────────
    waiver_pick_map: dict[tuple, dict] = {}  # (player_id, round) → {pick_num, day}
    if processed_waivers_json is not None:
        from pathlib import Path as _Path
        _jpath = _Path(processed_waivers_json)
        if _jpath.exists():
            with open(_jpath, encoding="utf-8") as _f:
                _raw_waivers = _json.load(_f)
            _batch_counter: dict[tuple, int] = {}
            for _item in (_raw_waivers or []):
                _pid  = _item.get("player_id")
                _rnd  = _item.get("round")
                _proc = str(_item.get("processed", ""))
                _local_date = _proc[:10] if len(_proc) >= 10 else ""
                if not _pid or not _rnd or not _local_date:
                    continue
                _batch_key = (int(_rnd), _local_date)
                _batch_counter[_batch_key] = _batch_counter.get(_batch_key, 0) + 1
                _day = pd.to_datetime(_local_date).strftime("%a")
                waiver_pick_map[(int(_pid), int(_rnd))] = {
                    "pick_num": _batch_counter[_batch_key],
                    "day":      _day,
                }

    # ── Coach ↔ team_id mappings ────────────────────────────────────────────────
    coach_to_team: dict[str, int] = {}
    if coach_list_csv.exists():
        cl = pd.read_csv(coach_list_csv)
        for _, r in cl.iterrows():
            tid  = pd.to_numeric(r.get("coach_team_id"), errors="coerce")
            name = r.get("coach_first_name")
            if pd.notna(tid) and name:
                coach_to_team[name] = int(tid)

    # ── All transactions per (player_id, team_id) from transactions ──────────────
    # Stores the full sorted list so the merge can filter by round.
    # Round-trip trades (traded away and back immediately) are cancelled out.
    all_txns: dict[tuple, list] = {}  # (pid, team_id) → [rows sorted by round]
    if transactions_csv.exists():
        txn = pd.read_csv(transactions_csv, low_memory=False)
        txn["player_id"]    = pd.to_numeric(txn["player_id"],    errors="coerce").astype("Int64")
        txn["user_team_id"] = pd.to_numeric(txn["user_team_id"], errors="coerce").astype("Int64")
        txn = txn.dropna(subset=["player_id"])
        txn["processed"] = pd.to_datetime(txn["processed"], errors="coerce", utc=True)

        def _tid(val):
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        for pid, group in txn.sort_values("processed").groupby("player_id"):
            rows = group.to_dict("records")

            # Detect and cancel round-trip: last two are Trades by different teams,
            # player returns to the team that had them before.
            if len(rows) >= 2:
                last, prev = rows[-1], rows[-2]
                last_team, prev_team = _tid(last.get("user_team_id")), _tid(prev.get("user_team_id"))
                if (str(last.get("source", "")) == "Trade"
                        and str(prev.get("source", "")) == "Trade"
                        and last_team is not None and prev_team is not None
                        and last_team != prev_team):
                    pre_team = _tid(rows[-3]["user_team_id"]) if len(rows) >= 3 else None
                    if pre_team is None or pre_team == last_team:
                        rows = rows[:-2]

            for row in rows:
                t = _tid(row.get("user_team_id"))
                if t is not None:
                    all_txns.setdefault((int(pid), t), []).append(row)
    else:
        log.warning("transactions.csv not found — all players will show as Draft")

    # ── Merge acquisition source onto (player, coach) pairs ────────────────────
    result_rows = []
    for _, r in pairs.iterrows():
        pid   = int(r["player_id"]) if pd.notna(r["player_id"]) else None
        fid   = int(r["feed_id"])   if pd.notna(r["feed_id"])   else None
        coach = r["coach"]
        if pid is None:
            continue

        team_id  = coach_to_team.get(coach)
        min_rnd  = min_round_map.get((pid, coach))

        info = None
        if team_id is not None:
            txns = all_txns.get((pid, team_id), [])
            # Only transactions at or before the first round this coach played the player
            relevant = [
                t for t in txns
                if pd.notna(t.get("round")) and int(t["round"]) <= (min_rnd or 9999)
            ]
            if relevant:
                row = relevant[-1]
                rnd = row.get("round")
                info = {
                    "source":         str(row.get("source", "Draft")),
                    "round_acquired": int(rnd) if pd.notna(rnd) else None,
                }

        if info is None:
            if fid in draft_source and draft_source[fid]["coach"] == coach:
                info = draft_source[fid]
            else:
                info = {"source": "Draft", "round_acquired": 0}

        rnd_acq = info["round_acquired"]
        wp = waiver_pick_map.get((pid, rnd_acq), {}) if info["source"] == "Waiver" else {}
        result_rows.append({
            "player_id":       pid,
            "feed_id":         fid,
            "first_name":      r["first_name"],
            "last_name":       r["last_name"],
            "coach":           coach,
            "source":          info["source"],
            "round_acquired":  rnd_acq,
            "waiver_pick_num": wp.get("pick_num"),
            "waiver_day":      wp.get("day"),
        })

    return pd.DataFrame(result_rows)


def build_acquisition_stats(
    player_sources: pd.DataFrame,
    player_match_csv,
) -> dict:
    """
    Returns nested dict: {coach: {display_cat: {avg, count, players: [...]}}}

    display_cat is one of "Draft", "Waiver", "FA", "Trade".
    Avg is computed from on-field rounds in player_match_results.csv.
    Players with zero on-field appearances are included with avg=0.
    """
    if player_sources.empty:
        return {}

    avg_map:    dict[tuple, float]     = {}   # keyed by (player_id, coach)
    total_map:  dict[tuple, int]       = {}   # keyed by (player_id, coach)
    games_map:  dict[tuple, int]       = {}   # keyed by (player_id, coach)
    rounds_map: dict[tuple, list]      = {}   # keyed by (player_id, coach)

    if player_match_csv.exists():
        pm = pd.read_csv(player_match_csv, low_memory=False)
        pm["player_id"] = pd.to_numeric(pm["player_id"], errors="coerce").astype("Int64")
        pm["points_x"]  = pd.to_numeric(pm["points_x"],  errors="coerce")

        on_field = pm[pm["on_field"] == True]

        # Per-(player, coach): avg, total, games, and sorted list of round numbers
        coach_agg = (
            on_field.groupby(["player_id", "coach_first_name"])["points_x"]
            .agg(avg="mean", total="sum", games="count")
            .reset_index()
        )
        for _, r in coach_agg.iterrows():
            key = (int(r["player_id"]), r["coach_first_name"])
            avg_map[key]   = round(float(r["avg"]), 1)
            total_map[key] = int(r["total"])
            games_map[key] = int(r["games"])

        rounds_series = (
            on_field.groupby(["player_id", "coach_first_name"])["round_x"]
            .apply(lambda s: sorted(s.dropna().astype(int).tolist()))
        )
        for (pid, coach), rnds in rounds_series.items():
            rounds_map[(int(pid), coach)] = rnds


    stats: dict = {}
    for _, row in player_sources.iterrows():
        coach  = row["coach"]
        source = row["source"]
        cat    = "FA" if source == "Free Agent" else source
        pid    = int(row["player_id"]) if pd.notna(row["player_id"]) else None
        if pid is None:
            continue

        avg       = avg_map.get((pid, coach), 0.0)
        if avg <= 0:
            continue  # only include players with recorded on-field scores
        total_pts    = total_map.get((pid, coach), 0)
        games        = games_map.get((pid, coach), 0)
        rounds_played = rounds_map.get((pid, coach), [])
        name         = f"{row['first_name']} {row['last_name']}"
        rnd_acq      = row.get("round_acquired")

        stats.setdefault(coach, {})
        stats[coach].setdefault(cat, {"avg": 0.0, "count": 0, "players": []})
        stats[coach][cat]["players"].append({
            "name":            name,
            "avg":             avg,
            "total_pts":       total_pts,
            "games":           games,
            "rounds_played":   rounds_played,
            "round_acquired":  rnd_acq,
            "waiver_pick_num": row.get("waiver_pick_num"),
            "waiver_day":      row.get("waiver_day"),
        })

    # Compute avg (total_pts / total_games) and count per (coach, cat)
    for coach, cats in stats.items():
        for cat, data in cats.items():
            total_pts   = sum(p["total_pts"] for p in data["players"])
            total_games = sum(p["games"]     for p in data["players"])
            data["avg"]       = round(total_pts / total_games, 1) if total_games > 0 else 0.0
            data["count"]     = len(data["players"])
            data["total_pts"] = total_pts
            data["players"].sort(key=lambda p: p["avg"], reverse=True)

    return stats


def get_acquisition_kings(acquisition_stats: dict) -> dict:
    """
    Returns {category: {coach, avg, count}} for each of the 3 display categories.
    Coach with the highest avg (≥1 player) wins. Returns None if no coach qualifies.
    """
    kings: dict = {}
    for cat in _DISPLAY_CATS:
        best_coach, best_avg, best_count = None, -1.0, 0
        for coach, cats in acquisition_stats.items():
            data = cats.get(cat)
            if data and data["count"] >= 1 and data["avg"] > best_avg:
                best_avg   = data["avg"]
                best_coach = coach
                best_count = data["count"]
        kings[cat] = {"coach": best_coach, "avg": best_avg, "count": best_count}
    return kings


def build_acquisition_heatmap(
    acquisition_stats: dict,
    coach_order: list[str],
) -> list[dict]:
    """
    Returns a list of row dicts (one per coach) ready for the template.

    Each row: {coach, cells: {cat: {avg, count, players, color, text_color, is_king}}}

    Colour scale is per-column (category), same red→yellow→green as power_rankings.
    """
    kings = get_acquisition_kings(acquisition_stats)

    # Compute lo/hi per category for colour scale
    col_vals: dict[str, list[float]] = {cat: [] for cat in _DISPLAY_CATS}
    for coach in coach_order:
        for cat in _DISPLAY_CATS:
            data = acquisition_stats.get(coach, {}).get(cat)
            if data and data["count"] >= 1:
                col_vals[cat].append(data["avg"])

    col_lo = {cat: min(v) if v else 0.0 for cat, v in col_vals.items()}
    col_hi = {cat: max(v) if v else 0.0 for cat, v in col_vals.items()}

    rows = []
    for coach in coach_order:
        cells: dict = {}
        for cat in _DISPLAY_CATS:
            data = acquisition_stats.get(coach, {}).get(cat)
            if data and data["count"] >= 1:
                avg   = data["avg"]
                color = _cell_color(avg, col_lo[cat], col_hi[cat])
                tcol  = _text_color(avg, col_lo[cat], col_hi[cat])
            else:
                avg, color, tcol = None, "rgb(230,230,230)", "#aaa"

            cells[cat] = {
                "avg":        avg,
                "count":      data["count"] if data else 0,
                "total_pts":  data["total_pts"] if data else 0,
                "players":    data["players"] if data else [],
                "color":      color,
                "text_color": tcol,
                "is_king":    kings.get(cat, {}).get("coach") == coach and avg is not None,
            }
        rows.append({"coach": coach, "cells": cells})

    return rows
