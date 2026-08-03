"""
draft_data.py — Draft Heat Map using position-normalised z-score ranking.

Algorithm (mirrors heat_map_2024_v3.ipynb):
  1. Compute effective SC average per player from player_stats_current.csv (SC primary).
  2. For each position (DEF=40, MID=56, RUC=8, FWD=40):
       - Filter eligible players (pos_1 OR pos_2)
       - Top N by effective average
       - Z-score: (score - mean) / stdev
  3. final_score = max z-score across all positions
  4. Within each draft round, rank 1-8 by final_score (NaN → last)
  5. Map rank to colour tier: 1=dark green, 2-3=green, 4-5=yellow, 6-7=orange, 8=red

Data sources:
  Primary:    data/live/player_stats_current.csv  (SC Draft API — official scores + positions)
  Complement: draft_prep/SC 2026/2026_SC_Player_list.csv  (team info for zero-game players)

Outputs (via load_draft_board):
  - coach_summary: list of dicts per coach with rank tier counts + avg rank
  - board_rows:    list of {round, <coach>: {player, color, missed_bye, missed_non_bye}}
  - n_rounds:      int
  - coach_order:   list[str]
"""
import html
import logging

import pandas as pd

log = logging.getLogger(__name__)

POS_COUNT = {"DEF": 40, "MID": 56, "RUC": 16, "FWD": 40}

RANK_TIERS = [
    ("rank_1",   lambda r: r == 1),
    ("rank_2_3", lambda r: 2 <= r <= 3),
    ("rank_4_5", lambda r: 4 <= r <= 5),
    ("rank_6_7", lambda r: 6 <= r <= 7),
    ("rank_8",   lambda r: r == 8),
]


def _rank_color(rank: int) -> str:
    if rank == 1:
        return "rank-1"
    elif rank <= 3:
        return "rank-2-3"
    elif rank <= 5:
        return "rank-4-5"
    elif rank <= 7:
        return "rank-6-7"
    return "rank-8"


def _rank_label(rank: int) -> str:
    if rank == 1:
        return "1"
    elif rank <= 3:
        return "2-3"
    elif rank <= 5:
        return "4-5"
    elif rank <= 7:
        return "6-7"
    return "8"


def _compute_effective_avg(sc_df: pd.DataFrame,
                           player_list_df: pd.DataFrame | None = None,
                           fanfooty_df: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """
    Compute a TPOR-adjusted effective average using SC-official data as primary source.

        adjusted_score = (actual_total_points + replacement_level[pos] × missed_non_bye)
                         / total_rounds

    Bye detection: if a team has no players with played=1 in a given round, that round
    is a bye for that team (derived entirely from SC data — no fanfooty needed).

    missed_non_bye must only count rounds the player genuinely didn't take the field.
    sc_df is built from fantasy-roster snapshots, so a round where the player wasn't on
    ANY coach's roster (e.g. mid-season waiver pickup/drop) is simply absent from sc_df —
    indistinguishable, from sc_df alone, from an actual injury/omission. fanfooty_df
    records every player who featured in a match regardless of fantasy ownership, so it's
    used here as the ground truth for "did this player actually register a score" —
    fanfooty rounds are unioned into the played-rounds set before computing misses.

    Zero-game players: looked up in player_list_df using SC team codes (same codes as
    sc_df.team_abbrev), so no cross-system mapping is required.
    """
    # Only rows where the player actually took the AFL field (played == 1; 0 = DNP)
    played = sc_df[sc_df["played"] == 1].copy()
    played["points"]  = pd.to_numeric(played["points"],  errors="coerce")
    played["feed_id"] = played["feed_id"].astype("Int64")
    played = played.dropna(subset=["feed_id", "points"])

    all_rounds    = set(played["round"].unique())
    total_rounds  = len(all_rounds)

    # Which rounds each team played (SC team codes throughout — no mapping needed)
    team_played_rounds: dict[str, set] = {
        team: set(grp["round"])
        for team, grp in played.groupby("team_abbrev")
    }

    # Rounds each player is known to have played from sc_df (fantasy-roster-based)
    sc_played_rounds: dict = {
        fid: set(grp["round"])
        for fid, grp in played.groupby("feed_id")
    }

    # Rounds each player registered a real score in fanfooty, independent of any
    # fantasy roster — the ground truth for "did this player actually play".
    ff_played_rounds: dict = {}
    if fanfooty_df is not None and not fanfooty_df.empty:
        ff = fanfooty_df.dropna(subset=["Player ID", "round_num"]).copy()
        ff["Player ID"] = pd.to_numeric(ff["Player ID"], errors="coerce").astype("Int64")
        ff_played_rounds = {
            fid: set(grp["round_num"])
            for fid, grp in ff.groupby("Player ID")
        }

    # Per-player team (most recent round's team_abbrev)
    player_team = (
        played.sort_values("round")
        .groupby("feed_id")["team_abbrev"]
        .last()
        .reset_index()
    )

    # Per-player positions (most recent round's pos_1/pos_2 from SC data)
    player_pos = (
        played.sort_values("round")
        .groupby("feed_id")[["pos_1", "pos_2"]]
        .last()
        .reset_index()
    )

    # Raw aggregates
    agg = (
        played.groupby("feed_id")
        .agg(total_sc=("points", "sum"), games_played=("points", "count"))
        .reset_index()
    )
    agg["feed_id"] = agg["feed_id"].astype("Int64")
    agg["raw_avg"] = agg["total_sc"] / agg["games_played"]
    agg = agg.merge(player_team, on="feed_id", how="left")
    agg = agg.merge(player_pos,  on="feed_id", how="left")

    # Split missed games: bye (team didn't play) vs injury/omission (team played, player didn't)
    def _missed_breakdown(row):
        bye_rds    = all_rounds - team_played_rounds.get(row["team_abbrev"], all_rounds)
        missed_bye = len(bye_rds)
        actual_played = sc_played_rounds.get(row["feed_id"], set()) | ff_played_rounds.get(row["feed_id"], set())
        missed_non_bye = len(all_rounds - bye_rds - actual_played)
        return missed_bye, missed_non_bye

    agg[["missed_bye", "missed_non_bye"]] = agg.apply(
        lambda r: pd.Series(_missed_breakdown(r)), axis=1
    )

    # Replacement level = mean raw_avg of players ranked N+1 to N+8 per position
    replacement: dict[str, float] = {}
    for pos, n in POS_COUNT.items():
        eligible = agg[(agg["pos_1"] == pos) & (agg["games_played"] > 0)].nlargest(n + 8, "raw_avg")
        fringe = eligible.iloc[n:] if len(eligible) > n else eligible
        replacement[pos] = float(fringe["raw_avg"].mean()) if not fringe.empty else 0.0
        log.debug("Replacement level [%s]: %.1f", pos, replacement[pos])

    # Impute non-bye misses at replacement level; bye rounds score 0 (known absence)
    def _adjusted(row):
        repl = replacement.get(row["pos_1"], 0.0)
        total_missed = total_rounds - row["games_played"]
        return (row["total_sc"] + repl * total_missed) / total_rounds

    agg["median_score"] = agg.apply(_adjusted, axis=1).round(2)
    result = agg[["feed_id", "pos_1", "pos_2", "raw_avg", "median_score", "missed_bye", "missed_non_bye"]].copy()

    # Zero-game players: in draft but never appeared in SC current data.
    # team_abbrev from player_list uses the same SC codes as sc_df — no mapping needed.
    if player_list_df is not None and not player_list_df.empty:
        pl = player_list_df[["feed_id", "team", "position"]].copy()
        pl["feed_id"] = pd.to_numeric(pl["feed_id"], errors="coerce").astype("Int64")
        pl = pl.dropna(subset=["feed_id"])
        missing = pl[~pl["feed_id"].isin(result["feed_id"])].copy()
        if not missing.empty:
            def _zero_breakdown(row):
                bye_rds    = all_rounds - team_played_rounds.get(row["team"], all_rounds)
                missed_bye = len(bye_rds)
                actual_played = ff_played_rounds.get(row["feed_id"], set())
                missed_non_bye = len(all_rounds - bye_rds - actual_played)
                return missed_bye, missed_non_bye

            missing[["missed_bye", "missed_non_bye"]] = missing.apply(
                lambda r: pd.Series(_zero_breakdown(r)), axis=1
            )
            missing["raw_avg"]      = 0.0
            missing["median_score"] = 0.0
            missing = missing.rename(columns={"position": "pos_1"})
            missing["pos_2"] = None
            result = pd.concat(
                [result, missing[["feed_id", "pos_1", "pos_2", "raw_avg", "median_score", "missed_bye", "missed_non_bye"]]],
                ignore_index=True,
            )

    return result


def _compute_zscores(player_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds {POS}_norm columns and final_score (max z-score) to player_data.
    player_data must have: feed_id, pos_1, pos_2, median_score.
    """
    df = player_data.copy()
    df["median_score"] = df["median_score"].fillna(0)

    for pos, count in POS_COUNT.items():
        all_pos = df[(df["pos_1"] == pos) | (df["pos_2"] == pos)].copy()
        top_n   = all_pos.nlargest(count, "median_score")
        mean_s  = top_n["median_score"].mean()
        std_s   = top_n["median_score"].std()
        col     = f"{pos}_norm"
        if std_s and std_s > 0:
            all_pos[col] = ((all_pos["median_score"] - mean_s) / std_s).round(3)
        else:
            all_pos[col] = 0.0
        df = df.merge(all_pos[["feed_id", col]], on="feed_id", how="left")

    norm_cols = [f"{p}_norm" for p in POS_COUNT]
    df["final_score"] = df[norm_cols].max(axis=1)
    return df


def _load_draft(draft_csv, alias_map: dict) -> pd.DataFrame:
    df = pd.read_csv(draft_csv)
    df.columns = [c.strip() for c in df.columns]
    df["coach_name"]  = df["Coach"].map(alias_map)
    df["draft_round"] = pd.to_numeric(df["Round"], errors="coerce").astype(int)
    df["pick"]        = pd.to_numeric(df["Pick"],  errors="coerce").astype(int)
    df["feed_id"]     = pd.to_numeric(df["feed_id"], errors="coerce").astype("Int64")
    return df.dropna(subset=["draft_round", "pick", "coach_name"]).reset_index(drop=True)


def _build_peers_html(pick_num: int, df: pd.DataFrame, max_pick: int, win_size: int = 8) -> str:
    """
    Returns a Bootstrap HTML table listing all players in the same rolling window as pick_num.
    The row for pick_num is highlighted. Returns "" if the window is empty.
    """
    start = min(pick_num, max_pick - win_size + 1)
    end = start + win_size - 1
    window = (
        df[(df["pick"] >= start) & (df["pick"] <= end)]
        .sort_values("final_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    if window.empty:
        return ""

    rows = []
    for i, row in window.iterrows():
        is_current = int(row["pick"]) == pick_num
        name = html.escape(str(row.get("Player", "—")))
        pos  = html.escape(str(row["pos_1"])) if pd.notna(row.get("pos_1")) else "—"
        avg  = f"{row['raw_avg']:.1f}"        if pd.notna(row.get("raw_avg"))        else "—"
        tpor = f"{row['median_score']:.1f}"   if pd.notna(row.get("median_score"))   else "—"
        z    = f"{row['final_score']:.2f}"    if pd.notna(row.get("final_score"))    else "—"
        tr_class = ' class="table-active fw-bold"' if is_current else ""
        rows.append(
            f"<tr{tr_class}>"
            f"<td>{i + 1}</td><td>{name}</td><td>{pos}</td><td>{avg}</td><td>{tpor}</td><td>{z}</td>"
            f"</tr>"
        )

    return (
        "<table class=\"table table-sm table-bordered mb-0\" style=\"font-size:0.75rem;min-width:310px\">"
        "<thead><tr><th>#</th><th>Player</th><th>Pos</th><th>Avg</th><th>TPOR</th><th>Z</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_board_table(draft_df: pd.DataFrame, coach_order: list[str]) -> tuple[list[dict], int]:
    """
    Returns (rows, n_rounds).
    Each row: {"round": int, <coach>: {"player": str, "color": str, "rank_label": str,
                                        "missed_bye": int, "missed_non_bye": int}}
    """
    n_rounds = int(draft_df["draft_round"].max()) if not draft_df.empty else 0
    max_pick = int(draft_df["pick"].max()) if not draft_df.empty else 0
    win_size = 8
    rows = []
    for rnd in range(1, n_rounds + 1):
        rnd_picks = draft_df[draft_df["draft_round"] == rnd]
        row: dict = {"round": rnd}
        for coach in coach_order:
            pick = rnd_picks[rnd_picks["coach_name"] == coach]
            if pick.empty:
                row[coach] = {"player": "", "color": "", "rank_label": "",
                              "missed_bye": 0, "missed_non_bye": 0, "peers_html": ""}
            else:
                p    = pick.iloc[0]
                rank = int(p["rank_in_round"]) if pd.notna(p.get("rank_in_round")) else 8
                row[coach] = {
                    "player":         str(p["Player"]),
                    "color":          _rank_color(rank),
                    "rank_label":     _rank_label(rank),
                    "missed_bye":     int(p["missed_bye"])     if pd.notna(p.get("missed_bye"))     else 0,
                    "missed_non_bye": int(p["missed_non_bye"]) if pd.notna(p.get("missed_non_bye")) else 0,
                    "peers_html":     _build_peers_html(int(p["pick"]), draft_df, max_pick, win_size),
                }
        rows.append(row)
    return rows, n_rounds


def build_coach_summary(draft_df: pd.DataFrame, coach_order: list[str]) -> list[dict]:
    """
    Returns one row per coach:
      coach, rank_1, rank_2_3, rank_4_5, rank_6_7, rank_8, avg_rank
    """
    rows = []
    for coach in coach_order:
        cdf   = draft_df[draft_df["coach_name"] == coach]
        ranks = cdf["rank_in_round"].dropna().astype(int)
        row   = {"coach": coach}
        row["rank_1"]   = int((ranks == 1).sum())
        row["rank_2_3"] = int(((ranks >= 2) & (ranks <= 3)).sum())
        row["rank_4_5"] = int(((ranks >= 4) & (ranks <= 5)).sum())
        row["rank_6_7"] = int(((ranks >= 6) & (ranks <= 7)).sum())
        row["rank_8"]   = int((ranks == 8).sum())
        row["avg_rank"] = round(ranks.mean(), 2) if not ranks.empty else None
        rows.append(row)
    # Sort by avg_rank ascending (lower = stronger draft), None last
    rows.sort(key=lambda r: (r["avg_rank"] is None, r["avg_rank"] or 0))
    return rows


def load_draft_board(draft_csv, player_list_csv, alias_map: dict,
                     coach_order: list[str], sc_df: pd.DataFrame,
                     fanfooty_df: "pd.DataFrame | None" = None):
    """
    Main entry point. Returns (coach_summary, board_rows, n_rounds, coach_order).

    sc_df: output of load_sc_current() — SC-official per-round scores (primary source).
    fanfooty_df: output of load_fanfooty_season() — used as ground truth for whether a
                 player actually played a round, independent of fantasy roster status.
    """
    if not draft_csv.exists():
        log.error("Draft CSV not found: %s", draft_csv)
        return [], [], 0, coach_order

    draft_df = _load_draft(draft_csv, alias_map)

    if sc_df is not None and not sc_df.empty and player_list_csv.exists():
        player_list_df = pd.read_csv(player_list_csv, low_memory=False)
        player_data    = _compute_effective_avg(sc_df, player_list_df, fanfooty_df)
        player_data    = _compute_zscores(player_data)
        draft_df = draft_df.merge(
            player_data[["feed_id", "pos_1", "raw_avg", "median_score", "final_score", "missed_bye", "missed_non_bye"]],
            on="feed_id", how="left",
        )
    else:
        log.warning("SC current data or player list unavailable — final_score will be NaN.")
        draft_df["final_score"]    = float("nan")
        draft_df["missed_bye"]     = 0
        draft_df["missed_non_bye"] = 0

    # Rank 1-8 within a forward window of 8 picks (the pick + next 7),
    # clamped at the boundary so the window is always exactly 8 wide.
    draft_df = draft_df.sort_values("pick").reset_index(drop=True)
    max_pick  = int(draft_df["pick"].max())
    win_size  = 8

    def _window_rank(pick_num: int, df: pd.DataFrame) -> int:
        start = min(pick_num, max_pick - win_size + 1)
        end   = start + win_size - 1
        window = (
            df[(df["pick"] >= start) & (df["pick"] <= end)]
            .sort_values("final_score", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        idx = window.index[window["pick"] == pick_num].tolist()
        return idx[0] + 1 if idx else win_size

    draft_df["rank_in_round"] = draft_df["pick"].apply(
        lambda p: _window_rank(int(p), draft_df)
    )

    summary  = build_coach_summary(draft_df, coach_order)
    board, n = build_board_table(draft_df, coach_order)
    return summary, board, n, coach_order
