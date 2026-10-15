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

# Injected from app.py after Flask-Caching is set up. build_round_value_baseline()
# loops every historical season's draft data, so it's memoized the same way
# loader.py and awards.py memoize their own expensive calls.
cache = None


def _memoize(fn):
    def wrapper(*args, **kwargs):
        if cache is not None:
            return cache.memoize()(fn)(*args, **kwargs)
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


POS_COUNT = {"DEF": 40, "MID": 56, "RUC": 16, "FWD": 40}

# Fanfooty's own Position vocabulary, used only as a last-resort position
# source (see _compute_effective_avg's final fallback tier) — different
# words than SC's DEF/MID/RUC/FWD scheme, mapped onto it here.
_FANFOOTY_POS_MAP = {"Back": "DEF", "Midfielder": "MID", "Ruck": "RUC", "Forward": "FWD"}


def _split_fanfooty_position(raw) -> tuple[str | None, str | None]:
    if not isinstance(raw, str):
        return None, None
    tokens = [_FANFOOTY_POS_MAP[t] for t in raw.split() if t in _FANFOOTY_POS_MAP]
    if not tokens:
        return None, None
    return tokens[0], (tokens[1] if len(tokens) > 1 else None)

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
                           sc_round_df: "pd.DataFrame | None" = None,
                           year: int | None = None) -> pd.DataFrame:
    """
    Compute a TPOR-adjusted effective average using SC-official data as primary source.

        adjusted_score = (actual_total_points + replacement_level[pos] × missed_non_bye)
                         / total_rounds

    Bye detection: if a team has no players with played=1 in a given round, that round
    is a bye for that team (derived entirely from SC data — no fanfooty needed).

    missed_non_bye must only count rounds the player genuinely didn't take the field.
    sc_df (master_player_data.csv) is built from fantasy-roster snapshots, so a round
    where the player wasn't on ANY coach's roster (e.g. mid-season waiver pickup/drop)
    is simply absent from sc_df — indistinguishable, from sc_df alone, from an actual
    injury/omission. sc_round_df (data/raw/supercoach/{year}/players_round_N.json,
    via load_sc_round_files()) is the SC-official bulk endpoint covering EVERY player
    each round regardless of fantasy ownership, so it's used here as the ground truth
    for "did this player actually play" — those rounds are unioned into the
    played-rounds set before computing misses.

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

    # Rounds each player actually played per the SC-official bulk round endpoint
    # (covers every player, not just those on a fantasy roster) — ground truth
    # for "did this player actually play", independent of fantasy ownership.
    sc_round_played_rounds: dict = {}
    if sc_round_df is not None and not sc_round_df.empty:
        srd = sc_round_df[sc_round_df["played"] == 1].dropna(subset=["feed_id", "round"])
        sc_round_played_rounds = {
            fid: set(grp["round"])
            for fid, grp in srd.groupby("feed_id")
        }

    # Per-player team (most recent round's team_abbrev)
    player_team = (
        played.sort_values("round")
        .groupby("feed_id")["team_abbrev"]
        .last()
        .reset_index()
    )

    # Per-player positions (most recent round's pos_1/pos_2 from SC data).
    # sc_df's own pos_1 sometimes carries BOTH eligible positions as one
    # compound string (e.g. "MID FWD") instead of a single token, with pos_2
    # always empty — unlike every other position source this function uses
    # (player_list_df, the legacy fallback), which are always a single clean
    # token. Split those into pos_1/pos_2 so dual-position players actually
    # get matched into both position groups below instead of neither
    # (a compound string like "MID FWD" matches no POS_COUNT key exactly,
    # which silently dropped the player out of z-scoring entirely).
    def _split_pos(row):
        p1 = row["pos_1"]
        if isinstance(p1, str) and " " in p1:
            parts = p1.split()
            return pd.Series([parts[0], parts[1]])
        return pd.Series([p1, row["pos_2"]])

    player_pos = (
        played.sort_values("round")
        .groupby("feed_id")[["pos_1", "pos_2"]]
        .last()
        .reset_index()
    )
    player_pos[["pos_1", "pos_2"]] = player_pos.apply(_split_pos, axis=1)

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
        actual_played = sc_played_rounds.get(row["feed_id"], set()) | sc_round_played_rounds.get(row["feed_id"], set())
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

    # Ground-truth per-player totals from the SC-official bulk round endpoint
    # (covers every player regardless of fantasy rostering), used below by
    # both fallback tiers as a score source for players absent from sc_df.
    round_totals: dict = {}
    if sc_round_df is not None and not sc_round_df.empty:
        srd_played = sc_round_df[sc_round_df["played"] == 1].dropna(subset=["feed_id"]).copy()
        srd_played["points"] = pd.to_numeric(srd_played["points"], errors="coerce")
        round_totals = (
            srd_played.groupby("feed_id")
            .agg(total_sc=("points", "sum"), games_played=("points", "count"))
            .to_dict("index")
        )

    # Players in the draft but absent from sc_df entirely — not necessarily
    # zero-game: sc_df is built from fantasy-roster snapshots, so a player who
    # was drafted and then never actually started/benched on any coach's
    # roster again (e.g. dropped before round 1) is just as absent as one who
    # genuinely never played. sc_round_df (the SC-official bulk per-round
    # endpoint, covering every player regardless of fantasy rostering) is
    # ground truth for what they actually scored, and is used here as a
    # fallback score source instead of assuming 0 — a real AFL performance
    # the player's own coach never started still reflects on the pick.
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
                actual_played = sc_round_played_rounds.get(row["feed_id"], set())
                missed_non_bye = len(all_rounds - bye_rds - actual_played)
                return missed_bye, missed_non_bye

            missing[["missed_bye", "missed_non_bye"]] = missing.apply(
                lambda r: pd.Series(_zero_breakdown(r)), axis=1
            )

            # player_list_df's own "position" can be a compound string too
            # (e.g. "MID FWD" — confirmed for several players, not just a
            # 2019-2021 thing: Dustin Martin 2022, Nat Fyfe 2023). Split the
            # same way as sc_df's pos_1 above, for the same reason.
            def _split_position(pos):
                if isinstance(pos, str) and " " in pos:
                    parts = pos.split()
                    return pd.Series([parts[0], parts[1]])
                return pd.Series([pos, None])

            missing[["pos_1", "pos_2"]] = missing["position"].apply(_split_position)

            def _fallback_scores(row):
                rt = round_totals.get(row["feed_id"])
                if not rt or not rt["games_played"]:
                    return 0.0, 0.0
                raw_avg = rt["total_sc"] / rt["games_played"]
                repl = replacement.get(row["pos_1"], 0.0)
                median = (rt["total_sc"] + repl * (total_rounds - rt["games_played"])) / total_rounds
                return round(raw_avg, 2), round(median, 2)

            missing[["raw_avg", "median_score"]] = missing.apply(
                lambda r: pd.Series(_fallback_scores(r)), axis=1
            )
            result = pd.concat(
                [result, missing[["feed_id", "pos_1", "pos_2", "raw_avg", "median_score", "missed_bye", "missed_non_bye"]]],
                ignore_index=True,
            )

    # Last resort: a small number of players have real games in sc_round_df
    # (confirmed AFL regulars, not scratch picks) but never appear in either
    # source above — never rostered by any coach at all, not even once, so
    # there's no team/position record of them anywhere in this pipeline's
    # usual sources. sc_round_df itself carries a native "position" column
    # for 2020+ (the raw SC dumps request it explicitly, per
    # ingest_historical_season.py's _positions()) — preferred when present,
    # since it's SC's own classification. Falls back to fanfooty's per-round
    # data (every player regardless of fantasy involvement, just in a
    # different vocabulary — Back/Midfielder/Forward/Ruck, sometimes
    # compound — mapped onto DEF/MID/RUC/FWD) for years with no native
    # position in sc_round_df at all (currently: 2019 and earlier, which
    # predate any SC archive).
    still_missing = set(round_totals) - set(result["feed_id"].dropna())
    if still_missing:
        srd_team = pd.Series(dtype=object)
        srd_pos: dict = {}
        if sc_round_df is not None and not sc_round_df.empty:
            srd_sorted = sc_round_df.dropna(subset=["feed_id"]).sort_values("round")
            srd_team = srd_sorted.groupby("feed_id")["team_abbrev"].last()
            if "position" in sc_round_df.columns:
                srd_pos = (
                    srd_sorted.dropna(subset=["position"])
                    .groupby("feed_id")[["position", "position_2"]]
                    .last()
                    .to_dict("index")
                )

        fpos = pd.Series(dtype=object)
        fanfooty_fids = still_missing - set(srd_pos)
        if fanfooty_fids and year is not None:
            from website.data.loader import load_fanfooty_season
            fdf = load_fanfooty_season(year)
            if not fdf.empty:
                fpos = (
                    fdf.dropna(subset=["Player ID"])
                    .sort_values("round_num")
                    .groupby("Player ID")["Position"]
                    .last()
                )

        rows = []
        for fid in still_missing:
            if fid in srd_pos:
                pos_1, pos_2 = srd_pos[fid]["position"], srd_pos[fid]["position_2"]
            else:
                pos_1, pos_2 = _split_fanfooty_position(fpos.get(fid))
            if pos_1 is None:
                continue
            rt = round_totals[fid]
            team = srd_team.get(fid)
            bye_rds = all_rounds - team_played_rounds.get(team, all_rounds)
            actual_played = sc_round_played_rounds.get(fid, set())
            repl = replacement.get(pos_1, 0.0)
            rows.append({
                "feed_id":        fid,
                "pos_1":          pos_1,
                "pos_2":          pos_2,
                "raw_avg":        round(rt["total_sc"] / rt["games_played"], 2),
                "median_score":   round((rt["total_sc"] + repl * (total_rounds - rt["games_played"])) / total_rounds, 2),
                "missed_bye":     len(bye_rds),
                "missed_non_bye": len(all_rounds - bye_rds - actual_played),
            })
        if rows:
            result = pd.concat([result, pd.DataFrame(rows)], ignore_index=True)

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
                                        "missed_bye": int, "missed_non_bye": int, "pick": int | None}}
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
                              "missed_bye": 0, "missed_non_bye": 0, "peers_html": "", "pick": None}
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
                    "pick":           int(p["pick"]),
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


def _build_draft_df_with_ranks(draft_csv, player_list_csv, alias_map: dict,
                                sc_df: pd.DataFrame,
                                sc_round_df: "pd.DataFrame | None" = None,
                                year: int | None = None) -> "pd.DataFrame | None":
    """
    Shared pipeline behind both load_draft_board() (the Heat Map page) and
    get_pick_value_extremes() (the Best/Worst Draft Pick award): loads the
    draft CSV, attaches each pick's position-normalised Z-score (final_score),
    and computes rank_in_round + pick_value, both from the same forward
    8-pick sliding window (the pick + next 7, clamped at the boundary).

    rank_in_round: 1-8 position within the window, sorted by final_score —
                   drives the Heat Map's colour tiers.
    pick_value:    final_score minus the window's own average final_score —
                   how much better/worse this pick was than its immediate
                   neighbours, i.e. value relative to draft slot rather than
                   raw player quality (a 1st-overall pick being a great player
                   isn't a "steal"; outperforming the picks around it is).

    Returns None if draft_csv doesn't exist.
    """
    if not draft_csv.exists():
        log.error("Draft CSV not found: %s", draft_csv)
        return None

    draft_df = _load_draft(draft_csv, alias_map)

    if sc_df is not None and not sc_df.empty:
        # player_list_csv only exists for 2022+ (draft_prep/SC {year}/) — it's
        # the preferred source _compute_effective_avg uses for team/position
        # of players absent from sc_df entirely (see its docstring). Years
        # without it (2019-2021) fall back to player_match_results.csv's most
        # recent roster appearance instead of going without — leaving this
        # None was why most picks in those years had no score at all (e.g.
        # 2021 was ~40% missing vs ~11% for a year with player_list_csv).
        if player_list_csv.exists():
            player_list_df = pd.read_csv(player_list_csv, low_memory=False)
        else:
            from website.data.loader import load_legacy_player_positions
            player_list_df = load_legacy_player_positions(year)
        player_data    = _compute_effective_avg(sc_df, player_list_df, sc_round_df, year)
        player_data    = _compute_zscores(player_data)
        draft_df = draft_df.merge(
            player_data[["feed_id", "pos_1", "raw_avg", "median_score", "final_score", "missed_bye", "missed_non_bye"]],
            on="feed_id", how="left",
        )
        # missed_non_bye is informational only (shown as 🚑 on the Heat Map) —
        # it says a player didn't take the field, not why (injury, being
        # dropped to the reserves for form, suspension, etc. all look
        # identical in the data, and SC's own injury_suspension_status field
        # is unpopulated for all of them). A past version of this code
        # treated a high missed_non_bye count as a confirmed long-term
        # injury and excluded those picks from ranking/Worst Draft Pick —
        # reverted after real cases (a player who played one game for 8
        # points before being sent down, clearly not an injury) showed that
        # assumption doesn't hold.
    else:
        log.warning("SC current data unavailable - final_score will be NaN.")
        draft_df["final_score"]    = float("nan")
        draft_df["missed_bye"]     = 0
        draft_df["missed_non_bye"] = 0

    # Rank 1-8 within a forward window of 8 picks (the pick + next 7),
    # clamped at the boundary so the window is always exactly 8 wide.
    draft_df = draft_df.sort_values("pick").reset_index(drop=True)
    max_pick  = int(draft_df["pick"].max())
    win_size  = 8

    def _window_bounds(pick_num: int) -> tuple[int, int]:
        start = min(pick_num, max_pick - win_size + 1)
        return start, start + win_size - 1

    def _window_rank(pick_num: int, df: pd.DataFrame) -> int:
        start, end = _window_bounds(pick_num)
        window = (
            df[(df["pick"] >= start) & (df["pick"] <= end)]
            .sort_values("final_score", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        idx = window.index[window["pick"] == pick_num].tolist()
        return idx[0] + 1 if idx else win_size

    def _window_avg_score(pick_num: int, df: pd.DataFrame) -> float:
        start, end = _window_bounds(pick_num)
        window = df[(df["pick"] >= start) & (df["pick"] <= end)]
        return window["final_score"].mean()

    draft_df["rank_in_round"] = draft_df["pick"].apply(
        lambda p: _window_rank(int(p), draft_df)
    )
    draft_df["window_avg_score"] = draft_df["pick"].apply(
        lambda p: _window_avg_score(int(p), draft_df)
    )
    draft_df["pick_value"] = draft_df["final_score"] - draft_df["window_avg_score"]

    return draft_df


def load_draft_board(draft_csv, player_list_csv, alias_map: dict,
                     coach_order: list[str], sc_df: pd.DataFrame,
                     sc_round_df: "pd.DataFrame | None" = None,
                     year: int | None = None):
    """
    Main entry point for the Heat Map page. Returns (coach_summary, board_rows,
    n_rounds, coach_order).

    sc_df: output of load_sc_current() — SC-official per-round scores (primary source).
    sc_round_df: output of load_sc_round_files() — SC-official bulk round data covering
                 every player regardless of fantasy roster, used as ground truth for
                 whether a player actually played a round.
    """
    draft_df = _build_draft_df_with_ranks(draft_csv, player_list_csv, alias_map, sc_df, sc_round_df, year)
    if draft_df is None:
        return [], [], 0, coach_order

    summary  = build_coach_summary(draft_df, coach_order)
    board, n = build_board_table(draft_df, coach_order)
    return summary, board, n, coach_order


@_memoize
def build_round_value_baseline(alias_map: dict) -> dict[int, float]:
    """
    Pools final_score by draft_round across every season with usable draft
    data (see list_available_seasons), giving the Worst Draft Pick awards a
    baseline of dozens of historical picks per round instead of just the 7
    picks drafted immediately after it that one season. A true bust can hide
    behind a run of other weak picks drafted around it the same year (e.g.
    2024: M.Crouch's -8.59 z-score read as merely "average-bad" next to his
    round's other weak picks, while D.Martin's -8.35 stood out only because
    his neighbours happened to be stronger) — comparing against history
    instead of that year's immediate neighbours fixes this.

    Returns {draft_round: mean final_score}, pooled across all years.
    """
    from website.config import draft_csv_path, player_list_csv_path
    from website.data.loader import list_available_seasons, load_sc_current, load_sc_round_files

    frames = []
    for year in list_available_seasons():
        dcsv = draft_csv_path(year)
        if not dcsv.exists():
            continue
        sc_df = load_sc_current(year)
        if sc_df is None or sc_df.empty:
            continue
        plcsv = player_list_csv_path(year)
        sc_round_df = load_sc_round_files(year)
        ddf = _build_draft_df_with_ranks(dcsv, plcsv, alias_map, sc_df, sc_round_df, year)
        if ddf is not None:
            frames.append(ddf[["draft_round", "final_score"]].dropna())

    if not frames:
        return {}
    pooled = pd.concat(frames, ignore_index=True)
    return pooled.groupby("draft_round")["final_score"].mean().to_dict()


def get_pick_value_extremes(draft_csv, player_list_csv, alias_map: dict,
                             sc_df: pd.DataFrame,
                             sc_round_df: "pd.DataFrame | None" = None,
                             early_max_round: int = 8,
                             round_baseline: "dict[int, float] | None" = None,
                             year: int | None = None) -> dict | None:
    """
    For the Best/Worst Early/Late Draft Pick awards: splits picks into
    "early" (draft_round <= early_max_round, i.e. premium draft capital) and
    "late", and returns the single best and worst pick in each half. Returns
    {"best_early": {...}, "best_late": {...}, "worst_early": {...},
    "worst_late": {...}} (a half with no usable data is simply absent from
    the dict), or None if there's no usable draft data at all.

    All four are ranked by value vs. round_baseline when given (that round's
    pooled historical average final_score across every past draft, from
    build_round_value_baseline()) rather than pick_value (value vs. the 8
    picks drafted immediately around it that same season) — a single-season
    8-pick window of immediate neighbours can be unusually strong or weak by
    chance that year (e.g. 2023: Bontempelli had the higher raw score of the
    two, but R.Laird's pick_value looked better purely because Laird's own
    neighbours happened to be weaker than Bontempelli's, which included
    Laird himself). Comparing every pick to the same multi-season
    baseline for its round fixes this consistently for all four awards.
    Falls back to pick_value when no baseline is supplied (e.g. not enough
    historical seasons yet).

    worst_late additionally prefers picks that actually got a game (raw_avg
    > 0): a total zero from a late, speculative pick isn't a meaningful
    story (that's the norm for a flier that late, whatever the cause), while
    one that played and still badly underperformed reflects a real miss.

    Each entry: coach, player, draft_round, pick, final_score, value,
    value_label (the latter two describe whichever metric was used).
    """
    draft_df = _build_draft_df_with_ranks(draft_csv, player_list_csv, alias_map, sc_df, sc_round_df, year)
    if draft_df is None:
        return None

    valid = draft_df.dropna(subset=["pick_value"]).copy()
    if valid.empty:
        return None

    if round_baseline:
        valid["history_value"] = valid["final_score"] - valid["draft_round"].map(round_baseline)
        history_label = lambda rd: f"vs. the historical Rd {rd} average"
    else:
        valid["history_value"] = valid["pick_value"]
        history_label = lambda rd: "vs. the picks around it"
    valid = valid.dropna(subset=["history_value"])

    def _row_to_dict(row, value: float, label: str) -> dict:
        return {
            "coach":        row["coach_name"],
            "player":       str(row["Player"]),
            "draft_round":  int(row["draft_round"]),
            "pick":         int(row["pick"]),
            "final_score":  round(float(row["final_score"]), 2),
            "value":        round(float(value), 2),
            "value_label":  label,
        }

    early = valid[valid["draft_round"] <= early_max_round]
    late  = valid[valid["draft_round"] >  early_max_round]

    result: dict = {}
    if not early.empty:
        b = early.loc[early["history_value"].idxmax()]
        result["best_early"] = _row_to_dict(b, b["history_value"], history_label(int(b["draft_round"])))
        # Early picks carry real expectation, so even a total bust (e.g. a
        # season-ending injury before round 1) is still a meaningful story —
        # no games-played filter here.
        w = early.loc[early["history_value"].idxmin()]
        result["worst_early"] = _row_to_dict(w, w["history_value"], history_label(int(w["draft_round"])))
    if not late.empty:
        b = late.loc[late["history_value"].idxmax()]
        result["best_late"] = _row_to_dict(b, b["history_value"], history_label(int(b["draft_round"])))
        # A late pick that never got a single game (raw_avg == 0) isn't a
        # meaningful "worst pick" story — busting is the norm for a
        # speculative flier that late, regardless of cause (omission,
        # injury, etc. look identical in the data either way). Prefer a
        # late pick that actually played and still badly underperformed —
        # that reflects a real evaluation miss, not a coin-flip outcome.
        # Falls back to the full late pool if every late pick that year
        # was a zero-gamer.
        late_played = late[late["raw_avg"] > 0]
        late_worst_pool = late_played if not late_played.empty else late
        w = late_worst_pool.loc[late_worst_pool["history_value"].idxmin()]
        result["worst_late"] = _row_to_dict(w, w["history_value"], history_label(int(w["draft_round"])))

    return result or None
