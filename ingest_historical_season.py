"""
ingest_historical_season.py — One-off converter: raw archived SC API dumps
(data/raw/supercoach/{year}/*.rds, from the old R-based pipeline) into the
data/seasons/{year}/ CSV shape the website reads for historical seasons
(same file shape as data/live/, see website/config.py).

Usage: py ingest_historical_season.py 2022 2023 2024 2025

Primary source (richest — full regular season + finals + transactions + draft):
  Per round (1..21 regular, 22..24 finals), from data/raw/supercoach/{year}/:
    leagues_{league_id}_ladderAndFixtures_round={N}&scores=true.rds
        -> 'ladder' list: cumulative per-team standings as of round N
        -> 'fixtures' list: that round's matchups with per-round scores
    players-cf_embed=...round={N}.rds
        -> ~800 players per round: identity (name, feed_id, AFL team) + that
           round's fantasy points (player_stats[0].points)
    userteams_{team_id}_statsPlayers_round={N}.rds  (one per team per round)
        -> that team's lineup for the round: player_id, picked, on-field
           position slot (DEF/MID/RUC/FWD)
    leagues_{league_id}_{processedWaivers,trades,teamtrades}.rds
        -> transactions.csv (Waiver/Free Agent/Trade), ported 1:1 from
           website/data/api_ingest.py's live-API parsing logic
    archive/legacy/fantasy_banter_data/{year}/draft_{year}.csv (2022/2023) or
    ASLdraft_{year}.JSON (2025) -> draft_prep/SC {year}/draft_{year}_result.csv

  NOTE: this LEAGUE_ID matters — 2025's season actually ran under league_id
  13462, not the 5282 that data/raw/supercoach/2025/ originally had locally
  (5282 was a dead/abandoned league: 7 teams, all-zero scores, confirmed by
  hand). The 13462 files were missing locally entirely and were pulled from
  a teammate's backup, https://github.com/pmac09/Fantasy-Banter-old2/tree/
  master/data/2025/raw — if another year ever looks similarly broken, check
  that repo (and its sibling year folders) before assuming it's unrecoverable.

Fallback source (used only if the raw per-round dump above isn't usable):
  archive/legacy/fantasy_banter_data/{year}/league_data.RDS + player_data.RDS
  — clean pre-built tables, but truncated at whatever round the old pipeline
  last ran that year (e.g. round 2 of 21 for 2023 — not used since the raw
  path above covers 2023 fully; kept as a safety net for any future year
  where even the raw per-round dump turns out to be unusable).

Known gaps NOT covered by this script (confirmed missing/unrecoverable —
checked data/raw/supercoach, archive/legacy/fantasy_banter_data, draft_prep,
and the pmac09/Fantasy-Banter-old2 GitHub backup):
  - 2024 draft heat map: no draft source of any kind found anywhere.
  - 2022 Grand Final (round 24): the round-24 snapshot file simply doesn't
    exist in any copy found (QF/EF/PF for rounds 22-23 are present) — the
    champion/runner-up is still correct via honour_board.py's hardcoded
    result, just no bracket visual.
Historical pages that depend on those will show reduced/no data; the
website already handles this gracefully (see website/data/finals.py,
website/data/acquisition.py, website/data/draft_data.py).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyreadr
import rdata

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "supercoach"
LEGACY_DIR = PROJECT_ROOT / "archive" / "legacy" / "fantasy_banter_data"
SEASONS_DIR = PROJECT_ROOT / "data" / "seasons"

LEAGUE_IDS = {2022: 651660, 2023: 99, 2024: 1283, 2025: 13462}
REGULAR_ROUNDS = 21
FINALS_ROUNDS = [22, 23, 24]


def _v(x):
    """Unwrap a 1-element numpy array (as returned by rdata) to a plain scalar."""
    if x is None:
        return None
    try:
        return x[0]
    except (TypeError, IndexError):
        return x


# ── Raw-dump path (2022-2024) ───────────────────────────────────────────────

def _round_snapshot(year: int, round_num: int) -> dict | None:
    lid = LEAGUE_IDS[year]
    path = RAW_DIR / str(year) / f"leagues_{lid}_ladderAndFixtures_round={round_num}&scores=true.rds"
    if not path.exists():
        return None
    try:
        return rdata.read_rds(str(path))
    except Exception:
        return None


def _coach_map(snapshot: dict) -> dict[int, dict]:
    m = {}
    for team in snapshot["ladder"]:
        ut = team["userTeam"]
        team_id = int(_v(ut["id"]))
        m[team_id] = {
            "coach": str(_v(ut["user"]["first_name"])),
            "team_name": str(_v(ut["teamname"])),
            "user_id": int(_v(ut["user_id"])),
        }
    return m


def raw_dump_is_usable(year: int) -> bool:
    snap = _round_snapshot(year, 1)
    if not snap or not snap.get("ladder"):
        return False
    total_pf = sum(float(_v(t["points_for"]) or 0) for t in snap["ladder"])
    cf_rounds = list((RAW_DIR / str(year)).glob("players-cf_embed*round=10.rds"))
    return len(snap["ladder"]) == 8 and total_pf > 0 and bool(cf_rounds)


def build_ladder_and_fixtures_from_raw(year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ladder_rows, fixture_rows = [], []
    coach_map: dict[int, dict] = {}

    for rnd in range(1, REGULAR_ROUNDS + 1):
        snap = _round_snapshot(year, rnd)
        if not snap:
            continue
        cm = _coach_map(snap)
        coach_map.update(cm)

        for team in snap["ladder"]:
            team_id = int(_v(team["userTeam"]["id"]))
            ladder_rows.append({
                "round": rnd,
                "coach_id": team_id,
                "coach_team_id": team_id,
                "coach_first_name": cm[team_id]["coach"],
                "coach_team_name": cm[team_id]["team_name"],
                "wins": int(_v(team["wins"])),
                "draws": int(_v(team["draws"])),
                "losses": int(_v(team["losses"])),
                "points_for": float(_v(team["points_for"])),
                "points_against": float(_v(team["points_against"])),
                "league_points": int(_v(team["points"])),
                "position": int(_v(team["position"])),
            })

        for fx in (snap.get("fixtures") or []):
            for a, b in ((fx["user_team1"], fx["user_team2"]), (fx["user_team2"], fx["user_team1"])):
                id_a, id_b = int(_v(a["id"])), int(_v(b["id"]))
                name_a, name_b = str(_v(a["teamname"])), str(_v(b["teamname"]))
                stats_a = (a.get("stats") or [{}])[0]
                stats_b = (b.get("stats") or [{}])[0]
                pts_a = float(_v(stats_a.get("points")) or 0)
                pts_b = float(_v(stats_b.get("points")) or 0)
                fixture_rows.append({
                    "round_number":           rnd,
                    "match_name":             f"{name_a} vs {name_b}",
                    "team_id":                id_a,
                    "team_name":              name_a,
                    "team_coach":             cm.get(id_a, {}).get("coach", str(_v(a["user"]["first_name"]))),
                    "team_points":            pts_a,
                    "opposition_team_id":     id_b,
                    "opposition_team_name":   name_b,
                    "opposition_team_coach":  cm.get(id_b, {}).get("coach", str(_v(b["user"]["first_name"]))),
                    "opposition_team_points": pts_b,
                    "Rank":                   0,
                })

    ladder = pd.DataFrame(ladder_rows).drop_duplicates(subset=["round", "coach_first_name"], keep="first")
    fixtures = pd.DataFrame(fixture_rows).drop_duplicates(subset=["round_number", "team_id", "opposition_team_id"], keep="first")
    return ladder.reset_index(drop=True), fixtures.reset_index(drop=True), coach_map


def build_finals_json_from_raw(year: int, out_dir: Path) -> None:
    json_dir = out_dir / "json"
    for rnd in FINALS_ROUNDS:
        snap = _round_snapshot(year, rnd)
        if not snap or not snap.get("fixtures"):
            continue
        fixtures_out = []
        for fx in snap["fixtures"]:
            t1, t2 = fx["user_team1"], fx["user_team2"]
            pts1 = float(_v((t1.get("stats") or [{}])[0].get("points")) or 0)
            pts2 = float(_v((t2.get("stats") or [{}])[0].get("points")) or 0)
            fixtures_out.append({
                "user_team1": {"user": {"first_name": str(_v(t1["user"]["first_name"]))},
                               "stats": [{"points": pts1}]},
                "user_team2": {"user": {"first_name": str(_v(t2["user"]["first_name"]))},
                               "stats": [{"points": pts2}]},
            })
        json_dir.mkdir(parents=True, exist_ok=True)
        (json_dir / f"match_json_rd{rnd}.json").write_text(
            json.dumps({"fixtures": fixtures_out}), encoding="utf-8"
        )


def build_player_matches_from_raw(year: int, coach_map: dict[int, dict], max_round: int) -> pd.DataFrame:
    rows = []
    for rnd in range(1, max_round + 1):
        cf_path = next(iter((RAW_DIR / str(year)).glob(f"players-cf_embed*round={rnd}.rds")), None)
        if cf_path is None:
            continue
        try:
            players = rdata.read_rds(str(cf_path))
        except Exception:
            continue

        player_lookup: dict[int, dict] = {}
        for p in players:
            pid = int(_v(p["id"]))
            stats = p.get("player_stats") or []
            points = float(_v(stats[0].get("points")) or 0) if stats else 0.0
            player_lookup[pid] = {
                "first_name": str(_v(p["first_name"])),
                "last_name":  str(_v(p["last_name"])),
                "feed_id":    _v(p["feed_id"]),
                "team":       str(_v(p["team"]["abbrev"])) if p.get("team") else "",
                "points":     points,
            }

        for team_id in coach_map:
            lineup_path = RAW_DIR / str(year) / f"userteams_{team_id}_statsPlayers_round={rnd}.rds"
            if not lineup_path.exists():
                continue
            try:
                lineup = rdata.read_rds(str(lineup_path))
            except Exception:
                continue
            for p in lineup.get("players") or []:
                pid = int(_v(p["player_id"]))
                info = player_lookup.get(pid)
                if info is None:
                    continue
                picked_raw = _v(p.get("picked"))
                on_field = str(picked_raw).lower() == "true"
                rows.append({
                    "round_x":          rnd,
                    "coach_id":         team_id,
                    "coach_team_id":    team_id,
                    "coach_first_name": coach_map[team_id]["coach"],
                    "coach_team_name":  coach_map[team_id]["team_name"],
                    "player_id":        pid,
                    "played_position":  _v(p.get("position")),
                    "points_x":         info["points"],
                    "on_field":         on_field,
                    "picked":           on_field,
                    "feed_id":          info["feed_id"],
                    "first_name":       info["first_name"],
                    "last_name":        info["last_name"],
                    "team":             info["team"],
                    "position":         _v(p.get("position")),
                })
        print(f"    round {rnd}/{max_round} done")

    return pd.DataFrame(rows).reset_index(drop=True)


def build_transactions_from_raw(year: int, out_dir: Path) -> None:
    """
    Ports website/data/api_ingest.py's _fetch_and_save_transactions() endpoint
    parsing onto the archived RDS equivalents: processedWaivers.rds -> Waiver,
    trades.rds -> Free Agent, teamtrades.rds -> Trade. Same column shape as
    the live transactions.csv so website/data/acquisition.py needs no changes.
    Also writes a processedWaivers.json sidecar (same shape as the live SC API
    response) so config.processed_waivers_json_path(year) finds it.
    """
    lid = LEAGUE_IDS[year]
    src = RAW_DIR / str(year)
    rows: list[dict] = []

    waivers_path = src / f"leagues_{lid}_processedWaivers.rds"
    waivers_raw: list = []
    if waivers_path.exists():
        try:
            waivers_raw = rdata.read_rds(str(waivers_path))
        except Exception:
            waivers_raw = []
    for t in waivers_raw:
        pl, drop = t.get("player") or {}, t.get("dropped_player") or {}
        rows.append({
            "round":                      _v(t.get("round")),
            "user_team_id":               _v(t.get("user_team_id")),
            "processed":                  str(_v(t.get("processed"))),
            "player_id":                  _v(t.get("player_id")),
            "player.feed_id":             _v(pl.get("feed_id")),
            "player.first_name":          _v(pl.get("first_name")),
            "player.last_name":           _v(pl.get("last_name")),
            "player.team.abbrev":         _v((pl.get("team") or {}).get("abbrev")),
            "dropped_player_id":          _v(t.get("dropped_player_id")),
            "dropped_player.feed_id":     _v(drop.get("feed_id")),
            "dropped_player.first_name":  _v(drop.get("first_name")),
            "dropped_player.last_name":   _v(drop.get("last_name")),
            "dropped_player.team.abbrev": _v((drop.get("team") or {}).get("abbrev")),
            "source":                     "Waiver",
        })

    trades_path = src / f"leagues_{lid}_trades.rds"
    trades_raw: list = []
    if trades_path.exists():
        try:
            trades_raw = rdata.read_rds(str(trades_path))
        except Exception:
            trades_raw = []
    for t in trades_raw:
        buy, sell = t.get("buy_player") or {}, t.get("sell_player") or {}
        rows.append({
            "round":                      _v(t.get("round")),
            "user_team_id":               _v(t.get("user_id")),
            "processed":                  str(_v(t.get("action_time"))),
            "player_id":                  _v(t.get("buy_player_id")),
            "player.feed_id":             _v(buy.get("feed_id")),
            "player.first_name":          _v(buy.get("first_name")),
            "player.last_name":           _v(buy.get("last_name")),
            "player.team.abbrev":         _v((buy.get("team") or {}).get("abbrev")),
            "dropped_player_id":          _v(t.get("sell_player_id")),
            "dropped_player.feed_id":     _v(sell.get("feed_id")),
            "dropped_player.first_name":  _v(sell.get("first_name")),
            "dropped_player.last_name":   _v(sell.get("last_name")),
            "dropped_player.team.abbrev": _v((sell.get("team") or {}).get("abbrev")),
            "source":                     "Free Agent",
        })

    teamtrades_path = src / f"leagues_{lid}_teamtrades.rds"
    teamtrades_raw: list = []
    if teamtrades_path.exists():
        try:
            teamtrades_raw = rdata.read_rds(str(teamtrades_path))
        except Exception:
            teamtrades_raw = []
    seen_trade_ids: set = set()
    for trade in teamtrades_raw:
        if int(_v(trade.get("status_id")) or 0) != 7:
            continue
        trade_id = _v(trade.get("id"))
        if trade_id in seen_trade_ids:
            continue
        seen_trade_ids.add(trade_id)
        rnd = _v(trade.get("round"))
        team_a, team_b = _v(trade.get("user_team_id")), _v(trade.get("trade_user_team_id"))
        processed = str(_v(trade.get("action_time")))
        for tp in (trade.get("team_trade_players") or []):
            pl_a, pl_b = _v(tp.get("player_id")), _v(tp.get("trade_player_id"))
            if pl_a is not None and team_b is not None:
                rows.append({"round": rnd, "user_team_id": team_b, "processed": processed,
                             "player_id": pl_a, "player.feed_id": None, "player.first_name": None,
                             "player.last_name": None, "player.team.abbrev": None,
                             "dropped_player_id": pl_b, "dropped_player.feed_id": None,
                             "dropped_player.first_name": None, "dropped_player.last_name": None,
                             "dropped_player.team.abbrev": None, "source": "Trade"})
            if pl_b is not None and team_a is not None:
                rows.append({"round": rnd, "user_team_id": team_a, "processed": processed,
                             "player_id": pl_b, "player.feed_id": None, "player.first_name": None,
                             "player.last_name": None, "player.team.abbrev": None,
                             "dropped_player_id": pl_a, "dropped_player.feed_id": None,
                             "dropped_player.first_name": None, "dropped_player.last_name": None,
                             "dropped_player.team.abbrev": None, "source": "Trade"})

    if not rows:
        print(f"    (no transactions found — {len(waivers_raw)} waivers, {len(trades_raw)} FA, {len(teamtrades_raw)} trades raw)")
        return

    pd.DataFrame(rows).to_csv(out_dir / "transactions.csv", index=False)

    if waivers_raw:
        waivers_json = [
            {"player_id": int(_v(t.get("player_id"))), "round": int(_v(t.get("round"))),
             "processed": str(_v(t.get("processed")))}
            for t in waivers_raw
        ]
        (src / f"leagues_{lid}_processedWaivers.json").write_text(
            json.dumps(waivers_json), encoding="utf-8"
        )
    print(f"    transactions.csv: {len(rows)} rows "
          f"({len(waivers_raw)} waiver, {len(trades_raw)} FA, {len(seen_trade_ids)} approved trades)")


def _player_id_lookup(year: int) -> dict[int, dict]:
    """id -> {first_name, last_name, feed_id} from round-1 players-cf_embed."""
    cf_path = next(iter((RAW_DIR / str(year)).glob("players-cf_embed*round=1.rds")), None)
    if cf_path is None:
        return {}
    try:
        players = rdata.read_rds(str(cf_path))
    except Exception:
        return {}
    return {
        int(_v(p["id"])): {
            "first_name": str(_v(p["first_name"])),
            "last_name":  str(_v(p["last_name"])),
            "feed_id":    _v(p["feed_id"]),
        }
        for p in players
    }


def build_draft_csv_from_raw(year: int, coach_map: dict[int, dict]) -> None:
    """draft_{year}.csv (raw SC draft export: user_team_id, round, pick, player_id)
    -> draft_prep/SC {year}/draft_{year}_result.csv, matching the Round/Pick/Player/
    Coach/feed_id shape website/data/draft_data.py expects. Coach alias comes from
    config.DRAFT_ALIAS_MAP's real-name entries (added alongside the existing
    nickname entries)."""
    src = LEGACY_DIR / str(year) / f"draft_{year}.csv"
    if not src.exists():
        print(f"    no draft_{year}.csv found — trying ASLdraft_{year}.JSON instead")
        build_draft_csv_from_legacy_json(year)
        return
    draft = pd.read_csv(src)
    player_lookup = _player_id_lookup(year)
    if not player_lookup:
        print(f"    no players-cf_embed round=1 for {year} — can't resolve draft player names, skipping")
        return

    rows = []
    for _, r in draft.iterrows():
        team_id = int(r["user_team_id"])
        pid = int(r["player_id"])
        coach = coach_map.get(team_id, {}).get("coach")
        info = player_lookup.get(pid)
        if coach is None or info is None:
            continue
        name = f"{info['first_name'][0]}.{info['last_name']}" if info["first_name"] else info["last_name"]
        rows.append({
            "Round": int(r["round"]), "Pick": int(r["pick"]), "Player": name,
            "Coach": coach, "Round Coach Order": 0, "Dataset count": 1,
            "feed_id": info["feed_id"],
        })
    out_path = PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"draft_{year}_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    draft_{year}_result.csv: {len(rows)} picks -> {out_path}")


def build_draft_csv_from_legacy_json(year: int) -> None:
    """ASLdraft_{year}.JSON (gzip'd RDS-wrapped JSON string, alias-keyed coach
    column already matching config.DRAFT_ALIAS_MAP) -> draft_prep/SC {year}/
    draft_{year}_result.csv."""
    src = LEGACY_DIR / str(year) / f"ASLdraft_{year}.JSON"
    if not src.exists():
        print(f"    no ASLdraft_{year}.JSON found — skipping draft heat map for {year}")
        return
    try:
        d = rdata.read_rds(str(src))
        payload = json.loads(d["x"][0])
    except Exception as exc:
        print(f"    could not parse {src.name}: {exc}")
        return

    player_list_path = PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"{year}_SC_Player_list.csv"
    if not player_list_path.exists():
        print(f"    no {player_list_path.name} — can't resolve draft player names, skipping")
        return
    plist = pd.read_csv(player_list_path, low_memory=False)
    name_lookup = {
        int(row["feed_id"]): f"{str(row['first_name'])[0]}.{row['last_name']}"
        for _, row in plist.dropna(subset=["feed_id"]).iterrows()
    }

    rows = []
    for pick in payload["draft_table"]:
        fid = int(pick["player_id"])
        rows.append({
            "Round": int(pick["round"]), "Pick": int(pick["pick"]),
            "Player": name_lookup.get(fid, f"#{fid}"),
            "Coach": pick["coach"], "Round Coach Order": pick.get("order_id", 0),
            "Dataset count": 1, "feed_id": fid,
        })
    out_path = PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"draft_{year}_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    draft_{year}_result.csv: {len(rows)} picks -> {out_path}")


def build_coach_list(coach_map: dict[int, dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {"coach_team_id": tid, "coach_id": tid, "coach_first_name": c["coach"], "coach_team_name": c["team_name"]}
        for tid, c in coach_map.items()
    ])


def build_current_teams_from_raw(year: int, coach_map: dict[int, dict], max_round: int) -> pd.DataFrame:
    """Rebuilds the roster as of max_round directly from that round's
    players-cf_embed (for avg/avg3/price) + userteams_statsPlayers (for
    which coach held the player) — richer than player_match_results.csv,
    which doesn't carry avg/price."""
    cf_path = next(iter((RAW_DIR / str(year)).glob(f"players-cf_embed*round={max_round}.rds")), None)
    if cf_path is None:
        return pd.DataFrame(columns=["coach_first_name", "first_name", "last_name", "team", "position", "avg", "avg3", "price", "feed_id"])
    try:
        players = rdata.read_rds(str(cf_path))
    except Exception:
        return pd.DataFrame(columns=["coach_first_name", "first_name", "last_name", "team", "position", "avg", "avg3", "price", "feed_id"])

    lookup: dict[int, dict] = {}
    for p in players:
        pid = int(_v(p["id"]))
        stats = (p.get("player_stats") or [{}])[0]
        lookup[pid] = {
            "first_name": str(_v(p["first_name"])),
            "last_name":  str(_v(p["last_name"])),
            "feed_id":    _v(p["feed_id"]),
            "team":       str(_v(p["team"]["abbrev"])) if p.get("team") else "",
            "avg":        _v(stats.get("avg")),
            "avg3":       _v(stats.get("avg3")),
            "price":      _v(stats.get("price")),
        }

    rows = []
    for team_id, cm in coach_map.items():
        lineup_path = RAW_DIR / str(year) / f"userteams_{team_id}_statsPlayers_round={max_round}.rds"
        if not lineup_path.exists():
            continue
        try:
            lineup = rdata.read_rds(str(lineup_path))
        except Exception:
            continue
        for p in lineup.get("players") or []:
            pid = int(_v(p["player_id"]))
            info = lookup.get(pid)
            if info is None:
                continue
            rows.append({
                "coach_first_name": cm["coach"],
                "first_name":       info["first_name"],
                "last_name":        info["last_name"],
                "team":             info["team"],
                "position":         _v(p.get("position")),
                "avg":              info["avg"],
                "avg3":             info["avg3"],
                "price":            info["price"],
                "feed_id":          info["feed_id"],
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["coach_first_name", "feed_id"]).reset_index(drop=True)


def ingest_year_from_raw(year: int) -> None:
    out_dir = SEASONS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{year}] building ladder/fixtures from raw round snapshots...")
    ladder, fixtures, coach_map = build_ladder_and_fixtures_from_raw(year)
    if ladder.empty:
        print(f"  [{year}] SKIP — no usable ladder rounds found in raw dump")
        return
    max_round = int(ladder["round"].max())

    print(f"  [{year}] building player match results from raw per-round dumps ({max_round} rounds)...")
    matches = build_player_matches_from_raw(year, coach_map, max_round)
    teams = build_current_teams_from_raw(year, coach_map, max_round)
    coaches = build_coach_list(coach_map)

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    matches.to_csv(out_dir / "player_match_results.csv", index=False)
    teams.to_csv(out_dir / "current_teams.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)
    build_finals_json_from_raw(year, out_dir)

    print(f"  [{year}] building transactions from raw waiver/trade dumps...")
    build_transactions_from_raw(year, out_dir)

    print(f"  [{year}] building draft heat map data...")
    build_draft_csv_from_raw(year, coach_map)

    n_finals = len(list((out_dir / "json").glob("match_json_rd*.json"))) if (out_dir / "json").exists() else 0
    print(f"  [{year}] OK (raw) — {max_round} rounds, {len(matches)} player-match rows, "
          f"{n_finals} finals-round files -> {out_dir}")


# ── Legacy fantasy_banter_data path (fallback — 2025) ───────────────────────

def _read_rds_df(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    return list(result.values())[0]


def ingest_year_from_legacy(year: int) -> None:
    src_dir = LEGACY_DIR / str(year)
    league_path, player_path = src_dir / "league_data.RDS", src_dir / "player_data.RDS"
    if not league_path.exists() or not player_path.exists():
        print(f"  [{year}] SKIP — no usable source found (raw dump unreliable, "
              f"and {league_path.name}/{player_path.name} not found)")
        return

    league = _read_rds_df(league_path)
    player = _read_rds_df(player_path)

    league_good = league.dropna(subset=["wins", "draws", "losses", "points_for", "points_against", "points", "position"]).copy()
    if league_good["round"].nunique() < 2:
        print(f"  [{year}] SKIP — legacy league_data.RDS has too little usable data")
        return
    league_good["round"] = league_good["round"].astype(int)

    out_dir = SEASONS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    ladder = pd.DataFrame({
        "round":            league_good["round"],
        "coach_id":         league_good["team_id"].astype(int),
        "coach_team_id":    league_good["team_id"].astype(int),
        "coach_first_name": league_good["coach"],
        "coach_team_name":  league_good["team"],
        "wins":             league_good["wins"].astype(int),
        "draws":            league_good["draws"].astype(int),
        "losses":           league_good["losses"].astype(int),
        "points_for":       league_good["points_for"],
        "points_against":   league_good["points_against"],
        "league_points":    league_good["points"].astype(int),
        "position":         league_good["position"].astype(int),
    }).drop_duplicates(subset=["round", "coach_first_name"], keep="first").reset_index(drop=True)

    fixture_good = league.dropna(subset=["team_score", "opponent_score", "position"]).copy()
    fixture_good["round"] = fixture_good["round"].astype(int)
    fixtures = pd.DataFrame({
        "round_number":           fixture_good["round"],
        "match_name":             fixture_good["team"] + " vs " + fixture_good["opponent_team"],
        "team_id":                fixture_good["team_id"].astype(int),
        "team_name":              fixture_good["team"],
        "team_coach":             fixture_good["coach"],
        "team_points":            fixture_good["team_score"],
        "opposition_team_id":     fixture_good["opponent_team_id"].astype(int),
        "opposition_team_name":   fixture_good["opponent_team"],
        "opposition_team_coach":  fixture_good["opponent_coach"],
        "opposition_team_points": fixture_good["opponent_score"],
        "Rank":                   fixture_good["position"].astype(int),
    }).reset_index(drop=True)

    player_good = player.dropna(subset=["round", "coach", "feed_id"]).copy()
    player_good["round"] = player_good["round"].astype(int)
    matches = pd.DataFrame({
        "round_x":          player_good["round"],
        "coach_id":         player_good["user_team_id"].astype(int),
        "coach_team_id":    player_good["user_team_id"].astype(int),
        "coach_first_name": player_good["coach"],
        "coach_team_name":  player_good["team"],
        "player_id":        player_good["player_id"],
        "played_position":  player_good["position"],
        "points_x":         player_good["points"],
        "on_field":         player_good["picked"].astype("boolean").fillna(False).astype(bool),
        "picked":           player_good["picked"].astype("boolean").fillna(False).astype(bool),
        "feed_id":          player_good["feed_id"],
        "first_name":       player_good["first_name"],
        "last_name":        player_good["last_name"],
        "team":             player_good["team_abbrev"],
        "position":         player_good["pos_1"],
    }).reset_index(drop=True)

    last_round = player_good["round"].max()
    last = player_good[player_good["round"] == last_round].drop_duplicates(subset=["user_team_id", "player_id"], keep="last")
    teams = pd.DataFrame({
        "coach_first_name": last["coach"],
        "first_name":       last["first_name"],
        "last_name":        last["last_name"],
        "team":             last["team_abbrev"],
        "position":         last["pos_1"],
        "avg":              last["avg"],
        "avg3":             last["avg3"],
        "price":            last["price"],
        "feed_id":          last["feed_id"],
    }).reset_index(drop=True)

    coaches = league.drop_duplicates(subset=["team_id"], keep="last")
    coaches = pd.DataFrame({
        "coach_team_id":    coaches["team_id"].astype(int),
        "coach_id":         coaches["team_id"].astype(int),
        "coach_first_name": coaches["coach"],
        "coach_team_name":  coaches["team"],
    }).reset_index(drop=True)

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    matches.to_csv(out_dir / "player_match_results.csv", index=False)
    teams.to_csv(out_dir / "current_teams.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)

    print(f"  [{year}] building draft heat map data...")
    build_draft_csv_from_legacy_json(year)

    print(f"  [{year}] OK (legacy fallback) — {ladder['round'].nunique()} rounds "
          f"(max round {ladder['round'].max()}), {len(matches)} player-match rows -> {out_dir}")


def ingest_year(year: int) -> None:
    if year in LEAGUE_IDS and raw_dump_is_usable(year):
        ingest_year_from_raw(year)
    else:
        print(f"  [{year}] raw dump unusable/unavailable — falling back to legacy fantasy_banter_data")
        ingest_year_from_legacy(year)


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2022, 2023, 2024, 2025]
    print(f"Ingesting seasons: {years}")
    for y in years:
        ingest_year(y)
