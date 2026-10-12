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
import re
import sys
from collections import defaultdict
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

WORKBOOK_PATH = LEGACY_DIR / "2016 - 2019" / "Master Fantasy Banter Workbook - MR edit.xlsx"
WORKBOOK_YEARS = {2016, 2017, 2018, 2019}
# "Lest" (4-letter, used only in the 2019 draft sheet) doesn't match the
# standard DRAFT_ALIAS_MAP key "LESTER" — every other alias in this workbook
# (RICHO, KAPPAZ, JMERC, GARTER, MELONS, PMAC, CHIEF) matches directly.
_WORKBOOK_ALIAS_FIX = {"LEST": "LESTER"}


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


def build_player_matches_from_raw(year: int, coach_map: dict[int, dict], max_round: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (matches, sc_round_data):
      matches — one row per (round, coach, player) for players actually
        rostered by one of our coaches (the existing player_match_results.csv
        shape).
      sc_round_data — one row per (round, player) for EVERY player in the
        players-cf_embed bulk dump that round, regardless of fantasy
        ownership. Captured in the same pass (players-cf_embed is already
        being read for `matches`) rather than re-parsing the same ~6s/file
        RDS dump a second time. Feeds website/data/draft_data.py's
        _compute_effective_avg() so it can tell "missed because of a bye"
        apart from "missed because unowned" for historical seasons, same as
        load_sc_round_files() already does for the live season.
    """
    rows = []
    sc_round_rows = []
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
            s0 = stats[0] if stats else {}
            points = float(_v(s0.get("points")) or 0)
            fid = _v(p["feed_id"])
            team_abbrev = str(_v(p["team"]["abbrev"])) if p.get("team") else ""
            player_lookup[pid] = {
                "first_name": str(_v(p["first_name"])),
                "last_name":  str(_v(p["last_name"])),
                "feed_id":    fid,
                "team":       team_abbrev,
                "points":     points,
            }
            sc_round_rows.append({
                "feed_id": fid, "round": rnd, "team_abbrev": team_abbrev,
                "played": int(_v(s0.get("games")) or 0), "points": points,
            })

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

    return pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(sc_round_rows).reset_index(drop=True)


def _build_transactions_rows(waivers_raw: list, trades_raw: list, teamtrades_raw: list) -> tuple[list[dict], set]:
    """
    Ports website/data/api_ingest.py's _fetch_and_save_transactions() endpoint
    parsing onto archived RDS lists in the same shape as the live SC API's
    processedWaivers/trades/teamtrades endpoints -> Waiver/Free Agent/Trade.
    Same column shape as the live transactions.csv so
    website/data/acquisition.py needs no changes.
    """
    rows: list[dict] = []
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

    for t in trades_raw:
        buy, sell = t.get("buy_player") or {}, t.get("sell_player") or {}
        rows.append({
            "round":                      _v(t.get("round")),
            "user_team_id":               _v(t.get("user_id")),
            "processed":                  str(_v(t.get("action_time"))),
            "player_id":                  _v(t.get("buy_player_id")) if t.get("buy_player_id") is not None else _v(buy.get("id")),
            "player.feed_id":             _v(buy.get("feed_id")),
            "player.first_name":          _v(buy.get("first_name")),
            "player.last_name":           _v(buy.get("last_name")),
            "player.team.abbrev":         _v((buy.get("team") or {}).get("abbrev")),
            "dropped_player_id":          _v(t.get("sell_player_id")) if t.get("sell_player_id") is not None else _v(sell.get("id")),
            "dropped_player.feed_id":     _v(sell.get("feed_id")),
            "dropped_player.first_name":  _v(sell.get("first_name")),
            "dropped_player.last_name":   _v(sell.get("last_name")),
            "dropped_player.team.abbrev": _v((sell.get("team") or {}).get("abbrev")),
            "source":                     "Free Agent",
        })

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

    return rows, seen_trade_ids


def _write_transactions_outputs(rows: list[dict], waivers_raw: list, trades_raw: list, teamtrades_raw: list,
                                 seen_trade_ids: set, out_dir: Path, waivers_json_path: Path | None) -> None:
    if not rows:
        print(f"    (no transactions found — {len(waivers_raw)} waivers, {len(trades_raw)} FA, {len(teamtrades_raw)} trades raw)")
        return

    pd.DataFrame(rows).to_csv(out_dir / "transactions.csv", index=False)

    if waivers_raw and waivers_json_path is not None:
        waivers_json = [
            {"player_id": int(_v(t.get("player_id"))), "round": int(_v(t.get("round"))),
             "processed": str(_v(t.get("processed")))}
            for t in waivers_raw
        ]
        waivers_json_path.write_text(json.dumps(waivers_json), encoding="utf-8")
    print(f"    transactions.csv: {len(rows)} rows "
          f"({len(waivers_raw)} waiver, {len(trades_raw)} FA, {len(seen_trade_ids)} approved trades)")


def build_transactions_from_raw(year: int, out_dir: Path) -> None:
    """processedWaivers.rds/trades.rds/teamtrades.rds for a 2022+ raw-dump
    year (see _build_transactions_rows) -> transactions.csv + a
    processedWaivers.json sidecar so config.processed_waivers_json_path(year)
    finds it."""
    lid = LEAGUE_IDS[year]
    src = RAW_DIR / str(year)

    def _load(name):
        path = src / f"leagues_{lid}_{name}.rds"
        if not path.exists():
            return []
        try:
            return rdata.read_rds(str(path))
        except Exception:
            return []

    waivers_raw = _load("processedWaivers")
    trades_raw = _load("trades")
    teamtrades_raw = _load("teamtrades")
    rows, seen_trade_ids = _build_transactions_rows(waivers_raw, trades_raw, teamtrades_raw)
    _write_transactions_outputs(rows, waivers_raw, trades_raw, teamtrades_raw, seen_trade_ids, out_dir,
                                 src / f"leagues_{lid}_processedWaivers.json")


def build_transactions_2021(out_dir: Path) -> None:
    """Same schema as the 2022+ raw dumps, under sc_*.rds filenames instead
    of leagues_{id}_*.rds — archive/legacy/fantasy_banter_data/2021/raw/."""
    src = LEGACY_DIR / "2021" / "raw"

    def _load(name):
        path = src / f"sc_{name}.rds"
        if not path.exists():
            return []
        try:
            return rdata.read_rds(str(path))
        except Exception:
            return []

    waivers_raw = _load("processedWaivers")
    trades_raw = _load("trades")
    teamtrades_raw = _load("teamtrades")
    rows, seen_trade_ids = _build_transactions_rows(waivers_raw, trades_raw, teamtrades_raw)
    _write_transactions_outputs(rows, waivers_raw, trades_raw, teamtrades_raw, seen_trade_ids, out_dir,
                                 out_dir / "processedWaivers.json")


def build_transactions_2020(out_dir: Path) -> None:
    """Same schema as the 2022+ raw dumps (with trades missing the top-level
    buy_player_id/sell_player_id fields — _build_transactions_rows falls back
    to buy_player['id']/sell_player['id']) under 2020_SC_*_DATA.RDS filenames
    — archive/legacy/fantasy_banter_data/2020/raw/."""
    src = LEGACY_DIR / "2020" / "raw"

    def _load(name):
        path = src / f"2020_SC_{name}.RDS"
        if not path.exists():
            return []
        try:
            return rdata.read_rds(str(path))
        except Exception:
            return []

    waivers_raw = _load("WAIVERS_DATA")
    trades_raw = _load("TRADES_DATA")
    teamtrades_raw = _load("TEAM_TRADES_DATA")
    rows, seen_trade_ids = _build_transactions_rows(waivers_raw, trades_raw, teamtrades_raw)
    _write_transactions_outputs(rows, waivers_raw, trades_raw, teamtrades_raw, seen_trade_ids, out_dir,
                                 out_dir / "processedWaivers.json")


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
        if not build_draft_csv_from_legacy_json(year):
            print(f"    trying draft-prep workbook instead")
            build_draft_csv_from_workbook(year, coach_map)
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


def build_draft_csv_from_legacy_json(year: int) -> bool:
    """ASLdraft_{year}.JSON (gzip'd RDS-wrapped JSON string, alias-keyed coach
    column already matching config.DRAFT_ALIAS_MAP) -> draft_prep/SC {year}/
    draft_{year}_result.csv. Returns True on success."""
    src = LEGACY_DIR / str(year) / f"ASLdraft_{year}.JSON"
    if not src.exists():
        print(f"    no ASLdraft_{year}.JSON found")
        return False
    try:
        d = rdata.read_rds(str(src))
        payload = json.loads(d["x"][0])
    except Exception as exc:
        print(f"    could not parse {src.name}: {exc}")
        return False

    player_list_path = PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"{year}_SC_Player_list.csv"
    if not player_list_path.exists():
        print(f"    no {player_list_path.name} — can't resolve draft player names")
        return False
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
    return True


def build_draft_csv_from_workbook(year: int, coach_map: dict[int, dict]) -> bool:
    """
    Recovered source for 2024 (no other draft data exists anywhere): a sheet
    named str(year) in draft_prep/SC 2026/Previous drafts combined - 2026
    draft prep.xlsx, in the same raw-export shape as draft_{year}.csv but
    with player name/feed_id already resolved (columns: id, league_id,
    user_team_id, round, pick, player_id, position, autopicked, time,
    year_id, feed_id, name).
    """
    import openpyxl

    xlsx_path = PROJECT_ROOT / "draft_prep" / "SC 2026" / "Previous drafts combined - 2026 draft prep.xlsx"
    if not xlsx_path.exists():
        print(f"    no draft-prep workbook found")
        return False
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    if str(year) not in wb.sheetnames:
        print(f"    no '{year}' sheet in draft-prep workbook")
        return False

    ws = wb[str(year)]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        team_id, rnd, pick, fid, name = int(r[2]), int(r[3]), int(r[4]), int(r[10]), str(r[11])
        coach = coach_map.get(team_id, {}).get("coach")
        if coach is None:
            continue
        parts = name.split(" ", 1)
        display = f"{parts[0][0]}.{parts[1]}" if len(parts) == 2 else name
        rows.append({
            "Round": rnd, "Pick": pick, "Player": display, "Coach": coach,
            "Round Coach Order": 0, "Dataset count": 1, "feed_id": fid,
        })

    out_path = PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"draft_{year}_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    draft_{year}_result.csv: {len(rows)} picks -> {out_path} (from workbook)")
    return True


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
    matches, sc_round_data = build_player_matches_from_raw(year, coach_map, max_round)
    teams = build_current_teams_from_raw(year, coach_map, max_round)
    coaches = build_coach_list(coach_map)

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    matches.to_csv(out_dir / "player_match_results.csv", index=False)
    teams.to_csv(out_dir / "current_teams.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)
    sc_round_data.to_csv(out_dir / "sc_round_data.csv", index=False)
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


def _build_season_from_league_player_tables(league: pd.DataFrame, player: pd.DataFrame):
    """
    Shared transform for any source already in the league_data.RDS /
    player_data.RDS column shape (round, fixture, team_id, team, coach,
    team_score, ..., wins, draws, losses, points_for, points_against,
    position, points / feed_id, player_id, first_name, ..., picked, pos_1).
    Used by both the legacy RDS fallback (2025) and the 2020 clean CSVs.
    Returns (ladder, fixtures, matches, teams, coaches).
    """
    league_good = league.dropna(subset=["wins", "draws", "losses", "points_for", "points_against", "points", "position"]).copy()
    league_good["round"] = league_good["round"].astype(int)

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
        # NaN here means "rostered, no score recorded" (almost always a bye —
        # confirmed by cross-checking Footywire's real game log, e.g. Clayton
        # Oliver R14 2021 picked=True/points=NaN/type=nonscoring lines up with
        # a genuine Melbourne bye). Other rows represent the identical
        # situation with points=0 instead — normalise to 0 for consistency.
        "points_x":         player_good["points"].fillna(0),
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

    return ladder, fixtures, matches, teams, coaches


def ingest_year_from_legacy(year: int) -> None:
    src_dir = LEGACY_DIR / str(year)
    league_path, player_path = src_dir / "league_data.RDS", src_dir / "player_data.RDS"
    if not league_path.exists() or not player_path.exists():
        print(f"  [{year}] SKIP — no usable source found (raw dump unreliable, "
              f"and {league_path.name}/{player_path.name} not found)")
        return

    league = _read_rds_df(league_path)
    player = _read_rds_df(player_path)

    if league.dropna(subset=["wins"])["round"].nunique() < 2:
        print(f"  [{year}] SKIP — legacy league_data.RDS has too little usable data")
        return

    out_dir = SEASONS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder, fixtures, matches, teams, coaches = _build_season_from_league_player_tables(league, player)

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    matches.to_csv(out_dir / "player_match_results.csv", index=False)
    teams.to_csv(out_dir / "current_teams.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)

    print(f"  [{year}] building draft heat map data...")
    build_draft_csv_from_legacy_json(year)

    print(f"  [{year}] OK (legacy fallback) — {ladder['round'].nunique()} rounds "
          f"(max round {ladder['round'].max()}), {len(matches)} player-match rows -> {out_dir}")


def build_draft_csv_2020(player: pd.DataFrame) -> None:
    """2020_00_SC_DRAFT.RDS (raw SC draft export, same shape as draft_{year}.csv)
    joined against players_2020.csv's own player_id -> feed_id/name mapping
    (no separate players-cf_embed source needed — it's already in the table)."""
    src = LEGACY_DIR / "2020" / "raw" / "2020_00_SC_DRAFT.RDS"
    if not src.exists():
        print(f"    no 2020_00_SC_DRAFT.RDS found — skipping draft heat map for 2020")
        return
    try:
        picks = rdata.read_rds(str(src))
    except Exception as exc:
        print(f"    could not parse {src.name}: {exc}")
        return

    coach_by_team = (
        player.dropna(subset=["user_team_id", "coach"])
        .drop_duplicates(subset=["user_team_id"])
        .set_index("user_team_id")["coach"].to_dict()
    )
    player_lookup = (
        player.dropna(subset=["player_id", "feed_id"])
        .drop_duplicates(subset=["player_id"])
        .set_index("player_id")[["first_name", "last_name", "feed_id"]]
        .to_dict("index")
    )

    rows = []
    for p in picks:
        team_id = int(_v(p.get("user_team_id")))
        pid = int(_v(p.get("player_id")))
        coach = coach_by_team.get(team_id)
        info = player_lookup.get(pid)
        if coach is None or info is None:
            continue
        name = f"{info['first_name'][0]}.{info['last_name']}" if info["first_name"] else info["last_name"]
        rows.append({
            "Round": int(_v(p.get("round"))), "Pick": int(_v(p.get("pick"))), "Player": name,
            "Coach": coach, "Round Coach Order": 0, "Dataset count": 1, "feed_id": info["feed_id"],
        })
    out_path = PROJECT_ROOT / "draft_prep" / "SC 2020" / "draft_2020_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    draft_2020_result.csv: {len(rows)} picks -> {out_path}")


def _extract_sc_round_rows(players: list, rnd: int) -> list[dict]:
    """Same bulk-player extraction as build_player_matches_from_raw's inner
    loop, factored out so the 2020/2021 bulk RDS sources (same player_stats
    shape as players-cf_embed) can reuse it."""
    rows = []
    for p in players:
        stats = p.get("player_stats") or []
        s0 = stats[0] if stats else {}
        rows.append({
            "feed_id": _v(p.get("feed_id")), "round": rnd,
            "team_abbrev": str(_v(p["team"]["abbrev"])) if p.get("team") else "",
            "played": int(_v(s0.get("games")) or 0), "points": float(_v(s0.get("points")) or 0),
        })
    return rows


def build_sc_round_data_2020(out_dir: Path) -> None:
    """archive/legacy/fantasy_banter_data/2020/raw/2020_{NN}_SC_PLAYER_DATA.RDS
    (rounds 1-18, same bulk player_stats shape as the 2022+ players-cf_embed
    source) -> data/seasons/2020/sc_round_data.csv."""
    src = LEGACY_DIR / "2020" / "raw"
    rows = []
    for rnd in range(1, 19):
        path = src / f"2020_{rnd:02d}_SC_PLAYER_DATA.RDS"
        if not path.exists():
            continue
        try:
            players = rdata.read_rds(str(path))
        except Exception:
            continue
        rows.extend(_extract_sc_round_rows(players, rnd))
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "sc_round_data.csv", index=False)
        print(f"    sc_round_data.csv: {len(rows)} player-round rows")


def build_sc_round_data_2021(out_dir: Path) -> None:
    """archive/legacy/fantasy_banter_data/2021/raw/sc_players_{N}.rds
    (rounds 1-23) -> data/seasons/2021/sc_round_data.csv."""
    src = LEGACY_DIR / "2021" / "raw"
    rows = []
    for rnd in range(1, 24):
        path = src / f"sc_players_{rnd}.rds"
        if not path.exists():
            continue
        try:
            players = rdata.read_rds(str(path))
        except Exception:
            continue
        rows.extend(_extract_sc_round_rows(players, rnd))
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "sc_round_data.csv", index=False)
        print(f"    sc_round_data.csv: {len(rows)} player-round rows")


def build_sc_round_data_from_fanfooty(year: int, out_dir: Path, max_round: int) -> None:
    """
    data/processed/{year}_round_N_fanfooty_data.csv (covers 2011-2026,
    every AFL player every round regardless of fantasy ownership) used as a
    bulk sc_round_data.csv source for 2016-2019, which have no SC-native bulk
    endpoint at all. Fanfooty's own "Player ID" is confirmed (cross-checked
    against the SC player lists this session — Ed Curnow 270146, Rory Sloane
    270963, Dustin Martin 290847, Tom Hawkins 261510, all exact matches) to
    be the same numbering as SC's feed_id. A row's mere presence means the
    player featured in that round's match, so played=1 always.
    """
    import glob
    rows = []
    for rnd in range(1, max_round + 1):
        paths = glob.glob(str(PROJECT_ROOT / "data" / "processed" / f"{year}_round_{rnd}_fanfooty_data.csv"))
        if not paths:
            continue
        df = pd.read_csv(paths[0], low_memory=False, usecols=["Player ID", "Team", "SC"])
        df = df.dropna(subset=["Player ID"])
        for _, r in df.iterrows():
            rows.append({
                "feed_id": int(r["Player ID"]), "round": rnd,
                "team_abbrev": r["Team"], "played": 1,
                "points": float(r["SC"]) if pd.notna(r["SC"]) else 0.0,
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "sc_round_data.csv", index=False)
        print(f"    sc_round_data.csv: {len(rows)} player-round rows (from fanfooty)")


def ingest_year_2020() -> None:
    src_dir = LEGACY_DIR / "2020" / "clean"
    league_path, player_path = src_dir / "league_2020.csv", src_dir / "players_2020.csv"
    if not league_path.exists() or not player_path.exists():
        print(f"  [2020] SKIP — {league_path.name}/{player_path.name} not found")
        return

    league = pd.read_csv(league_path)
    player = pd.read_csv(player_path, low_memory=False)

    out_dir = SEASONS_DIR / "2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    # No FINALS FLAG column here (unlike the 2016-2019 workbook) — rounds
    # past the regular season (17-18) are excluded from the regular tables
    # and handled separately by build_finals_json_from_league_table below.
    ladder, fixtures, matches, teams, coaches = _build_season_from_league_player_tables(
        league[league["round"] <= 16].copy(), player[player["round"] <= 16].copy()
    )

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    matches.to_csv(out_dir / "player_match_results.csv", index=False)
    teams.to_csv(out_dir / "current_teams.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)

    print(f"  [2020] building finals bracket json...")
    build_finals_json_from_league_table(league, out_dir, total_regular_rounds=16)

    print(f"  [2020] building transactions from raw waiver/trade dumps...")
    build_transactions_2020(out_dir)

    print(f"  [2020] building draft heat map data...")
    build_draft_csv_2020(player)

    print(f"  [2020] building bulk sc_round_data (every player, every round)...")
    build_sc_round_data_2020(out_dir)

    print(f"  [2020] OK (clean CSV) — {ladder['round'].nunique()} rounds "
          f"(max round {ladder['round'].max()}), {len(matches)} player-match rows -> {out_dir}")


def build_finals_json_from_league_table(league: pd.DataFrame, out_dir: Path, total_regular_rounds: int) -> None:
    """Builds match_json_rd{N}.json for finals rounds found in a league_data.RDS
    -shaped table (used for 2020, whose finals rounds — unlike 2016-2019's
    workbook — aren't separately flagged, so anything past total_regular_rounds
    with real scores is treated as a finals round)."""
    finals = league[league["round"] > total_regular_rounds].dropna(subset=["team_score"])
    if finals.empty:
        return
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for rnd, rdf in finals.groupby("round"):
        seen_games = set()
        fixtures_out = []
        for _, r in rdf.iterrows():
            game_key = tuple(sorted([r["team"], r["opponent_team"]]))
            if game_key in seen_games:
                continue
            seen_games.add(game_key)
            fixtures_out.append({
                "user_team1": {"user": {"first_name": r["coach"]}, "stats": [{"points": float(r["team_score"])}]},
                "user_team2": {"user": {"first_name": r["opponent_coach"]}, "stats": [{"points": float(r["opponent_score"])}]},
            })
        (json_dir / f"match_json_rd{int(rnd)}.json").write_text(
            json.dumps({"fixtures": fixtures_out}), encoding="utf-8"
        )


def build_draft_csv_2021() -> None:
    """ASL_Draft_2021_data.csv is already clean: round, pick, order_id, coach
    (alias), player_id (= feed_id directly), player_name, team, position, ..."""
    src = LEGACY_DIR / "2021" / "ASL_Draft_2021_data.csv"
    if not src.exists():
        print(f"    no ASL_Draft_2021_data.csv found — skipping draft heat map for 2021")
        return
    df = pd.read_csv(src).dropna(subset=["round", "pick", "player_id"])
    rows = [{
        "Round": int(r["round"]), "Pick": int(r["pick"]), "Player": r["player_name"],
        "Coach": _resolve_coach_alias(r["coach"]), "Round Coach Order": int(r["order_id"]),
        "Dataset count": 1, "feed_id": int(r["player_id"]),
    } for _, r in df.iterrows()]
    out_path = PROJECT_ROOT / "draft_prep" / "SC 2021" / "draft_2021_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    draft_2021_result.csv: {len(rows)} picks -> {out_path}")


def ingest_year_2021() -> None:
    src_dir = LEGACY_DIR / "2021" / "clean"
    league_path, player_path = src_dir / "league_2021.csv", src_dir / "players_2021.csv"
    if not league_path.exists() or not player_path.exists():
        print(f"  [2021] SKIP — {league_path.name}/{player_path.name} not found")
        return

    league = pd.read_csv(league_path)
    player = pd.read_csv(player_path, low_memory=False)

    out_dir = SEASONS_DIR / "2021"
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder, fixtures, matches, teams, coaches = _build_season_from_league_player_tables(
        league[league["round"] <= 21].copy(), player[player["round"] <= 21].copy()
    )

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    matches.to_csv(out_dir / "player_match_results.csv", index=False)
    teams.to_csv(out_dir / "current_teams.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)

    print(f"  [2021] building finals bracket json...")
    build_finals_json_from_league_table(league, out_dir, total_regular_rounds=21)

    print(f"  [2021] building transactions from raw waiver/trade dumps...")
    build_transactions_2021(out_dir)

    print(f"  [2021] building draft heat map data...")
    build_draft_csv_2021()

    print(f"  [2021] building bulk sc_round_data (every player, every round)...")
    build_sc_round_data_2021(out_dir)

    print(f"  [2021] OK (clean CSV) — {ladder['round'].nunique()} rounds "
          f"(max round {ladder['round'].max()}), {len(matches)} player-match rows -> {out_dir}")


def _resolve_coach_alias(raw) -> str:
    from website.config import DRAFT_ALIAS_MAP
    key = str(raw).strip().upper()
    key = _WORKBOOK_ALIAS_FIX.get(key, key)
    return DRAFT_ALIAS_MAP.get(key, str(raw))


def _load_workbook_sheet(wb, sheet_name: str) -> pd.DataFrame:
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    headers = next(rows_iter)
    data = [r for r in rows_iter if any(v is not None for v in r)]
    return pd.DataFrame(data, columns=headers)


def _workbook_coach_map(teams_df: pd.DataFrame, year: int) -> dict[int, dict]:
    yr = teams_df[teams_df["SEASON"] == year]
    names = sorted(yr["SC TEAM"].unique())
    name_to_id = {name: 900000 + year * 100 + i for i, name in enumerate(names)}
    coach_map = {}
    for _, row in yr.iterrows():
        tid = name_to_id[row["SC TEAM"]]
        coach_map[tid] = {"coach": _resolve_coach_alias(row["SC COACH"]), "team_name": row["SC TEAM"]}
    return coach_map, {name: tid for tid, name in ((t, c["team_name"]) for t, c in coach_map.items())}


def build_ladder_from_workbook(fixture_df: pd.DataFrame, year: int, name_to_id: dict) -> pd.DataFrame:
    yr = fixture_df[(fixture_df["SEASON"] == year) & (fixture_df["FINALS FLAG"] == 0)].copy()
    for col in ["WIN", "DRAW", "LOSS", "TEAM SCORE", "OPPONENT SCORE", "ROUND"]:
        yr[col] = pd.to_numeric(yr[col], errors="coerce").fillna(0)
    yr = yr.sort_values(["TEAM NAME", "ROUND"])
    yr["league_points_round"] = yr["WIN"] * 4 + yr["DRAW"] * 2
    for col, src in [("cum_wins", "WIN"), ("cum_draws", "DRAW"), ("cum_losses", "LOSS"),
                      ("cum_pf", "TEAM SCORE"), ("cum_pa", "OPPONENT SCORE"),
                      ("cum_points", "league_points_round")]:
        yr[col] = yr.groupby("TEAM NAME")[src].cumsum()

    rows = []
    for rnd, rdf in yr.groupby("ROUND"):
        rdf = rdf.sort_values(["cum_points", "cum_pf"], ascending=[False, False]).reset_index(drop=True)
        for i, r in rdf.iterrows():
            tid = name_to_id[r["TEAM NAME"]]
            rows.append({
                "round": int(rnd), "coach_id": tid, "coach_team_id": tid,
                "coach_first_name": _resolve_coach_alias(r["COACH"]), "coach_team_name": r["TEAM NAME"],
                "wins": int(r["cum_wins"]), "draws": int(r["cum_draws"]), "losses": int(r["cum_losses"]),
                "points_for": r["cum_pf"], "points_against": r["cum_pa"],
                "league_points": int(r["cum_points"]), "position": i + 1,
            })
    return pd.DataFrame(rows)


def build_fixtures_from_workbook(fixture_df: pd.DataFrame, year: int, name_to_id: dict) -> pd.DataFrame:
    yr = fixture_df[(fixture_df["SEASON"] == year) & (fixture_df["FINALS FLAG"] == 0)].copy()
    for col in ["ROUND", "TEAM SCORE", "OPPONENT SCORE"]:
        yr[col] = pd.to_numeric(yr[col], errors="coerce").fillna(0)
    rows = []
    for _, r in yr.iterrows():
        rows.append({
            "round_number": int(r["ROUND"]),
            "match_name": f'{r["TEAM NAME"]} vs {r["OPPONENT NAME"]}',
            "team_id": name_to_id[r["TEAM NAME"]], "team_name": r["TEAM NAME"],
            "team_coach": _resolve_coach_alias(r["COACH"]), "team_points": r["TEAM SCORE"],
            "opposition_team_id": name_to_id[r["OPPONENT NAME"]], "opposition_team_name": r["OPPONENT NAME"],
            "opposition_team_coach": _resolve_coach_alias(r["OPPONENT COACH"]),
            "opposition_team_points": r["OPPONENT SCORE"], "Rank": 0,
        })
    return pd.DataFrame(rows)


def build_player_matches_from_workbook(scores_df: pd.DataFrame, year: int, name_to_id: dict) -> pd.DataFrame:
    yr = scores_df[scores_df["SEASON"] == year].copy()
    for col in ["ROUND", "PTS"]:
        yr[col] = pd.to_numeric(yr[col], errors="coerce").fillna(0)
    # DATA - Round Scores has no FINALS FLAG (unlike DATA - Fixture) — exclude
    # rounds past the regular season (22-23, 2-week finals) to match
    # build_ladder_from_workbook/build_fixtures_from_workbook and Footywire's
    # own season-average convention, which also excludes finals.
    yr = yr[yr["ROUND"] <= 21]
    rows = []
    for _, r in yr.iterrows():
        tid = name_to_id.get(r["SC TEAM"])
        if tid is None:
            continue
        name = str(r["PLAYER"])
        parts = name.split(".", 1)
        first, last = (parts[0], parts[1]) if len(parts) == 2 else ("", name)
        on_field = r["POS"] != "BEN"
        rows.append({
            "round_x": int(r["ROUND"]), "coach_id": tid, "coach_team_id": tid,
            "coach_first_name": _resolve_coach_alias(r["SC COACH"]), "coach_team_name": r["SC TEAM"],
            "player_id": pd.NA, "played_position": r["POS"] if on_field else None,
            "points_x": r["PTS"], "on_field": on_field, "picked": on_field,
            "feed_id": pd.NA, "first_name": first, "last_name": last,
            "team": r["TEAM"], "position": r["POS"],
        })
    return pd.DataFrame(rows)


def build_finals_json_from_workbook(fixture_df: pd.DataFrame, year: int, out_dir: Path) -> None:
    yr = fixture_df[(fixture_df["SEASON"] == year) & (fixture_df["FINALS FLAG"] == 1)].copy()
    if yr.empty:
        return
    for col in ["ROUND", "TEAM SCORE", "OPPONENT SCORE"]:
        yr[col] = pd.to_numeric(yr[col], errors="coerce").fillna(0)
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for rnd, rdf in yr.groupby("ROUND"):
        seen_games = set()
        fixtures_out = []
        for _, r in rdf.iterrows():
            game_key = tuple(sorted([r["TEAM NAME"], r["OPPONENT NAME"]]))
            if game_key in seen_games:
                continue
            seen_games.add(game_key)
            fixtures_out.append({
                "user_team1": {"user": {"first_name": _resolve_coach_alias(r["COACH"])},
                               "stats": [{"points": float(r["TEAM SCORE"])}]},
                "user_team2": {"user": {"first_name": _resolve_coach_alias(r["OPPONENT COACH"])},
                               "stats": [{"points": float(r["OPPONENT SCORE"])}]},
            })
        (json_dir / f"match_json_rd{int(rnd)}.json").write_text(
            json.dumps({"fixtures": fixtures_out}), encoding="utf-8"
        )


def _workbook_name_lookups() -> tuple[dict, dict, dict]:
    """
    Union of every player-identity source we have across years (feed_id is
    stable per physical person across seasons, so a player active in ANY of
    these years resolves): master_player_data.csv (2006-2026), the SC player
    lists for 2022-2026, and the archived 2020/2021 clean player CSVs. This
    matters because a single year's source alone is often missing players who
    retired before/after it — e.g. Nat Fyfe and Joel Selwood aren't in
    master_player_data.csv at all, but are in the 2022 SC player list.

    Returns (full_name_lookup, initials_lookup, first_initial_lookup):
      full_name_lookup:      (first_lower, last_lower) -> feed_id
      initials_lookup:       (initials e.g. "j.p.", last_lower) -> feed_id
      first_initial_lookup:  (first_initial, last_lower) -> set of feed_ids
    """
    full_name, initials, first_initial = {}, {}, defaultdict(set)

    def add(df: pd.DataFrame, fcol: str, lcol: str, idcol: str) -> None:
        for _, r in df.dropna(subset=[fcol, lcol, idcol]).iterrows():
            first, last = str(r[fcol]).strip(), str(r[lcol]).strip()
            if not first or not last:
                continue
            fid = int(r[idcol])
            full_name.setdefault((first.lower(), last.lower()), fid)
            init_key = (".".join(w[0].lower() for w in first.split() if w) + ".", last.lower())
            initials.setdefault(init_key, fid)
            first_initial[(first[0].lower(), last.lower())].add(fid)

    # Order matters: setdefault() means the FIRST source to claim a
    # (name, lastname) key wins. The SC-native sources (player lists,
    # 2020/2021 clean CSVs) consistently use a middle initial to disambiguate
    # same-name players (e.g. plain "Tom Lynch" = Adelaide's, "Tom J. Lynch" =
    # Richmond's) — master_player_data.csv does NOT make this distinction
    # (only one deduped "Tom"+"Lynch" row, pointing at Richmond's id), so it
    # goes last, as a fallback for names missing from every SC-native source
    # entirely (e.g. Nat Fyfe / Joel Selwood aren't in master_player_data.csv
    # at all, but are in the 2022 SC player list, so order doesn't matter for
    # them — it only matters for genuine same-name collisions like Lynch).
    for year in [2020, 2021]:
        players_path = LEGACY_DIR / str(year) / "clean" / f"players_{year}.csv"
        if players_path.exists():
            add(pd.read_csv(players_path, low_memory=False), "first_name", "last_name", "feed_id")

    # Fanfooty's own "Player ID" is, confirmed by direct cross-check against
    # the SC player lists (Ed Curnow 270146, Rory Sloane 270963, Dustin
    # Martin 290847, Tom Hawkins 261510 — all exact matches), the SAME
    # numbering as SC's feed_id. This single multi-year file (2011-2026)
    # resolves players who'd retired before any SC-native source we have
    # starts (e.g. Alex Rance, Jarrad McVeigh, Luke Hodge — all retired
    # 2019/2020, absent from master_player_data.csv and every SC player
    # list, but present here since fanfooty scrapes every AFL player
    # regardless of fantasy relevance).
    fanfooty_path = PROJECT_ROOT / "data" / "processed" / "master_fanfooty_data.csv"
    if fanfooty_path.exists():
        add(pd.read_csv(fanfooty_path, low_memory=False, usecols=["Player ID", "First Name", "Surname"]),
            "First Name", "Surname", "Player ID")

    for year in [2022, 2023, 2024, 2025, 2026]:
        plist_path = PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"{year}_SC_Player_list.csv"
        if plist_path.exists():
            add(pd.read_csv(plist_path, low_memory=False), "first_name", "last_name", "feed_id")

    master = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "master_player_data.csv",
        low_memory=False, usecols=["Player ID", "First Name", "Last Name"],
    )
    add(master, "First Name", "Last Name", "Player ID")

    return full_name, initials, first_initial


def _resolve_draft_player(footywire: str, short_name: str, full_name_lookup: dict,
                           initials_lookup: dict, first_initial_lookup: dict) -> int | None:
    """Tries, in order: exact initials+lastname match from the short Name
    column (most precise signal we have — e.g. "T.J. Lynch" vs "T.Lynch"
    deliberately disambiguates Richmond's Tom J. Lynch from Adelaide's plain
    Tom Lynch; checked first so an ambiguous full-name hit can't shadow it)
    -> exact full-name match (Footywire column) -> unique first-initial+
    lastname match (only if exactly one candidate across all sources —
    genuinely ambiguous multi-candidate cases are left unmatched)."""
    m = re.match(r"^((?:[A-Za-z]\.)+)\s*(.+)$", short_name.strip())
    if m:
        initials_part, last = m.group(1).lower(), m.group(2).strip().lower()
        fid = initials_lookup.get((initials_part, last))
        if fid is not None:
            return fid

    full = footywire.strip()
    if full and full not in ("#N/A", "nan"):
        parts = full.split(" ", 1)
        if len(parts) == 2:
            fid = full_name_lookup.get((parts[0].lower(), parts[1].lower()))
            if fid is not None:
                return fid

    if not m:
        return None
    first_initial = initials_part[0]
    candidates = first_initial_lookup.get((first_initial, last), set())
    return next(iter(candidates)) if len(candidates) == 1 else None


def build_draft_csv_from_workbook_sheet(draft_df: pd.DataFrame, year: int) -> None:
    draft_df = draft_df.dropna(subset=["Round", "Pick", "Name"])
    full_name_lookup, initials_lookup, first_initial_lookup = _workbook_name_lookups()
    rows = []
    matched = 0
    for _, r in draft_df.iterrows():
        fid = _resolve_draft_player(str(r["Footywire"]), str(r["Name"]),
                                     full_name_lookup, initials_lookup, first_initial_lookup)
        if fid is not None:
            matched += 1
        rows.append({
            "Round": int(r["Round"]), "Pick": int(r["Pick"]), "Player": r["Name"],
            "Coach": _resolve_coach_alias(r["Team"]), "Round Coach Order": 0,
            "Dataset count": 1, "feed_id": fid,
        })
    out_path = PROJECT_ROOT / "draft_prep" / f"SC {year}" / f"draft_{year}_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    draft_{year}_result.csv: {len(rows)} picks, {matched} feed_id matches -> {out_path}")


def ingest_year_from_workbook(year: int) -> None:
    import openpyxl

    if not WORKBOOK_PATH.exists():
        print(f"  [{year}] SKIP — workbook not found at {WORKBOOK_PATH}")
        return

    out_dir = SEASONS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.load_workbook(str(WORKBOOK_PATH), read_only=True, data_only=True)
    teams_df = _load_workbook_sheet(wb, "DATA - Teams")
    fixture_df = _load_workbook_sheet(wb, "DATA - Fixture")

    coach_map, name_to_id = _workbook_coach_map(teams_df, year)
    if not coach_map:
        print(f"  [{year}] SKIP — no teams found in workbook for this year")
        return

    ladder = build_ladder_from_workbook(fixture_df, year, name_to_id)
    fixtures = build_fixtures_from_workbook(fixture_df, year, name_to_id)
    coaches = build_coach_list(coach_map)

    ladder.to_csv(out_dir / "ladder.csv", index=False)
    fixtures.to_csv(out_dir / "fixture_results_by_team.csv", index=False)
    coaches.to_csv(out_dir / "coach_list.csv", index=False)
    build_finals_json_from_workbook(fixture_df, year, out_dir)

    scores_df = _load_workbook_sheet(wb, "DATA - Round Scores")
    if year in scores_df["SEASON"].unique():
        matches = build_player_matches_from_workbook(scores_df, year, name_to_id)
        matches.to_csv(out_dir / "player_match_results.csv", index=False)
        last_round = matches["round_x"].max()
        last = matches[matches["round_x"] == last_round].drop_duplicates(subset=["coach_team_id", "first_name", "last_name"], keep="last")
        teams = pd.DataFrame({
            "coach_first_name": last["coach_first_name"], "first_name": last["first_name"],
            "last_name": last["last_name"], "team": last["team"], "position": last["position"],
            "avg": pd.NA, "avg3": pd.NA, "price": pd.NA, "feed_id": pd.NA,
        })
        teams.to_csv(out_dir / "current_teams.csv", index=False)
        n_player_rows = len(matches)
    else:
        n_player_rows = 0
        print(f"    no player-level Round Scores for {year} — ladder/fixtures only")

    if year in {2017, 2018, 2019}:
        print(f"  [{year}] building bulk sc_round_data from fanfooty (every player, every round)...")
        build_sc_round_data_from_fanfooty(year, out_dir, max_round=21)

    draft_sheet = f"DATA - {year} Draft order"
    if draft_sheet in wb.sheetnames:
        draft_df = _load_workbook_sheet(wb, draft_sheet)
        build_draft_csv_from_workbook_sheet(draft_df, year)

    print(f"  [{year}] OK (workbook) — {ladder['round'].nunique()} rounds, "
          f"{n_player_rows} player-match rows -> {out_dir}")


def ingest_year(year: int) -> None:
    if year == 2020:
        ingest_year_2020()
    elif year == 2021:
        ingest_year_2021()
    elif year in WORKBOOK_YEARS:
        ingest_year_from_workbook(year)
    elif year in LEAGUE_IDS and raw_dump_is_usable(year):
        ingest_year_from_raw(year)
    else:
        print(f"  [{year}] raw dump unusable/unavailable — falling back to legacy fantasy_banter_data")
        ingest_year_from_legacy(year)


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2022, 2023, 2024, 2025]
    print(f"Ingesting seasons: {years}")
    for y in years:
        ingest_year(y)
