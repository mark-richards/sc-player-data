"""
Scrape the AFL official injury list and match players to the SC player database.
Output: data/live/afl_injury_list.csv

Columns: sc_id, feed_id, name, afl_name, team_abbrev, injury, est_return,
         match_confidence (exact/fuzzy/ambiguous), ambiguous_matches
"""
import re
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

AFL_INJURY_URL = "https://www.afl.com.au/matches/injury-list"
OUT_PATH = Path("data/live/afl_injury_list.csv")

AFL_TEAM_MAP = {
    "Adelaide Crows": "ADE",
    "Brisbane":       "BRL",
    "Carlton":        "CAR",
    "Collingwood":    "COL",
    "Essendon":       "ESS",
    "Fremantle":      "FRE",
    "Geelong":        "GEE",
    "Gold Coast SUNS": "GCS",
    "GWS GIANTS":     "GWS",
    "Hawthorn":       "HAW",
    "Melbourne":      "MEL",
    "North Melbourne": "NTH",
    "Port Adelaide":  "PTA",
    "Richmond":       "RIC",
    "St Kilda":       "STK",
    "Sydney Swans":   "SYD",
    "West Coast Eagles": "WCE",
    "Western Bulldogs":  "WBD",
}

# SC player list team abbreviations mapping (what's used in the CSV)
SC_TEAM_ALIASES = {
    "BRL": ["BRL", "BRI", "BRIS"],
    "GCS": ["GCS", "GCG", "GCFC"],
    "GWS": ["GWS", "GWSC"],
}


def _norm_name(s: str) -> str:
    """Lowercase, strip suffixes like Jr./Jr, normalise spacing."""
    s = s.lower().strip()
    s = re.sub(r"\bjr\.?\b|\bsnr\.?\b", "", s)
    s = re.sub(r"[^a-z\s']", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_name(a), _norm_name(b)).ratio()


def _team_matches(sc_team: str, afl_abbrev: str) -> bool:
    sc_up = sc_team.upper()
    afl_up = afl_abbrev.upper()
    if sc_up == afl_up:
        return True
    aliases = SC_TEAM_ALIASES.get(afl_up, [])
    return sc_up in aliases


def scrape_injury_list() -> list[dict]:
    """Return list of {afl_name, injury, est_return, afl_team_abbrev}."""
    log.info("Fetching AFL injury list...")
    r = requests.get(AFL_INJURY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Team names in alphabetical order (matches table order)
    seen = set()
    teams_ordered = []
    for span in soup.find_all("span", class_="club-list__club-name"):
        t = span.get_text(strip=True)
        if t not in seen:
            seen.add(t)
            teams_ordered.append(t)

    tables = soup.find_all("table")
    if len(teams_ordered) != len(tables):
        log.warning(
            "Team count (%d) != table count (%d) — using positional match",
            len(teams_ordered), len(tables)
        )

    rows = []
    for i, table in enumerate(tables):
        afl_team = teams_ordered[i] if i < len(teams_ordered) else f"UNKNOWN_{i}"
        afl_abbrev = AFL_TEAM_MAP.get(afl_team, afl_team[:3].upper())
        for tr in table.find_all("tr")[1:]:  # skip header
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cols) >= 2:
                rows.append({
                    "afl_name": cols[0],
                    "injury": cols[1] if len(cols) > 1 else "",
                    "est_return": cols[2] if len(cols) > 2 else "",
                    "afl_team": afl_team,
                    "afl_abbrev": afl_abbrev,
                })
    log.info("Found %d injured players across %d teams", len(rows), len(tables))
    return rows


def load_sc_players() -> pd.DataFrame:
    pl = pd.read_csv("draft_prep/SC 2026/2026_SC_Player_list.csv", low_memory=False)
    pl["full_name"] = (pl["first_name"].str.strip() + " " + pl["last_name"].str.strip()).str.strip()
    pl["_norm"] = pl["full_name"].apply(_norm_name)
    return pl


def match_player(afl_name: str, afl_abbrev: str, sc_players: pd.DataFrame) -> dict:
    """
    Match an AFL injury-list player name to a row in the SC player list.
    Returns a dict with sc_id, feed_id, name, match_confidence, ambiguous_matches.
    """
    norm = _norm_name(afl_name)

    # 1. Exact normalised name match
    exact = sc_players[sc_players["_norm"] == norm]
    if len(exact) == 1:
        row = exact.iloc[0]
        return {
            "sc_id": int(row["id"]),
            "feed_id": str(row["feed_id"]),
            "name": row["full_name"],
            "team_abbrev": str(row.get("team", "")),
            "match_confidence": "exact",
            "ambiguous_matches": "",
        }
    if len(exact) > 1:
        # Multiple exact matches — try to narrow by team
        team_match = exact[exact["team"].apply(lambda t: _team_matches(str(t), afl_abbrev))]
        if len(team_match) == 1:
            row = team_match.iloc[0]
            return {
                "sc_id": int(row["id"]),
                "feed_id": str(row["feed_id"]),
                "name": row["full_name"],
                "team_abbrev": str(row.get("team", "")),
                "match_confidence": "exact+team",
                "ambiguous_matches": "",
            }
        # Still ambiguous
        options = "; ".join(
            f"{r['full_name']} ({r.get('team','')})" for _, r in exact.iterrows()
        )
        return {
            "sc_id": None,
            "feed_id": None,
            "name": None,
            "team_abbrev": afl_abbrev,
            "match_confidence": "ambiguous",
            "ambiguous_matches": options,
        }

    # 2. Last-name match, then verify first name
    afl_parts = norm.split()
    if not afl_parts:
        return _no_match(afl_abbrev)

    last = afl_parts[-1]
    last_matches = sc_players[sc_players["_norm"].str.split().str[-1] == last]

    if len(last_matches) == 0:
        # Try fuzzy last name (handles typos / diacritics)
        scores = sc_players["_norm"].apply(
            lambda n: _name_similarity(last, n.split()[-1])
        )
        best = scores.max()
        if best >= 0.85:
            last_matches = sc_players[scores == best]

    if len(last_matches) == 1:
        row = last_matches.iloc[0]
        sim = _name_similarity(afl_name, row["full_name"])
        # Require first-name initial to match when last-name is fuzzy (not exact)
        afl_first_init = afl_parts[0][0] if afl_parts else ""
        sc_norm_parts = row["_norm"].split()
        sc_first_init = sc_norm_parts[0][0] if sc_norm_parts else ""
        team_ok = _team_matches(str(row.get("team", "")), afl_abbrev)
        if afl_first_init != sc_first_init and not team_ok:
            return _no_match(afl_abbrev)
        conf = "fuzzy" if sim >= 0.7 else "weak_fuzzy"
        return {
            "sc_id": int(row["id"]),
            "feed_id": str(row["feed_id"]),
            "name": row["full_name"],
            "team_abbrev": str(row.get("team", "")),
            "match_confidence": conf,
            "ambiguous_matches": "",
        }

    if len(last_matches) > 1:
        # Disambiguate by first initial + team
        if len(afl_parts) >= 2:
            first = afl_parts[0]
            init_matches = last_matches[
                last_matches["_norm"].apply(lambda n: n.split()[0].startswith(first[0]))
            ]
        else:
            init_matches = last_matches

        # Try team filter
        team_matches = init_matches[
            init_matches["team"].apply(lambda t: _team_matches(str(t), afl_abbrev))
        ]
        if len(team_matches) == 1:
            row = team_matches.iloc[0]
            return {
                "sc_id": int(row["id"]),
                "feed_id": str(row["feed_id"]),
                "name": row["full_name"],
                "team_abbrev": str(row.get("team", "")),
                "match_confidence": "fuzzy+team",
                "ambiguous_matches": "",
            }
        if len(init_matches) == 1:
            row = init_matches.iloc[0]
            return {
                "sc_id": int(row["id"]),
                "feed_id": str(row["feed_id"]),
                "name": row["full_name"],
                "team_abbrev": str(row.get("team", "")),
                "match_confidence": "fuzzy",
                "ambiguous_matches": "",
            }

        # Genuinely ambiguous — list all options
        pool = team_matches if len(team_matches) > 0 else last_matches
        options = "; ".join(
            f"{r['full_name']} ({r.get('team','')})" for _, r in pool.iterrows()
        )
        return {
            "sc_id": None,
            "feed_id": None,
            "name": None,
            "team_abbrev": afl_abbrev,
            "match_confidence": "ambiguous",
            "ambiguous_matches": options,
        }

    return _no_match(afl_abbrev)


def _no_match(afl_abbrev: str) -> dict:
    return {
        "sc_id": None,
        "feed_id": None,
        "name": None,
        "team_abbrev": afl_abbrev,
        "match_confidence": "no_match",
        "ambiguous_matches": "",
    }


def run():
    injured = scrape_injury_list()
    sc_players = load_sc_players()

    results = []
    for p in injured:
        match = match_player(p["afl_name"], p["afl_abbrev"], sc_players)
        results.append({
            "afl_name": p["afl_name"],
            "afl_team": p["afl_team"],
            "injury": p["injury"],
            "est_return": p["est_return"],
            **match,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    log.info("Saved %d entries to %s", len(df), OUT_PATH)

    # Print summary for review
    print(f"\n{'='*70}")
    print(f"AFL INJURY LIST — {len(df)} players")
    print(f"{'='*70}")
    for conf in ["ambiguous", "no_match", "weak_fuzzy"]:
        subset = df[df["match_confidence"] == conf]
        if len(subset):
            print(f"\n!!! {conf.upper()} ({len(subset)}) — requires review !!!")
            for _, r in subset.iterrows():
                print(f"  AFL: '{r['afl_name']}' ({r['afl_team']})")
                print(f"       injury={r['injury']}, return={r['est_return']}")
                if r['ambiguous_matches']:
                    print(f"       Options: {r['ambiguous_matches']}")

    print(f"\n--- Matched players by confidence ---")
    print(df["match_confidence"].value_counts().to_string())

    print(f"\n--- All matched players ---")
    show = df[["afl_name", "afl_team", "injury", "est_return", "name", "team_abbrev", "match_confidence", "ambiguous_matches"]].copy()
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 160)
    print(show.to_string(index=False))

    return df


if __name__ == "__main__":
    run()
