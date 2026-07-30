"""
role_watch.py — Detects recent role changes from Fanfooty Tag Notes.

What we scan for
----------------
Positive signals
  • CBA / midfield up:  "on ball", "starting on ball", "central mid", "clearance"
  • TOG increase:       "time on ground" keyword present, or TOG column trending up
  • Ruck elevation:     "first ruck", "lead ruck" (upgrade from second ruck)
  • Midfield injection: "midfield" (for a forward/defender picking up extra midfield time)
  • Role lift:          "playing a wing role", "coming off half back"

Negative signals
  • Tagged:            Tag 2 == "shovel" or "guard" pointing AT that player
  • TOG decline:       TOG column trending down vs baseline

Output
------
A list of dicts, one per notable player, with:
    name, team, feed_id, change_type, note, rounds_present
"""

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

# ── Keyword patterns ────────────────────────────────────────────────────────

_CBA_PATTERNS = re.compile(
    r"on\s+ball|starting\s+on\s+ball|central\s+mid|clearance|"
    r"midfield\s+time|getting\s+through\s+the\s+middle|role\s+in\s+midfield",
    re.IGNORECASE,
)

_RUCK_UPGRADE_PATTERNS = re.compile(
    r"first\s+ruck|lead\s+ruck",
    re.IGNORECASE,
)

_TOG_INCREASE_PATTERN = re.compile(
    r"time\s+on\s+ground|big\s+TOG|increased.*minutes|more\s+time\s+on",
    re.IGNORECASE,
)

_ROLE_LIFT_PATTERNS = re.compile(
    r"wing\s+role|half\s+back|coming\s+off\s+half|forward\s+role|"
    r"key\s+forward|spearhead|increased\s+role|elevated",
    re.IGNORECASE,
)

_TAG_NEGATIVE_PATTERNS = re.compile(
    r"shovel|tagged\s+by|guard|following|tagging",
    re.IGNORECASE,
)

_RUCK_DEMOTION_PATTERNS = re.compile(
    r"second\s+ruck|playing\s+forward|ruck.{0,20}benched",
    re.IGNORECASE,
)


def _classify_notes(tag_notes: str, tag2: str, tag2_notes: str) -> tuple[str | None, str]:
    """
    Returns (change_type, best_note_snippet) or (None, "") if nothing notable.
    change_type values: "CBA↑", "Ruck↑", "TOG↑", "Role↑", "Tagged↓", "Ruck↓"
    """
    combined = " | ".join(filter(None, [tag_notes, tag2, tag2_notes]))

    if _RUCK_UPGRADE_PATTERNS.search(combined):
        return "Ruck↑", _first_sentence(combined)
    if _CBA_PATTERNS.search(combined):
        return "CBA↑", _first_sentence(combined)
    if _TOG_INCREASE_PATTERN.search(combined):
        return "TOG↑", _first_sentence(combined)
    if _ROLE_LIFT_PATTERNS.search(combined):
        return "Role↑", _first_sentence(combined)
    if _TAG_NEGATIVE_PATTERNS.search(tag2 or ""):
        return "Tagged↓", _first_sentence(combined)
    if _RUCK_DEMOTION_PATTERNS.search(combined):
        return "Ruck↓", _first_sentence(combined)

    return None, ""


def _first_sentence(text: str, max_chars: int = 100) -> str:
    """Return up to the first sentence (or max_chars characters) of text."""
    text = text.strip()
    # Strip the % placeholder codes (e.g. %P, %M, %T) from fanfooty notes
    text = re.sub(r"%\w+", "", text).strip(" .|")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    for sep in ("...", ".", "!", "?"):
        idx = text.find(sep)
        if 0 < idx < max_chars:
            return text[: idx + len(sep)]
    return text[:max_chars]


# ── TOG trend detection ────────────────────────────────────────────────────

def _tog_trend(player_rounds: pd.DataFrame) -> tuple[str | None, str]:
    """
    Check if a player's Time on Ground has materially changed in the last
    2 rounds vs their earlier baseline.

    Returns (change_type, note) or (None, "").
    """
    tog_col = "tog"
    if tog_col not in player_rounds.columns:
        return None, ""

    valid = player_rounds[tog_col].dropna()
    if len(valid) < 2:
        return None, ""

    # Last round vs the rest
    last = float(valid.iloc[-1])
    rest_mean = float(valid.iloc[:-1].mean())

    if rest_mean == 0:
        return None, ""

    pct_change = (last - rest_mean) / rest_mean * 100

    if pct_change >= 8:  # ≥8% TOG increase
        return "TOG↑", f"TOG up {pct_change:.0f}% vs baseline ({rest_mean:.0f}→{last:.0f})"
    if pct_change <= -10:  # ≥10% TOG drop
        return "TOG↓", f"TOG down {abs(pct_change):.0f}% vs baseline ({rest_mean:.0f}→{last:.0f})"

    return None, ""


# ── Main public function ────────────────────────────────────────────────────

def detect_role_changes(
    fanfooty_recent_df: pd.DataFrame,
    relevant_feed_ids: set[int] | None = None,
    max_results: int = 25,
) -> list[dict]:
    """
    Analyses the last N rounds of fanfooty data and returns a list of
    notable role changes.

    Parameters
    ----------
    fanfooty_recent_df : DataFrame
        Output of data_loader.load_fanfooty_recent_rounds().
        Expected columns: feed_id, name, tag_notes, tag2_notes, tog,
                          clearances, round_num, year.
    relevant_feed_ids : set[int] | None
        If provided, only analyse players in this set (free agents + my roster).
        Reduces noise dramatically. If None, analyses all players.
    max_results : int
        Cap the output at this many entries.

    Returns
    -------
    list[dict] with keys: feed_id, name, team, change_type, note, rounds_present
    """
    if fanfooty_recent_df is None or fanfooty_recent_df.empty:
        log.warning("No fanfooty recent data provided to role_watch.")
        return []

    df = fanfooty_recent_df
    if relevant_feed_ids:
        df = df[df["feed_id"].isin(relevant_feed_ids)]

    if df.empty:
        log.info("No relevant players found in recent fanfooty data.")
        return []

    results: list[dict] = []
    seen: set[int] = set()  # track feed_ids already added to avoid duplicates

    for feed_id, group in df.groupby("feed_id"):
        feed_id = int(feed_id)

        name = group["name"].iloc[-1] if "name" in group.columns else str(feed_id)
        team = group.get("team", pd.Series([""] * len(group))).iloc[-1] if "team" in group.columns else ""
        rounds_present = sorted(group["round_num"].unique().tolist())

        # --- Text-based detection (Tag Notes from most recent round) -------
        latest = group.sort_values("round_num").iloc[-1]
        tag_notes = str(latest.get("tag_notes", "") or "")
        tag2 = str(latest.get("tag2_notes", "") or "")
        tag2_notes = str(latest.get("tag2_notes", "") or "")

        change_type, note = _classify_notes(tag_notes, tag2, tag2_notes)

        # --- TOG trend across available rounds --------------------------------
        if change_type is None:
            change_type, note = _tog_trend(group)

        if change_type and feed_id not in seen:
            seen.add(feed_id)
            results.append(
                {
                    "feed_id": feed_id,
                    "name": name,
                    "team": team,
                    "change_type": change_type,
                    "note": note,
                    "rounds_present": rounds_present,
                }
            )

    # Prioritise positive signals (↑) over negative (↓), then cap
    positive = [r for r in results if "↑" in r["change_type"]]
    negative = [r for r in results if "↓" in r["change_type"]]
    ordered = positive + negative
    ordered = ordered[:max_results]

    log.info(
        "Role Watch: detected %d notable role changes (%d positive, %d negative).",
        len(ordered), len(positive), len(negative),
    )
    return ordered
