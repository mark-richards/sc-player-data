"""
llm_narrator.py — AI-generated prose blurbs for newsletter sections.

Requires ANTHROPIC_API_KEY in .env. If missing or call fails, returns "".
Uses prompt caching on the shared system prefix to reduce cost across sections.
"""

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError:
            log.warning("anthropic package not installed — LLM blurbs disabled")
            return None
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            log.debug("ANTHROPIC_API_KEY not set — LLM blurbs disabled")
            return None
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _df_to_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    try:
        from tabulate import tabulate
        return tabulate(df.head(max_rows), headers="keys", tablefmt="simple",
                        floatfmt=".1f", showindex=False)
    except ImportError:
        return df.head(max_rows).to_string(index=False)


# Shared league context — cached on the API side across all section calls.
_LEAGUE_SYSTEM = (
    "You write sections for a weekly AFL SuperCoach fantasy draft newsletter. "
    "League: 8 teams, 19 on-field starters (5 DEF + 7 MID + 2 RUC + 5 FWD) + 4 bench. "
    "Free agents are unowned players available on the waiver wire. "
    "Rules: Australian English, direct tone, no filler, max word count enforced, "
    "never invent numbers not present in the data table."
)

_SECTION_INSTRUCTIONS = {
    "ceiling_chasers": (
        "Write the Ceiling Chasers section blurb. "
        "These players are ranked by how often they score 120+ (Ceiling Index %). "
        "Name the top 1–2 players, cite their CI% and projected score if present. "
        "Give a one-sentence pickup verdict. Flag any risk (low 2026 avg vs career, tough opponent). "
        "Max 70 words."
    ),
    "mid_year_movers": (
        "Write the Mid-Year Movers section blurb. "
        "These free agents average under 80 but trajectory signals suggest 85+ is imminent. "
        "Key signals: CBA% gaining, TOG rising, SC momentum. "
        "Explain WHY the top pick is poised to break out — cite the specific signal. "
        "Be bold. Max 70 words."
    ),
    "weekly_streamers": (
        "Write the Weekly Streamers section blurb. "
        "These are 1–3 week pickups based on sudden role changes, CBA spikes, or TOG jumps. "
        "Name the trigger for the top pick. Warn if the signal may not sustain long-term. "
        "Max 70 words."
    ),
    "positional_targets": (
        "Write a two-sentence intro for the Positional Targets section. "
        "Pick the single most compelling player from the table (any position). "
        "P80pct is the % chance of scoring 80+ points (e.g. 46.4 = 46.4%). "
        "next_rd_proj_pts is the projected score in points. These are NOT the same thing — do not confuse them. "
        "Cite their P80pct (as a %) or next_rd_proj_pts (in points). One pick, one reason. Max 50 words."
    ),
}

_COLS_BY_SECTION = {
    "ceiling_chasers":    ["name", "pos_bucket", "team", "opponent", "ceiling_index",
                           "avg_fanfooty_2026", "games_played_2026",
                           "projected_score", "calibrated_p80", "probability_gt_120"],
    "mid_year_movers":    ["name", "pos_bucket", "team", "season_avg", "career_avg_prior",
                           "cba_avg", "cba_trend", "tog_trend", "jumper_reason"],
    "weekly_streamers":   ["name", "pos_bucket", "team", "avg3", "season_avg",
                           "streamer_trigger", "streamer_score"],
    "positional_targets": ["name", "pos_bucket", "team", "opponent",
                           "projected_score", "calibrated_p80"],
}

# Rename ambiguous column names before building the LLM table so it can't confuse
# a points average (e.g. 79.2) with a probability percentage.
_COL_RENAMES = {
    "calibrated_p80":    "P80pct",       # e.g. 46.4 means 46.4% chance of 80+
    "probability_gt_80": "P80pct",
    "probability_gt_100": "P100pct",
    "probability_gt_120": "P120pct",
    "ceiling_index":     "CI_pct_120plus",
    "bayesian_avg":      "career_adj_avg_pts",
    "avg_fanfooty_2026": "season_avg_pts",
    "projected_score":   "next_rd_proj_pts",
    "effective_avg":     "effective_avg_pts",
}


def generate_section_blurb(
    section_key: str,
    data_df: pd.DataFrame,
    round_num: "int | None" = None,
    news_context: str = "",
) -> str:
    """
    Generate an AI narrative blurb for a newsletter section.
    Returns empty string if API key not set, section not configured, or call fails.
    """
    client = _get_client()
    if client is None:
        return ""

    instructions = _SECTION_INSTRUCTIONS.get(section_key)
    if instructions is None or data_df.empty:
        return ""

    # Trim to only the columns that exist and are relevant for this section
    wanted_cols = _COLS_BY_SECTION.get(section_key, list(data_df.columns))
    available_cols = [c for c in wanted_cols if c in data_df.columns]
    table_df = data_df[available_cols].copy() if available_cols else data_df.copy()

    # Round numeric columns to 1 dp for readability
    for col in table_df.select_dtypes(include="number").columns:
        table_df[col] = table_df[col].round(1)

    # Rename columns so the LLM can't confuse points averages with percentages
    table_df = table_df.rename(columns={k: v for k, v in _COL_RENAMES.items() if k in table_df.columns})

    table_str = _df_to_table(table_df)
    round_label = f"Round {round_num}" if round_num else "Pre-Season"

    user_content = (
        f"Current round: {round_label}\n\n"
        f"Data (use only these numbers — do not invent stats):\n{table_str}"
    )
    if news_context:
        user_content += f"\n\nRecent news:\n{news_context}"

    # System message with cache_control on the shared league context prefix
    try:
        import anthropic
        system_blocks = [
            {
                "type": "text",
                "text": _LEAGUE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": instructions,
            },
        ]
        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=180,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        log.warning("LLM blurb generation failed for '%s': %s", section_key, e)
        return ""
