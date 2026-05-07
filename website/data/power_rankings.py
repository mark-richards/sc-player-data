"""
power_rankings.py — Power Rankings computations.

Produces:
  1. Rolling 3-round average line chart  (Plotly, one line per coach)
  2. Per-round heat map table data       (coach × round, value = team_pts/19)
  3. Player rankings table               (top on-field players per position)

Metric: team_points / 19  (best-19-of-20 scoring — same as Fantasy Banter).
"""
import logging

import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)

_SCORING_PLAYERS_DEFAULT = 19   # fallback when player data unavailable

# Plotly qualitative colour palette — one per coach (up to 8)
_COACH_COLOURS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
    "#984ea3", "#a65628", "#f781bf", "#999999",
]


# ── Colour helpers ────────────────────────────────────────────────────────────

def _cell_color(val: float, lo: float, hi: float) -> str:
    """
    Returns CSS rgb() colour on a red → yellow → green scale.
    lo maps to red, hi maps to green.
    """
    if hi <= lo or pd.isna(val):
        return "rgb(255,255,191)"   # neutral yellow
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    if t < 0.5:
        t2 = t * 2                                      # 0→1 through red→yellow
        r = int(215 + (255 - 215) * t2)
        g = int(48  + (255 - 48)  * t2)
        b = int(39  - 39 * t2)
    else:
        t2 = (t - 0.5) * 2                             # 0→1 through yellow→green
        r = int(255 - 255 * t2)
        g = int(255 - (255 - 104) * t2)
        b = int(191 - 191 * t2)
    return f"rgb({r},{g},{b})"


def _text_color(bg_val: float, lo: float, hi: float) -> str:
    """Black text on light backgrounds, white on dark ends."""
    if hi <= lo or pd.isna(bg_val):
        return "#000"
    t = (bg_val - lo) / (hi - lo)
    return "#fff" if (t < 0.15 or t > 0.85) else "#000"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _on_field_counts(players_df: pd.DataFrame) -> dict:
    """
    Returns {(coach_first_name, round_number): n_on_field} from player match data.
    Falls back to empty dict if players_df is empty or missing required columns.
    """
    if players_df is None or players_df.empty:
        return {}
    needed = {"coach_first_name", "round_x", "on_field"}
    if not needed.issubset(players_df.columns):
        return {}
    df = players_df[players_df["on_field"] == True]
    counts = df.groupby(["coach_first_name", "round_x"]).size()
    return {(coach, rnd): int(n) for (coach, rnd), n in counts.items()}


def _expected_on_field_per_round(on_field: dict) -> dict:
    """Returns {round: max_on_field_count} across all coaches — the expected on-field count for that round."""
    from collections import defaultdict
    per_round: dict = defaultdict(int)
    for (coach, rnd), n in on_field.items():
        if n > per_round[rnd]:
            per_round[rnd] = n
    return dict(per_round)


def _donut_counts(on_field: dict, expected_per_round: dict) -> dict:
    """Returns {(coach, round): donut_count}."""
    return {
        (coach, rnd): max(0, expected_per_round.get(rnd, _SCORING_PLAYERS_DEFAULT) - n)
        for (coach, rnd), n in on_field.items()
    }


_INJURY_TAGS = {"injured", "sore"}


def _match_detail_map(fixtures_df: pd.DataFrame) -> dict:
    """Returns {(coach, round): {"score", "opp_score", "opp_coach"}} from fixtures."""
    if fixtures_df is None or fixtures_df.empty:
        return {}
    result = {}
    for _, row in fixtures_df.iterrows():
        coach = row.get("coach_first_name", row.get("team_coach", ""))
        try:
            rnd = int(row["round_number"])
        except (TypeError, ValueError):
            continue
        result[(coach, rnd)] = {
            "score":     int(row.get("team_points", 0) or 0),
            "opp_score": int(row.get("opposition_team_points", 0) or 0),
            "opp_coach": str(row.get("opposition_team_coach", "")),
        }
    return result


def _injury_player_details(players_df: pd.DataFrame, fanfooty_df: pd.DataFrame) -> dict:
    """Returns {(coach, round): [{"name": str, "score": int}]} for on-field injured players."""
    if players_df is None or players_df.empty or fanfooty_df is None or fanfooty_df.empty:
        return {}
    needed_p = {"feed_id", "coach_first_name", "round_x", "on_field", "first_name", "last_name", "points_x"}
    needed_f = {"Player ID", "round_num", "Tag", "Tag 2"}
    if not needed_p.issubset(players_df.columns) or not needed_f.issubset(fanfooty_df.columns):
        return {}
    on_field = players_df[players_df["on_field"] == True][
        ["feed_id", "coach_first_name", "round_x", "first_name", "last_name", "points_x"]
    ].copy()
    on_field["feed_id"]  = pd.to_numeric(on_field["feed_id"],  errors="coerce").astype("Int64")
    on_field["points_x"] = pd.to_numeric(on_field["points_x"], errors="coerce").fillna(0)
    merged = on_field.merge(
        fanfooty_df[["Player ID", "round_num", "Tag", "Tag 2"]],
        left_on=["feed_id", "round_x"],
        right_on=["Player ID", "round_num"],
        how="left",
    )
    mask = merged["Tag"].str.lower().isin(_INJURY_TAGS) | merged["Tag 2"].str.lower().isin(_INJURY_TAGS)
    result: dict = {}
    for _, row in merged[mask].iterrows():
        key = (row["coach_first_name"], int(row["round_x"]))
        result.setdefault(key, []).append({
            "name":  f"{str(row['first_name'])[0]}. {row['last_name']}",
            "score": int(row["points_x"]),
        })
    return result


def _donut_player_details(players_df: pd.DataFrame) -> dict:
    """Returns {(coach, round): [{"name": str}]} for on-field players who scored 0."""
    if players_df is None or players_df.empty:
        return {}
    needed = {"coach_first_name", "round_x", "on_field", "first_name", "last_name", "points_x"}
    if not needed.issubset(players_df.columns):
        return {}
    df = players_df[players_df["on_field"] == True].copy()
    df["points_x"] = pd.to_numeric(df["points_x"], errors="coerce").fillna(0)
    result: dict = {}
    for _, row in df[df["points_x"] == 0].iterrows():
        key = (row["coach_first_name"], int(row["round_x"]))
        result.setdefault(key, []).append({
            "name": f"{str(row['first_name'])[0]}. {row['last_name']}",
        })
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def build_power_rankings_chart(
    fixtures_df: pd.DataFrame,
    players_df: pd.DataFrame | None = None,
    portraits: dict | None = None,
) -> dict:
    """
    Returns a Plotly figure dict: line chart of rolling 3-round average team
    score per player for each coach, with coach portraits at each data point.
    """
    if fixtures_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig.to_dict()

    on_field = _on_field_counts(players_df)

    df = fixtures_df[["coach_first_name", "round_number", "team_points"]].copy()
    df["spp"] = df.apply(
        lambda r: r["team_points"] / on_field.get(
            (r["coach_first_name"], r["round_number"]), _SCORING_PLAYERS_DEFAULT
        ),
        axis=1,
    )

    coaches_alpha = sorted(df["coach_first_name"].unique())
    rounds = sorted(df["round_number"].unique())

    # Build per-coach rolling-3-round averages
    coach_y: dict[str, list] = {}
    for coach in coaches_alpha:
        cdf = df[df["coach_first_name"] == coach].set_index("round_number")["spp"]
        y_vals = []
        for r in rounds:
            window = [cdf.get(w) for w in [r - 2, r - 1, r] if w in cdf.index]
            y_vals.append(round(sum(window) / len(window), 1) if window else None)
        coach_y[coach] = y_vals

    # Sort coaches by their most-recent non-None value (highest first)
    def _last_val(c):
        valid = [v for v in coach_y[c] if v is not None]
        return valid[-1] if valid else 0.0

    coaches_sorted = sorted(coaches_alpha, key=_last_val, reverse=True)

    fig = go.Figure()
    layout_images = []

    for coach in coaches_sorted:
        y_vals = coach_y[coach]
        x_vals = [int(r) for r in rounds]
        colour = _COACH_COLOURS[coaches_alpha.index(coach) % len(_COACH_COLOURS)]

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",  # Images will serve as markers
            name=coach,
            line={"color": colour, "width": 2},
            connectgaps=True,
            hovertemplate=f"<b>{coach}</b><br>Round %{{x}}<br>Avg SPP: %{{y:.1f}}<extra></extra>",
        ))

        if portraits and coach in portraits:
            for i, r in enumerate(rounds):
                y = y_vals[i]
                if y is not None:
                    layout_images.append(go.layout.Image(
                        source=portraits[coach],
                        xref="x", yref="y",
                        x=r, y=y,
                        sizex=0.5, sizey=4,
                        xanchor="center", yanchor="middle",
                        layer="above"
                    ))

    fig.update_layout(
        title={
            "text": "Power Rankings",
            "x": 0.5,
            "font": {"size": 16},
        },
        xaxis={
            "title": "Round",
            "dtick": 1,
            "tickvals": [int(r) for r in rounds],
            "gridcolor": "#e0e0e0",
        },
        yaxis={"title": "Avg Score per Player", "gridcolor": "#e0e0e0"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
        height=380,
        hovermode="x unified",
        images=layout_images,
        annotations=[{
            "text": "3-Round Rolling Average per Player",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 1.0,
            "xanchor": "center", "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 11, "color": "#888"},
        }],
    )

    return fig.to_dict()


def build_power_rankings_table(
    fixtures_df: pd.DataFrame,
    players_df: pd.DataFrame | None = None,
    fanfooty_df: pd.DataFrame | None = None,
) -> dict:
    """
    Returns a dict for rendering the coach × round heat-map table.

    Structure:
      {
        "rounds":       [int, ...]  (newest first),
        "coaches":      [
          {
            "coach":    str,
            "avg":      float,
            "avg_3rd":  float,
            "cells": [
              {"round": int, "val": float|None, "color": str, "text_color": str},
              ...  (newest first)
            ],
          },
          ...  (sorted by avg_3rd descending)
        ],
        "lo": float, "hi": float,
      }
    """
    if fixtures_df.empty:
        return {"rounds": [], "coaches": []}

    on_field = _on_field_counts(players_df)
    expected_per_round = _expected_on_field_per_round(on_field)
    match_details = _match_detail_map(fixtures_df)
    injured_detail = _injury_player_details(players_df, fanfooty_df)
    donut_detail = _donut_player_details(players_df)

    result_map = {
        (row["coach_first_name"], int(row["round_number"])): (
            "W" if row["win"] else ("D" if row["draw"] else "L")
        )
        for _, row in fixtures_df.iterrows()
    }

    df = fixtures_df[["coach_first_name", "round_number", "team_points"]].copy()
    df["spp"] = df.apply(
        lambda r: round(r["team_points"] / on_field.get(
            (r["coach_first_name"], r["round_number"]), _SCORING_PLAYERS_DEFAULT
        ), 1),
        axis=1,
    )

    rounds_asc = sorted(df["round_number"].unique())
    rounds_desc = list(reversed(rounds_asc))

    pivot = df.pivot_table(index="coach_first_name", columns="round_number",
                           values="spp", aggfunc="first")

    # Global colour range
    all_vals = df["spp"].dropna().values
    lo, hi = float(all_vals.min()), float(all_vals.max())

    coach_rows = []
    for coach in pivot.index:
        row_vals = [pivot.loc[coach, r] if r in pivot.columns else None
                    for r in rounds_asc]
        valid = [v for v in row_vals if v is not None and not pd.isna(v)]
        season_avg = round(sum(valid) / len(valid), 1) if valid else 0.0
        last3 = valid[-3:] if len(valid) >= 1 else valid
        avg_3rd = round(sum(last3) / len(last3), 1) if last3 else 0.0

        cells = []
        for r in rounds_desc:
            v = pivot.loc[coach, r] if r in pivot.columns else None
            if v is not None and pd.isna(v):
                v = None
            key = (coach, r)
            result_code = result_map.get(key)
            injured = injured_detail.get(key, [])
            donuts = donut_detail.get(key, [])
            match = match_details.get(key)

            # Build tooltip HTML
            tooltip_parts: list[str] = []
            if match and v is not None:
                verb = "def" if result_code == "W" else ("drew" if result_code == "D" else "lost to")
                tooltip_parts.append(
                    f"<strong>{match['score']:,} {verb} {match['opp_score']:,} vs {match['opp_coach']}</strong>"
                )
            for p in injured:
                tooltip_parts.append(f"🤕 {p['name']} ({p['score']})")
            for p in donuts:
                tooltip_parts.append(f"🍩 {p['name']}")

            cells.append({
                "round":           int(r),
                "val":             float(v) if v is not None else None,
                "donuts":          len(donuts),
                "injuries":        len(injured),
                "result":          result_code,
                "color":           _cell_color(v, lo, hi) if v is not None else "rgb(230,230,230)",
                "text_color":      _text_color(v, lo, hi) if v is not None else "#aaa",
                "tooltip":         "<br>".join(tooltip_parts),
            })

        coach_rows.append({
            "coach":   coach,
            "avg":     season_avg,
            "avg_3rd": avg_3rd,
            "cells":   cells,
        })

    coach_rows.sort(key=lambda x: x["avg_3rd"], reverse=True)
    return {
        "rounds": [int(r) for r in rounds_desc],
        "coaches": coach_rows,
        "lo": lo,
        "hi": hi,
    }


def build_player_rankings(players_df: pd.DataFrame, position: str | None = None) -> list[dict]:
    """
    Returns a sorted list of player dicts for the rankings table.

    Filters:
    - on_field == True only
    - minimum 2 games played

    Computes per player:
    - games, avg (season), form (last-3-round avg), best, total

    position: one of "DEF", "MID", "RUC", "FWD" or None (all positions).
    """
    if players_df.empty:
        return []

    df = players_df[players_df["on_field"] == True].copy()

    if position:
        df = df[df["played_position"] == position]

    if df.empty:
        return []

    # Include team if column is present (may be absent in older CSV snapshots)
    base_cols = ["player_id", "first_name", "last_name", "coach_first_name"]
    if "team" in df.columns:
        base_cols.append("team")

    # For position-filtered tabs, include played_position in the group key.
    # For Overall, group by player only to avoid one row per position for dual-position players.
    group_cols = base_cols + ["played_position"] if position else base_cols

    agg = (
        df.groupby(group_cols)
        .agg(
            games=("points_x", "count"),
            total=("points_x", "sum"),
            avg=("points_x", "mean"),
            best=("points_x", "max"),
        )
        .reset_index()
    )
    agg = agg[agg["games"] >= 2]

    if not position:
        # Attach the dominant position (most games played in a single position)
        dominant = (
            df.groupby(["player_id", "played_position"])
            .size()
            .reset_index(name="_n")
            .sort_values("_n", ascending=False)
            .drop_duplicates("player_id")[["player_id", "played_position"]]
        )
        agg = agg.merge(dominant, on="player_id", how="left")

    # Compute form: average of last 3 rounds per player
    df_sorted = df.sort_values("round_x")
    last3 = (
        df_sorted
        .groupby(["player_id"])["points_x"]
        .apply(lambda s: s.tail(3).mean())
        .reset_index()
        .rename(columns={"points_x": "form_3rnd"})
    )
    agg = agg.merge(last3, on="player_id", how="left")
    agg["avg"]       = agg["avg"].round(1)
    agg["form_3rnd"] = agg["form_3rnd"].round(1)

    agg = agg.sort_values("avg", ascending=False)
    return agg.to_dict(orient="records")


def build_power_rankings_table(
    fixtures_df: pd.DataFrame,
    players_df: pd.DataFrame | None = None,
    fanfooty_df: pd.DataFrame | None = None,
) -> dict:
    """
    Returns a dict for rendering the coach × round heat-map table.

    Structure:
      {
        "rounds":       [int, ...]  (newest first),
        "coaches":      [
          {
            "coach":    str,
            "avg":      float,
            "avg_3rd":  float,
            "cells": [
              {"round": int, "val": float|None, "color": str, "text_color": str},
              ...  (newest first)
            ],
          },
          ...  (sorted by avg_3rd descending)
        ],
        "lo": float, "hi": float,
      }
    """
    if fixtures_df.empty:
        return {"rounds": [], "coaches": []}

    on_field = _on_field_counts(players_df)
    expected_per_round = _expected_on_field_per_round(on_field)
    match_details  = _match_detail_map(fixtures_df)
    injured_detail = _injury_player_details(players_df, fanfooty_df)
    donut_detail   = _donut_player_details(players_df)

    result_map = {
        (row["coach_first_name"], int(row["round_number"])): (
            "W" if row["win"] else ("D" if row["draw"] else "L")
        )
        for _, row in fixtures_df.iterrows()
    }

    df = fixtures_df[["coach_first_name", "round_number", "team_points"]].copy()
    df["spp"] = df.apply(
        lambda r: round(r["team_points"] / on_field.get(
            (r["coach_first_name"], r["round_number"]), _SCORING_PLAYERS_DEFAULT
        ), 1),
        axis=1,
    )

    rounds_asc = sorted(df["round_number"].unique())
    rounds_desc = list(reversed(rounds_asc))

    pivot = df.pivot_table(index="coach_first_name", columns="round_number",
                           values="spp", aggfunc="first")

    # Global colour range
    all_vals = df["spp"].dropna().values
    lo, hi = float(all_vals.min()), float(all_vals.max())

    coach_rows = []
    for coach in pivot.index:
        row_vals = [pivot.loc[coach, r] if r in pivot.columns else None
                    for r in rounds_asc]
        valid = [v for v in row_vals if v is not None and not pd.isna(v)]
        season_avg = round(sum(valid) / len(valid), 1) if valid else 0.0
        last3 = valid[-3:] if len(valid) >= 1 else valid
        avg_3rd = round(sum(last3) / len(last3), 1) if last3 else 0.0

        cells = []
        for r in rounds_desc:
            v = pivot.loc[coach, r] if r in pivot.columns else None
            if v is not None and pd.isna(v):
                v = None
            key            = (coach, r)
            result_code    = result_map.get(key)
            injured        = injured_detail.get(key, [])
            donuts         = donut_detail.get(key, [])
            match          = match_details.get(key)

            # Build tooltip HTML
            tooltip_parts: list[str] = []
            if match and v is not None:
                verb = "def" if result_code == "W" else ("drew" if result_code == "D" else "lost to")
                tooltip_parts.append(
                    f"<strong>{match['score']:,} {verb} {match['opp_score']:,} vs {match['opp_coach']}</strong>"
                )
            for p in injured:
                tooltip_parts.append(f"🤕 {p['name']} ({p['score']})")
            for p in donuts:
                tooltip_parts.append(f"🍩 {p['name']}")

            cells.append({
                "round":           int(r),
                "val":             float(v) if v is not None else None,
                "donuts":          len(donuts),
                "injuries":        len(injured),
                "result":          result_code,
                "color":           _cell_color(v, lo, hi) if v is not None else "rgb(230,230,230)",
                "text_color":      _text_color(v, lo, hi) if v is not None else "#aaa",
                "tooltip":         "<br>".join(tooltip_parts),
            })

        coach_rows.append({
            "coach":   coach,
            "avg":     season_avg,
            "avg_3rd": avg_3rd,
            "cells":   cells,
        })

    coach_rows.sort(key=lambda x: x["avg_3rd"], reverse=True)
    return {
        "rounds": [int(r) for r in rounds_desc],
        "coaches": coach_rows,
        "lo": lo,
        "hi": hi,
    }


def build_player_rankings(players_df: pd.DataFrame, position: str | None = None) -> list[dict]:
    """
    Returns a sorted list of player dicts for the rankings table.

    Filters:
    - on_field == True only
    - minimum 2 games played

    Computes per player:
    - games, avg (season), form (last-3-round avg), best, total

    position: one of "DEF", "MID", "RUC", "FWD" or None (all positions).
    """
    if players_df.empty:
        return []

    df = players_df[players_df["on_field"] == True].copy()

    if position:
        df = df[df["played_position"] == position]

    if df.empty:
        return []

    # Include team if column is present (may be absent in older CSV snapshots)
    base_cols = ["player_id", "first_name", "last_name", "coach_first_name"]
    if "team" in df.columns:
        base_cols.append("team")

    # For position-filtered tabs, include played_position in the group key.
    # For Overall, group by player only to avoid one row per position for dual-position players.
    group_cols = base_cols + ["played_position"] if position else base_cols

    agg = (
        df.groupby(group_cols)
        .agg(
            games=("points_x", "count"),
            total=("points_x", "sum"),
            avg=("points_x", "mean"),
            best=("points_x", "max"),
        )
        .reset_index()
    )
    agg = agg[agg["games"] >= 2]

    if not position:
        # Attach the dominant position (most games played in a single position)
        dominant = (
            df.groupby(["player_id", "played_position"])
            .size()
            .reset_index(name="_n")
            .sort_values("_n", ascending=False)
            .drop_duplicates("player_id")[["player_id", "played_position"]]
        )
        agg = agg.merge(dominant, on="player_id", how="left")

    # Compute form: average of last 3 rounds per player
    df_sorted = df.sort_values("round_x")
    last3 = (
        df_sorted
        .groupby(["player_id"])["points_x"]
        .apply(lambda s: s.tail(3).mean())
        .reset_index()
        .rename(columns={"points_x": "form_3rnd"})
    )
    agg = agg.merge(last3, on="player_id", how="left")
    agg["avg"]       = agg["avg"].round(1)
    agg["form_3rnd"] = agg["form_3rnd"].round(1)

    agg = agg.sort_values("avg", ascending=False)
    return agg.to_dict(orient="records")
