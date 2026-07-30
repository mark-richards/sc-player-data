"""
routes/coach.py — Per-coach profile page (/coach/<name>).
"""
import json

import pandas as pd
from flask import Blueprint, abort, render_template

from website.data.loader import load_current_teams, load_fixtures, load_player_matches, load_coach_portraits
from website.data.charts import COACH_COLOURS
from website.routes.field_view import fetch_current_team, load_weekly_changes

bp = Blueprint("coach", __name__)


def _positional_line_chart(player_df: pd.DataFrame, coach: str) -> dict:
    """Per-round DEF/MID/RUC/FWD on-field points for this coach."""
    import plotly.graph_objects as go

    df = player_df[
        (player_df["coach_first_name"] == coach) &
        (player_df["on_field"] == True) &
        (player_df["played_position"].isin(["DEF", "MID", "RUC", "FWD"]))
    ]
    agg = (
        df.groupby(["round_x", "played_position"])["points_x"]
        .sum()
        .reset_index()
        .pivot(index="round_x", columns="played_position", values="points_x")
        .fillna(0)
        .sort_index()
    )

    pos_colours = {"DEF": "#1f77b4", "MID": "#2ca02c", "RUC": "#9467bd", "FWD": "#ff7f0e"}
    fig = go.Figure()
    for pos in ["DEF", "MID", "RUC", "FWD"]:
        if pos in agg.columns:
            fig.add_trace(go.Scatter(
                x=agg.index, y=agg[pos], mode="lines+markers",
                name=pos, line=dict(color=pos_colours.get(pos), width=2),
                hovertemplate=f"<b>{pos}</b> R%{{x}}: %{{y:.0f}}<extra></extra>",
            ))

    fig.update_layout(
        title=f"{coach} — Positional Scores by Round",
        xaxis=dict(title="Round", dtick=1, gridcolor="#e0e0e0"),
        yaxis=dict(title="On-field Points", gridcolor="#e0e0e0"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40), height=320,
    )
    return fig.to_dict()


@bp.get("/<coach_name>")
def coach_profile(coach_name: str):
    # Validate coach name
    valid_coaches = ["Anthony", "James", "Jordan", "Lester", "Luke", "Mark", "Paul", "Simon"]
    if coach_name not in valid_coaches:
        abort(404)

    fixtures  = load_fixtures()
    players   = load_player_matches()
    roster_df = load_current_teams()
    portraits = load_coach_portraits()

    # Round-by-round scores
    scores = (
        fixtures[fixtures["coach_first_name"] == coach_name]
        [["round_number", "team_points", "opposition_team_coach", "opposition_team_points", "win", "draw"]]
        .sort_values("round_number")
    )
    scores["result"] = scores.apply(
        lambda r: "W" if r["win"] else ("D" if r["draw"] else "L"), axis=1
    )

    # Current roster
    roster = roster_df[roster_df["coach_first_name"] == coach_name][
        ["first_name", "last_name", "team", "position", "avg", "price", "avg3"]
    ].copy()
    roster = roster.sort_values("avg", ascending=False)

    # Top scorers this season (on-field only)
    top_scorers = (
        players[
            (players["coach_first_name"] == coach_name) &
            (players["on_field"] == True)
        ]
        .groupby(["first_name", "last_name", "played_position"])
        .agg(
            games=("points_x", "count"),
            avg=("points_x", "mean"),
            total=("points_x", "sum"),
            best=("points_x", "max"),
        )
        .reset_index()
        .sort_values("avg", ascending=False)
        .head(15)
    )
    top_scorers["avg"] = top_scorers["avg"].round(1)

    # Season total
    season_total = scores["team_points"].sum()
    season_avg   = round(scores["team_points"].mean(), 1) if len(scores) else 0
    season_best  = scores["team_points"].max() if len(scores) else 0

    portrait_src = portraits.get(coach_name, "")
    pos_chart = _positional_line_chart(players, coach_name)
    field_data = fetch_current_team(coach_name)
    weekly_changes = load_weekly_changes(coach_name, field_data)

    return render_template(
        "coach_profile.html",
        coach=coach_name,
        portrait=portrait_src,
        scores=scores.to_dict(orient="records"),
        roster=roster.to_dict(orient="records"),
        top_scorers=top_scorers.to_dict(orient="records"),
        season_total=season_total,
        season_avg=season_avg,
        season_best=int(season_best) if season_best else 0,
        pos_chart=json.dumps(pos_chart),
        field_data=field_data,
        weekly_changes=weekly_changes,
    )
