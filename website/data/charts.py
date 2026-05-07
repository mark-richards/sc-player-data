"""
charts.py — Build Plotly figure dicts for the season summary page.
All functions return a plain Python dict (via fig.to_dict()) suitable
for json.dumps() and Plotly.react() on the client.
"""
import pandas as pd
import plotly.graph_objects as go

# Consistent coach colours (same order used across all charts)
COACH_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _coach_colour_map(coaches: list[str]) -> dict[str, str]:
    return {c: COACH_COLOURS[i % len(COACH_COLOURS)] for i, c in enumerate(sorted(coaches))}


# ── Ladder Journey ─────────────────────────────────────────────────────────

def ladder_journey_chart(ladder_df: pd.DataFrame) -> dict:
    """
    Line chart: x = round, y = ladder position (1 at top, 8 at bottom).
    One line per coach.
    """
    coaches = sorted(ladder_df["coach_first_name"].unique())
    colours = _coach_colour_map(coaches)

    fig = go.Figure()
    for coach in coaches:
        cdf = ladder_df[ladder_df["coach_first_name"] == coach].sort_values("round")
        fig.add_trace(go.Scatter(
            x=cdf["round"],
            y=cdf["position"],
            mode="lines+markers",
            name=coach,
            line=dict(color=colours[coach], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{coach}</b><br>Round %{{x}}<br>Position %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title="Ladder Journey",
        xaxis=dict(title="Round", dtick=1, gridcolor="#e0e0e0"),
        yaxis=dict(
            title="Position",
            autorange="reversed",
            tickvals=list(range(1, 9)),
            ticktext=[str(i) for i in range(1, 9)],
            gridcolor="#e0e0e0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=350,
    )
    return fig.to_dict()


# ── Score Box Plot ──────────────────────────────────────────────────────────

def score_boxplot(fixture_df: pd.DataFrame) -> dict:
    """
    Box plot per coach showing distribution of team_points.
    Ordered by median score descending.
    """
    coaches = fixture_df.groupby("coach_first_name")["team_points"].median().sort_values(ascending=False).index.tolist()
    colours = _coach_colour_map(coaches)

    fig = go.Figure()
    for coach in coaches:
        scores = fixture_df[fixture_df["coach_first_name"] == coach]["team_points"].dropna()
        fig.add_trace(go.Box(
            y=scores,
            name=coach,
            marker_color=colours[coach],
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            hovertemplate=f"<b>{coach}</b><br>%{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title="Score Distribution",
        yaxis=dict(title="Score", gridcolor="#e0e0e0"),
        xaxis=dict(title=""),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40),
        height=350,
    )
    return fig.to_dict()


# ── Score Scatter ──────────────────────────────────────────────────────────

def score_scatter(fixture_df: pd.DataFrame, portraits: dict | None = None) -> dict:
    """
    Scatter: x = avg points against, y = avg points for.
    One dot per coach, labelled by name. Quadrant lines show league averages.
    portraits arg kept for API compatibility but no longer used.
    """
    summary = (
        fixture_df.groupby("coach_first_name")
        .agg(avg_for=("team_points", "mean"), avg_against=("opposition_team_points", "mean"))
        .reset_index()
    )

    colours = _coach_colour_map(summary["coach_first_name"].tolist())
    league_avg_for = summary["avg_for"].mean()
    league_avg_against = summary["avg_against"].mean()

    pad = 30
    x_min = summary["avg_against"].min() - pad
    x_max = summary["avg_against"].max() + pad
    y_min = summary["avg_for"].min() - pad
    y_max = summary["avg_for"].max() + pad

    fig = go.Figure()

    # Quadrant reference lines (league averages)
    fig.add_hline(y=league_avg_for, line_dash="dot", line_color="#cccccc", line_width=1)
    fig.add_vline(x=league_avg_against, line_dash="dot", line_color="#cccccc", line_width=1)

    # One trace per coach so legend/hover works correctly
    for _, row in summary.iterrows():
        coach = row["coach_first_name"]
        fig.add_trace(go.Scatter(
            x=[row["avg_against"]],
            y=[row["avg_for"]],
            mode="markers+text",
            name=coach,
            marker=dict(size=14, color=colours[coach]),
            text=[coach],
            textposition="top center",
            textfont=dict(size=11),
            hovertemplate=(
                f"<b>{coach}</b><br>"
                f"Scored: {row['avg_for']:.0f}<br>"
                f"Conceded: {row['avg_against']:.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(title="Avg Conceded", gridcolor="#f0f0f0", range=[x_min, x_max]),
        yaxis=dict(title="Avg Scored", gridcolor="#f0f0f0", range=[y_min, y_max]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=50),
        height=350,
    )
    return fig.to_dict()
