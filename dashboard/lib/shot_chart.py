"""Plotly half-court shot chart with SkillCorner colors."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .colors import SkillCornerColors, load_colors
from .court import COURT_WIDTH, HALF_COURT_LENGTH, court_shapes


def build_shot_chart(
    shots: pd.DataFrame,
    colors: SkillCornerColors | None = None,
    title: str | None = None,
) -> go.Figure:
    colors = colors or load_colors()
    fig = go.Figure()

    made = shots[shots["result"] == "Made"] if len(shots) else shots
    missed = shots[shots["result"] == "Missed"] if len(shots) else shots

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Team: %{customdata[1]}<br>"
        "Type: %{customdata[2]}<br>"
        "Result: %{customdata[3]}<br>"
        "Distance: %{customdata[4]} ft<br>"
        "Confidence: %{customdata[5]}"
        "<extra></extra>"
    )

    def _customdata(df: pd.DataFrame) -> list[list]:
        rows = []
        for row in df.to_dict("records"):
            dist = row.get("distance")
            conf = row.get("confidence")
            dist_s = f"{dist:.1f}" if pd.notna(dist) else "—"
            conf_s = f"{conf:.3f}" if pd.notna(conf) else "—"
            rows.append(
                [
                    row.get("name") or "Unknown",
                    row.get("team") or "—",
                    row.get("shot_type") or "—",
                    row.get("result") or "—",
                    dist_s,
                    conf_s,
                ]
            )
        return rows

    if len(made):
        fig.add_trace(
            go.Scatter(
                x=made["x_court"],
                y=made["y_court"],
                mode="markers",
                name="Made",
                marker={
                    "symbol": "circle",
                    "size": 11,
                    "color": colors.made,
                    "line": {"width": 0.6, "color": colors.text},
                },
                customdata=_customdata(made),
                hovertemplate=hover,
            )
        )
    if len(missed):
        fig.add_trace(
            go.Scatter(
                x=missed["x_court"],
                y=missed["y_court"],
                mode="markers",
                name="Missed",
                marker={
                    "symbol": "x",
                    "size": 10,
                    "color": colors.miss,
                    "line": {"width": 2, "color": colors.miss},
                },
                customdata=_customdata(missed),
                hovertemplate=hover,
            )
        )

    margin = 1.5
    fig.update_layout(
        title=title or "Shot Chart",
        title_font={"family": colors.font, "color": colors.text, "size": 20},
        font={"family": colors.font, "color": colors.text},
        paper_bgcolor=colors.background,
        plot_bgcolor=colors.background,
        shapes=court_shapes(colors),
        xaxis={
            "range": [-margin, HALF_COURT_LENGTH + margin],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": None,
            "fixedrange": True,
        },
        yaxis={
            "range": [-margin, COURT_WIDTH + margin],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": None,
            "scaleanchor": "x",
            "scaleratio": 1,
            "fixedrange": True,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=720,
    )
    return fig
