"""Plotly half-court shot chart with hoop at the bottom."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .colors import SkillCornerColors, load_colors
from .court import COURT_WIDTH, HALF_COURT_LENGTH, court_shapes, to_display


def build_shot_chart(
    shots: pd.DataFrame,
    colors: SkillCornerColors | None = None,
    title: str | None = None,
    height: int = 560,
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
        "Confidence: %{customdata[5]}<br>"
        "Clip: %{customdata[6]} · frame %{customdata[7]}"
        "<extra></extra>"
    )

    def _customdata(df: pd.DataFrame) -> list[list]:
        rows = []
        for row in df.to_dict("records"):
            dist = row.get("distance")
            conf = row.get("confidence")
            dist_s = f"{dist:.1f}" if pd.notna(dist) else "—"
            conf_s = f"{conf:.3f}" if pd.notna(conf) else "—"
            frame = row.get("frame")
            rows.append(
                [
                    row.get("name") or "Unknown",
                    row.get("team") or "—",
                    row.get("shot_type") or "—",
                    row.get("result") or "—",
                    dist_s,
                    conf_s,
                    row.get("video") if pd.notna(row.get("video")) else "—",
                    int(frame) if pd.notna(frame) else "—",
                ]
            )
        return rows

    def _xy(df: pd.DataFrame) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for x, y in zip(df["x_court"], df["y_court"]):
            dx, dy = to_display(float(x), float(y))
            xs.append(dx)
            ys.append(dy)
        return xs, ys

    if len(made):
        mx, my = _xy(made)
        fig.add_trace(
            go.Scatter(
                x=mx,
                y=my,
                mode="markers",
                name="Made",
                marker={
                    "symbol": "circle",
                    "size": 10,
                    "color": colors.made,
                    "line": {"width": 0.5, "color": colors.black},
                },
                customdata=_customdata(made),
                hovertemplate=hover,
            )
        )
    if len(missed):
        xx, xy = _xy(missed)
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=xy,
                mode="markers",
                name="Missed",
                marker={
                    "symbol": "x",
                    "size": 9,
                    "color": colors.miss,
                    "line": {"width": 2, "color": colors.miss},
                },
                customdata=_customdata(missed),
                hovertemplate=hover,
            )
        )

    margin = 1.0
    layout = {
        "font": {"family": colors.font, "color": colors.black},
        "paper_bgcolor": colors.white,
        "plot_bgcolor": colors.white,
        "shapes": court_shapes(colors),
        "xaxis": {
            "range": [-margin, COURT_WIDTH + margin],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": None,
            "fixedrange": True,
        },
        "yaxis": {
            "range": [-margin, HALF_COURT_LENGTH + margin],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": None,
            "scaleanchor": "x",
            "scaleratio": 1,
            "fixedrange": True,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
            "font": {"color": colors.black},
            "bgcolor": colors.white,
        },
        "margin": {"l": 4, "r": 4, "t": 24 if title else 4, "b": 4},
        "height": height,
        "autosize": True,
    }
    if title:
        layout["title"] = title
        layout["title_font"] = {"family": colors.font, "color": colors.black, "size": 18}
    fig.update_layout(**layout)
    return fig
