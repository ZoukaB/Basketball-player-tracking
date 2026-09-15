"""NBA half-court geometry in feet for Plotly shot charts."""

from __future__ import annotations

import math

import numpy as np

from .colors import SkillCornerColors

# One half of an NBA 94 x 50 ft floor.
HALF_COURT_LENGTH = 47.0
COURT_WIDTH = 50.0
HOOP_X = 5.25
HOOP_Y = 25.0
HOOP_RADIUS = 0.75
BACKBOARD_X = 4.0
BACKBOARD_WIDTH = 6.0
RESTRICTED_RADIUS = 4.0
PAINT_LENGTH = 19.0
PAINT_WIDTH = 16.0
FT_CIRCLE_RADIUS = 6.0
THREE_POINT_RADIUS = 23.75
CORNER_THREE_FROM_SIDELINE = 3.0
CENTER_CIRCLE_RADIUS = 6.0

_LINE_WIDTH = 1.6


def _line(x0: float, y0: float, x1: float, y1: float, color: str) -> dict:
    return {
        "type": "line",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "line": {"color": color, "width": _LINE_WIDTH},
        "layer": "below",
    }


def _rect(x0: float, y0: float, x1: float, y1: float, color: str) -> dict:
    return {
        "type": "rect",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "line": {"color": color, "width": _LINE_WIDTH},
        "fillcolor": "rgba(0,0,0,0)",
        "layer": "below",
    }


def _circle(x: float, y: float, r: float, color: str, dash: str | None = None) -> dict:
    line: dict = {"color": color, "width": _LINE_WIDTH}
    if dash:
        line["dash"] = dash
    return {
        "type": "circle",
        "x0": x - r,
        "y0": y - r,
        "x1": x + r,
        "y1": y + r,
        "line": line,
        "fillcolor": "rgba(0,0,0,0)",
        "layer": "below",
    }


def _polyline(xs: np.ndarray, ys: np.ndarray, color: str, dash: str | None = None) -> dict:
    line: dict = {"color": color, "width": _LINE_WIDTH}
    if dash:
        line["dash"] = dash
    return {
        "type": "path",
        "path": _path(xs, ys),
        "line": line,
        "fillcolor": "rgba(0,0,0,0)",
        "layer": "below",
    }


def _path(xs: np.ndarray, ys: np.ndarray) -> str:
    parts = [f"M {xs[0]},{ys[0]}"]
    parts.extend(f"L {x},{y}" for x, y in zip(xs[1:], ys[1:]))
    return " ".join(parts)


def _arc(
    cx: float,
    cy: float,
    radius: float,
    theta0: float,
    theta1: float,
    color: str,
    n: int = 80,
    dash: str | None = None,
) -> dict:
    angles = np.linspace(theta0, theta1, n)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return _polyline(xs, ys, color, dash=dash)


def court_shapes(colors: SkillCornerColors) -> list[dict]:
    """Plotly layout shapes for a 47 x 50 ft NBA half-court."""
    line = colors.court_line
    paint_y0 = HOOP_Y - PAINT_WIDTH / 2
    paint_y1 = HOOP_Y + PAINT_WIDTH / 2
    corner_y0 = CORNER_THREE_FROM_SIDELINE
    corner_y1 = COURT_WIDTH - CORNER_THREE_FROM_SIDELINE
    corner_dx = math.sqrt(
        max(THREE_POINT_RADIUS**2 - (HOOP_Y - corner_y0) ** 2, 0.0)
    )
    corner_x = HOOP_X + corner_dx
    theta = math.atan2(HOOP_Y - corner_y0, corner_dx)

    shapes = [
        _rect(0, 0, HALF_COURT_LENGTH, COURT_WIDTH, line),
        _rect(0, paint_y0, PAINT_LENGTH, paint_y1, line),
        _circle(HOOP_X, HOOP_Y, HOOP_RADIUS, line),
        _line(
            BACKBOARD_X,
            HOOP_Y - BACKBOARD_WIDTH / 2,
            BACKBOARD_X,
            HOOP_Y + BACKBOARD_WIDTH / 2,
            line,
        ),
        _arc(HOOP_X, HOOP_Y, RESTRICTED_RADIUS, -math.pi / 2, math.pi / 2, line),
        _arc(PAINT_LENGTH, HOOP_Y, FT_CIRCLE_RADIUS, -math.pi / 2, math.pi / 2, line),
        _arc(
            PAINT_LENGTH,
            HOOP_Y,
            FT_CIRCLE_RADIUS,
            math.pi / 2,
            3 * math.pi / 2,
            line,
            dash="dash",
        ),
        _line(0, corner_y0, corner_x, corner_y0, line),
        _line(0, corner_y1, corner_x, corner_y1, line),
        _arc(HOOP_X, HOOP_Y, THREE_POINT_RADIUS, -theta, theta, line),
        _arc(
            HALF_COURT_LENGTH,
            HOOP_Y,
            CENTER_CIRCLE_RADIUS,
            math.pi / 2,
            3 * math.pi / 2,
            line,
        ),
    ]
    return shapes
