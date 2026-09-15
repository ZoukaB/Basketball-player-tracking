"""SkillCorner viz palette for dashboard charts.

Pulls live values from ``skillcornerviz.utils.constants`` and
``skillcornerviz.utils.skillcorner_colors``. Fallbacks match v1.2.2.
"""

from __future__ import annotations

from dataclasses import dataclass

# skillcornerviz 1.2.2 (PHYSICAL PITCH / PITCH SHADOW / GREEN_TO_RED_SCALE).
_FALLBACK = {
    "PRIMARY_HIGHLIGHT_COLOR": "#00C800",
    "TEXT_COLOR": "#001400",
    "BASE_COLOR": "#D9D9D6",
    "MISS_COLOR": "#FF1A1A",
    "LIME": "#32FE6B",
    "BLUE": "#4D5CFF",
    "TEAL": "#17D9BA",
}


@dataclass(frozen=True)
class SkillCornerColors:
    made: str
    miss: str
    court_line: str
    background: str
    text: str
    lime: str
    blue: str
    teal: str
    font: str = "Roboto, Helvetica, Arial, sans-serif"


def _from_skillcornerviz() -> dict[str, str] | None:
    try:
        from skillcornerviz.utils import constants
        from skillcornerviz.utils import skillcorner_colors as sc_colors
    except ImportError:
        return None

    values = dict(_FALLBACK)
    values["PRIMARY_HIGHLIGHT_COLOR"] = str(constants.PRIMARY_HIGHLIGHT_COLOR)
    values["TEXT_COLOR"] = str(constants.TEXT_COLOR)
    values["BASE_COLOR"] = str(constants.BASE_COLOR)

    reds = getattr(sc_colors, "red_hex_codes", {}) or {}
    if "SCREEN HEX" in reds:
        values["MISS_COLOR"] = str(reds["SCREEN HEX"])
    elif getattr(constants, "GREEN_TO_RED_SCALE", None):
        values["MISS_COLOR"] = str(constants.GREEN_TO_RED_SCALE[0])

    greens = getattr(sc_colors, "greens", {}) or {}
    digital = greens.get("DIGITAL PITCH") or {}
    if "BASE" in digital:
        values["LIME"] = str(digital["BASE"])
    else:
        values["LIME"] = str(getattr(constants, "DARK_PRIMARY_HIGHLIGHT_COLOR", values["LIME"]))

    blues = getattr(sc_colors, "blue_hex_codes", {}) or {}
    if "SCREEN HEX" in blues:
        values["BLUE"] = str(blues["SCREEN HEX"])
    teals = getattr(sc_colors, "teal_hex_codes", {}) or {}
    if "SCREEN HEX" in teals:
        values["TEAL"] = str(teals["SCREEN HEX"])
    return values


def load_colors() -> SkillCornerColors:
    values = _from_skillcornerviz() or dict(_FALLBACK)
    return SkillCornerColors(
        made=values["PRIMARY_HIGHLIGHT_COLOR"],
        miss=values["MISS_COLOR"],
        court_line=values["TEXT_COLOR"],
        background=values["BASE_COLOR"],
        text=values["TEXT_COLOR"],
        lime=values["LIME"],
        blue=values["BLUE"],
        teal=values["TEAL"],
    )
