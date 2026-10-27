"""Dashboard palette: SkillCorner green plus white, gray, and black."""

from __future__ import annotations

from dataclasses import dataclass

GREEN = "#006D00"
WHITE = "#FFFFFF"
GRAY = "#E8E8E8"
MUTED = "#6B6B6B"
BLACK = "#111111"


@dataclass(frozen=True)
class SkillCornerColors:
    green: str = GREEN
    white: str = WHITE
    gray: str = GRAY
    muted: str = MUTED
    black: str = BLACK
    made: str = GREEN
    miss: str = BLACK
    court_line: str = BLACK
    background: str = WHITE
    text: str = BLACK
    banner: str = GREEN
    font: str = "Roboto, Helvetica, Arial, sans-serif"


def load_colors() -> SkillCornerColors:
    return SkillCornerColors()


def theme_css(colors: SkillCornerColors | None = None) -> str:
    colors = colors or load_colors()
    return f"""
    <style>
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], [data-testid="stMainBlockContainer"],
        .stApp, .main {{
            background-color: {colors.white} !important;
            color: {colors.black};
        }}
        /* No horizontal scroll, and always reserve the vertical scrollbar's
           space: otherwise the responsive chart and the scrollbar keep resizing
           each other (the page "wiggles" sideways). */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            overflow-x: hidden !important;
        }}
        html, [data-testid="stMain"] {{
            scrollbar-gutter: stable;
        }}
        /* Shot type / Result filters: same left inset as the info panel. */
        .st-key-shot-filters {{
            padding: 4px 16px 0 20px;
        }}
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stBottom"],
        [data-testid="stStatusWidget"],
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarNav"],
        footer {{
            display: none !important;
        }}
        .block-container {{
            padding: 0 !important;
            max-width: 100% !important;
        }}
        [data-testid="stVerticalBlock"] {{
            gap: 0 !important;
        }}
        [data-testid="stElementContainer"] {{
            margin-bottom: 0 !important;
        }}
        iframe {{
            overflow: hidden !important;
        }}
        [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) {{
            background: {colors.banner} !important;
            padding: 10px 16px !important;
            min-height: 56px;
            align-items: center;
        }}
        [data-testid="stPopover"] {{
            width: 100%;
        }}
        [data-testid="stPopover"] button {{
            background-color: {colors.white} !important;
            color: {colors.black} !important;
            border: 1px solid {colors.gray} !important;
            font-weight: 600;
            min-width: 10rem;
            width: 100% !important;
            height: 2.4rem !important;
        }}
        [data-testid="stPopover"] button p,
        [data-testid="stPopover"] button span,
        [data-testid="stPopover"] button div {{
            color: {colors.black} !important;
        }}
        div[data-baseweb="popover"] {{
            background-color: {colors.white} !important;
        }}
        div[data-baseweb="popover"] label,
        div[data-baseweb="popover"] p,
        div[data-baseweb="popover"] span {{
            color: {colors.black} !important;
        }}
        .stSelectbox label {{
            color: {colors.black} !important;
            font-weight: 600;
        }}
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: {colors.white} !important;
            color: {colors.black} !important;
            border: 1px solid {colors.muted} !important;
        }}
        .stSelectbox [data-baseweb="select"] span {{
            color: {colors.black} !important;
        }}
        [data-baseweb="menu"] {{
            background-color: {colors.white} !important;
        }}
        [data-baseweb="menu"] li {{
            color: {colors.black} !important;
        }}
        .placeholder-bust {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: {colors.gray};
            border: 2px solid {colors.black};
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .placeholder-bust svg {{ width: 36px; height: 36px; fill: {colors.black}; }}
        /* Selected player: large headshot, name + stats in a column to its right
           (wraps under the photo when the panel is narrow). */
        .player-card {{
            display: flex;
            flex-wrap: wrap;
            align-items: flex-start;
            gap: 16px;
        }}
        .headshot {{
            width: 210px;
            max-width: 100%;
            height: auto;
            flex-shrink: 0;
        }}
        .player-card .info-meta {{
            margin-top: 0.35rem;
        }}
        .team-logo {{
            width: 64px;
            height: 64px;
            object-fit: contain;
            flex-shrink: 0;
        }}
        .logo-inline {{
            height: 1.2em;
            width: auto;
            vertical-align: -0.25em;
            margin-right: 6px;
        }}
        .info-name {{
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
            color: {colors.black};
            font-family: {colors.font};
        }}
        .info-meta {{
            margin: 0.45rem 0 0.6rem 0;
            font-size: 0.92rem;
            line-height: 1.55;
            color: {colors.black};
            font-family: {colors.font};
        }}
        .info-panel {{
            padding: 12px 16px 8px 20px;
            background: {colors.white};
            overflow: visible;
        }}
    </style>
    """
