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
            overflow: hidden !important;
            height: 100vh !important;
            max-height: 100vh !important;
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
            overflow: hidden !important;
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
        [data-testid="stHorizontalBlock"]:not(:has([data-testid="stPopover"])) {{
            height: calc(100vh - 56px) !important;
            max-height: calc(100vh - 56px) !important;
            overflow: hidden !important;
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
        .stPlotlyChart,
        .stPlotlyChart > div,
        .js-plotly-plot,
        .plot-container {{
            height: calc(100vh - 56px) !important;
            max-height: calc(100vh - 56px) !important;
            overflow: hidden !important;
        }}
    </style>
    """
