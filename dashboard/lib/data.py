"""Load and normalize shot CSVs produced by dashboard/build_data.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
DASHBOARD_CSV = "dashboard_shots.csv"

NORMALIZED_COLUMNS = [
    "video",
    "frame",
    "quarter",
    "clock_s",
    "player_id",
    "name",
    "team",
    "shot_type",
    "result",
    "x_court",
    "y_court",
    "distance",
    "confidence",
]

_COLUMN_ALIASES = {
    "video": ("video", "clip"),
    "frame": ("frame",),
    "quarter": ("quarter", "period"),
    "clock_s": ("clock_s", "clock"),
    "player_id": ("player_id", "player_Id", "playerId", "number", "tracker_id"),
    "name": ("name", "Name", "player", "player_name"),
    "team": ("team", "Team"),
    "shot_type": ("shot_type", "Shot type", "shotType", "type"),
    "result": ("result", "Result", "outcome"),
    "x_court": ("x_court", "court_x", "x"),
    "y_court": ("y_court", "court_y", "y"),
    "distance": ("distance", "Distance", "distance_ft"),
    "confidence": ("confidence", "Confidence"),
}

_THREE_POINT_TYPES = {"3pt", "3-pt", "three", "three pointer", "three-pointer"}
_MADE_VALUES = {"made", "make", "1", "true"}

# Full court is 94 x 50 ft (sports CourtConfiguration, NBA, feet); the chart is
# the left half with the hoop at (5.25, 25) like lib/court.py.
FULL_LENGTH = 94.0
COURT_WIDTH = 50.0
HOOP = (5.25, 25.0)
THREE_RADIUS = 23.75
CORNER_THREE_Y = 22.0   # |y - 25| beyond this in the corner = 3pt
CORNER_THREE_X = 14.0   # corner zone depth from the baseline


def to_half_court(x: float, y: float) -> tuple[float, float]:
    """Mirror right-half shots so every shot lands on the left half court.

    Uses the shot's own location, not `attacking_basket`: the latter comes from
    the possession box and can disagree with where the shot was taken.
    """
    if x > FULL_LENGTH / 2:
        return FULL_LENGTH - x, COURT_WIDTH - y
    return x, y


def hoop_distance(half_x: float, half_y: float) -> float:
    return float(np.hypot(half_x - HOOP[0], half_y - HOOP[1]))


def classify_shot_type(event_type: str, half_x: float | None, half_y: float | None) -> str:
    """collect_shots event type (JUMP / LAYUP) -> layup / 2pt / 3pt."""
    if str(event_type).upper() != "JUMP":
        return "layup"
    if half_x is None or half_y is None:
        return "2pt"
    corner = half_x <= CORNER_THREE_X and abs(half_y - HOOP[1]) >= CORNER_THREE_Y
    return "3pt" if corner or hoop_distance(half_x, half_y) >= THREE_RADIUS else "2pt"


def list_dashboard_files() -> list[Path]:
    """Mock data first (default), then outputs/<run>/dashboard_shots.csv, newest first."""
    if not OUTPUTS_DIR.is_dir():
        return []
    mock = sorted(p for p in (OUTPUTS_DIR / "mock_data").glob("*.csv") if p.is_file())
    runs = [p for p in OUTPUTS_DIR.glob(f"*/{DASHBOARD_CSV}") if p.is_file() and p not in mock]
    return mock + sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _pick_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _normalize_result(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return "Made" if text.lower() in _MADE_VALUES else "Missed"


def _normalize_shot_type(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in _THREE_POINT_TYPES:
        return "3pt"
    return text.lower()


def normalize_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Map mock or pipeline columns onto a single shot-chart schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    mapped = {}
    for dest, aliases in _COLUMN_ALIASES.items():
        src = _pick_column(df, aliases)
        mapped[dest] = df[src] if src is not None else pd.NA

    out = pd.DataFrame(mapped)
    out["result"] = out["result"].map(_normalize_result)
    out["shot_type"] = out["shot_type"].map(_normalize_shot_type)
    out["x_court"] = pd.to_numeric(out["x_court"], errors="coerce")
    out["y_court"] = pd.to_numeric(out["y_court"], errors="coerce")
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out["quarter"] = pd.to_numeric(out["quarter"], errors="coerce").astype("Int64")
    out["clock_s"] = pd.to_numeric(out["clock_s"], errors="coerce")
    if "name" in out.columns:
        out["name"] = out["name"].astype("string")
    if "team" in out.columns:
        out["team"] = out["team"].astype("string")
    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype("string")
    return out[NORMALIZED_COLUMNS]


def load_shots(path: str | Path) -> pd.DataFrame:
    return normalize_shots(pd.read_csv(path, dtype={"player_id": "string"}))


def quarter_label(quarter) -> str:
    if pd.isna(quarter):
        return "—"
    quarter = int(quarter)
    return f"Q{quarter}" if quarter <= 4 else ("OT" if quarter == 5 else f"OT{quarter - 4}")


def clock_label(seconds) -> str:
    """Game clock (seconds left in the quarter) -> M:SS."""
    if pd.isna(seconds):
        return "—"
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def time_label(quarter, seconds) -> str:
    return f"{quarter_label(quarter)} {clock_label(seconds)}"


def filter_shots(
    df: pd.DataFrame,
    *,
    team: str | None = None,
    player: str | None = None,
    shot_type: str | None = None,
    result: str | None = None,
    quarters: list[int] | None = None,
    clock_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Keep shots for one team, optional player, type/result and time window.

    `quarters`: keep only these quarters (None = all). `clock_range`: (high, low)
    seconds left on the game clock, inclusive, applied within each quarter.
    Shots without a time are dropped only when a time filter is active.
    """
    subset = df
    if team:
        subset = subset[subset["team"] == team]
    if player:
        subset = subset[subset["name"] == player]
    if quarters is not None:
        subset = subset[subset["quarter"].isin(quarters)]
    if clock_range is not None:
        high, low = max(clock_range), min(clock_range)
        subset = subset[subset["clock_s"].between(low, high)]
    if shot_type and shot_type != "All":
        subset = subset[subset["shot_type"] == shot_type]
    if result and result != "All":
        subset = subset[subset["result"] == result]
    return subset.reset_index(drop=True)


def shot_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    fga = int(len(df))
    made = df["result"].eq("Made")
    fgm = int(made.sum())
    threes = df["shot_type"].eq("3pt")
    tpa = int(threes.sum())
    tpm = int((threes & made).sum())
    twos = df["shot_type"].eq("2pt")
    layups = df["shot_type"].eq("layup")
    twopm = int((twos & made).sum())
    layupm = int((layups & made).sum())
    return {
        # Field-goal points only: the data has no free throws.
        "points": 3 * tpm + 2 * (twopm + layupm),
        "fga": fga,
        "fgm": fgm,
        "fg_pct": (fgm / fga) if fga else 0.0,
        "tpa": tpa,
        "tpm": tpm,
        "tp_pct": (tpm / tpa) if tpa else 0.0,
        "twopa": int(twos.sum()),
        "twopm": twopm,
        "layupa": int(layups.sum()),
        "layupm": layupm,
    }
