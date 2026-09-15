"""Load and normalize shot CSVs from mock data or pipeline output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_DIR = REPO_ROOT / "outputs" / "mock_data"

NORMALIZED_COLUMNS = [
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
    "player_id": ("player_id", "player_Id", "playerId", "number", "tracker_id"),
    "name": ("name", "Name", "player", "player_name"),
    "team": ("team", "Team"),
    "shot_type": ("shot_type", "Shot type", "shotType", "type"),
    "result": ("result", "Result", "outcome"),
    "x_court": ("x_court", "court_x", "x"),
    "y_court": ("y_court", "court_y", "y"),
    "distance": ("distance", "Distance"),
    "confidence": ("confidence", "Confidence"),
}

_THREE_POINT_TYPES = {"3pt", "3-pt", "three", "three pointer", "three-pointer"}
_MADE_VALUES = {"made", "make", "1", "true"}


def default_shots_path() -> Path | None:
    if not MOCK_DIR.is_dir():
        return None
    csvs = sorted(p for p in MOCK_DIR.glob("*.csv") if p.is_file())
    if not csvs:
        return None
    preferred = [p for p in csvs if p.name.lower().startswith("mock_shots")]
    return (preferred or csvs)[0]


def list_mock_shot_files() -> list[Path]:
    if not MOCK_DIR.is_dir():
        return []
    return sorted(p for p in MOCK_DIR.glob("*.csv") if p.is_file())


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
    if "name" in out.columns:
        out["name"] = out["name"].astype("string")
    if "team" in out.columns:
        out["team"] = out["team"].astype("string")
    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype("string")
    return out[NORMALIZED_COLUMNS]


def load_shots(path: str | Path) -> pd.DataFrame:
    return normalize_shots(pd.read_csv(path))


def filter_shots(
    df: pd.DataFrame,
    *,
    teams: list[str] | None = None,
    players: list[str] | None = None,
    shot_types: list[str] | None = None,
    results: list[str] | None = None,
    min_confidence: float | None = None,
) -> pd.DataFrame:
    subset = df
    if teams:
        subset = subset[subset["team"].isin(teams)]
    if players:
        subset = subset[subset["name"].isin(players)]
    if shot_types:
        subset = subset[subset["shot_type"].isin(shot_types)]
    if results:
        subset = subset[subset["result"].isin(results)]
    if min_confidence is not None and subset["confidence"].notna().any():
        subset = subset[
            subset["confidence"].isna() | (subset["confidence"] >= min_confidence)
        ]
    return subset.reset_index(drop=True)


def shot_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    fga = int(len(df))
    made = df["result"].eq("Made")
    fgm = int(made.sum())
    threes = df["shot_type"].eq("3pt")
    tpa = int(threes.sum())
    tpm = int((threes & made).sum())
    return {
        "fga": fga,
        "fgm": fgm,
        "fg_pct": (fgm / fga) if fga else 0.0,
        "tpa": tpa,
        "tpm": tpm,
        "tp_pct": (tpm / tpa) if tpa else 0.0,
    }
