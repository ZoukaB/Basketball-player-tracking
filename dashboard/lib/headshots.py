"""Player headshots and team logos from the repo's assets/ folder.

    assets/headshots/<Team_Name>/<number>_<First>_<Last>.png   e.g. Boston_Celtics/0_Jayson_Tatum.png
    assets/team_logos/<nickname>.png                           e.g. celtics.png -> "Boston Celtics"

Headshots are matched on team + jersey number (unique; see
assets/headshots/ids_resolved.csv). Missing files fall back to the placeholder.
Images are returned as inline data URIs because they are rendered inside HTML.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
HEADSHOT_DIR = ASSETS_DIR / "headshots"
LOGO_DIR = ASSETS_DIR / "team_logos"
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _MIME)


def find_headshot(team: str | None, number: str | None) -> Path | None:
    if not team or number in (None, ""):
        return None
    folder = HEADSHOT_DIR / str(team).strip().replace(" ", "_")
    prefix = f"{str(number).strip()}_"
    return next((p for p in _images(folder) if p.name.startswith(prefix)), None)


def find_logo(team: str | None) -> Path | None:
    if not team:
        return None
    team_lower = str(team).lower()
    return next((p for p in _images(LOGO_DIR) if p.stem.lower() in team_lower), None)


@lru_cache(maxsize=256)
def _data_uri(path: str, mtime: float) -> str:
    # mtime in the key picks up a replaced image without restarting the server.
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{_MIME[Path(path).suffix.lower()]};base64,{data}"


def _uri(path: Path | None) -> str | None:
    return _data_uri(str(path), path.stat().st_mtime) if path else None


def headshot_uri(team: str | None, number: str | None) -> str | None:
    """Inline data URI for the player's headshot, or None if there is no file."""
    return _uri(find_headshot(team, number))


def logo_uri(team: str | None) -> str | None:
    """Inline data URI for the team logo, or None if there is no file."""
    return _uri(find_logo(team))
