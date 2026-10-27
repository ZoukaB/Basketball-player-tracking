"""Turn a collect_shots.py shots.csv into the dashboard's dashboard_shots.csv.

Per shot:
- team: the shooter's team, else the offense team, else a random team;
- player: a random (number, name) from that team's roster in src/config.yaml
  (seeded, `identity=random`) until OCR gives stable identities;
- location mirrored onto the left half court by where the shot was taken;
- shot_type: layup / 2pt / 3pt (collect_shots only gives JUMP / LAYUP);
- quarter + game clock from the clip name (`...-q2-10.36-10.32.mp4` = Q2 starting
  at 10:36) minus frame / fps. Confidence stays empty until the pipeline has one.

Usage (from repo root):
    .venv/Scripts/python.exe dashboard/build_data.py outputs/dashboard_3clips/shots.csv
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.data import DASHBOARD_CSV, REPO_ROOT, classify_shot_type, hoop_distance, to_half_court  # noqa: E402

COLUMNS = ["video", "frame", "quarter", "clock_s", "team", "team_source", "player_id", "name",
           "identity", "shot_type", "result", "x_court", "y_court", "distance_ft", "confidence",
           "court_x", "court_y"]

# "...-q2-10.36-10.32.mp4": quarter 2, clip starts at 10:36 on the game clock.
_CLIP_TIME = re.compile(r"-q(\d)-(\d{1,2})\.(\d{2})-\d{1,2}\.\d{2}", re.IGNORECASE)


def clip_start(video: str) -> tuple[int | None, float | None]:
    """(quarter, game-clock seconds left at the clip's first frame) from its name."""
    match = _CLIP_TIME.search(str(video))
    if not match:
        return None, None
    quarter, minutes, seconds = (int(g) for g in match.groups())
    return quarter, minutes * 60.0 + seconds


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def build_rows(shots: pd.DataFrame, rosters: dict[str, dict], seed: int, fps: float) -> list[dict]:
    rng = random.Random(seed)
    teams = sorted(rosters)
    rows = []
    for shot in shots.to_dict("records"):
        team, team_source = _text(shot.get("team")), "classifier"
        if not team:
            team, team_source = _text(shot.get("offense_team")), "offense"
        if not team:
            team, team_source = rng.choice(teams), "random"

        roster = sorted((rosters.get(team) or {}).items())
        number, name = rng.choice(roster) if roster else ("", "Unknown")

        cx, cy = pd.to_numeric(shot.get("court_x"), errors="coerce"), pd.to_numeric(shot.get("court_y"), errors="coerce")
        has_xy = pd.notna(cx) and pd.notna(cy)
        hx, hy = to_half_court(float(cx), float(cy)) if has_xy else (None, None)
        quarter, start_s = clip_start(shot.get("video", ""))
        frame = pd.to_numeric(shot.get("frame"), errors="coerce")
        clock_s = None if start_s is None or pd.isna(frame) else max(0.0, start_s - float(frame) / fps)
        rows.append({
            "video": shot.get("video", ""),
            "frame": shot.get("frame", ""),
            "quarter": quarter,
            "clock_s": None if clock_s is None else round(clock_s, 1),
            "team": team,
            "team_source": team_source,
            "player_id": str(number),
            "name": name,
            "identity": "random",
            "shot_type": classify_shot_type(shot.get("shot_type", ""), hx, hy),
            "result": _text(shot.get("result")),
            "x_court": None if hx is None else round(hx, 2),
            "y_court": None if hy is None else round(hy, 2),
            "distance_ft": None if hx is None else round(hoop_distance(hx, hy), 2),
            "confidence": shot.get("confidence"),
            "court_x": None if not has_xy else round(float(cx), 2),
            "court_y": None if not has_xy else round(float(cy), 2),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build dashboard data from collect_shots output")
    p.add_argument("shots_csv", help="shots.csv written by src/collect_shots.py")
    p.add_argument("--out", default=None, help=f"Output CSV (default: <same dir>/{DASHBOARD_CSV})")
    p.add_argument("--config", default=str(REPO_ROOT / "src" / "config.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0, help="Clip frame rate (frame -> game clock)")
    args = p.parse_args()

    shots_path = Path(args.shots_csv)
    shots = pd.read_csv(shots_path)
    with open(args.config, "r", encoding="utf-8") as handle:
        rosters = yaml.safe_load(handle)["teams"]["rosters"]

    out = pd.DataFrame(build_rows(shots, rosters, args.seed, args.fps), columns=COLUMNS)
    out_path = Path(args.out) if args.out else shots_path.parent / DASHBOARD_CSV
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Wrote {len(out)} shots -> {out_path}")


if __name__ == "__main__":
    main()
