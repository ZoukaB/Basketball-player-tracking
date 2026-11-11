"""Demo render: game footage with shot banners and a live court map inset.

The video keeps its size; when a shot is decided, a MADE / MISSED banner (team
colour) shows at the top for `--banner-s`. A full court (same drawing as
shot_map.png) sits in the bottom-right corner: each shot's marker (made o /
missed x, colour of the shot's team, `--team-column`) appears at the frame the
pipeline decided it,
with a running per-team tally above the court. Built from a collect_shots.py
shots.csv: no model is run.

Usage (from repo root):
    .venv/Scripts/python.exe src/render_demo.py --video data/10_possessions_1video.mp4 \
        --shots outputs/continuous_10poss/shots.csv --out outputs/demo/10_possessions_demo.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports import MeasurementUnit  # noqa: E402
from sports.basketball import CourtConfiguration, League, draw_court, draw_made_and_miss_on_court  # noqa: E402

from src.utils import load_config, resolve_path  # noqa: E402

# Court drawing geometry (sports defaults, also used by collect_shots' overlay).
SCALE, PADDING = 20, 50
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, DARK = (255, 255, 255), (30, 30, 30)
POINTS = {"JUMP": 2, "LAYUP": 2}
BASKETS = ((5.25, 25.0), (88.75, 25.0))  # NBA full court, feet (as src/shots.py)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side demo: video + live shot map")
    p.add_argument("--video", required=True)
    p.add_argument("--shots", required=True, help="shots.csv from collect_shots.py")
    p.add_argument("--out", default="outputs/demo/demo.mp4")
    p.add_argument("--banner-s", type=float, default=1.5, help="Seconds a shot banner / highlight stays")
    p.add_argument("--inset-width", type=float, default=0.32, help="Map inset width, as a fraction of the frame")
    p.add_argument("--team-column", default="offense_team",
                   help="shots.csv column giving each shot's team (use 'team' when the run had "
                        "shots.enforce_offense_team off, e.g. continuous footage)")
    p.add_argument("--possession-label", action=argparse.BooleanOptionalAction, default=True,
                   help="Show 'Possession k/n' from the segment columns (off when segments are not possessions)")
    p.add_argument("--three-ft", type=float, default=23.75, help="Distance from which a jump shot counts 3")
    return p.parse_args()


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def shot_points(shot: pd.Series, three_ft: float) -> int:
    if shot["result"] != "made":
        return 0
    if shot["shot_type"] == "JUMP" and pd.notna(shot["court_x"]) and pd.notna(shot["court_y"]):
        # Distance to the nearer basket from the shot's own location (the CSV's
        # distance_ft uses attacking_basket, which can be the wrong end).
        x, y = float(shot["court_x"]), float(shot["court_y"])
        if min(np.hypot(x - bx, y - by) for bx, by in BASKETS) >= three_ft:
            return 3
    return POINTS.get(shot["shot_type"], 2)


def text(img, msg, org, scale, color, thickness=2):
    cv2.putText(img, msg, org, FONT, scale, DARK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, msg, org, FONT, scale, color, thickness, cv2.LINE_AA)


def court_map(config, shots: pd.DataFrame, team_colors: dict[str, str]) -> np.ndarray:
    """Full court with every shot so far, coloured by offense team."""
    court = draw_court(config=config)
    for team, hex_color in team_colors.items():
        sub = shots[(shots["shot_team"] == team) & shots["court_x"].notna()]
        made = sub[sub["result"] == "made"][["court_x", "court_y"]].to_numpy(np.float32)
        missed = sub[sub["result"] == "missed"][["court_x", "court_y"]].to_numpy(np.float32)
        if len(made) or len(missed):
            color = sv.Color.from_hex(hex_color)
            court = draw_made_and_miss_on_court(
                config=config, made_xy=made if len(made) else None, miss_xy=missed if len(missed) else None,
                made_color=color, miss_color=color, made_size=18, miss_size=18, line_thickness=4, court=court,
            )
    return court


def map_inset(court: np.ndarray, shots: pd.DataFrame, team_colors: dict[str, str], three_ft: float,
              highlight: pd.Series | None, pulse: float, width: int) -> np.ndarray:
    """Court (with the newest shot ringed) under a per-team tally strip, `width` px wide."""
    court = court.copy()
    if highlight is not None and pd.notna(highlight["court_x"]):
        cx = int(round(float(highlight["court_x"]) * SCALE + PADDING))
        cy = int(round(float(highlight["court_y"]) * SCALE + PADDING))
        color = hex_to_bgr(team_colors.get(highlight["shot_team"], "#ffffff"))
        cv2.circle(court, (cx, cy), int(30 + 25 * pulse), color, 6, cv2.LINE_AA)
    court = cv2.resize(court, (width, int(court.shape[0] * width / court.shape[1])), interpolation=cv2.INTER_AREA)

    # Tally strip: one half per team, made / attempts and points.
    u = width / 600  # text scale: tuned for a ~600 px wide inset
    strip_h = int(38 * u)
    strip = np.full((strip_h, width, 3), 35, np.uint8)
    x = int(10 * u)
    for team, hex_color in team_colors.items():
        sub = shots[shots["shot_team"] == team]
        made = int((sub["result"] == "made").sum())
        pts = int(sum(shot_points(r, three_ft) for _, r in sub.iterrows()))
        box = int(16 * u)
        y = (strip_h - box) // 2
        cv2.rectangle(strip, (x, y), (x + box, y + box), hex_to_bgr(hex_color), -1)
        text(strip, f"{team.split()[-1]}  {made}/{len(sub)}  {pts} pts", (x + box + int(8 * u), y + box - 1),
             0.6 * u, WHITE, max(1, int(round(1.5 * u))))
        x += width // 2
    inset = np.vstack([strip, court])
    cv2.rectangle(inset, (0, 0), (inset.shape[1] - 1, inset.shape[0] - 1), WHITE, 2)
    return inset


def annotate(frame: np.ndarray, banner: pd.Series | None, team_colors: dict[str, str],
             possession: tuple[int, int] | None, inset: np.ndarray) -> np.ndarray:
    """Banner on top, possession label bottom-left, map inset bottom-right."""
    out = frame.copy()
    h, w = out.shape[:2]
    u = h / 720  # text scale relative to a 720p frame
    margin = int(20 * u)
    if possession is not None:
        text(out, f"Possession {possession[0]}/{possession[1]}", (margin, h - margin), 0.8 * u, WHITE,
             max(2, int(2 * u)))
    if banner is not None:
        color = hex_to_bgr(team_colors.get(banner["shot_team"], "#888888"))
        kind = "Jump shot" if banner["shot_type"] == "JUMP" else "Layup"
        msg = f"{banner['result'].upper()}  -  {banner['shot_team']}  -  {kind}"
        cv2.rectangle(out, (0, 0), (w, int(70 * u)), color, -1)
        text(out, msg, (int(24 * u), int(48 * u)), 1.2 * u, WHITE, max(3, int(3 * u)))
    ih, iw = inset.shape[:2]
    out[h - margin - ih:h - margin, w - margin - iw:w - margin] = inset
    return out


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    """H.264 via imageio-ffmpeg when available (plays everywhere), else OpenCV mp4v."""
    try:
        import imageio_ffmpeg

        gen = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", pix_fmt_in="bgr24",
                                          output_params=["-crf", "20", "-preset", "medium"], macro_block_size=1)
        gen.send(None)
        return (lambda frame: gen.send(np.ascontiguousarray(frame).tobytes())), gen.close, "H.264"
    except ImportError:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        return writer.write, writer.release, "mp4v"


def main() -> None:
    args = parse_args()
    cfg = load_config()
    team_colors = dict(cfg["teams"]["team_colors"])
    kcfg = cfg["keypoints"]
    config = CourtConfiguration(league=getattr(League, kcfg.get("league", "NBA")),
                                measurement_unit=getattr(MeasurementUnit, kcfg.get("measurement_unit", "FEET")))

    shots = pd.read_csv(resolve_path(args.shots)).sort_values("frame").reset_index(drop=True)
    video_name = Path(args.video).name
    if "video" in shots.columns and (shots["video"] == video_name).any():
        shots = shots[shots["video"] == video_name].reset_index(drop=True)

    shots["shot_team"] = shots[args.team_column]

    # Possession starts (segment start frames) when the CSV has segment columns.
    starts = []
    if args.possession_label and {"segment", "segment_frame"} <= set(shots.columns):
        starts = sorted((shots["frame"] - shots["segment_frame"]).unique().tolist())

    cap = cv2.VideoCapture(str(resolve_path(args.video)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    inset_w = int(round(width * args.inset_width))
    hold = int(round(args.banner_s * fps))

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write, close, codec = open_writer(out, fps, (width, height))

    shown = 0  # shots already on the map
    court = court_map(config, shots.iloc[:0], team_colors)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Shots decided up to this frame join the map (redrawn only when it changes).
            n_now = int((shots["frame"] <= index).sum())
            if n_now != shown:
                shown = n_now
                court = court_map(config, shots.iloc[:shown], team_colors)
            latest = shots.iloc[shown - 1] if shown else None
            recent = latest if latest is not None and index - int(latest["frame"]) < hold else None
            pulse = 0.0 if recent is None else 0.5 + 0.5 * np.sin((index - int(recent["frame"])) * 0.4)

            possession = None
            if starts:
                k = sum(1 for s in starts if s <= index)
                possession = (max(k, 1), len(starts))

            inset = map_inset(court, shots.iloc[:shown], team_colors, args.three_ft, recent, pulse, inset_w)
            write(annotate(frame, recent, team_colors, possession, inset))
            index += 1
    finally:
        cap.release()
        close()
    print(f"Wrote {out} ({index}/{n_frames} frames, {fps:.0f} fps, {codec}, {len(shots)} shots)")


if __name__ == "__main__":
    main()
