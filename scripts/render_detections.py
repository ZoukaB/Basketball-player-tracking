"""Render annotated camera and court videos from pipeline outputs.

Reads history + ``player_df.csv`` under ``outputs/<video_name>/`` and writes:

- ``<video_name>-detections.mp4`` — SAM2 masks/boxes/IDs on the camera view
- ``<video_name>-court.mp4`` — players on a 2D court with tracker IDs

Usage (from the repo root):

    python scripts/render_detections.py --run-dir outputs/my_clip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from sports.basketball import draw_court, draw_points_on_court
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.court import default_court_config
from src.pipeline.history import FRAME_DIR, iter_history, load_history_meta
from src.pipeline.rosters import TEAM_COLORS

# Same palette as the notebook tracking preview.
COLOR = sv.ColorPalette.from_hex(
    [
        "#ffff00",
        "#ff9b00",
        "#ff66ff",
        "#3399ff",
        "#ff66b2",
        "#ff8080",
        "#b266ff",
        "#9999ff",
        "#66ffff",
        "#33ff99",
        "#66ff66",
        "#99ff00",
    ]
)
FALLBACK_COURT_COLOR = "#888888"


def compress_h264(source: Path, target: Path) -> None:
    """Re-encode OpenCV ``mp4v`` output the same way the notebook does.

    ``sv.VideoSink`` writes MPEG-4 Part 2 (``mp4v``), which most players and
    browsers cannot play. ffmpeg ``libx264`` + ``yuv420p`` is the readable file.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found. Install it to write playable mp4s.")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )


def write_frames_to_mp4(
    frames,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    desc: str,
) -> None:
    """Write frames with VideoSink, then ffmpeg-compress like the notebook."""
    width -= width % 2
    height -= height % 2
    raw_path = output_path.with_name(output_path.stem + "-raw.mp4")
    video_info = sv.VideoInfo(
        width=width,
        height=height,
        fps=int(round(fps)),
        total_frames=total_frames,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(raw_path), video_info) as sink:
        for frame in tqdm(frames, total=total_frames, desc=desc):
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            sink.write_frame(frame)
    compress_h264(raw_path, output_path)
    raw_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render camera and court videos from pipeline history",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Pipeline output folder, e.g. outputs/my_clip",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="History folder (defaults to <run-dir>/history)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS from history/meta.json",
    )
    parser.add_argument(
        "--skip-detections",
        action="store_true",
        help="Do not write the camera detections video",
    )
    parser.add_argument(
        "--skip-court",
        action="store_true",
        help="Do not write the 2D court video",
    )
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.history:
        history_dir = Path(args.history)
        run_dir = history_dir.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        history_dir = run_dir / "history"
    else:
        raise SystemExit("Pass --run-dir outputs/<video_name> or --history .../history")
    return run_dir, history_dir


def render_detections_video(
    history_dir: Path,
    output_path: Path,
    fps: float,
) -> int:
    frame_paths = sorted((history_dir / FRAME_DIR).glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No history frames in {history_dir / FRAME_DIR}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    mask_annotator = sv.MaskAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.TRACK,
        opacity=0.5,
    )
    box_annotator = sv.BoxAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.TRACK,
        text_color=sv.Color.BLACK,
    )

    def annotated_frames():
        for _idx, frame, detections in iter_history(history_dir):
            annotated = frame.copy()
            if len(detections) > 0:
                if detections.mask is not None:
                    annotated = mask_annotator.annotate(
                        scene=annotated,
                        detections=detections,
                    )
                annotated = box_annotator.annotate(
                    scene=annotated,
                    detections=detections,
                )
                labels = (
                    [str(int(tid)) for tid in detections.tracker_id]
                    if detections.tracker_id is not None
                    else None
                )
                annotated = label_annotator.annotate(
                    scene=annotated,
                    detections=detections,
                    labels=labels,
                )
            yield annotated

    write_frames_to_mp4(
        annotated_frames(),
        output_path,
        fps=fps,
        width=width,
        height=height,
        total_frames=len(frame_paths),
        desc="render camera",
    )
    return len(frame_paths)


def _valid_court_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[np.isfinite(rows["court_x"]) & np.isfinite(rows["court_y"])]


def draw_frame_on_court(
    rows: pd.DataFrame,
    config,
    court: np.ndarray,
) -> np.ndarray:
    """Draw one frame of players onto a court image, labeled by tracker_id."""
    rows = _valid_court_rows(rows)
    if len(rows) == 0:
        return court

    if "team" in rows.columns:
        groups = rows.groupby("team", dropna=False)
    else:
        groups = [(None, rows)]

    for team_name, group in groups:
        xy = group[["court_x", "court_y"]].to_numpy(dtype=float)
        labels = [str(int(tid)) for tid in group["tracker_id"]]
        if isinstance(team_name, str) and team_name in TEAM_COLORS:
            fill = sv.Color.from_hex(TEAM_COLORS[team_name])
        else:
            fill = sv.Color.from_hex(FALLBACK_COURT_COLOR)
        court = draw_points_on_court(
            config=config,
            xy=xy,
            labels=labels,
            fill_color=fill,
            court=court,
        )
    return court


def render_court_video(
    player_df: pd.DataFrame,
    identity_df: pd.DataFrame | None,
    output_path: Path,
    fps: float,
) -> int:
    """2D court video using cleaned ``court_x`` / ``court_y`` and tracker IDs."""
    if len(player_df) == 0:
        raise RuntimeError("player_df is empty; run the pipeline first.")

    if identity_df is not None and "team" in identity_df.columns:
        player_df = player_df.merge(
            identity_df[["tracker_id", "team"]],
            on="tracker_id",
            how="left",
        )

    config = default_court_config()
    blank = draw_court(config=config)
    court_h, court_w = blank.shape[:2]
    n_frames = int(player_df["frame_idx"].max()) + 1
    grouped = player_df.groupby("frame_idx")

    def court_frames():
        for frame_idx in range(n_frames):
            court = draw_court(config=config)
            if frame_idx in grouped.groups:
                court = draw_frame_on_court(
                    grouped.get_group(frame_idx),
                    config,
                    court,
                )
            yield court

    write_frames_to_mp4(
        court_frames(),
        output_path,
        fps=fps,
        width=court_w,
        height=court_h,
        total_frames=n_frames,
        desc="render court",
    )
    return n_frames


def main() -> None:
    args = parse_args()
    run_dir, history_dir = resolve_run_dir(args)
    meta = load_history_meta(history_dir) if history_dir.is_dir() else {}
    video_name = meta.get("video_name") or run_dir.name
    fps = args.fps or float(meta.get("fps") or 30)

    if not args.skip_detections:
        if not history_dir.is_dir():
            raise FileNotFoundError(f"History folder not found: {history_dir}")
        det_path = run_dir / f"{video_name}-detections.mp4"
        n = render_detections_video(history_dir, det_path, fps)
        print(f"wrote {det_path}  ({n} frames)")

    if not args.skip_court:
        player_path = run_dir / "player_df.csv"
        if not player_path.is_file():
            raise FileNotFoundError(
                f"Missing {player_path}. Court render needs cleaned court_x/court_y."
            )
        player_df = pd.read_csv(player_path)
        identity_path = run_dir / "identity_df.csv"
        identity_df = pd.read_csv(identity_path) if identity_path.is_file() else None
        court_path = run_dir / f"{video_name}-court.mp4"
        n = render_court_video(player_df, identity_df, court_path, fps)
        print(f"wrote {court_path}  ({n} frames)")

    print(f"fps: {fps}")


if __name__ == "__main__":
    main()
