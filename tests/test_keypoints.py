"""Smoke test for the court keypoints + homography brick.

Renders start/middle/end frames with:
  - the video frame overlaid with the homography-warped court lines (verification)
  - players marked in their team color
  - next to it, the top-down 2D court with team-colored projected players
Saves one side-by-side image per target frame.

Usage (from repo root):
    .venv/Scripts/python.exe tests/test_keypoints.py --video data/boston-celtics-new-york-knicks-game-1-q2-10.36-10.32.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector  # noqa: E402
from src.keypoints import KeypointBrick  # noqa: E402
from src.teams import TeamBrick  # noqa: E402
from src.utils import analysis_stride, load_config, resolve_path  # noqa: E402

import supervision as sv  # noqa: E402
from sports.basketball import draw_court, draw_points_on_court  # noqa: E402

SCALE, PADDING = 20, 50
LINE_COLOR_BGR = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Court keypoints brick smoke test")
    parser.add_argument("--video", default=None, help="Path to a video (default: config input.video)")
    parser.add_argument("--seconds", type=float, default=0.0, help="Process N seconds from --start (0 = to end)")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default 0)")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: outputs/<video stem>)")
    parser.add_argument("--min-crops", type=int, default=120, help="Team bootstrap crops")
    return parser.parse_args()


def court_to_frame_matrix(m2f: np.ndarray) -> np.ndarray:
    """Compose court(feet)->frame with court-image-pixel->feet so we can warp the court image."""
    a = np.array(
        [[1.0 / SCALE, 0.0, -PADDING / SCALE], [0.0, 1.0 / SCALE, -PADDING / SCALE], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return m2f @ a


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg["teams"]["min_crops"] = args.min_crops

    video_path = resolve_path(args.video) if args.video else resolve_path(cfg["input"]["video"])
    assert video_path.exists(), f"Missing video: {video_path}"
    out_dir = resolve_path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    info = sv.VideoInfo.from_video_path(str(video_path))
    stride = analysis_stride(cfg, info.fps)
    start_frame = int(round(args.start * info.fps)) if args.start else 0
    span_frames = int(round(args.seconds * info.fps)) if args.seconds else 0
    end_frame = start_frame + span_frames if span_frames else 0
    max_frames = (end_frame - start_frame) // stride if end_frame else 0

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    last_frame = min(end_frame, total_frames) if end_frame else total_frames
    n_analysis = max(1, (last_frame - start_frame) // stride)
    target_indices = sorted({0, n_analysis // 2, n_analysis - 1})

    detector = Detector(cfg)
    keypoints = KeypointBrick(cfg)
    teams = TeamBrick(cfg)

    palette = {}
    hex_values = list(cfg["teams"]["team_colors"].values())
    for i in range(2):
        palette[i] = sv.Color.from_hex(hex_values[i % len(hex_values)])

    print(f"Video: {video_path}")
    print(
        f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride}, "
        f"{total_frames} frames; targets (analysis idx) {target_indices} "
        f"-> video frames {[start_frame + i * stride for i in target_indices]}"
    )

    # --- pass 1: bootstrap the team classifier ---
    gen1 = sv.get_video_frames_generator(source_path=str(video_path), start=start_frame, stride=stride)
    for index, frame in enumerate(gen1):
        if max_frames and index >= max_frames:
            break
        dets = detector.detect(frame)
        players = dets[np.isin(dets.class_id, cfg["classes"]["player_ids"])]
        teams.update(frame, players, index=index)
        if teams.fitted:
            break
    print(f"  team classifier fitted: {teams.fitted} at analysis frame {teams.fit_frame}")

    vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=6)

    # --- pass 2: render the target frames ---
    gen2 = sv.get_video_frames_generator(source_path=str(video_path), start=start_frame, stride=stride)
    for index, frame in enumerate(gen2):
        if max_frames and index >= max_frames:
            break
        if index not in target_indices:
            continue

        video_frame = start_frame + index * stride
        kpts, mask = keypoints.landmarks(frame)
        n_anchors = int(np.count_nonzero(mask))
        transformer = keypoints.transformer_from(kpts, mask)
        m2f = keypoints.matrix_to_frame(kpts, mask)

        dets = detector.detect(frame)
        players = dets[np.isin(dets.class_id, cfg["classes"]["player_ids"])]
        frame_xy = (
            players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            if len(players)
            else np.empty((0, 2))
        )
        court_xy = keypoints.transform_points(transformer, frame_xy) if transformer is not None else np.empty((0, 2))
        team_ids = teams.update(frame, players, index=index) if len(players) else None

        # --- frame annotation ---
        annotated = frame.copy()
        if m2f is not None:
            court_lines = keypoints.draw_court_lines(scale=SCALE, padding=PADDING)
            warped = cv2.warpPerspective(court_lines, court_to_frame_matrix(m2f), (frame.shape[1], frame.shape[0]))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            mask_warp = gray > 60
            annotated[mask_warp] = LINE_COLOR_BGR
        annotated = vertex_annotator.annotate(scene=annotated, key_points=kpts)

        for i in range(len(frame_xy)):
            if team_ids is not None:
                t = int(team_ids[i])
                color = palette[t if t >= 0 else 0].as_bgr()
            else:
                color = (0, 255, 255)
            center = tuple(np.round(frame_xy[i]).astype(int))
            cv2.circle(annotated, center, 12, color, -1)
            cv2.circle(annotated, center, 12, (0, 0, 0), 2)

        # --- 2D court map with team-colored points ---
        court = draw_court(config=keypoints.config)
        for team_id, color in palette.items():
            if team_ids is None:
                continue
            sel = np.asarray(team_ids) == team_id
            if sel.any() and len(court_xy) == len(team_ids):
                court = draw_points_on_court(
                    config=keypoints.config,
                    xy=np.asarray(court_xy)[sel],
                    fill_color=color,
                    court=court,
                )

        court_resized = cv2.resize(
            court, (int(court.shape[1] * annotated.shape[0] / court.shape[0]), annotated.shape[0])
        )
        composite = np.hstack([annotated, court_resized])
        out_path = out_dir / f"projection_{video_frame:04d}.jpg"
        cv2.imwrite(str(out_path), composite)
        print(
            f"  analysis {index:3d} (video {video_frame:4d}): anchors={n_anchors} "
            f"homography={'yes' if transformer is not None else 'no'} -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
