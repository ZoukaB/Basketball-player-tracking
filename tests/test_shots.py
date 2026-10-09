"""Smoke test for the shot detection brick.

Runs detection + keypoints + (optional) teams, feeds the ShotEventTracker, then
writes outputs/shots.json and outputs/shot_map.png.

Usage (from repo root):
    .venv/Scripts/python.exe tests/test_shots.py --video data/Bad_detections_game2_10s.mp4
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
from src.shots import ShotBrick  # noqa: E402
from src.teams import TeamBrick  # noqa: E402
from src.utils import analysis_stride, load_config, resolve_path  # noqa: E402

import supervision as sv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shot detection brick smoke test")
    parser.add_argument("--video", default=None, help="Path to a video (default: config input.video)")
    parser.add_argument("--seconds", type=float, default=0.0, help="Process N seconds from --start (0 = to end)")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default 0)")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: outputs/<video stem>)")
    parser.add_argument("--min-crops", type=int, default=120, help="Team bootstrap crops")
    return parser.parse_args()


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
    analysis_fps = info.fps / stride
    start_frame = int(round(args.start * info.fps)) if args.start else 0
    span_frames = int(round(args.seconds * info.fps)) if args.seconds else 0
    end_frame = start_frame + span_frames if span_frames else 0
    max_frames = (end_frame - start_frame) // stride if end_frame else 0

    detector = Detector(cfg)
    keypoints = KeypointBrick(cfg)
    teams = TeamBrick(cfg)
    shots = ShotBrick(cfg, analysis_fps=analysis_fps)

    print(f"Video: {video_path}")
    print(f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride}, analysis fps={analysis_fps:.1f}")
    print(f"  tracker: reset={shots.tracker.reset_time_frames} min_between={shots.tracker.minimum_frames_between_starts} "
          f"cooldown={shots.tracker.cooldown_frames_after_made} frames")

    # Bootstrap teams (optional; shots still work without team labels).
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

    gen2 = sv.get_video_frames_generator(source_path=str(video_path), start=start_frame, stride=stride)
    n_events = 0
    for index, frame in enumerate(gen2):
        if max_frames and index >= max_frames:
            break
        video_frame = start_frame + index * stride

        dets = detector.detect(frame)
        kpts, mask = keypoints.landmarks(frame)
        transformer = keypoints.transformer_from(kpts, mask)

        events = shots.update(frame, index, dets, transformer=transformer, team_brick=teams)
        for event in events:
            n_events += 1
            print(
                f"  [SHOT] analysis {index:3d} (video {video_frame:4d}): "
                f"{event['event']:6s} type={event['type']}"
            )

    print("\n--- summary ---")
    print(f"Shot pixel events: {n_events}")
    print(f"Recorded shots: {len(shots.shots)}")
    for s in shots.shots:
        loc = "n/a" if s["court_x"] is None else f"({s['court_x']:.1f},{s['court_y']:.1f})"
        print(f"  video {start_frame + s['frame'] * stride:4d}: {s['outcome']:6s} {s['type']:5s} "
              f"team={s['team']} track={s['tracker_id']} court={loc}")

    shots.to_json(out_dir / "shots.json")
    court = shots.draw_shot_map(keypoints.config, cfg["teams"])
    cv2.imwrite(str(out_dir / "shot_map.png"), court)
    print(f"Wrote {out_dir / 'shots.json'} and {out_dir / 'shot_map.png'}")


if __name__ == "__main__":
    main()
