"""Smoke test for the team classification brick (SigLIP + KMeans, CPU).

Bootstraps on crops from the first frames, fits once when the trigger fires,
then only predicts. Saves a montage of sample crops per cluster so you can map
cluster index -> team name.

Usage (from repo root):
    .venv/Scripts/python.exe tests/test_teams.py
    .venv/Scripts/python.exe tests/test_teams.py --video data/Bad_detections_game2_10s.mp4 --min-crops 200
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
from src.teams import TeamBrick  # noqa: E402
from src.utils import analysis_stride, load_config, resolve_path  # noqa: E402

import supervision as sv  # noqa: E402

MAX_SAMPLES_PER_TEAM = 12
CROP_W, CROP_H = 72, 144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Team classification brick smoke test")
    parser.add_argument("--video", default=None, help="Path to a video (default: config input.video)")
    parser.add_argument("--seconds", type=float, default=0.0, help="Process N seconds from --start (0 = to end)")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default 0)")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: outputs/<video stem>)")
    parser.add_argument("--min-crops", type=int, default=None, help="Override teams.min_crops")
    return parser.parse_args()


def montage(samples: dict[int, list[np.ndarray]]) -> np.ndarray:
    rows = []
    for team_id in sorted(samples):
        crops = samples[team_id][:MAX_SAMPLES_PER_TEAM]
        resized = [cv2.resize(c, (CROP_W, CROP_H)) for c in crops]
        while len(resized) < MAX_SAMPLES_PER_TEAM:
            resized.append(np.zeros((CROP_H, CROP_W, 3), np.uint8))
        rows.append(np.hstack(resized))
    if not rows:
        return np.zeros((CROP_H, CROP_W * MAX_SAMPLES_PER_TEAM, 3), np.uint8)
    return np.vstack(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.min_crops is not None:
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

    detector = Detector(cfg)
    teams = TeamBrick(cfg)
    print(f"Video: {video_path}")
    print(
        f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride}, "
        f"fit_mode={teams.fit_mode}, min_crops={teams.min_crops}, "
        f"max_bootstrap_frames={teams.max_bootstrap_frames}"
    )

    samples: dict[int, list[np.ndarray]] = {}
    predicted = 0
    class_counts: dict[int, int] = {}
    n_frames = 0

    frame_generator = sv.get_video_frames_generator(
        source_path=str(video_path), start=start_frame, stride=stride
    )
    for index, frame in enumerate(frame_generator):
        if max_frames and index >= max_frames:
            break
        n_frames += 1

        dets = detector.detect(frame)
        players = dets[np.isin(dets.class_id, cfg["classes"]["player_ids"])]
        if len(players) == 0:
            continue

        teams_ids = teams.update(frame, players, index=index)
        if teams_ids is None:
            continue

        crops = teams.crops_from_detections(frame, players)
        for team_id, crop in zip(teams_ids, crops):
            class_counts[int(team_id)] = class_counts.get(int(team_id), 0) + 1
            samples.setdefault(int(team_id), []).append(crop)
        predicted += len(teams_ids)

        if index % 10 == 0:
            print(f"  analysis {index:4d}: {len(players)} dets -> team counts so far {class_counts}")

    print("\n--- summary ---")
    print(f"Frames processed: {n_frames}")
    print(f"Fitted: {teams.fitted} (at analysis frame {teams.fit_frame})")
    print(f"Predicted detections: {predicted}")
    if class_counts:
        total = sum(class_counts.values())
        for team_id in sorted(class_counts):
            print(
                f"  cluster {team_id}: {class_counts[team_id]} crops "
                f"({100.0 * class_counts[team_id] / total:.1f}%)  "
                f"config name -> {teams.team_name(team_id)}"
            )
        out_path = out_dir / "team_samples.jpg"
        cv2.imwrite(str(out_path), montage(samples))
        print(f"  sample montage (row per cluster) -> {out_path}")


if __name__ == "__main__":
    main()
