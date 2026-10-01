"""Smoke test for the SAM2 tracking brick on the first N seconds of a clip.

Usage (from repo root):
    .venv/Scripts/python.exe tests/test_tracking.py
    .venv/Scripts/python.exe tests/test_tracking.py --video data/Bad_detections_game2_10s.mp4 --seconds 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector  # noqa: E402
from src.tracking import SAM2Tracker  # noqa: E402
from src.utils import (  # noqa: E402
    FrameCutDetector,
    analysis_stride,
    load_config,
    resolve_path,
)

import supervision as sv  # noqa: E402

SAMPLE_EVERY = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 tracking brick smoke test")
    parser.add_argument("--video", default=None, help="Path to a video (default: config input.video)")
    parser.add_argument("--seconds", type=float, default=8.0, help="Only process the first N seconds")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: outputs/<video stem>)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 camera predictor requires CUDA; none available")

    video_path = resolve_path(args.video) if args.video else resolve_path(cfg["input"]["video"])
    assert video_path.exists(), f"Missing video: {video_path}"

    out_dir = resolve_path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    info = sv.VideoInfo.from_video_path(str(video_path))
    stride = analysis_stride(cfg, info.fps)
    max_video_frames = int(round(args.seconds * info.fps)) if args.seconds else 0
    max_frames = max_video_frames // stride if max_video_frames else 0

    detector = Detector(cfg)
    tracker = SAM2Tracker(cfg)
    cut_detector = FrameCutDetector(cfg["cuts"]["frame_diff_threshold"])

    print(f"Video: {video_path}")
    print(
        f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride} "
        f"(analysis {info.fps / stride:.1f} fps), {max_frames} analysis frames ({args.seconds}s)"
    )
    print(
        f"  SAM2: {cfg['sam2']['variant']} ckpt={cfg['sam2']['checkpoint']} "
        f"seed_class_ids={[int(c) for c in tracker.seed_class_ids]}"
    )
    print(f"  cut threshold={cut_detector.threshold}  Outputs -> {out_dir}")

    mask_annotator = sv.MaskAnnotator(opacity=0.5, color_lookup=sv.ColorLookup.TRACK)
    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.TRACK
    )

    needs_prompt = True
    seeded_at: int | None = None
    all_track_ids: set[int] = set()
    counts: list[int] = []
    scores: list[float] = []
    times_ms: list[float] = []

    frame_generator = sv.get_video_frames_generator(source_path=str(video_path), stride=stride)
    for index, frame in enumerate(frame_generator):
        if max_frames and index >= max_frames:
            break

        video_frame = index * stride
        t0 = time.perf_counter()

        is_cut, score = cut_detector.update(frame)
        if is_cut:
            print(f"  [WARN] hard cut @ analysis {index} (video {video_frame}) diff={score:.1f} -> re-prompt")
            tracker.reset()
            needs_prompt = True

        detections = detector.detect(frame)
        seed_dets = detections[np.isin(detections.class_id, tracker.seed_class_ids)]

        if needs_prompt:
            if not tracker.prompt_first_frame(frame, seed_dets):
                print(f"  analysis {index:4d}: no seed detections, waiting to prompt")
                continue
            needs_prompt = False
            seeded_at = index
            print(f"  analysis {index:4d} (video {video_frame:4d}): seeded {len(seed_dets)} tracks")

        tracked = tracker.propagate(frame)
        tracker_ids = np.asarray(tracked.tracker_id) if tracked.tracker_id is not None else np.array([])
        all_track_ids.update(int(t) for t in tracker_ids)
        counts.append(len(tracked))

        scores.append((time.perf_counter() - t0) * 1000)

        if index % SAMPLE_EVERY == 0 and len(tracked) > 0:
            labels = [str(int(t)) for t in tracker_ids]
            annotated = mask_annotator.annotate(scene=frame.copy(), detections=tracked)
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
            out_path = out_dir / f"track_frame_{video_frame:04d}.jpg"
            cv2.imwrite(str(out_path), annotated)

    print("\n--- summary ---")
    print(f"Seeded at analysis frame: {seeded_at}")
    print(f"Unique tracker ids: {len(all_track_ids)} -> {sorted(all_track_ids)}")
    if counts:
        print(f"Masks/analysis frame: min={min(counts)} max={max(counts)} avg={np.mean(counts):.1f}")
    if scores:
        print(f"ms/analysis frame (detect+SAM2): avg={np.mean(scores):.1f} min={min(scores):.1f} max={max(scores):.1f}")
        print(f"  => {1000.0 / np.mean(scores):.1f} analysis fps")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
