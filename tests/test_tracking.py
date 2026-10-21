"""Smoke test for the SAM2 tracking brick (stable seed-once baseline).

Assumes a continuous shot (no cuts) with 10 detectable players on the first
frame. SAM2 is seeded once from the first frame's `player` detections and then
simply propagated; there is no cut detection and no re-prompting. The
cut/re-prompt machinery is kept dormant in src/ (see notes/future_ideas.md).

Usage (from repo root):
    .venv/Scripts/python.exe tests/test_tracking.py
    .venv/Scripts/python.exe tests/test_tracking.py --video data/x.mp4 --start 6 --seconds 4
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
from src.utils import analysis_stride, load_config, resolve_path  # noqa: E402

import supervision as sv  # noqa: E402

SAMPLE_EVERY = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 tracking brick smoke test")
    parser.add_argument("--video", default=None, help="Path to a video (default: config input.video)")
    parser.add_argument("--seconds", type=float, default=0.0, help="Process N seconds from --start (0 = to end)")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default 0)")
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
    start_frame = int(round(args.start * info.fps)) if args.start else 0
    span_frames = int(round(args.seconds * info.fps)) if args.seconds else 0
    end_frame = start_frame + span_frames if span_frames else 0
    max_frames = (end_frame - start_frame) // stride if end_frame else 0

    detector = Detector(cfg)
    tracker = SAM2Tracker(cfg)
    expected_players = cfg["sam2"].get("expected_players")

    end_label = f"{end_frame / info.fps:.1f}s" if end_frame else "end"
    print(f"Video: {video_path}")
    print(
        f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride} "
        f"(analysis {info.fps / stride:.1f} fps), window {args.start:.1f}s..{end_label}, "
        f"{max_frames if max_frames else 'all'} analysis frames"
    )
    print(
        f"  SAM2: {cfg['sam2']['variant']} ckpt={cfg['sam2']['checkpoint']} "
        f"seed_class_ids={[int(c) for c in tracker.seed_class_ids]}"
    )
    print(f"  Outputs -> {out_dir}")

    mask_annotator = sv.MaskAnnotator(opacity=0.5, color_lookup=sv.ColorLookup.TRACK)
    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.TRACK
    )

    frame_generator = sv.get_video_frames_generator(
        source_path=str(video_path), start=start_frame, stride=stride
    )

    # --- seed once on the first frame ---
    try:
        first_frame = next(frame_generator)
    except StopIteration:
        print("No frames to process.")
        return

    first_dets = detector.detect(first_frame)
    seed_dets = first_dets[np.isin(first_dets.class_id, tracker.seed_class_ids)]
    tracker.prompt_first_frame(first_frame, seed_dets)

    print(f"  seeded {len(seed_dets)} tracks on first frame (video {start_frame})")
    if expected_players and len(seed_dets) != expected_players:
        print(
            f"  [WARN] expected {expected_players} detectable players on frame 0, "
            f"got {len(seed_dets)} (seeding what is there)"
        )

    seed_ids = seed_dets.tracker_id
    all_track_ids: set[int] = set(int(t) for t in seed_ids) if seed_ids is not None else set()
    counts: list[int] = []
    scores: list[float] = []

    # --- propagate for the rest of the video ---
    for index, frame in enumerate(frame_generator, start=1):
        if max_frames and index >= max_frames:
            break

        t0 = time.perf_counter()
        tracked = tracker.propagate(frame)
        scores.append((time.perf_counter() - t0) * 1000)

        tracker_ids = np.asarray(tracked.tracker_id) if tracked.tracker_id is not None else np.array([])
        all_track_ids.update(int(t) for t in tracker_ids)
        counts.append(len(tracked))

        if index % SAMPLE_EVERY == 0 and len(tracked) > 0:
            labels = [str(int(t)) for t in tracker_ids]
            annotated = mask_annotator.annotate(scene=frame.copy(), detections=tracked)
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
            video_frame = start_frame + index * stride
            cv2.imwrite(str(out_dir / f"track_frame_{video_frame:04d}.jpg"), annotated)

    print("\n--- summary ---")
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
