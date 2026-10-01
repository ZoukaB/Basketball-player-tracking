"""Smoke test for the detection brick on the first N seconds of a clip.

Usage (from repo root):
    .venv/Scripts/python.exe tests/test_detection.py
    .venv/Scripts/python.exe tests/test_detection.py --video data/Bad_detections_game2_10s.mp4 --seconds 8
    .venv/Scripts/python.exe tests/test_detection.py --video data/foo.mp4 --raw
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector  # noqa: E402
from src.utils import analysis_stride, load_config, resolve_path  # noqa: E402

import supervision as sv  # noqa: E402

SAMPLE_EVERY = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection brick smoke test")
    parser.add_argument("--video", default=None, help="Path to a video (default: config input.video)")
    parser.add_argument("--seconds", type=float, default=8.0, help="Only process the first N seconds")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: outputs/<video stem>)")
    parser.add_argument("--raw", action="store_true", help="Disable class_floor cleaning (raw detections)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    video_path = resolve_path(args.video) if args.video else resolve_path(cfg["input"]["video"])
    assert video_path.exists(), f"Missing video: {video_path}"

    out_dir = resolve_path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    info = sv.VideoInfo.from_video_path(str(video_path))
    stride = analysis_stride(cfg, info.fps)
    max_video_frames = int(round(args.seconds * info.fps)) if args.seconds else 0
    max_frames = max_video_frames // stride if max_video_frames else 0

    detector = Detector(cfg)
    print(f"Video: {video_path}")
    print(
        f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride} "
        f"(analysis {info.fps / stride:.1f} fps), processing {max_frames} analysis frames ({args.seconds}s)"
    )
    print(f"  Model: {detector.model_id} (conf={detector.confidence}, iou={detector.iou_threshold})")
    print(f"  Cleaning: {'OFF (raw)' if args.raw else 'ON'}  Outputs -> {out_dir}")

    dropped_by_reason: Counter = Counter()
    class_names: dict[int, str] = {}
    raw_total = clean_total = 0
    players_per_frame: list[int] = []

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    frame_generator = sv.get_video_frames_generator(source_path=str(video_path), stride=stride)
    for index, frame in enumerate(frame_generator):
        if max_frames and index >= max_frames:
            break

        raw = detector.detect_raw(frame)
        if args.raw:
            cleaned, stats = raw, {}
        else:
            cleaned, stats = detector.clean(raw, frame.shape)

        dropped_by_reason.update(stats)
        raw_total += len(raw)
        clean_total += len(cleaned)

        for cid, cname in zip(raw.class_id, raw.data.get("class_name", [])):
            class_names[int(cid)] = str(cname)

        players = cleaned[np.isin(cleaned.class_id, detector.player_class_ids)]
        players_per_frame.append(len(players))

        if index % SAMPLE_EVERY == 0:
            text = [class_names.get(int(c), str(int(c))) for c in cleaned.class_id]
            annotated = box_annotator.annotate(scene=frame.copy(), detections=cleaned)
            annotated = label_annotator.annotate(scene=annotated, detections=cleaned, labels=text)
            video_frame = index * stride
            out_path = out_dir / f"frame_{video_frame:04d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(
                f"  analysis {index:4d} (video {video_frame:4d}): raw={len(raw):3d} "
                f"clean={len(cleaned):3d} players={len(players):2d} -> {out_path.name}"
            )

    print("\nClass id -> name:", dict(sorted(class_names.items())))
    print(f"Boxes: raw={raw_total}  clean={clean_total}  dropped={raw_total - clean_total}")
    if dropped_by_reason:
        print("Dropped by reason:", dict(dropped_by_reason))
    if players_per_frame:
        print(
            f"Players/frame (clean): min={min(players_per_frame)} "
            f"max={max(players_per_frame)} avg={np.mean(players_per_frame):.1f}"
        )
    print(f"Frames processed: {len(players_per_frame)}")


if __name__ == "__main__":
    main()
