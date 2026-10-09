"""Minimal smoke test for the jersey-number OCR brick.

Default: verifies OCR is disabled -> the model is not loaded and no VRAM is used.
With --enabled: runs SAM2 tracking on a few frames and reads numbers every frame.

Usage (from repo root):
    .venv/Scripts/python.exe tests/test_ocr.py                       # disabled check
    .venv/Scripts/python.exe tests/test_ocr.py --enabled --frames 15 --every 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector  # noqa: E402
from src.ocr import OCRBrick  # noqa: E402
from src.teams import TeamBrick  # noqa: E402
from src.tracking import SAM2Tracker  # noqa: E402
from src.utils import analysis_stride, load_config, resolve_path  # noqa: E402

import supervision as sv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jersey-number OCR minimal test")
    parser.add_argument("--video", default=None, help="Path to a video (default: config input.video)")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default 0)")
    parser.add_argument("--frames", type=int, default=15, help="Analysis frames to process (default 15)")
    parser.add_argument("--every", type=int, default=1, help="Run OCR every N analysis frames (default 1)")
    parser.add_argument("--min-crops", type=int, default=120, help="Team bootstrap crops")
    parser.add_argument("--enabled", action="store_true", help="Run OCR (loads the model)")
    return parser.parse_args()


def check_disabled(cfg: dict) -> None:
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    brick = OCRBrick({**cfg, "ocr": {**cfg["ocr"], "enabled": False}})
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"[disabled] model={brick.model} should_run(0)={brick.should_run(0)} "
          f"vram_delta={(after - before) / 1e9:.4f} GB")
    assert brick.model is None, "OCR model must not be loaded when disabled"
    assert brick.should_run(0) is False
    assert brick.update(np.zeros((16, 16, 3), np.uint8), sv.Detections.empty(), sv.Detections.empty()) == {}
    print("[disabled] OK: no model loaded, no-op update, 0 VRAM.")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg["teams"]["min_crops"] = args.min_crops

    check_disabled(cfg)
    if not args.enabled:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("OCR/tracking test requires CUDA")

    cfg["ocr"]["enabled"] = True
    cfg["ocr"]["stride_frames"] = args.every
    cfg["ocr"]["verbose"] = True

    video_path = resolve_path(args.video) if args.video else resolve_path(cfg["input"]["video"])
    assert video_path.exists(), f"Missing video: {video_path}"

    info = sv.VideoInfo.from_video_path(str(video_path))
    stride = analysis_stride(cfg, info.fps)
    start_frame = int(round(args.start * info.fps)) if args.start else 0
    print(f"[test] video={video_path.name} stride={stride} frames={args.frames} ocr_every={args.every}")

    print("[test] building Detector ...")
    detector = Detector(cfg)
    print("[test] building TeamBrick ...")
    teams = TeamBrick(cfg)
    print("[test] building SAM2Tracker ...")
    tracker = SAM2Tracker(cfg)
    print("[test] building OCRBrick (enabled) ...")
    ocr = OCRBrick(cfg)
    print("[test] all bricks ready")

    # Bootstrap teams on the first few frames.
    gen1 = sv.get_video_frames_generator(source_path=str(video_path), start=start_frame, stride=stride)
    for index, frame in enumerate(gen1):
        if index >= args.frames:
            break
        dets = detector.detect(frame)
        players = dets[np.isin(dets.class_id, cfg["classes"]["player_ids"])]
        teams.update(frame, players, index=index)
        if teams.fitted:
            break
    print(f"[test] teams fitted: {teams.fitted}")

    gen2 = sv.get_video_frames_generator(source_path=str(video_path), start=start_frame, stride=stride)
    team_by_tid: dict[int, int] = {}
    seeded = False
    for index, frame in enumerate(gen2):
        if index >= args.frames:
            break
        video_frame = start_frame + index * stride

        dets = detector.detect(frame)
        if not seeded:
            seed = dets[np.isin(dets.class_id, tracker.seed_class_ids)]
            if not tracker.prompt_first_frame(frame, seed):
                print(f"[test] analysis {index}: no seed players, skipping")
                continue
            seeded = True
            print(f"[test] analysis {index} (video {video_frame}): seeded {len(seed)} tracks")
            continue

        tracked = tracker.propagate(frame)
        team_ids = teams.update(frame, tracked, tracker_ids=tracked.tracker_id, index=index) if len(tracked) else None
        if team_ids is not None:
            for tid, team in zip(np.asarray(tracked.tracker_id), np.asarray(team_ids)):
                if int(team) >= 0:
                    team_by_tid[int(tid)] = int(team)

        number_dets = dets[dets.class_id == ocr.number_class_id]
        print(f"[test] analysis {index} (video {video_frame}): tracks={len(tracked)} "
              f"number_boxes={len(number_dets)} ocr_run={ocr.should_run(index)}")

        if ocr.should_run(index):
            ocr.update(frame, number_dets, tracked)
            print(f"[test]   validated_numbers={ocr.validated_numbers}")

    print("\n--- identities ---")
    for tid in sorted(ocr.validated_numbers):
        number = ocr.validated_numbers[tid]
        team = team_by_tid.get(tid)
        print(f"  track {tid}: #{number} team={team} name={ocr.resolve_name(team, number)}")


if __name__ == "__main__":
    main()
