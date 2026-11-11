"""Save per-frame detections of a video once, for fast offline tuning.

Runs the same detector as collect_shots.py (Detector.detect: cleaned boxes) on
every analysis frame and writes one row per box of the shot-related classes
(ball, ball-in-basket, player-in-possession, jump shot, layup, rim), plus one
row per frame flagging hard cuts (class_id -1, same FrameCutDetector as
collect_shots). The shot logic can then be replayed from this file in seconds.

Output: CSV (gzip) with columns frame, class_id, confidence, x1, y1, x2, y2, cut.

Usage (from repo root):
    .venv/Scripts/python.exe src/dump_detections.py --video data/Q1_part1_game2.mp4 \
        --out outputs/detections/Q1_part1_game2.csv.gz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector  # noqa: E402
from src.utils import FrameCutDetector, analysis_stride, load_config, resolve_path, setup_env  # noqa: E402

import supervision as sv  # noqa: E402

KEEP_CLASSES = [0, 1, 4, 5, 6, 9]  # ball, ball-in-basket, possession, jump, layup, rim


def main() -> None:
    p = argparse.ArgumentParser(description="Dump per-frame detections for offline tuning")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="Output .csv.gz")
    args = p.parse_args()

    setup_env()
    cfg = load_config()
    video = resolve_path(args.video)
    info = sv.VideoInfo.from_video_path(str(video))
    stride = analysis_stride(cfg, info.fps)
    detector = Detector(cfg)
    cuts_cfg = cfg["cuts"]
    cut_detector = FrameCutDetector(cuts_cfg["z_meandiff_threshold"], cuts_cfg.get("min_gap_frames", 0))

    rows = []
    t0 = time.perf_counter()
    for index, frame in enumerate(sv.get_video_frames_generator(str(video), stride=stride)):
        is_cut = cut_detector.update(frame)[0]
        rows.append((index, -1, 0.0, 0.0, 0.0, 0.0, 0.0, int(is_cut)))
        d = detector.detect(frame)
        keep = np.isin(d.class_id, KEEP_CLASSES) if len(d) else np.zeros(0, bool)
        for cid, conf, box in zip(d.class_id[keep], d.confidence[keep], d.xyxy[keep]):
            rows.append((index, int(cid), float(conf), *(float(v) for v in box), 0))
        if index and index % 1000 == 0:
            print(f"  frame {index}/{info.total_frames // stride} ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["frame", "class_id", "confidence", "x1", "y1", "x2", "y2", "cut"])
    df.to_csv(out, index=False, float_format="%.2f")
    print(f"Wrote {out}: {df['frame'].nunique()} frames, {int((df['class_id'] >= 0).sum())} boxes, "
          f"{int(df['cut'].sum())} cuts, stride {stride}, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
