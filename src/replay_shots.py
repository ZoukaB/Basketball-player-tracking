"""Replay the shot logic on saved detections (src/dump_detections.py), in seconds.

Runs ShotBrick per segment (hard cuts, as collect_shots.py) without the team
classifier or homography: only detection and made/missed are replayed. For each
shot decided MADE it measures the ball-in-basket evidence:
- bib_frames / bib_run / bib_conf: ball-in-basket frames, longest consecutive
  run and max confidence between the shot start and the decision (+ margin);
- rim_share: share of those boxes whose centre lies inside a rim box (padded);
- ball_above: ball seen above a rim (horizontally over it) in the second before
  the first ball-in-basket box.
With --gt, each make is labelled true/false via the eval's time matching.

Usage (from repo root):
    .venv/Scripts/python.exe src/replay_shots.py --dump outputs/detections/Q1_part1_game2.csv.gz \
        --gt data/Groundtruth/Q1_part1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_shots import load_gt, match  # noqa: E402
from src.shots import ShotBrick  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

BALL, BASKET, RIM = 0, 1, 9
_BLANK = np.zeros((2, 2, 3), np.uint8)


def load_dump(path: Path) -> tuple[dict[int, sv.Detections], list[int], int]:
    """(detections per frame, cut frames, number of frames)."""
    df = pd.read_csv(path)
    n_frames = int(df["frame"].max()) + 1
    cuts = sorted(df.loc[(df["class_id"] == -1) & (df["cut"] == 1), "frame"].tolist())
    boxes = df[df["class_id"] >= 0]
    per_frame = {}
    for frame, g in boxes.groupby("frame"):
        per_frame[int(frame)] = sv.Detections(
            xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
            class_id=g["class_id"].to_numpy(int),
            confidence=g["confidence"].to_numpy(np.float32),
        )
    return per_frame, cuts, n_frames


def replay(per_frame: dict[int, sv.Detections], cuts: list[int], n_frames: int, cfg: dict,
           fps: float = 30.0) -> list[dict]:
    """Shots of the whole video, one ShotBrick per segment (as collect_shots)."""
    shots, brick = [], ShotBrick(cfg, analysis_fps=fps)
    cut_set = set(cuts)
    for frame in range(n_frames):
        if frame in cut_set:
            shots.extend(brick.shots)
            brick = ShotBrick(cfg, analysis_fps=fps)
        brick.update(_BLANK, frame, per_frame.get(frame, sv.Detections.empty()))
    shots.extend(brick.shots)
    return shots


def _inside(point: np.ndarray, boxes: np.ndarray, pad: float) -> bool:
    x, y = point
    return bool(np.any((boxes[:, 0] - pad <= x) & (x <= boxes[:, 2] + pad) &
                       (boxes[:, 1] - pad <= y) & (y <= boxes[:, 3] + pad)))


def make_features(per_frame: dict[int, sv.Detections], shot: dict, fps: float = 30.0) -> dict:
    start, end = int(shot["start_frame"]), int(shot["frame"])
    bib_frames, confs, in_rim, run, best_run = [], [], [], 0, 0
    for f in range(start, end + 1):
        d = per_frame.get(f)
        bib = d[d.class_id == BASKET] if d is not None and len(d) else None
        if bib is not None and len(bib):
            bib_frames.append(f)
            confs.append(float(bib.confidence.max()))
            rims = d[d.class_id == RIM]
            centre = bib.get_anchors_coordinates(sv.Position.CENTER)[int(np.argmax(bib.confidence))]
            pad = 0.25 * float(np.mean(rims.xyxy[:, 2] - rims.xyxy[:, 0])) if len(rims) else 0.0
            in_rim.append(len(rims) > 0 and _inside(centre, rims.xyxy, pad))
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0

    ball_above = False
    if bib_frames:
        for f in range(max(0, bib_frames[0] - int(fps)), bib_frames[0]):
            d = per_frame.get(f)
            if d is None or not len(d):
                continue
            balls, rims = d[d.class_id == BALL], d[d.class_id == RIM]
            for b in balls.xyxy:
                for r in rims.xyxy:
                    cx = 0.5 * (b[0] + b[2])
                    if r[0] <= cx <= r[2] and b[3] <= r[1] + 0.5 * (r[3] - r[1]):
                        ball_above = True
    return {
        "bib_frames": len(bib_frames), "bib_run": best_run,
        "bib_conf": max(confs) if confs else 0.0,
        "rim_share": float(np.mean(in_rim)) if in_rim else 0.0,
        "ball_above": ball_above,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Replay shot logic on saved detections")
    p.add_argument("--dump", required=True)
    p.add_argument("--gt", default=None, help="Ground-truth CSV, to label makes true/false")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tolerance-s", type=float, default=3.0)
    args = p.parse_args()

    cfg = load_config()
    per_frame, cuts, n_frames = load_dump(resolve_path(args.dump))
    shots = replay(per_frame, cuts, n_frames, cfg, args.fps)
    table = pd.DataFrame([{"start": s["start_frame"], "frame": s["frame"], "type": s["type"],
                           "outcome": s["outcome"], **(make_features(per_frame, s, args.fps)
                                                      if s["made"] else {})} for s in shots])
    print(f"{n_frames} frames, {len(cuts)} cuts, {len(shots)} shots "
          f"({int((table['outcome'] == 'made').sum())} made)")

    if args.gt:
        gt = load_gt(resolve_path(args.gt))
        pred = pd.DataFrame({"time_s": table["frame"] / args.fps})
        pairs = dict((p_, g) for g, p_ in match(gt, pred, args.tolerance_s))
        table["gt"] = [f"{gt.loc[pairs[i], 'result']}" if i in pairs else "no shot" for i in range(len(table))]
        made = table[table["outcome"] == "made"].copy()
        made["label"] = np.where(made["gt"] == "Made", "TRUE make", "FALSE make")
        cols = ["start", "frame", "type", "gt", "label", "bib_frames", "bib_run", "bib_conf", "rim_share", "ball_above"]
        print(made.sort_values(["label", "frame"])[cols].to_string(index=False))
        n_gt_made = int((gt["result"] == "Made").sum())
        n_true = int((made["label"] == "TRUE make").sum())
        print(f"\nmakes: {len(made)} predicted, {n_true} true, {len(made) - n_true} false; "
              f"GT makes {n_gt_made} -> made recall {n_true}/{n_gt_made}")
    else:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
