"""Evaluate pipeline shots against a ground-truth shot list.

Ground truth (semicolon CSV, as data/Groundtruth/*.csv): one row per shot with
`Video time` (time in the video file: "m:ss" or seconds), `Shot type`
(3Pt / Midrange / Layup), `Result` (Made / Missed), `Team` (NY / Boston), plus
optional `Game clock` (or `Time`; shown in the report), Posession / Number / Name.
Header whitespace, case and unit suffixes like "(s)" are ignored. The game clock cannot be used for
matching: broadcast edits and clock stoppages break its link to video time.

Predictions: shots.csv from src/collect_shots.py (frame, shot_type JUMP/LAYUP,
result, team / offense_team, court_x, court_y).

Matching: one-to-one, minimising |time difference|, only pairs within
`--tolerance-s` (Hungarian assignment). A prediction's time is its decision
frame / fps (make confirmed, or miss decided).

Reports:
- detection: precision, recall, F1, missed shots (FN), false detections (FP);
- on matched shots: accuracy + confusion matrix for result, shot type, team.
Prediction shot type: LAYUP -> Layup; JUMP -> 3Pt beyond the arc (or corner
three) from the nearer basket, else Midrange.

Usage (from repo root):
    .venv/Scripts/python.exe src/eval_shots.py --gt data/Groundtruth/Q1_part1.csv \
        --pred outputs/game2_q1part1/shots.csv --video data/Q1_part1_game2.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import resolve_path  # noqa: E402

# Same court geometry as src/shots.py (not imported: it pulls in sports/transformers).
LEFT_BASKET = (5.25, 25.0)
THREE_RADIUS_FT = 23.75
CORNER_THREE_Y_FT = 22.0   # |y - 25| beyond this, in the corner zone = 3pt
CORNER_THREE_X_FT = 14.0   # corner zone depth from the baseline
COURT_LENGTH_FT = 94.0

TEAM_ALIASES = {"ny": "New York Knicks", "nyk": "New York Knicks", "knicks": "New York Knicks",
                "new york knicks": "New York Knicks", "boston": "Boston Celtics", "bos": "Boston Celtics",
                "celtics": "Boston Celtics", "boston celtics": "Boston Celtics"}
TYPE_ALIASES = {"3pt": "3Pt", "3": "3Pt", "three": "3Pt", "midrange": "Midrange", "mid": "Midrange",
                "2pt": "Midrange", "layup": "Layup", "dunk": "Layup"}
LABELS = {
    "result": ["Made", "Missed"],
    "shot_type": ["3Pt", "Midrange", "Layup"],
    "team": ["Boston Celtics", "New York Knicks"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pipeline shots against ground truth")
    p.add_argument("--gt", required=True, help="Ground-truth CSV (semicolon-separated, with a Time column)")
    p.add_argument("--pred", required=True, help="shots.csv from collect_shots.py")
    p.add_argument("--video", default=None, help="Video, to read its fps (default: --fps)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--video-name", default=None, help="Keep only predictions of this video (default: all)")
    # 3 s: a prediction is timed at its decision frame (a miss is decided up to
    # reset_time_frames_s = 1.7 s after the start), GT at the rim, in whole
    # seconds. On Q1_part1 the scores are stable from 3 s to 4 s; 2 s splits
    # ~6 true shots into a miss + a false detection.
    p.add_argument("--tolerance-s", type=float, default=3.0, help="Max |time difference| for a match")
    p.add_argument("--team-column", default="team", help="Prediction column giving the shot's team")
    p.add_argument("--out", default=None, help="Write metrics JSON + confusion-matrix PNG with this path stem")
    return p.parse_args()


# ------------------------------------------------------------------ loading
def parse_time(value) -> float:
    """'m:ss', 'h:mm:ss' or seconds -> seconds."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return float("nan")
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part.replace(",", "."))
    return seconds


def load_gt(path: Path) -> pd.DataFrame:
    # utf-8-sig: files saved by Excel start with a BOM that would rename the first column.
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig").dropna(how="all")
    # Normalise headers: lower case, no unit suffix ("Video Time (s)" -> "video time").
    df.columns = [re.sub(r"\(.*?\)", "", c).strip().lower() for c in df.columns]
    if "video time" not in df.columns:
        sys.exit(f"{path}: no 'Video time' column. Add the time of each shot in the video file "
                 f"(m:ss or seconds); the game clock can't be matched to the video.")
    clock_col = next((c for c in ("game clock", "time", "clock") if c in df.columns), None)
    gt = pd.DataFrame({
        "time_s": df["video time"].map(parse_time),
        "clock": df[clock_col].str.strip() if clock_col else "",
        "shot_type": df["shot type"].str.strip().str.lower().map(TYPE_ALIASES),
        "result": df["result"].str.strip().str.capitalize(),
        "team": df["team"].str.strip().str.lower().map(TEAM_ALIASES),
        "possession": df.get("posession", df.get("possession")),
        "name": df.get("name"),
    })
    bad = gt[gt[["time_s", "shot_type", "result", "team"]].isna().any(axis=1)]
    if len(bad):
        sys.exit(f"{path}: rows with missing/unknown values:\n{df.loc[bad.index].to_string()}")
    return gt.sort_values("time_s").reset_index(drop=True)


def predicted_type(event_type: str, x, y) -> str:
    if str(event_type).upper() != "JUMP":
        return "Layup"
    if pd.isna(x) or pd.isna(y):
        return "Midrange"
    x, y = float(x), float(y)
    if x > COURT_LENGTH_FT / 2:  # measure from the nearer basket
        x, y = COURT_LENGTH_FT - x, 2 * LEFT_BASKET[1] - y
    corner = x <= CORNER_THREE_X_FT and abs(y - LEFT_BASKET[1]) >= CORNER_THREE_Y_FT
    far = float(np.hypot(x - LEFT_BASKET[0], y - LEFT_BASKET[1])) >= THREE_RADIUS_FT
    return "3Pt" if corner or far else "Midrange"


def load_pred(path: Path, fps: float, team_column: str, video_name: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if video_name:
        df = df[df["video"] == video_name]
    return pd.DataFrame({
        "time_s": df["frame"].astype(float) / fps,
        "frame": df["frame"].astype(int),
        "shot_type": [predicted_type(t, x, y) for t, x, y in zip(df["shot_type"], df["court_x"], df["court_y"])],
        "result": df["result"].str.strip().str.capitalize(),
        "team": df[team_column].fillna(""),
    }).sort_values("time_s").reset_index(drop=True)


# ----------------------------------------------------------------- matching
def match(gt: pd.DataFrame, pred: pd.DataFrame, tolerance_s: float) -> list[tuple[int, int]]:
    """One-to-one (gt_index, pred_index) pairs minimising total |dt|, all within tolerance."""
    if gt.empty or pred.empty:
        return []
    dt = np.abs(gt["time_s"].to_numpy()[:, None] - pred["time_s"].to_numpy()[None, :])
    big = 1e6
    rows, cols = linear_sum_assignment(np.where(dt <= tolerance_s, dt, big))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if dt[r, c] <= tolerance_s]


# ------------------------------------------------------------------ metrics
def confusion(true: list[str], pred: list[str], labels: list[str]) -> pd.DataFrame:
    cols = labels + sorted(set(pred) - set(labels))  # e.g. an empty/unknown team
    m = pd.DataFrame(0, index=labels, columns=cols)
    for t, p in zip(true, pred):
        if t in m.index:
            m.loc[t, p] += 1
    m.index.name, m.columns.name = "true", "pred"
    return m


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame, pairs: list[tuple[int, int]]) -> dict:
    tp = len(pairs)
    fn, fp = len(gt) - tp, len(pred) - tp
    precision = tp / len(pred) if len(pred) else 0.0
    recall = tp / len(gt) if len(gt) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "n_gt": len(gt), "n_pred": len(pred),
        "detection": {"tp": tp, "fn_missed": fn, "fp_extra": fp,
                      "precision": precision, "recall": recall, "f1": f1},
        "attributes": {},
    }
    for attr, labels in LABELS.items():
        true = [gt.loc[g, attr] for g, _ in pairs]
        guess = [pred.loc[p, attr] for _, p in pairs]
        acc = float(np.mean([t == p for t, p in zip(true, guess)])) if pairs else 0.0
        report["attributes"][attr] = {"accuracy": acc, "confusion": confusion(true, guess, labels)}
    return report


# ------------------------------------------------------------------ output
def fmt_time(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def print_report(gt, pred, pairs, report) -> None:
    det = report["detection"]
    print(f"Ground truth: {report['n_gt']} shots | predictions: {report['n_pred']} shots")
    print(f"\nDETECTION  precision {det['precision']:.2f}  recall {det['recall']:.2f}  F1 {det['f1']:.2f}"
          f"   (matched {det['tp']}, missed {det['fn_missed']}, false {det['fp_extra']})")
    for attr, res in report["attributes"].items():
        print(f"\n{attr.upper()}  accuracy {res['accuracy']:.2f} on {det['tp']} matched shots")
        print(res["confusion"].to_string())

    print("\nPER SHOT (time order)")
    matched_gt = {g: p for g, p in pairs}
    matched_pred = {p for _, p in pairs}
    rows = []
    for g in range(len(gt)):
        if g in matched_gt:
            p = matched_gt[g]
            errs = [a for a in LABELS if gt.loc[g, a] != pred.loc[p, a]]
            status = "OK" if not errs else "wrong " + ",".join(errs)
            rows.append((gt.loc[g, "time_s"], f"{fmt_time(gt.loc[g, 'time_s']):>7} {gt.loc[g, 'clock']:>5}  "
                         f"GT {gt.loc[g, 'shot_type']:8s} {gt.loc[g, 'result']:6s} {gt.loc[g, 'team']:16s} | "
                         f"pred f{pred.loc[p, 'frame']:<5d} {pred.loc[p, 'shot_type']:8s} {pred.loc[p, 'result']:6s} "
                         f"{pred.loc[p, 'team']:16s} {status}"))
        else:
            rows.append((gt.loc[g, "time_s"], f"{fmt_time(gt.loc[g, 'time_s']):>7} {gt.loc[g, 'clock']:>5}  "
                         f"GT {gt.loc[g, 'shot_type']:8s} {gt.loc[g, 'result']:6s} {gt.loc[g, 'team']:16s} | "
                         f"MISSED (no detection)"))
    for p in range(len(pred)):
        if p not in matched_pred:
            rows.append((pred.loc[p, "time_s"], f"{fmt_time(pred.loc[p, 'time_s']):>7} {'':5s}  {'':37s}"
                         f"| pred f{pred.loc[p, 'frame']:<5d} {pred.loc[p, 'shot_type']:8s} "
                         f"{pred.loc[p, 'result']:6s} {pred.loc[p, 'team']:16s} FALSE (no GT shot)"))
    for _, line in sorted(rows, key=lambda r: r[0]):
        print(line)


def save(report: dict, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in report.items() if k != "attributes"}
    data["attributes"] = {a: {"accuracy": r["accuracy"], "confusion": r["confusion"].to_dict(orient="index")}
                          for a, r in report["attributes"].items()}
    stem.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(report["attributes"]), figsize=(5 * len(report["attributes"]), 4.2))
    for ax, (attr, res) in zip(axes, report["attributes"].items()):
        m = res["confusion"]
        ax.imshow(m.to_numpy(), cmap="Blues")
        ax.set_xticks(range(m.shape[1]), [c.replace(" ", "\n") for c in m.columns], fontsize=8)
        ax.set_yticks(range(m.shape[0]), [r.replace(" ", "\n") for r in m.index], fontsize=8)
        ax.set_xlabel("predicted"), ax.set_ylabel("ground truth")
        ax.set_title(f"{attr}  (acc {res['accuracy']:.2f})")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, int(m.iat[i, j]), ha="center", va="center",
                        color="white" if m.iat[i, j] > m.to_numpy().max() / 2 else "black")
    det = report["detection"]
    fig.suptitle(f"Detection: P {det['precision']:.2f}  R {det['recall']:.2f}  F1 {det['f1']:.2f}  "
                 f"(matched {det['tp']}, missed {det['fn_missed']}, false {det['fp_extra']})")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=130)
    print(f"\nWrote {stem.with_suffix('.json')} and {stem.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    fps = args.fps
    if args.video:
        import cv2

        cap = cv2.VideoCapture(str(resolve_path(args.video)))
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
    gt = load_gt(resolve_path(args.gt))
    pred = load_pred(resolve_path(args.pred), fps, args.team_column, args.video_name)
    pairs = match(gt, pred, args.tolerance_s)
    report = evaluate(gt, pred, pairs)
    print_report(gt, pred, pairs, report)
    if args.out:
        save(report, resolve_path(args.out))


if __name__ == "__main__":
    main()
