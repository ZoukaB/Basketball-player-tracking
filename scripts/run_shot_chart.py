"""Detect, SAM2-track, and plot a shot chart for a clip.

Runs RF-DETR + SAM2, records each shot (tracker ID, court xy, made/miss),
writes ``shots_df.csv``, and draws the court map.

Usage (from the repo root, with .env containing ROBOFLOW_API_KEY):

    python scripts/run_shot_chart.py
    python scripts/run_shot_chart.py --video data/Q1_1st_part_compressed.mp4
    python scripts/run_shot_chart.py --fps 0          # every source frame
    python scripts/run_shot_chart.py --tracker-id 3
    python scripts/run_shot_chart.py --from-csv outputs/Q1_1st_part_compressed/shots_df.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_api_keys

load_api_keys(REPO_ROOT / ".env")

from src.pipeline import DEFAULT_TEAM_NAMES, BasketballPipeline
from src.pipeline.shots import plot_shot_chart

DEFAULT_VIDEO = REPO_ROOT / "data" / "Q1_1st_part_compressed.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shot chart from detections + SAM2")
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help="Path to an .mp4 clip (default: data/Q1_1st_part_compressed.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames. Omit to process the whole clip.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10,
        help="Tracking sample rate. Source video is strided to this FPS (default: 10). Pass 0 for every frame.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs"),
        help="Root output directory (a <video_name> folder is created inside)",
    )
    parser.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run jersey OCR to fill number/name on shots (default: off)",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Also dump per-frame JPEGs + SAM2 detections (large on disk)",
    )
    parser.add_argument(
        "--from-csv",
        default=None,
        help="Skip inference and plot an existing shots_df.csv",
    )
    parser.add_argument(
        "--tracker-id",
        type=int,
        default=None,
        help="Only plot shots for this SAM2 tracker ID",
    )
    parser.add_argument(
        "--team",
        default=None,
        help='Only plot shots for this team, e.g. "Boston Celtics"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_csv:
        shots_path = Path(args.from_csv)
        if not shots_path.is_file():
            raise SystemExit(f"shots csv not found: {shots_path}")
        shots_df = pd.read_csv(shots_path)
        run_dir = shots_path.parent
    else:
        video_path = Path(args.video)
        if not video_path.is_file():
            raise SystemExit(f"Video not found: {video_path}")
        run_dir = Path(args.output_dir) / video_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        history_dir = (run_dir / "history") if args.save_history else None

        pipeline = BasketballPipeline(
            team_names=dict(DEFAULT_TEAM_NAMES),
            use_ocr=args.ocr,
        )
        result = pipeline.run(
            video_path,
            max_frames=args.max_frames,
            history_dir=history_dir,
            target_fps=None if args.fps == 0 else args.fps,
        )
        shots_df = result.shots_df
        shots_path = run_dir / "shots_df.csv"
        shots_df.to_csv(shots_path, index=False)
        result.player_df.to_csv(run_dir / "player_df.csv", index=False)
        result.event_df.to_csv(run_dir / "event_df.csv", index=False)
        result.identity_df.to_csv(run_dir / "identity_df.csv", index=False)

        print("identity_df")
        print(result.identity_df.to_string(index=False))
        print()
        print(f"wrote {shots_path}")
        print(f"wrote {run_dir / 'player_df.csv'}")
        print(f"wrote {run_dir / 'event_df.csv'}")
        print(f"wrote {run_dir / 'identity_df.csv'}")
        if history_dir is not None:
            print(f"history: {history_dir}")

    chart_name = "shot_chart"
    if args.tracker_id is not None:
        chart_name += f"_tracker{args.tracker_id}"
    if args.team:
        chart_name += "_" + args.team.replace(" ", "_").lower()
    chart_path = run_dir / f"{chart_name}.jpg"

    plot_shot_chart(
        shots_df,
        chart_path,
        tracker_id=args.tracker_id,
        team=args.team,
    )

    print()
    print("shots_df")
    if len(shots_df) == 0:
        print("(no shots detected)")
    else:
        print(shots_df.to_string(index=False))
    print()
    print(f"wrote {chart_path}")
    print(f"shots: {len(shots_df)}")
    if "outcome" in shots_df.columns and len(shots_df) > 0:
        print(shots_df["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
