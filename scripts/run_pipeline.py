"""Run the basketball CV pipeline on a video.

Writes tables and history under ``outputs/<video_name>/``.

Usage (from the repo root, with .env containing ROBOFLOW_API_KEY):

    python scripts/run_pipeline.py --video path/to/clip.mp4 --max-frames 30
    python scripts/run_pipeline.py --video path/to/clip.mp4 --fps 10
    python scripts/run_pipeline.py --video path/to/clip.mp4 --ocr
    python scripts/render_detections.py --run-dir outputs/clip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_api_keys

load_api_keys(REPO_ROOT / ".env")

from src.pipeline import DEFAULT_TEAM_NAMES, BasketballPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basketball player-tracking pipeline")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to an .mp4 (or similar) clip",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames. Use ~30 for a quick test.",
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
        help="Run jersey OCR to fill number/name in identity_df (default: off)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    run_dir = Path(args.output_dir) / video_path.stem
    history_dir = run_dir / "history"
    run_dir.mkdir(parents=True, exist_ok=True)

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

    player_path = run_dir / "player_df.csv"
    event_path = run_dir / "event_df.csv"
    identity_path = run_dir / "identity_df.csv"
    shots_path = run_dir / "shots_df.csv"
    result.player_df.to_csv(player_path, index=False)
    result.event_df.to_csv(event_path, index=False)
    result.identity_df.to_csv(identity_path, index=False)
    result.shots_df.to_csv(shots_path, index=False)

    print("identity_df")
    print(result.identity_df.to_string(index=False))
    print()
    print("player_df (head)")
    print(result.player_df.head(20).to_string(index=False))
    print()
    print("event_df (head)")
    print(result.event_df.head(20).to_string(index=False))
    print()
    print(f"wrote {player_path}")
    print(f"wrote {event_path}")
    print(f"wrote {identity_path}")
    print(f"wrote {shots_path}")
    print(f"history: {history_dir}")
    print(f"player_df rows: {len(result.player_df)}")
    print(f"event_df rows:  {len(result.event_df)}")
    print(f"shots_df rows:  {len(result.shots_df)}")


if __name__ == "__main__":
    main()
