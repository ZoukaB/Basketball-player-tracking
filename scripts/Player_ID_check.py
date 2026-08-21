"""Check first-frame player IDs from the Roboflow detector.

Runs RF-DETR on frame 0, assigns the same 1-based tracker IDs SAM2 would
use, crops each player box, and saves a labeled grid.

Usage (from the repo root, with .env containing ROBOFLOW_API_KEY):

    python scripts/Player_ID_check.py --video path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_api_keys

load_api_keys(REPO_ROOT / ".env")

from src.detection import BasketballDetector


CROP_HEIGHT = 240
LABEL_BAR = 36
GRID_GAP = 8
GRID_BG = (20, 20, 20)
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save first-frame player crops labeled by tracker_id",
    )
    parser.add_argument("--video", required=True, help="Path to a video clip")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "player_id_check.jpg"),
        help="Output image path",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Grid columns (default: auto)",
    )
    return parser.parse_args()


def crop_players(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """Full-body crops from player boxes, clipped to the frame."""
    if len(detections) == 0:
        return []
    frame_h, frame_w = frame.shape[:2]
    boxes = sv.clip_boxes(detections.xyxy, (frame_w, frame_h))
    return [sv.crop_image(frame, box) for box in boxes]


def label_crop(crop: np.ndarray, tracker_id: int, height: int = CROP_HEIGHT) -> np.ndarray:
    """Resize a crop and draw ``id=<tracker_id>`` under it."""
    if crop.size == 0:
        crop = np.zeros((height, height, 3), dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = height / max(h, 1)
    resized = cv2.resize(crop, (max(int(w * scale), 1), height))

    labeled = np.full(
        (height + LABEL_BAR, resized.shape[1], 3),
        LABEL_BG,
        dtype=np.uint8,
    )
    labeled[:height] = resized
    text = f"id={int(tracker_id)}"
    cv2.putText(
        labeled,
        text,
        (8, height + LABEL_BAR - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        LABEL_FG,
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_grid(
    tiles: list[np.ndarray],
    cols: int | None = None,
    gap: int = GRID_GAP,
) -> np.ndarray:
    """Pack labeled crops into a single image."""
    n = len(tiles)
    cols = cols or min(5, n)
    rows = math.ceil(n / cols)
    tile_h = max(t.shape[0] for t in tiles)
    tile_w = max(t.shape[1] for t in tiles)

    canvas_h = rows * tile_h + (rows + 1) * gap
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), GRID_BG, dtype=np.uint8)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = gap + r * (tile_h + gap)
        x = gap + c * (tile_w + gap)
        padded = np.full((tile_h, tile_w, 3), GRID_BG, dtype=np.uint8)
        padded[: tile.shape[0], : tile.shape[1]] = tile
        canvas[y : y + tile_h, x : x + tile_w] = padded
    return canvas


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame = next(sv.get_video_frames_generator(str(video_path)))

    detector = BasketballDetector()
    players = detector.detect_players(frame, assign_tracker_ids=True)
    if len(players) == 0:
        raise RuntimeError("No players detected on the first frame.")

    crops = crop_players(frame, players)
    tiles = [
        label_crop(crop, int(tracker_id))
        for crop, tracker_id in zip(crops, players.tracker_id)
    ]
    grid = make_grid(tiles, cols=args.cols)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)

    print(f"players: {len(players)}")
    print(f"tracker_ids: {players.tracker_id.tolist()}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
