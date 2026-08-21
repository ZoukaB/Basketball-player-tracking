"""Save and load pipeline ``frame_history`` / ``detections_history``.

Layout::

    outputs/<video_name>/history/
        meta.json
        frame_history/000000.jpg
        detections_history/000000.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

FRAME_DIR = "frame_history"
DET_DIR = "detections_history"
META_NAME = "meta.json"
JPEG_QUALITY = 95


def prepare_history_dir(history_dir: str | Path) -> Path:
    history_dir = Path(history_dir)
    (history_dir / FRAME_DIR).mkdir(parents=True, exist_ok=True)
    (history_dir / DET_DIR).mkdir(parents=True, exist_ok=True)
    return history_dir


def save_history_frame(
    history_dir: Path,
    frame_idx: int,
    frame: np.ndarray,
) -> Path:
    path = history_dir / FRAME_DIR / f"{frame_idx:06d}.jpg"
    cv2.imwrite(
        str(path),
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    return path


def save_history_detections(
    history_dir: Path,
    frame_idx: int,
    detections: sv.Detections,
) -> Path:
    path = history_dir / DET_DIR / f"{frame_idx:06d}.npz"
    payload: dict[str, np.ndarray] = {
        "xyxy": np.asarray(detections.xyxy, dtype=np.float32),
    }
    if detections.mask is not None:
        payload["mask"] = detections.mask.astype(np.uint8)
    if detections.tracker_id is not None:
        payload["tracker_id"] = np.asarray(detections.tracker_id, dtype=np.int32)
    if detections.class_id is not None:
        payload["class_id"] = np.asarray(detections.class_id, dtype=np.int32)
    if detections.confidence is not None:
        payload["confidence"] = np.asarray(detections.confidence, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


def save_history_meta(history_dir: Path, **meta) -> Path:
    path = history_dir / META_NAME
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_history_meta(history_dir: Path) -> dict:
    path = Path(history_dir) / META_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_history_detections(path: str | Path) -> sv.Detections:
    data = np.load(path, allow_pickle=False)
    xyxy = data["xyxy"]
    kwargs: dict = {"xyxy": xyxy}
    names = set(data.files)
    if "mask" in names and data["mask"].size > 0:
        kwargs["mask"] = data["mask"].astype(bool)
    if "tracker_id" in names and data["tracker_id"].size > 0:
        kwargs["tracker_id"] = data["tracker_id"]
    if "class_id" in names and data["class_id"].size > 0:
        kwargs["class_id"] = data["class_id"]
    if "confidence" in names and data["confidence"].size > 0:
        kwargs["confidence"] = data["confidence"]
    return sv.Detections(**kwargs)


def iter_history(history_dir: str | Path):
    """Yield ``(frame_idx, frame, detections)`` in order."""
    history_dir = Path(history_dir)
    frame_paths = sorted((history_dir / FRAME_DIR).glob("*.jpg"))
    det_dir = history_dir / DET_DIR
    for frame_path in frame_paths:
        frame_idx = int(frame_path.stem)
        det_path = det_dir / f"{frame_path.stem}.npz"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read {frame_path}")
        if not det_path.is_file():
            raise FileNotFoundError(f"Missing detections for frame {frame_idx}: {det_path}")
        yield frame_idx, frame, load_history_detections(det_path)
