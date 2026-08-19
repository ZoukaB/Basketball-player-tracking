"""SAM2 player tracking.

Wraps the real-time SAM2 camera predictor used in the original notebook.
Prompt the first frame with RF-DETR boxes, then propagate masks and stable
track IDs through the rest of the video.

The SAM2 fork is expected at ``$SAM2_DIR`` or, by default, next to this repo:

    <parent>/segment-anything-2-real-time
"""

from __future__ import annotations

import os
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv
import torch

DEFAULT_SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_EDGE_DISTANCE = 0.03
MATCH_IOU_THRESHOLD = 0.3


def _default_sam2_dir() -> Path:
    if os.environ.get("SAM2_DIR"):
        return Path(os.environ["SAM2_DIR"])
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "segment-anything-2-real-time"


def _ensure_sam2_on_path(sam2_dir: Path) -> None:
    if not sam2_dir.is_dir():
        raise FileNotFoundError(
            f"SAM2 repo not found at {sam2_dir}. "
            "Clone segment-anything-2-real-time there or set SAM2_DIR."
        )
    sam2_path = str(sam2_dir.resolve())
    if sam2_path not in sys.path:
        sys.path.insert(0, sam2_path)
    for module_name in list(sys.modules):
        if module_name == "sam2" or module_name.startswith("sam2."):
            del sys.modules[module_name]


def load_sam2_predictor(
    sam2_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    config: str = DEFAULT_SAM2_CONFIG,
):
    """Build the SAM2 camera predictor from the real-time fork."""
    sam2_dir = Path(sam2_dir) if sam2_dir is not None else _default_sam2_dir()
    _ensure_sam2_on_path(sam2_dir)

    checkpoint_path = (
        Path(checkpoint)
        if checkpoint is not None
        else sam2_dir / DEFAULT_SAM2_CHECKPOINT
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path}")

    from sam2.build_sam import build_sam2_camera_predictor

    return build_sam2_camera_predictor(config, str(checkpoint_path))


def _track_context():
    stack = ExitStack()
    stack.enter_context(torch.inference_mode())
    if torch.cuda.is_available():
        stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
    else:
        stack.enter_context(nullcontext())
    return stack


def match_detector_to_tracker_id(
    event_detections: sv.Detections,
    player_detections: sv.Detections,
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> Optional[int]:
    """Map the best detector box for an event class to a SAM2 tracker_id."""
    if len(event_detections) == 0 or len(player_detections) == 0:
        return None
    if player_detections.tracker_id is None:
        return None

    best_idx = int(np.argmax(event_detections.confidence))
    event_box = event_detections.xyxy[best_idx : best_idx + 1]
    ious = sv.box_iou_batch(player_detections.xyxy, event_box).reshape(-1)
    matched_idx = int(np.argmax(ious))

    if ious[matched_idx] >= iou_threshold:
        return int(player_detections.tracker_id[matched_idx])
    return None


class SAM2Tracker:
    """Prompt SAM2 on the first frame, then track player masks over time."""

    def __init__(
        self,
        predictor=None,
        *,
        sam2_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        config: str = DEFAULT_SAM2_CONFIG,
        mask_edge_distance: float = MASK_EDGE_DISTANCE,
    ) -> None:
        self.predictor = predictor or load_sam2_predictor(
            sam2_dir=sam2_dir,
            checkpoint=checkpoint,
            config=config,
        )
        self.mask_edge_distance = mask_edge_distance
        self._prompted = False

    def prompt_first_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> None:
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")

        if detections.tracker_id is None:
            detections.tracker_id = np.arange(1, len(detections) + 1)

        with _track_context():
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
                bbox = np.asarray([xyxy], dtype=np.float32)
                self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(obj_id),
                    bbox=bbox,
                )

        self._prompted = True

    def propagate(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        with _track_context():
            tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        if masks.ndim == 2:
            masks = masks[None, ...]

        masks = np.array(
            [
                sv.filter_segments_by_distance(
                    mask,
                    relative_distance=self.mask_edge_distance,
                    mode="edge",
                )
                for mask in masks
            ]
        )

        xyxy = sv.mask_to_xyxy(masks=masks)
        return sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)

    def reset(self) -> None:
        self._prompted = False
