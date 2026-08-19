"""RF-DETR basketball object detection.

Wraps the Roboflow Universe model used in the original notebook:

    basketball-player-detection-3-ycjdo/4

Detected classes:

    ball, ball-in-basket, number, player, player-in-possession,
    player-jump-shot, player-layup-dunk, player-shot-block, referee, rim

Set ``ONNXRUNTIME_EXECUTION_PROVIDERS`` before importing this module if you
need a specific ONNX Runtime backend. CUDA is preferred when available.
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Optional, Sequence

os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "CUDAExecutionProvider,CPUExecutionProvider",
)

import numpy as np
import supervision as sv
from inference import get_model

DEFAULT_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
DEFAULT_CONFIDENCE = 0.4
DEFAULT_IOU_THRESHOLD = 0.9
DEFAULT_NMS_THRESHOLD = 0.5
JERSEY_CROP_SCALE = 0.4


class DetectionClass(IntEnum):
    BALL = 0
    BALL_IN_BASKET = 1
    NUMBER = 2
    PLAYER = 3
    PLAYER_IN_POSSESSION = 4
    PLAYER_JUMP_SHOT = 5
    PLAYER_LAYUP_DUNK = 6
    PLAYER_SHOT_BLOCK = 7
    REFEREE = 8
    RIM = 9


CLASS_NAMES = {
    DetectionClass.BALL: "ball",
    DetectionClass.BALL_IN_BASKET: "ball-in-basket",
    DetectionClass.NUMBER: "number",
    DetectionClass.PLAYER: "player",
    DetectionClass.PLAYER_IN_POSSESSION: "player-in-possession",
    DetectionClass.PLAYER_JUMP_SHOT: "player-jump-shot",
    DetectionClass.PLAYER_LAYUP_DUNK: "player-layup-dunk",
    DetectionClass.PLAYER_SHOT_BLOCK: "player-shot-block",
    DetectionClass.REFEREE: "referee",
    DetectionClass.RIM: "rim",
}

PLAYER_CLASS_IDS: tuple[int, ...] = (
    DetectionClass.PLAYER,
    DetectionClass.PLAYER_IN_POSSESSION,
    DetectionClass.PLAYER_JUMP_SHOT,
    DetectionClass.PLAYER_LAYUP_DUNK,
    DetectionClass.PLAYER_SHOT_BLOCK,
)

OTHER_CLASS_IDS: tuple[int, ...] = (
    DetectionClass.BALL,
    DetectionClass.BALL_IN_BASKET,
    DetectionClass.RIM,
)


def filter_class_ids(
    detections: sv.Detections,
    class_ids: int | Sequence[int],
) -> sv.Detections:
    ids = np.atleast_1d(np.asarray(class_ids, dtype=int))
    if detections.class_id is None or len(detections) == 0:
        return detections[np.array([], dtype=bool)]
    return detections[np.isin(detections.class_id, ids)]


def assign_sequential_tracker_ids(detections: sv.Detections) -> sv.Detections:
    """Give each box a 1-based ID for SAM2 prompting on the first frame."""
    detections.tracker_id = np.arange(1, len(detections) + 1)
    return detections


def jersey_crops(
    frame: np.ndarray,
    detections: sv.Detections,
    scale: float = JERSEY_CROP_SCALE,
) -> list[np.ndarray]:
    """Crop the central region of each player box for jersey/team clustering."""
    if len(detections) == 0:
        return []
    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=scale)
    return [sv.crop_image(frame, box) for box in boxes]


class BasketballDetector:
    """Load the RF-DETR basketball detector and run inference on frames."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        class_agnostic_nms: bool = False,
    ) -> None:
        self.model_id = model_id
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.class_agnostic_nms = class_agnostic_nms
        self.model = get_model(model_id=model_id)

    def infer(self, frame: np.ndarray) -> sv.Detections:
        """Run the detector and return all classes as ``sv.Detections``."""
        result = self.model.infer(
            frame,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold,
            class_agnostic_nms=self.class_agnostic_nms,
        )[0]
        return sv.Detections.from_inference(result)

    def detect_players(
        self,
        frame: np.ndarray,
        assign_tracker_ids: bool = False,
        nms_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Detect player-related classes (standing, possession, shot actions)."""
        detections = filter_class_ids(self.infer(frame), PLAYER_CLASS_IDS)
        if nms_threshold is not None and len(detections) > 0:
            detections = detections.with_nms(
                threshold=nms_threshold,
                class_agnostic=True,
            )
        if assign_tracker_ids:
            detections = assign_sequential_tracker_ids(detections)
        return detections

    def detect_numbers(self, frame: np.ndarray) -> sv.Detections:
        """Detect jersey-number boxes."""
        return filter_class_ids(self.infer(frame), DetectionClass.NUMBER)

    def detect_other(self, frame: np.ndarray) -> sv.Detections:
        """Detect ball, ball-in-basket, and rim."""
        return filter_class_ids(self.infer(frame), OTHER_CLASS_IDS)

    def split(
        self,
        detections: sv.Detections,
    ) -> tuple[sv.Detections, sv.Detections, sv.Detections]:
        """Split a full inference result into players, numbers, and other."""
        players = filter_class_ids(detections, PLAYER_CLASS_IDS)
        numbers = filter_class_ids(detections, DetectionClass.NUMBER)
        other = filter_class_ids(detections, OTHER_CLASS_IDS)
        return players, numbers, other
