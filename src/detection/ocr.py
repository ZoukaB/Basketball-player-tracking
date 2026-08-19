"""Jersey-number OCR.

Wraps the Roboflow Universe SmolVLM2 model used in the original notebook:

    basketball-jersey-numbers-ocr/7

Number boxes come from ``BasketballDetector``. This module crops them, reads
the digits, and optionally matches each crop to a SAM2 player mask via IoS.
"""

from __future__ import annotations

import os
from typing import Sequence

os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "CUDAExecutionProvider,CPUExecutionProvider",
)

import numpy as np
import supervision as sv
from inference import get_model

DEFAULT_OCR_MODEL_ID = "basketball-jersey-numbers-ocr/7"
DEFAULT_OCR_PROMPT = "Read the number."
CROP_PAD_PX = 10
CROP_PAD_PY = 10
MATCH_IOS_THRESHOLD = 0.9


def coords_above_threshold(
    matrix: np.ndarray,
    threshold: float,
    sort_desc: bool = True,
) -> list[tuple[int, int]]:
    """Return (row, col) pairs where ``matrix > threshold``."""
    values = np.asarray(matrix)
    rows, cols = np.where(values > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    if sort_desc:
        pairs.sort(key=lambda rc: values[rc[0], rc[1]], reverse=True)
    return pairs


def crop_number_boxes(
    frame: np.ndarray,
    detections: sv.Detections,
    pad_px: int = CROP_PAD_PX,
    pad_py: int = CROP_PAD_PY,
) -> list[np.ndarray]:
    """Pad, clip, and crop jersey-number boxes from a frame."""
    if len(detections) == 0:
        return []
    frame_h, frame_w = frame.shape[:2]
    boxes = sv.clip_boxes(
        sv.pad_boxes(xyxy=detections.xyxy, px=pad_px, py=pad_py),
        (frame_w, frame_h),
    )
    return [sv.crop_image(frame, xyxy) for xyxy in boxes]


def _ensure_masks(
    detections: sv.Detections,
    resolution_wh: tuple[int, int],
) -> sv.Detections:
    if detections.mask is not None:
        return detections
    detections.mask = sv.xyxy_to_mask(
        boxes=detections.xyxy,
        resolution_wh=resolution_wh,
    )
    return detections


def match_numbers_to_players(
    player_detections: sv.Detections,
    number_detections: sv.Detections,
    resolution_wh: tuple[int, int],
    ios_threshold: float = MATCH_IOS_THRESHOLD,
) -> list[tuple[int, int]]:
    """Match number boxes to player masks with mask IoS.

    Returns ``(player_idx, number_idx)`` pairs, highest overlap first.
    """
    if len(player_detections) == 0 or len(number_detections) == 0:
        return []

    players = _ensure_masks(player_detections, resolution_wh)
    numbers = _ensure_masks(number_detections, resolution_wh)
    overlap = sv.mask_iou_batch(
        masks_true=players.mask,
        masks_detection=numbers.mask,
        overlap_metric=sv.OverlapMetric.IOS,
    )
    return coords_above_threshold(overlap, ios_threshold)


class JerseyOCR:
    """Load the SmolVLM2 jersey-number recognizer and read cropped digits."""

    def __init__(
        self,
        model_id: str = DEFAULT_OCR_MODEL_ID,
        prompt: str = DEFAULT_OCR_PROMPT,
        pad_px: int = CROP_PAD_PX,
        pad_py: int = CROP_PAD_PY,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.pad_px = pad_px
        self.pad_py = pad_py
        self.model = get_model(model_id=model_id)

    def recognize(self, crop: np.ndarray) -> str:
        """Read a single jersey-number crop."""
        result = self.model.infer(crop, prompt=self.prompt)[0]
        return result.response.strip()

    def recognize_crops(self, crops: Sequence[np.ndarray]) -> list[str]:
        return [self.recognize(crop) for crop in crops]

    def recognize_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> list[str]:
        """Crop number boxes from ``frame`` and read each one."""
        crops = crop_number_boxes(
            frame,
            detections,
            pad_px=self.pad_px,
            pad_py=self.pad_py,
        )
        return self.recognize_crops(crops)

    def recognize_and_match(
        self,
        frame: np.ndarray,
        number_detections: sv.Detections,
        player_detections: sv.Detections,
        ios_threshold: float = MATCH_IOS_THRESHOLD,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """OCR number boxes and match them to player masks.

        Returns recognized strings (one per number box) and
        ``(player_idx, number_idx)`` pairs.
        """
        texts = self.recognize_detections(frame, number_detections)
        frame_h, frame_w = frame.shape[:2]
        pairs = match_numbers_to_players(
            player_detections,
            number_detections,
            resolution_wh=(frame_w, frame_h),
            ios_threshold=ios_threshold,
        )
        return texts, pairs
