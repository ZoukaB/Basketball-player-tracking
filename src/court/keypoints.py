"""Court keypoint detection and player-to-court mapping.

Wraps the Roboflow Universe model used in the original notebook:

    basketball-court-detection-2/14

Confident court landmarks (anchor confidence 0.5) are used to estimate a
homography. Player feet (box bottom-center) are then mapped into court
coordinates. ``clean_court_paths`` applies the notebook's ``clean_paths``
smoothing to a full-video ``(frames, players, 2)`` array.
"""

from __future__ import annotations

import os
from typing import Optional

os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "CUDAExecutionProvider,CPUExecutionProvider",
)

import numpy as np
import supervision as sv
from inference import get_model
from sports import MeasurementUnit, ViewTransformer, clean_paths
from sports.basketball import CourtConfiguration, League

DEFAULT_KEYPOINT_MODEL_ID = "basketball-court-detection-2/14"
DEFAULT_KEYPOINT_CONFIDENCE = 0.3
DEFAULT_ANCHOR_CONFIDENCE = 0.5
MIN_LANDMARKS = 4

CLEAN_PATH_JUMP_SIGMA = 3.5
CLEAN_PATH_MIN_JUMP_DIST = 0.6
CLEAN_PATH_MAX_JUMP_RUN = 18
CLEAN_PATH_PAD_AROUND_RUNS = 2
CLEAN_PATH_SMOOTH_WINDOW = 9
CLEAN_PATH_SMOOTH_POLY = 2


def default_court_config() -> CourtConfiguration:
    return CourtConfiguration(
        league=League.NBA,
        measurement_unit=MeasurementUnit.FEET,
    )


def landmarks_mask(
    key_points: sv.KeyPoints,
    anchor_confidence: float = DEFAULT_ANCHOR_CONFIDENCE,
) -> np.ndarray:
    """Boolean mask of keypoints above the anchor-confidence threshold."""
    if key_points.keypoint_confidence is None or len(key_points) == 0:
        return np.zeros((0,), dtype=bool)
    return key_points.keypoint_confidence[0] > anchor_confidence


def court_transformer(
    key_points: sv.KeyPoints,
    config: CourtConfiguration | None = None,
    anchor_confidence: float = DEFAULT_ANCHOR_CONFIDENCE,
    min_landmarks: int = MIN_LANDMARKS,
) -> Optional[ViewTransformer]:
    """Homography from image keypoints to court vertices, or None if too few."""
    config = config or default_court_config()
    mask = landmarks_mask(key_points, anchor_confidence)
    if np.count_nonzero(mask) < min_landmarks:
        return None
    return ViewTransformer(
        source=key_points.xy[0][mask],
        target=np.asarray(config.vertices)[mask],
    )


def detections_to_court_xy(
    key_points: sv.KeyPoints,
    detections: sv.Detections,
    config: CourtConfiguration | None = None,
    anchor_confidence: float = DEFAULT_ANCHOR_CONFIDENCE,
    min_landmarks: int = MIN_LANDMARKS,
) -> np.ndarray:
    """Map detection feet to court coordinates.

    Returns an ``(N, 2)`` array. Rows are NaN when homography cannot be
    estimated or there are no detections.
    """
    court_xy = np.full((len(detections), 2), np.nan, dtype=float)
    if len(detections) == 0:
        return court_xy

    transformer = court_transformer(
        key_points,
        config=config,
        anchor_confidence=anchor_confidence,
        min_landmarks=min_landmarks,
    )
    if transformer is None:
        return court_xy

    frame_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    return transformer.transform_points(points=frame_xy)


def clean_court_paths(
    video_xy: np.ndarray,
    jump_sigma: float = CLEAN_PATH_JUMP_SIGMA,
    min_jump_dist: float = CLEAN_PATH_MIN_JUMP_DIST,
    max_jump_run: int = CLEAN_PATH_MAX_JUMP_RUN,
    pad_around_runs: int = CLEAN_PATH_PAD_AROUND_RUNS,
    smooth_window: int = CLEAN_PATH_SMOOTH_WINDOW,
    smooth_poly: int = CLEAN_PATH_SMOOTH_POLY,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth mapped court trajectories with ``sports.clean_paths``.

    ``video_xy`` is ``(frames, players, 2)``. Returns ``(cleaned_xy, edited_mask)``.
    """
    return clean_paths(
        video_xy,
        jump_sigma=jump_sigma,
        min_jump_dist=min_jump_dist,
        max_jump_run=max_jump_run,
        pad_around_runs=pad_around_runs,
        smooth_window=smooth_window,
        smooth_poly=smooth_poly,
    )


class CourtKeypointDetector:
    """Load the court-keypoint model and run inference on frames."""

    def __init__(
        self,
        model_id: str = DEFAULT_KEYPOINT_MODEL_ID,
        confidence: float = DEFAULT_KEYPOINT_CONFIDENCE,
        anchor_confidence: float = DEFAULT_ANCHOR_CONFIDENCE,
        config: CourtConfiguration | None = None,
    ) -> None:
        self.model_id = model_id
        self.confidence = confidence
        self.anchor_confidence = anchor_confidence
        self.config = config or default_court_config()
        self.model = get_model(model_id=model_id)

    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        """Detect all court keypoints (unfiltered)."""
        result = self.model.infer(frame, confidence=self.confidence)[0]
        return sv.KeyPoints.from_inference(result)

    def filter_confident(self, key_points: sv.KeyPoints) -> sv.KeyPoints:
        """Keep keypoints above ``anchor_confidence`` (for visualization)."""
        mask = landmarks_mask(key_points, self.anchor_confidence)
        if mask.size == 0:
            return key_points
        return key_points[:, mask]

    def map_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> np.ndarray:
        """Detect court landmarks and map player boxes to court xy."""
        key_points = self.detect(frame)
        return detections_to_court_xy(
            key_points,
            detections,
            config=self.config,
            anchor_confidence=self.anchor_confidence,
        )
