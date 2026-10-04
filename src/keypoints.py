"""Court keypoints + homography brick.

Notebook cells reused: 66 (model load), 70 (confidence filtering), 73/75
(CourtConfiguration + ViewTransformer homography, BOTTOM_CENTER anchors).

Model: basketball-court-detection-2/14, confidence=0.3, anchor=0.5.
Court: CourtConfiguration(league=NBA, measurement_unit=FEET).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import supervision as sv

from .utils import setup_env

setup_env()

from inference import get_model  # noqa: E402  (import after env setup)
from sports import MeasurementUnit, ViewTransformer  # noqa: E402
from sports.basketball import (  # noqa: E402
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court,
)


class KeypointBrick:
    def __init__(self, cfg: dict) -> None:
        kcfg = cfg["keypoints"]
        self.model_id = kcfg["model_id"]
        self.confidence = float(kcfg.get("confidence", 0.3))
        self.anchor_confidence = float(kcfg.get("anchor_confidence", 0.5))

        self.config = CourtConfiguration(
            league=getattr(League, kcfg.get("league", "NBA")),
            measurement_unit=getattr(MeasurementUnit, kcfg.get("measurement_unit", "FEET")),
        )
        self.model = get_model(model_id=self.model_id)

    # --------------------------------------------------------------- detection
    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        result = self.model.infer(frame, confidence=self.confidence)[0]
        return sv.KeyPoints.from_inference(result)

    def landmarks(self, frame: np.ndarray) -> tuple[sv.KeyPoints, np.ndarray]:
        """Return (keypoints, boolean mask of anchors above confidence)."""
        key_points = self.detect(frame)
        # supervision renamed keypoint_confidence -> confidence across versions.
        confidence = getattr(key_points, "keypoint_confidence", None)
        if confidence is None:
            confidence = key_points.confidence
        mask = np.asarray(confidence[0]) > self.anchor_confidence
        return key_points, mask

    def transformer_from(
        self, key_points: sv.KeyPoints, mask: np.ndarray
    ) -> Optional[ViewTransformer]:
        """Build frame->court homography from already-detected keypoints; None if <4 anchors."""
        if int(np.count_nonzero(mask)) < 4:
            return None
        court_landmarks = np.array(self.config.vertices)[mask]
        frame_landmarks = key_points[:, mask].xy[0]
        return ViewTransformer(source=frame_landmarks, target=court_landmarks)

    def get_transformer(self, frame: np.ndarray) -> Optional[ViewTransformer]:
        """Build frame->court homography; None if fewer than 4 anchors."""
        key_points, mask = self.landmarks(frame)
        return self.transformer_from(key_points, mask)

    def matrix_to_frame(self, key_points: sv.KeyPoints, mask: np.ndarray) -> Optional[np.ndarray]:
        """Return a 3x3 court(feet)->frame(pixels) homography; None if <4 anchors.

        Used to warp court lines onto the video frame for visual verification.
        """
        if int(np.count_nonzero(mask)) < 4:
            return None
        court_landmarks = np.array(self.config.vertices)[mask].astype(np.float32)
        frame_landmarks = key_points[:, mask].xy[0].astype(np.float32)
        matrix, _ = cv2.findHomography(court_landmarks, frame_landmarks)
        return matrix

    # --------------------------------------------------------------- projection
    @staticmethod
    def transform_points(transformer: ViewTransformer, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(transformer.transform_points(points=np.asarray(points)))

    def draw_court_points(self, court_xy: np.ndarray, colors=None) -> np.ndarray:
        """Return an RGB court image with projected points drawn on it."""
        court = draw_court(config=self.config)
        if len(court_xy) > 0:
            court = draw_points_on_court(
                config=self.config,
                xy=np.asarray(court_xy),
                fill_color=sv.Color.from_hex("#FF1493") if colors is None else colors,
                court=court,
            )
        return court

    def draw_court_lines(self, scale: int = 20, padding: int = 50) -> np.ndarray:
        """Court lines (white) on black, used for warping onto the video frame."""
        return draw_court(
            config=self.config,
            scale=scale,
            padding=padding,
            background_color=sv.Color(0, 0, 0),
            paint_color=None,
        )
