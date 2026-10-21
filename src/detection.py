"""Object detection brick: RF-DETR basketball player/ball/rim detector.

Notebook cells reused: 25 (model load), 32/33 (seed prompt), 48/62/91 (per-frame).
Model ID: basketball-player-detection-3-ycjdo/4
Params: confidence=0.4, iou_threshold=0.9
Class map: 0 ball, 1 ball-in-basket, 2 number, 3 player, 4 player-in-possession,
           5 player-jump-shot, 6 player-layup-dunk, 7 player-shot-block, 8 referee, 9 rim

Cleaning (config `detection.class_floor`): per-class confidence + geometry floors
plus a border-sliver rule, so degenerate RF-DETR boxes (e.g. players exiting the
frame) don't reach the tracker.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import supervision as sv

from .utils import setup_env

setup_env()

from inference import get_model  # noqa: E402  (import after env setup)

_EPS = 1e-6


class Detector:
    def __init__(self, cfg: dict) -> None:
        det_cfg = cfg["detection"]
        self.model_id = det_cfg["model_id"]
        self.confidence = float(det_cfg.get("confidence", 0.4))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.9))
        self.class_agnostic_nms = bool(det_cfg.get("class_agnostic_nms", False))

        classes_cfg = cfg["classes"]
        self.player_class_ids = list(classes_cfg["player_ids"])
        self.id_to_name = {
            int(cid): name
            for name, cid in classes_cfg.items()
            if name != "player_ids" and isinstance(cid, int)
        }

        floor_cfg = det_cfg.get("class_floor", {}) or {}
        self.default_floor = dict(floor_cfg.get("default", {}))
        self.filters = self._build_filters(floor_cfg)

        self.edge_margin = int(det_cfg.get("edge_margin_px", 0))
        self.edge_min_width = float(det_cfg.get("edge_min_width", 0))

        nms_cfg = det_cfg.get("nms", {}) or {}
        self.nms_enabled = bool(nms_cfg.get("enabled", False))
        self.nms_iou = float(nms_cfg.get("iou_threshold", 0.6))
        self.nms_class_agnostic = bool(nms_cfg.get("class_agnostic", True))

        ref_cfg = det_cfg.get("referee_suppression", {}) or {}
        self.referee_suppression_enabled = bool(ref_cfg.get("enabled", False))
        self.referee_iou_threshold = float(ref_cfg.get("iou_threshold", 0.5))
        self.referee_suppress_classes = list(ref_cfg.get("classes", [3]))
        self.referee_class_id = next(
            (cid for cid, name in self.id_to_name.items() if name == "referee"), None
        )

        self.model = get_model(model_id=self.model_id)

    # ------------------------------------------------------------------ setup
    def _build_filters(self, floor_cfg: dict) -> dict:
        filters: dict[int, dict] = {}
        for cid, name in self.id_to_name.items():
            spec = floor_cfg.get(name)
            filters[cid] = {**self.default_floor, **(spec or {})}
        return filters

    # --------------------------------------------------------------- inference
    def detect_raw(self, frame: np.ndarray, class_agnostic_nms: bool | None = None) -> sv.Detections:
        agnostic = self.class_agnostic_nms if class_agnostic_nms is None else bool(class_agnostic_nms)
        result = self.model.infer(
            frame,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold,
            class_agnostic_nms=agnostic,
        )[0]
        return sv.Detections.from_inference(result)

    # --------------------------------------------------------------- cleaning
    def clean(
        self, detections: sv.Detections, frame_shape: tuple
    ) -> tuple[sv.Detections, dict]:
        """Apply per-class floors; return (cleaned, {reason: dropped_count})."""
        if len(detections) == 0:
            return detections, {}

        frame_h, frame_w = frame_shape[:2]
        keep = np.ones(len(detections), dtype=bool)
        stats: Counter = Counter()

        for i in range(len(detections)):
            cid = int(detections.class_id[i])
            conf = float(detections.confidence[i])
            x1, y1, x2, y2 = (float(v) for v in detections.xyxy[i])
            width, height = x2 - x1, y2 - y1
            area = width * height
            aspect = max(width / (height + _EPS), height / (width + _EPS))

            floor = self.filters.get(cid, self.default_floor)
            reason = None
            if conf < floor.get("min_confidence", 0.0):
                reason = "confidence"
            elif width < floor.get("min_width", 0.0):
                reason = "width"
            elif height < floor.get("min_height", 0.0):
                reason = "height"
            elif area < floor.get("min_area", 0.0):
                reason = "area"
            elif aspect > floor.get("max_aspect", float("inf")):
                reason = "aspect"
            elif (
                self.edge_min_width > 0
                and width < self.edge_min_width
                and (x1 <= self.edge_margin or x2 >= frame_w - self.edge_margin)
            ):
                reason = "edge"

            if reason is not None:
                keep[i] = False
                stats[reason] += 1

        cleaned = detections[keep]
        if self.referee_suppression_enabled and len(cleaned) > 0:
            cleaned, n_ref = self._suppress_referees(cleaned)
            if n_ref:
                stats["referee"] = stats.get("referee", 0) + n_ref
        if self.nms_enabled and len(cleaned) > 0:
            cleaned = cleaned.with_nms(
                threshold=self.nms_iou, class_agnostic=self.nms_class_agnostic
            )
        return cleaned, dict(stats)

    def _suppress_referees(self, detections: sv.Detections) -> tuple[sv.Detections, int]:
        """Drop player-family boxes that overlap a referee box (RF-DETR double-labels)."""
        if self.referee_class_id is None:
            return detections, 0
        ref_mask = detections.class_id == self.referee_class_id
        if not ref_mask.any():
            return detections, 0

        ref_boxes = detections.xyxy[ref_mask]
        keep = np.ones(len(detections), dtype=bool)
        dropped = 0
        for i in range(len(detections)):
            if int(detections.class_id[i]) not in self.referee_suppress_classes:
                continue
            ious = sv.box_iou_batch(detections.xyxy[i : i + 1], ref_boxes).reshape(-1)
            if ious.size and float(ious.max()) >= self.referee_iou_threshold:
                keep[i] = False
                dropped += 1
        return detections[keep], dropped

    # ------------------------------------------------------------------ public
    def detect(self, frame: np.ndarray) -> sv.Detections:
        cleaned, _ = self.clean(self.detect_raw(frame), frame.shape)
        return cleaned

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detect(frame)
        return detections[np.isin(detections.class_id, self.player_class_ids)]
