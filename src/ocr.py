"""Jersey-number OCR brick.

Notebook cells reused: 37 (model id + prompt), 38 (helpers), 41/48 (per-frame
number flow: number class 2, pad 10 + clip, mask IoS match at 0.9,
ConsecutiveValueTracker(n_consecutive=3)).

DISABLED by default (ocr.enabled: false). When disabled the recognition model is
NOT loaded at all (no VRAM used) and all methods are no-ops.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import supervision as sv
from sports import ConsecutiveValueTracker

from .utils import setup_env

setup_env()

from inference import get_model  # noqa: E402  (import after env setup)


def coords_above_threshold(matrix: np.ndarray, threshold: float, sort_desc: bool = True):
    """Return (row, col) pairs where matrix > threshold, optionally by value desc."""
    a = np.asarray(matrix)
    rows, cols = np.where(a > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    if sort_desc:
        pairs.sort(key=lambda rc: a[rc[0], rc[1]], reverse=True)
    return pairs


class OCRBrick:
    def __init__(self, cfg: dict) -> None:
        ocfg = cfg["ocr"]
        self.enabled = bool(ocfg.get("enabled", False))
        self.model_id = ocfg["model_id"]
        self.prompt = ocfg.get("prompt", "Read the number.")
        self.number_class_id = int(ocfg.get("number_class_id", 2))
        self.match_threshold = float(ocfg.get("match_threshold", 0.9))
        self.pad_px = int(ocfg.get("pad_px", 10))
        self.n_consecutive = int(ocfg.get("n_consecutive", 3))
        self.stride_frames = int(ocfg.get("stride_frames", 5))
        self.min_number_size_px = float(ocfg.get("min_number_size_px", 12))
        self.verbose = bool(ocfg.get("verbose", False))

        teams_cfg = cfg["teams"]
        self.team_names = {int(k): v for k, v in (teams_cfg.get("team_names") or {}).items()}
        self.rosters = {
            team: {str(num): name for num, name in (roster or {}).items()}
            for team, roster in (teams_cfg.get("rosters") or {}).items()
        }

        self.model = None
        self.validator: Optional[ConsecutiveValueTracker] = None
        if self.enabled:
            if self.verbose:
                print(f"[OCR] loading model {self.model_id} ...")
            self.model = get_model(model_id=self.model_id)
            self.validator = ConsecutiveValueTracker(n_consecutive=self.n_consecutive)
            if self.verbose:
                print(f"[OCR] model ready; n_consecutive={self.n_consecutive} "
                      f"match_threshold={self.match_threshold} min_size={self.min_number_size_px}")
        self.validated_numbers: dict[int, str] = {}

    # ------------------------------------------------------------------ control
    def should_run(self, index: int) -> bool:
        return self.enabled and (index % self.stride_frames == 0)

    def reset(self) -> None:
        """Reset per-clip OCR state (validator + validated numbers)."""
        if self.validator is not None:
            self.validator.reset_all()
        self.validated_numbers = {}

    # --------------------------------------------------------------------- main
    def update(
        self,
        frame: np.ndarray,
        number_detections: sv.Detections,
        player_detections: sv.Detections,
    ) -> dict:
        """Read numbers and match to SAM2 tracks. Returns {tracker_id: number}.

        `number_detections` are the cleaned class-2 detections; `player_detections`
        are SAM2-tracked players (must have `.mask` and `.tracker_id`).
        """
        if not self.enabled:
            return {}
        if number_detections is None or len(number_detections) == 0:
            return {}
        if (
            player_detections is None
            or len(player_detections) == 0
            or player_detections.mask is None
            or player_detections.tracker_id is None
        ):
            return {}

        height, width = frame.shape[:2]

        # Safeguard: drop number boxes that are too small (bad jersey angle).
        keep = np.ones(len(number_detections), dtype=bool)
        for i, box in enumerate(number_detections.xyxy):
            bw = float(box[2] - box[0])
            bh = float(box[3] - box[1])
            if min(bw, bh) < self.min_number_size_px:
                keep[i] = False
        n_total = len(number_detections)
        number_detections = number_detections[keep]
        if self.verbose:
            print(f"[OCR] number boxes: {n_total} detected, {len(number_detections)} kept "
                  f"(min_size={self.min_number_size_px})")
        if len(number_detections) == 0:
            return {}

        number_detections.mask = sv.xyxy_to_mask(
            boxes=number_detections.xyxy, resolution_wh=(width, height)
        )
        boxes = sv.clip_boxes(
            sv.pad_boxes(xyxy=number_detections.xyxy, px=self.pad_px, py=self.pad_px),
            (width, height),
        )
        crops = [sv.crop_image(frame, box) for box in boxes]
        numbers = [
            self.model.infer(crop, prompt=self.prompt)[0].response.strip()
            for crop in crops
        ]
        if self.verbose:
            print(f"[OCR] raw responses: {numbers}")

        iou = sv.mask_iou_batch(
            masks_true=player_detections.mask,
            masks_detection=number_detections.mask,
            overlap_metric=sv.OverlapMetric.IOS,
        )
        pairs = coords_above_threshold(iou, self.match_threshold)
        if self.verbose:
            print(f"[OCR] IoS matches (>={self.match_threshold}): {len(pairs)}")
        if pairs:
            player_idx, number_idx = zip(*pairs)
            tracker_ids = np.asarray(player_detections.tracker_id)[list(player_idx)]
            values = [numbers[int(i)] for i in number_idx]
            if self.verbose:
                print(f"[OCR] matches: {[(int(t), v) for t, v in zip(tracker_ids, values)]}")
            self.validator.update(tracker_ids=tracker_ids, values=values)
            self.validated_numbers = self.validator.validated_dict()

        return dict(self.validated_numbers)

    # ------------------------------------------------------------------ output
    def resolve_name(self, team_id: Optional[int], number: Optional[str]) -> Optional[str]:
        if team_id is None or number is None:
            return None
        team_name = self.team_names.get(int(team_id))
        return self.rosters.get(team_name, {}).get(str(number))
