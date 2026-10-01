"""Tracking brick: SAM2 real-time mask propagation seeded by RF-DETR detections.

Notebook cells reused: 28 (fork import), 29 (SAM2Tracker), 33 (seed + propagate).

The real-time fork lives outside this repo (see `sam2.repo_dir`) and provides
`build_sam2_camera_predictor`, which the installed `sam2` package lacks. We load
it explicitly and purge any previously imported `sam2` modules.

We build the predictor with `apply_postprocessing=False` so that
`model.fill_hole_area` stays 0: the tracking path then never calls the compiled
`_C` extension (`get_connected_componnets`), which is not built locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv
import torch

from .utils import REPO_ROOT


def _load_build_fn(cfg: dict):
    """Put the SAM2 fork on sys.path and return its predictor builder."""
    repo_dir = (REPO_ROOT / cfg["sam2"]["repo_dir"]).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"SAM2 fork repo not found: {repo_dir}")

    repo_str = str(repo_dir)
    if sys.path[0] != repo_str:
        if repo_str in sys.path:
            sys.path.remove(repo_str)
        sys.path.insert(0, repo_str)

    # Drop any `sam2` already imported from site-packages so the fork wins.
    for name in [m for m in sys.modules if m == "sam2" or m.startswith("sam2.")]:
        del sys.modules[name]

    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    from sam2.build_sam import build_sam2_camera_predictor

    return build_sam2_camera_predictor, repo_dir


class SAM2Tracker:
    def __init__(self, cfg: dict) -> None:
        sam2_cfg = cfg["sam2"]
        self.seed_class_ids = np.asarray(sam2_cfg.get("seed_class_ids", [3]), dtype=int)
        mask_filter = sam2_cfg.get("mask_filter", {})
        self.mask_relative_distance = float(mask_filter.get("relative_distance", 0.03))
        self.mask_mode = mask_filter.get("mode", "edge")

        build_fn, repo_dir = _load_build_fn(cfg)
        checkpoint = str((repo_dir / sam2_cfg["checkpoint"]).resolve())
        self.predictor = build_fn(
            sam2_cfg["config"],
            checkpoint,
            apply_postprocessing=bool(sam2_cfg.get("apply_postprocessing", False)),
        )
        self._prompted = False

    # ------------------------------------------------------------- prompting
    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> bool:
        """Seed the tracker from a frame. Returns False if there is nothing to track."""
        if detections is None or len(detections) == 0:
            self._prompted = False
            return False

        if detections.tracker_id is None:
            detections.tracker_id = np.arange(1, len(detections) + 1)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
                bbox = np.asarray([xyxy], dtype=np.float32)
                self.predictor.add_new_prompt(
                    frame_idx=0, obj_id=int(obj_id), bbox=bbox
                )

        self._prompted = True
        return True

    def reprompt(self, frame: np.ndarray, detections: sv.Detections) -> bool:
        """Clear state and re-seed from the current frame (used after a hard cut)."""
        try:
            self.predictor.reset_state()
        except Exception:
            pass
        self._prompted = False
        return self.prompt_first_frame(frame, detections)

    # ------------------------------------------------------------ propagation
    def propagate(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Tracker not prompted: call prompt_first_frame first")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        if tracker_ids.size == 0:
            return sv.Detections.empty()

        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]

        masks = np.array(
            [
                sv.filter_segments_by_distance(
                    mask, relative_distance=self.mask_relative_distance, mode=self.mask_mode
                )
                for mask in masks
            ]
        )

        xyxy = sv.mask_to_xyxy(masks=masks)
        return sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)

    def reset(self) -> None:
        self._prompted = False
