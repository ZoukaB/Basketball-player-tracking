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

    # -------------------------------------------------- additive object growth
    def _next_obj_id(self) -> int:
        ids = [int(i) for i in self.predictor.condition_state.get("obj_ids", [])]
        return max(ids) + 1 if ids else 1

    def add_object(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """Add a new tracked object mid-stream WITHOUT resetting existing tracks.

        NOTE: currently UNUSED and NOT SAFE with this fork. It accepts the new
        obj_id (by caching the current frame's features and briefly clearing
        `tracking_has_started`), but the next `track()` crashes because past
        memory frames hold fewer object pointers:
            RuntimeError: stack expects each tensor to be equal size ...
        Supporting this requires padding historical memory frames. See
        notes/future_ideas.md ("Incremental object add"). Kept as the basis for a
        future fork patch.
        """
        if not self._prompted:
            raise RuntimeError("Tracker not prompted: call prompt_first_frame first")

        predictor = self.predictor
        frame_idx = int(predictor.frame_idx)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prepared, _, _ = predictor.perpare_data(frame, image_size=predictor.image_size)
            image = prepared.cuda().float().unsqueeze(0)
            backbone_out = predictor.forward_image(image)
            predictor.condition_state["cached_features"] = {
                frame_idx: (image, backbone_out)
            }
            predictor.condition_state["tracking_has_started"] = False
            obj_id = self._next_obj_id()
            predictor.add_new_prompt(
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                bbox=np.asarray([bbox], dtype=np.float32),
            )

        return obj_id

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


class RepromptPolicy:
    """Decide when to re-seed SAM2 as more players become visible.

    `grow_on_new_max`: re-prompt when the cleaned detected-player count reaches a
    NEW maximum above the count at the last prompt, sustained for
    `debounce_frames` and respecting `min_frames_between_reprompts` (and capped
    by `max_tracks`). This avoids thrash from merge/split flicker: a count that
    has already triggered (or that drops back) does not re-trigger.

    Other policies (`mismatch`, `full_scene`, `never`) are documented in
    notes/future_ideas.md; only `grow_on_new_max` is implemented here.
    """

    def __init__(self, cfg: dict) -> None:
        sam2_cfg = cfg["sam2"]
        self.policy = sam2_cfg.get("reprompt_policy", "never")
        self.debounce_frames = int(sam2_cfg.get("debounce_frames", 3))
        self.min_frames_between_reprompts = int(
            sam2_cfg.get("min_frames_between_reprompts", 15)
        )
        self.max_tracks = int(sam2_cfg.get("max_tracks", 12))
        self.reset_state(seeded_n=0, index=0)

    def reset_state(self, seeded_n: int, index: int) -> None:
        self.best_n = int(seeded_n)
        self.last_prompt_index = int(index)
        self._pending_value: int | None = None
        self._pending_streak = 0

    def should_reprompt(self, index: int, n_det: int) -> bool:
        """Pure decision (no state commit): has the count reached a debounced new max?"""
        if self.policy != "grow_on_new_max":
            return False

        n_det = int(n_det)
        if n_det <= self.best_n or n_det > self.max_tracks:
            self._pending_value = None
            self._pending_streak = 0
            return False

        if n_det == self._pending_value:
            self._pending_streak += 1
        else:
            self._pending_value = n_det
            self._pending_streak = 1

        return (
            self._pending_streak >= self.debounce_frames
            and (index - self.last_prompt_index) >= self.min_frames_between_reprompts
        )

    def commit(self, index: int, n_det: int) -> None:
        """Record that a re-prompt actually happened at this count."""
        self.best_n = int(n_det)
        self.last_prompt_index = int(index)
        self._pending_value = None
        self._pending_streak = 0


def tracks_all_matched(
    track_boxes: np.ndarray, det_boxes: np.ndarray, match_iou: float
) -> bool:
    """True if every track box overlaps some detection box with IoU >= match_iou.

    Used as a safety guard before a reset-based re-prompt: if any track is not
    re-detectable (e.g. occluded but still tracked well by SAM2), the reset is
    vetoed so that track is not lost.
    """
    track_boxes = np.asarray(track_boxes)
    det_boxes = np.asarray(det_boxes)
    if track_boxes.size == 0:
        return True
    if det_boxes.size == 0:
        return False
    ious = sv.box_iou_batch(track_boxes, det_boxes)
    return bool((ious.max(axis=1) >= match_iou).all())
