"""Team classification brick: SigLIP embeddings + KMeans (k=2) via `sports`.

Notebook cells reused: 32/62 (crops: sv.scale_boxes(factor=0.4) + sv.crop_image),
35 (TeamClassifier.fit), 44/62 (predict), 48 (ConsecutiveValueTracker n_consecutive=1).

Uses `sports.TeamClassifier(device=cfg.teams.device, batch_size=...)`, which runs
on CPU per the hardware constraints. In `bootstrap_online` mode it buffers crops
from the first frames, fits once when a trigger fires, then only predicts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import supervision as sv
from sports import ConsecutiveValueTracker, TeamClassifier


class TeamBrick:
    def __init__(self, cfg: dict) -> None:
        tcfg = cfg["teams"]
        self.device = tcfg.get("device", "cpu")
        self.batch_size = int(tcfg.get("batch_size", 32))
        self.fit_mode = tcfg.get("fit_mode", "bootstrap_online")
        self.min_crops = int(tcfg.get("min_crops", 300))
        self.max_bootstrap_frames = int(tcfg.get("max_bootstrap_frames", 60))
        self.scale_factor = float(tcfg.get("crop_scale_factor", 0.4))

        self.team_names = {int(k): v for k, v in (tcfg.get("team_names") or {}).items()}
        self.team_colors = tcfg.get("team_colors") or {}

        self.classifier = TeamClassifier(device=self.device, batch_size=self.batch_size)
        self.track_validator = ConsecutiveValueTracker(n_consecutive=1)

        self.fitted = False
        self.fit_frame: Optional[int] = None
        self._buffer: list[np.ndarray] = []
        self._buffer_frames = 0
        # Optional pre-collected crops (offline_stride mode).
        self._offline_crops: list[np.ndarray] = []

    # ------------------------------------------------------------------ crops
    def crops_from_detections(self, frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
        if detections is None or len(detections) == 0:
            return []
        boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=self.scale_factor)
        return [sv.crop_image(frame, box) for box in boxes]

    # --------------------------------------------------------------- training
    def collect_offline(self, crops: list[np.ndarray]) -> None:
        """Offline_stride mode: accumulate crops from a pre-pass, then fit once."""
        self._offline_crops.extend(crops)

    def fit(self, index: int | None = None) -> bool:
        crops = self._offline_crops if self.fit_mode == "offline_stride" else self._buffer
        if len(crops) < 2:
            return False
        self.classifier.fit(crops)
        self.fitted = True
        self.fit_frame = index
        self._buffer = []
        self._offline_crops = []
        return True

    # ------------------------------------------------------------------- main
    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        tracker_ids: Optional[np.ndarray] = None,
        index: int | None = None,
    ) -> Optional[np.ndarray]:
        """Return per-detection team ids once fitted, else None (still bootstrapping)."""
        crops = self.crops_from_detections(frame, detections)
        if not crops:
            return None if not self.fitted else np.empty(0, dtype=int)

        if not self.fitted:
            self._buffer.extend(crops)
            self._buffer_frames += 1
            if len(self._buffer) >= self.min_crops or self._buffer_frames >= self.max_bootstrap_frames:
                self.fit(index)
            return None

        teams = np.asarray(self.classifier.predict(crops)).astype(int)
        if tracker_ids is not None and len(tracker_ids) == len(teams):
            self.track_validator.update(tracker_ids=np.asarray(tracker_ids), values=teams)
            teams = np.asarray(self.track_validator.get_validated(np.asarray(tracker_ids))).astype(int)
        return teams

    # --------------------------------------------------------------- helpers
    def team_name(self, team_id: int) -> str:
        return self.team_names.get(int(team_id), f"team{team_id}")

    def team_color_hex(self, team_id: int) -> str:
        return self.team_colors.get(self.team_name(team_id), "#ffffff")
