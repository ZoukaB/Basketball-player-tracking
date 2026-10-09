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
        self._track_team: dict[int, int] = {}

        self.fitted = False
        self.fit_frame: Optional[int] = None
        self._buffer: list[np.ndarray] = []
        self._buffer_frames = 0
        # Optional pre-collected crops (offline_stride mode).
        self._offline_crops: list[np.ndarray] = []

    # ------------------------------------------------------------------ crops
    def crops_with_indices(
        self, frame: np.ndarray, detections: sv.Detections, min_size: int = 4
    ) -> tuple[list[np.ndarray], list[int]]:
        """Return (valid crops, detection indices). Skips degenerate/empty boxes."""
        if detections is None or len(detections) == 0:
            return [], []
        boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=self.scale_factor)
        h, w = frame.shape[:2]
        crops: list[np.ndarray] = []
        indices: list[int] = []
        for i, box in enumerate(boxes):
            x1 = max(0, int(round(float(box[0]))))
            y1 = max(0, int(round(float(box[1]))))
            x2 = min(w, int(round(float(box[2]))))
            y2 = min(h, int(round(float(box[3]))))
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            indices.append(i)
        return crops, indices

    def crops_from_detections(self, frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
        return self.crops_with_indices(frame, detections)[0]

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

    def fit_offline_from_videos(
        self,
        detector,
        video_paths,
        stride: int,
        player_class_ids,
        class_agnostic_nms: bool = True,
        max_seconds: float = 0.0,
    ) -> int:
        """Notebook-style offline fit: sample crops across videos (stride) and fit once."""
        player_class_ids = list(player_class_ids)
        crops: list[np.ndarray] = []
        for path in video_paths:
            info = sv.VideoInfo.from_video_path(str(path))
            max_frames = int(round(max_seconds * info.fps / stride)) if max_seconds else 0
            generator = sv.get_video_frames_generator(source_path=str(path), stride=stride)
            for i, frame in enumerate(generator):
                if max_frames and i >= max_frames:
                    break
                detections = detector.detect_raw(frame, class_agnostic_nms=class_agnostic_nms)
                players = detections[np.isin(detections.class_id, player_class_ids)]
                crops_frame, _ = self.crops_with_indices(frame, players)
                crops.extend(crops_frame)
        if len(crops) >= 2:
            self.classifier.fit(crops)
            self.fitted = True
            self.fit_frame = -1
        return len(crops)

    # ------------------------------------------------------------------- main
    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        tracker_ids: Optional[np.ndarray] = None,
        index: int | None = None,
    ) -> Optional[np.ndarray]:
        """Return per-detection team ids once fitted, else None (still bootstrapping).

        Unknown/unmatched detections get -1 (e.g. crop too small to classify).
        """
        n = len(detections) if detections is not None else 0
        crops, idx = self.crops_with_indices(frame, detections)
        if not crops:
            return None if not self.fitted else np.full(n, -1, dtype=int)

        if not self.fitted:
            self._buffer.extend(crops)
            self._buffer_frames += 1
            if len(self._buffer) >= self.min_crops or self._buffer_frames >= self.max_bootstrap_frames:
                self.fit(index)
            return None

        if tracker_ids is not None and len(np.asarray(tracker_ids)) == n:
            tids = np.asarray(tracker_ids)
            teams_full = np.full(n, -1, dtype=int)
            # Predict only for tracks we have not classified yet, then cache.
            unknown = [p for p, i in enumerate(idx) if int(tids[i]) not in self._track_team]
            if unknown:
                preds = np.asarray(self.classifier.predict([crops[p] for p in unknown])).astype(int)
                for p, pred in zip(unknown, preds):
                    self._track_team[int(tids[idx[p]])] = int(pred)
            for p, i in enumerate(idx):
                teams_full[i] = self._track_team[int(tids[i])]
            return teams_full

        preds = np.asarray(self.classifier.predict(crops)).astype(int)
        teams_full = np.full(n, -1, dtype=int)
        teams_full[idx] = preds
        return teams_full

    # --------------------------------------------------------------- helpers
    def team_name(self, team_id: int) -> str:
        return self.team_names.get(int(team_id), f"team{team_id}")

    def team_color_hex(self, team_id: int) -> str:
        return self.team_colors.get(self.team_name(team_id), "#ffffff")
