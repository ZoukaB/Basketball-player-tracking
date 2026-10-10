"""Team classification brick: SigLIP embeddings + KMeans (k=2) via `sports`.

Notebook cells reused: 32/62 (crops: sv.scale_boxes(factor=0.4) + sv.crop_image),
35 (TeamClassifier.fit), 44/62 (predict), 48 (ConsecutiveValueTracker n_consecutive=1).

Uses `sports.TeamClassifier(device=cfg.teams.device, batch_size=...)`, which runs
on CPU per the hardware constraints. In `bootstrap_online` mode it buffers crops
from the first frames, fits once when a trigger fires, then only predicts.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
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
        self.team_votes_target = int(tcfg.get("team_votes_target", 5))
        self._track_votes: dict[int, dict[int, int]] = {}
        self._track_team: dict[int, int] = {}

        self.fitted = False
        self.fit_frame: Optional[int] = None
        self._buffer: list[np.ndarray] = []
        self._buffer_frames = 0
        # Optional pre-collected crops (offline_stride mode).
        self._offline_crops: list[np.ndarray] = []
        # Small retained subset used to preview clusters and suggest the mapping.
        self._probe_crops: list[np.ndarray] = []

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
        if crops:
            self._probe_crops = random.sample(crops, min(len(crops), 400))
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
            # Predict each track up to `team_votes_target` times, then lock the majority.
            need = [
                p for p, i in enumerate(idx)
                if sum(self._track_votes.get(int(tids[i]), {}).values()) < self.team_votes_target
            ]
            if need:
                preds = np.asarray(self.classifier.predict([crops[p] for p in need])).astype(int)
                for p, pred in zip(need, preds):
                    votes = self._track_votes.setdefault(int(tids[idx[p]]), {})
                    votes[int(pred)] = votes.get(int(pred), 0) + 1
            for p, i in enumerate(idx):
                votes = self._track_votes.get(int(tids[i]))
                if votes:
                    team = max(votes, key=votes.get)
                    self._track_team[int(tids[i])] = int(team)
                    teams_full[i] = int(team)
            return teams_full

        preds = np.asarray(self.classifier.predict(crops)).astype(int)
        teams_full = np.full(n, -1, dtype=int)
        teams_full[idx] = preds
        return teams_full

    # --------------------------------------------------------------- helpers
    def team_of(self, tracker_id: int) -> Optional[int]:
        """Majority team for a SAM2 track id, or None if unseen."""
        return self._track_team.get(int(tracker_id))

    def reset_tracks(self) -> None:
        """Clear per-clip track team/vote state (call between clips)."""
        self._track_votes.clear()
        self._track_team.clear()

    # ---------------------------------------------------- cluster validation
    def _cluster_labels(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.array([], dtype=int)
        return np.asarray(self.classifier.predict(crops)).astype(int)

    def cluster_sizes(self) -> dict[int, int]:
        labels = self._cluster_labels(self._probe_crops)
        return {int(c): int((labels == c).sum()) for c in sorted(set(labels.tolist()))}

    def cluster_montage(self, per_cluster: int = 12, path=None,
                        crop_w: int = 72, crop_h: int = 144) -> np.ndarray:
        """One row of sample crops per cluster; saves to `path` when given."""
        labels = self._cluster_labels(self._probe_crops)
        clusters = sorted(set(labels.tolist())) if labels.size else []
        rows = []
        for cid in clusters:
            idxs = np.where(labels == cid)[0][:per_cluster]
            cells = [cv2.resize(self._probe_crops[i], (crop_w, crop_h)) for i in idxs]
            while len(cells) < per_cluster:
                cells.append(np.zeros((crop_h, crop_w, 3), np.uint8))
            rows.append(np.hstack(cells))
        image = np.vstack(rows) if rows else np.zeros((crop_h, crop_w * per_cluster, 3), np.uint8)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
        return image

    def cluster_mean_color(self) -> dict[int, tuple[float, float, float]]:
        """Mean central-region colour (BGR) per cluster, from probe crops."""
        labels = self._cluster_labels(self._probe_crops)
        out: dict[int, tuple[float, float, float]] = {}
        for cid in (sorted(set(labels.tolist())) if labels.size else []):
            means = []
            for i in np.where(labels == cid)[0]:
                crop = self._probe_crops[i]
                h, w = crop.shape[:2]
                centre = crop[h // 4:max(h // 4 + 1, 3 * h // 4), w // 4:max(w // 4 + 1, 3 * w // 4)]
                if centre.size:
                    means.append(centre.reshape(-1, 3).mean(axis=0))
            if means:
                out[int(cid)] = tuple(float(v) for v in np.mean(means, axis=0))
        return out

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> np.ndarray:
        h = hex_color.lstrip("#")
        return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=float)

    def suggest_mapping(self) -> dict[int, str]:
        """Suggest cluster -> team name by nearest jersey colour (2-team assignment)."""
        clusters = sorted(self.team_names.keys())
        colours = self.cluster_mean_color()
        if len(clusters) != 2 or len(colours) != 2:
            return dict(self.team_names)
        names = [self.team_names[c] for c in clusters]
        refs = [self._hex_to_rgb(self.team_colors.get(n, "#ffffff")) for n in names]
        means = {c: np.array(colours[c][::-1], dtype=float) for c in clusters}  # BGR -> RGB
        d = np.array([[np.linalg.norm(means[c] - refs[j]) for j in range(2)] for c in clusters])
        if d[0, 0] + d[1, 1] <= d[0, 1] + d[1, 0]:
            return {clusters[0]: names[0], clusters[1]: names[1]}
        return {clusters[0]: names[1], clusters[1]: names[0]}

    def set_mapping(self, mapping: dict) -> None:
        self.team_names = {int(k): v for k, v in mapping.items()}

    def mapping_summary(self) -> dict[int, str]:
        return dict(self.team_names)

    def team_name(self, team_id: int) -> str:
        return self.team_names.get(int(team_id), f"team{team_id}")

    def team_color_hex(self, team_id: int) -> str:
        return self.team_colors.get(self.team_name(team_id), "#ffffff")


def resolve_team_mapping(
    teams: "TeamBrick",
    montage_path: str | Path | None,
    mapping_path: str | Path | None,
    ask: bool,
    auto_color: bool,
    log=print,
) -> dict:
    """Show cluster samples, suggest/validate the cluster->team mapping, persist it."""
    if not teams.fitted or not teams._probe_crops:
        return teams.mapping_summary()

    teams.cluster_montage(path=montage_path)
    sizes = teams.cluster_sizes()
    suggest = teams.suggest_mapping()
    log(f"Team cluster samples -> {montage_path}")
    log(f"  cluster sizes: {sizes}")
    log(f"  suggested mapping: {suggest}")

    mapping = None
    mp = Path(mapping_path) if mapping_path else None
    if mp and mp.exists():
        try:
            mapping = {int(k): v for k, v in json.loads(mp.read_text(encoding="utf-8")).items()}
            log(f"  loaded mapping from {mp}: {mapping}")
        except Exception:
            mapping = None
    if mapping is None:
        mapping = dict(suggest)

    if ask and sys.stdin is not None and sys.stdin.isatty():
        for cid in sorted(mapping.keys()):
            default = mapping.get(cid, "")
            response = input(f"  Team for cluster {cid} [{default}]: ").strip()
            if response:
                mapping[cid] = response
    elif auto_color:
        mapping = dict(suggest)

    teams.set_mapping(mapping)
    if mp:
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2), encoding="utf-8")
    log(f"  applied mapping: {teams.mapping_summary()}")
    return teams.mapping_summary()
