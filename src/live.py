"""Live shot session: the collect_shots loop, run frame by frame in a background thread.

Single pass, no pre-pass: teams are bootstrapped online from the first frames
(`teams.fit_mode: bootstrap_online`), then the cluster->team mapping is picked by
jersey colour. Each finalized shot becomes a row immediately; the dashboard polls
`snapshot()` for the latest annotated frame + rows.

No stable identity yet: each shot gets a random (number, name) from the team's
roster in config (`identity="random"`), to be replaced by OCR later. OCR and SAM2
are not loaded.

Source: a video file, a folder of clips (smallest first), a camera index or a URL.

Importing this module is cheap: the bricks (torch, inference, transformers,
sports) are imported in the worker thread, so the dashboard page shows up at
once and `preload()` warms the models in the background.
"""
from __future__ import annotations

import copy
import csv
import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

# Leave two cores to the desktop/editor: torch, numba (UMAP) and BLAS otherwise
# grab every core during imports and the team fit. Must be set before they import.
_THREADS = str(max(1, (os.cpu_count() or 4) - 2))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
# The sports TeamClassifier prints one tqdm bar per predict call (every frame).
os.environ.setdefault("TQDM_DISABLE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import VIDEO_EXTS, analysis_stride, list_videos, resolve_path, setup_env  # noqa: E402

# Half-court geometry (feet), hoop at (5.25, 25) like dashboard/lib/court.py.
FULL_LENGTH = 94.0
COURT_WIDTH = 50.0
HOOP = (5.25, 25.0)
THREE_RADIUS = 23.75
CORNER_THREE_Y = 22.0   # |y - 25| beyond this in the corner = 3pt
CORNER_THREE_X = 14.0   # corner zone depth from the baseline

SHOT_CLASS_IDS = [1, 5, 6]  # ball-in-basket, jump shot, layup/dunk
ROW_FIELDS = ["video", "frame", "team", "team_source", "player_id", "name", "identity",
              "shot_type", "result", "x_court", "y_court", "court_x", "court_y",
              "attacking_basket", "distance_ft"]


def to_half_court(x: float, y: float) -> tuple[float, float]:
    """Mirror right-half shots so every shot lands on the left half court.

    Uses the shot's own location, not `attacking_basket`: the latter comes from
    the possession box and can disagree (e.g. a layup at x=6 tagged "right").
    """
    if x > FULL_LENGTH / 2:
        return FULL_LENGTH - x, COURT_WIDTH - y
    return x, y


def classify_shot_type(event_type: str, half_x: Optional[float], half_y: Optional[float]) -> str:
    if event_type != "JUMP":
        return "layup"
    if half_x is None or half_y is None:
        return "2pt"
    corner = half_x <= CORNER_THREE_X and abs(half_y - HOOP[1]) >= CORNER_THREE_Y
    far = float(np.hypot(half_x - HOOP[0], half_y - HOOP[1])) >= THREE_RADIUS
    return "3pt" if corner or far else "2pt"


def list_sources(source: str) -> list[tuple[str, object]]:
    """(display name, cv2.VideoCapture argument) per clip."""
    source = str(source).strip()
    if source.isdigit():
        return [(f"camera {source}", int(source))]
    if "://" in source:
        return [(source, source)]
    path = resolve_path(source)
    if path.is_dir():
        return [(p.name, str(p)) for p in list_videos(path, None)]
    if path.suffix.lower() in VIDEO_EXTS and path.exists():
        return [(path.name, str(path))]
    raise FileNotFoundError(f"No video source at {source}")


class LiveShotSession:
    def __init__(self, cfg: dict, out_csv: str | Path | None = "outputs/live_shots.csv",
                 seed: int = 0, dedup_window_s: float = 0.5, dedup_dist_ft: float = 5.0,
                 frame_width: int = 960, log=print) -> None:
        self.cfg = copy.deepcopy(cfg)
        self.cfg["ocr"]["enabled"] = False
        self.out_csv = resolve_path(out_csv) if out_csv else None
        self.seed = seed
        self.dedup_window_s = dedup_window_s
        self.dedup_dist_ft = dedup_dist_ft
        self.frame_width = frame_width
        self.log = log

        self.rosters = self.cfg["teams"].get("rosters") or {}
        self.player_class_ids = list(self.cfg["classes"]["player_ids"])
        self.box_annotator = None

        self._detector = None
        self._keypoints = None
        self._teams = None

        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.models_ready = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._reset_state()

    # ------------------------------------------------------------------ control
    def _reset_state(self) -> None:
        self._rng = random.Random(self.seed)
        self._rows: list[dict] = []
        self._jpeg: Optional[bytes] = None
        self._banner: Optional[tuple[str, float]] = None
        self._status = {"state": "idle", "clip": "", "clip_index": 0, "n_clips": 0,
                        "frame": 0, "fps": 0.0, "teams_fitted": False, "mapping": {},
                        "error": ""}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def preload(self) -> None:
        """Import + load the detection/keypoint models in the background."""
        if self.models_ready or self.running:
            return
        self._set_status(state="loading models")
        threading.Thread(target=self._preload_safe, daemon=True).start()

    def _preload_safe(self) -> None:
        try:
            self._load_models()
            if not self.running:
                self._set_status(state="idle")
        except Exception as exc:
            self._set_status(state="error", error=f"{type(exc).__name__}: {exc}")
            raise

    def start(self, source: str) -> None:
        if self.running:
            return
        self._stop.clear()
        self._resume.set()
        with self._lock:
            self._reset_state()
            self._status["state"] = "starting"
        self._thread = threading.Thread(target=self._run_safe, args=(source,), daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._resume.clear()
        self._set_status(state="paused")

    def resume(self) -> None:
        self._resume.set()
        self._set_status(state="running")

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            return {"jpeg": self._jpeg, "rows": list(self._rows), "status": dict(self._status)}

    def _set_status(self, **kwargs) -> None:
        with self._lock:
            self._status.update(kwargs)

    # --------------------------------------------------------------------- loop
    def _load_models(self) -> None:
        # The lock makes Start wait for an in-flight preload instead of loading twice.
        with self._load_lock:
            if self.models_ready:
                return
            self._set_status(state="loading models")
            # Env (HF offline, ONNX providers) must be set before huggingface_hub /
            # inference import; supervision and torch would otherwise import it first.
            setup_env()
            import supervision as sv
            import torch
            from src.detection import Detector
            from src.keypoints import KeypointBrick

            torch.set_num_threads(int(_THREADS))

            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self._detector = Detector(self.cfg)
            self._keypoints = KeypointBrick(self.cfg)
            self._warm_up_team_fit()
            self.models_ready = True

    def _warm_up_team_fit(self) -> None:
        """Run the TeamClassifier's UMAP + KMeans once on random data.

        The first fit in a process JIT-compiles UMAP (numba/LLVM) and makes sklearn
        scan every loaded DLL for thread pools: minutes with CUDA/torch loaded, which
        froze the video at "fitting teams". Doing it here moves that cost to preload.
        Same calls/dtypes as sports.TeamClassifier.fit/predict (SigLIP: 768-d float32).
        """
        self._set_status(state="warming up team classifier")
        import umap
        from sklearn.cluster import KMeans

        rng = np.random.default_rng(0)
        data = rng.random((300, 768), dtype=np.float32)
        reducer = umap.UMAP(n_components=3)
        projections = reducer.fit_transform(data)
        reducer.transform(data[:8])
        KMeans(n_clusters=2).fit(projections).predict(projections[:8])

    def _load_bricks(self) -> None:
        self._load_models()  # first: runs setup_env before transformers/sports import
        from src.teams import TeamBrick

        # Teams are refit per session (KMeans cluster ids are not stable across fits).
        self._set_status(state="loading team classifier")
        self._teams = TeamBrick(self.cfg)

    def _run_safe(self, source: str) -> None:
        try:
            self.run(source)
        except Exception as exc:  # surface errors in the dashboard instead of a dead thread
            self._set_status(state="error", error=f"{type(exc).__name__}: {exc}")
            raise

    def run(self, source: str) -> None:
        """Process every clip of `source` synchronously (the thread target)."""
        clips = list_sources(source)
        self._load_bricks()
        self._set_status(state="running", n_clips=len(clips))
        for clip_index, (name, arg) in enumerate(clips):
            if self._stop.is_set():
                break
            self._set_status(clip=name, clip_index=clip_index + 1, frame=0)
            self._run_clip(name, arg)
        self._set_status(state="stopped" if self._stop.is_set() else "done")

    def _run_clip(self, name: str, arg) -> None:
        from src.shots import ShotBrick

        detector, keypoints, teams = self._detector, self._keypoints, self._teams
        cap = cv2.VideoCapture(arg)
        if not cap.isOpened():
            self.log(f"[live] cannot open {name}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = analysis_stride(self.cfg, fps)
        analysis_fps = fps / stride
        shots = ShotBrick(self.cfg, analysis_fps=analysis_fps)
        window_frames = max(1, int(round(analysis_fps * self.dedup_window_s)))
        kept: list[tuple[dict, int]] = []  # (shot, row index) for this clip's dedup

        transformer = None
        raw_index, index = 0, 0
        t_last = time.perf_counter()
        try:
            while not self._stop.is_set():
                self._resume.wait()
                ok, frame = cap.read()
                if not ok:
                    break
                raw_index += 1
                if (raw_index - 1) % stride:
                    continue

                detections = detector.detect(frame)
                if not teams.fitted:
                    players = detections[np.isin(detections.class_id, self.player_class_ids)]
                    self._set_status(state="fitting teams")
                    teams.update(frame, players, index=index)
                # ShotBrick may also trigger the fit through possession crops.
                if teams.fitted and not self._status["teams_fitted"]:
                    # Jersey-colour mapping only (one SigLIP pass over the probe crops);
                    # resolve_team_mapping would also run the montage + cluster sizes.
                    self._set_status(state="mapping teams")
                    teams.set_mapping(teams.suggest_mapping())
                    mapping = teams.mapping_summary()
                    self.log(f"[live] teams fitted at frame {index}, mapping {mapping}")
                    self._set_status(state="running", teams_fitted=True, mapping=mapping)

                # Same homography reuse as collect_shots: recompute on shot-relevant frames.
                shot_relevant = bool(np.any(np.isin(detections.class_id, SHOT_CLASS_IDS))) if len(detections) else False
                if shot_relevant or transformer is None:
                    kpts, mask = keypoints.landmarks(frame)
                    transformer = keypoints.transformer_from(kpts, mask)

                n_before = len(shots.shots)
                shots.update(frame, index, detections, transformer=transformer, team_brick=teams)
                for shot in shots.shots[n_before:]:
                    self._add_shot(name, shot, kept, window_frames)

                now = time.perf_counter()
                self._publish_frame(frame, detections)
                self._set_status(frame=index, fps=1.0 / max(now - t_last, 1e-6))
                t_last = now
                index += 1
        finally:
            cap.release()

    # -------------------------------------------------------------------- shots
    def _add_shot(self, clip: str, shot: dict, kept: list[tuple[dict, int]], window_frames: int) -> None:
        """Incremental version of dedup_clip_shots: keep the higher-priority duplicate."""
        from src.shots import _is_duplicate, shot_priority

        for k, (other, row_index) in enumerate(kept):
            if _is_duplicate(shot, other, window_frames, self.dedup_dist_ft):
                if shot_priority(shot) > shot_priority(other):
                    with self._lock:
                        self._rows[row_index] = self._shot_row(clip, shot)
                    kept[k] = (shot, row_index)
                    self._write_csv()
                return

        row = self._shot_row(clip, shot)
        with self._lock:
            self._rows.append(row)
            row_index = len(self._rows) - 1
            self._banner = (f"{row['result'].upper()}  {row['name']} ({row['team']})", time.perf_counter())
        kept.append((shot, row_index))
        self._write_csv()
        self.log(f"[live] {clip} f{row['frame']} {row['shot_type']} {row['result']} "
                 f"{row['team']} #{row['player_id']} {row['name']} @({row['x_court']},{row['y_court']})")

    def _shot_row(self, clip: str, shot: dict) -> dict:
        teams = self._teams
        team_id, team_source = shot.get("team"), "classifier"
        if team_id is None:
            team_id, team_source = shot.get("offense_team"), "offense"
        if team_id is not None and teams.fitted:
            team = teams.team_name(team_id)
        else:
            team, team_source = self._rng.choice(sorted(self.rosters)), "random"

        roster = sorted((self.rosters.get(team) or {}).items())
        number, name = self._rng.choice(roster) if roster else ("", "Unknown")

        cx, cy = shot.get("court_x"), shot.get("court_y")
        basket = shot.get("attacking_basket") or shot.get("shot_basket")
        hx, hy = to_half_court(cx, cy) if cx is not None else (None, None)
        # Distance to the hoop the shot was taken at (half-court hoop after mirroring).
        distance = None if hx is None else float(np.hypot(hx - HOOP[0], hy - HOOP[1]))
        return {
            "video": clip,
            "frame": shot["frame"],
            "team": team,
            "team_source": team_source,
            "player_id": number,
            "name": name,
            "identity": "random",
            "shot_type": classify_shot_type(shot.get("type", ""), hx, hy),
            "result": shot.get("outcome", ""),
            "x_court": None if hx is None else round(hx, 2),
            "y_court": None if hy is None else round(hy, 2),
            "court_x": None if cx is None else round(cx, 2),
            "court_y": None if cy is None else round(cy, 2),
            "attacking_basket": basket or "",
            "distance_ft": None if distance is None else round(distance, 2),
        }

    def _write_csv(self) -> None:
        if self.out_csv is None:
            return
        with self._lock:
            rows = list(self._rows)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    # -------------------------------------------------------------------- frame
    def _publish_frame(self, frame: np.ndarray, detections) -> None:
        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        with self._lock:
            banner = self._banner
        if banner is not None and time.perf_counter() - banner[1] < 1.5:
            text = banner[0]
            colour = (0, 160, 0) if text.startswith("MADE") else (0, 0, 200)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 70), colour, -1)
            cv2.putText(annotated, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (255, 255, 255), 3, cv2.LINE_AA)

        h, w = annotated.shape[:2]
        if w > self.frame_width:
            annotated = cv2.resize(annotated, (self.frame_width, int(h * self.frame_width / w)),
                                   interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
