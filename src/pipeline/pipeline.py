"""End-to-end basketball CV pipeline.

Assembles the notebook into one pass over a video:

1. Roboflow RF-DETR object detection (players, jersey numbers, events)
2. SAM2 tracking (stable ``tracker_id`` + masks)
3. Jersey OCR + team clustering (name / number from the roster)
4. Court keypoints + homography, then ``clean_paths`` smoothing

Outputs
-------
player_df
    One row per tracked player per frame:
    ``frame_idx, tracker_id, name, number, court_x, court_y``
    ``court_x`` / ``court_y`` come from ``cleaned_xy``.

event_df
    One row per frame:
    ``frame_idx, possession, layup_dunk, jumpshot, ball_in_basket``

history (``outputs/<video_name>/history``)
    ``frame_history/`` — JPEG per frame
    ``detections_history/`` — SAM2 ``sv.Detections`` (boxes, masks, tracker_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import supervision as sv
import torch
from sports import ConsecutiveValueTracker, TeamClassifier
from tqdm import tqdm

from src.court import CourtKeypointDetector, clean_court_paths
from src.detection import (
    BasketballDetector,
    DetectionClass,
    JerseyOCR,
    jersey_crops,
)
from src.pipeline.history import (
    prepare_history_dir,
    save_history_detections,
    save_history_frame,
    save_history_meta,
)
from src.pipeline.rosters import DEFAULT_TEAM_NAMES, TEAM_ROSTERS
from src.tracking import SAM2Tracker, get_state_matches

# --- notebook constants -------------------------------------------------------
TEAM_CROP_STRIDE = 30          # ~1 FPS at 30 FPS, used to fit TeamClassifier
OCR_STRIDE = 5                 # run jersey OCR every N frames
NUMBER_VALIDATE_STREAK = 3     # ConsecutiveValueTracker for OCR
TEAM_VALIDATE_STREAK = 1       # teams are assigned once on the first frame
STATE_NMS_THRESHOLD = 0.5      # class-agnostic NMS on player-state boxes
STATE_IOU_THRESHOLD = 0.3      # IoU to attach RF-DETR states onto SAM2 tracks


@dataclass
class PipelineResult:
    """Tables produced by ``BasketballPipeline.run``."""

    player_df: pd.DataFrame
    event_df: pd.DataFrame
    identity_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_dir: Path | None = None


class BasketballPipeline:
    """Load models once, then run them on a video."""

    def __init__(
        self,
        team_names: dict[int, str] | None = None,
        team_rosters: dict[str, dict[str, str]] | None = None,
        ocr_stride: int = OCR_STRIDE,
        team_crop_stride: int = TEAM_CROP_STRIDE,
    ) -> None:
        # Cluster id -> official team name. Flip this if names look swapped.
        self.team_names = team_names if team_names is not None else dict(DEFAULT_TEAM_NAMES)
        self.team_rosters = team_rosters if team_rosters is not None else TEAM_ROSTERS
        self.ocr_stride = ocr_stride
        self.team_crop_stride = team_crop_stride

        # 1) Roboflow RF-DETR: players, numbers, ball-in-basket, shot actions.
        self.detector = BasketballDetector()

        # 2) SmolVLM2: read digits from jersey-number crops.
        self.ocr = JerseyOCR()

        # 3) SAM2: prompt with first-frame boxes, then track masks + IDs.
        self.tracker = SAM2Tracker()

        # 4) Court keypoints: homography from image feet -> court feet.
        self.court = CourtKeypointDetector()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 5) SigLIP + k-means (k=2) jersey-color team clustering.
        self.team_classifier = TeamClassifier(device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
        history_dir: str | Path | None = None,
    ) -> PipelineResult:
        """Process ``video_path`` and return player / event dataframes.

        When ``history_dir`` is set, each frame and its SAM2 detections are
        written to ``frame_history/`` and ``detections_history/`` as the loop
        runs (same idea as the notebook lists, but on disk).
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_info = sv.VideoInfo.from_video_path(str(video_path))
        if history_dir is not None:
            history_dir = prepare_history_dir(history_dir)

        # Train the team classifier on jersey crops sampled from this clip.
        self._fit_team_classifier(video_path, max_frames=max_frames)

        # Prompt SAM2 from the first frame that has player boxes.
        first_frame = self._first_frame(video_path)
        first_players = self.detector.detect_players(
            first_frame,
            assign_tracker_ids=True,
        )
        if len(first_players) == 0:
            raise RuntimeError("No players detected on the first frame.")

        tracker_ids = np.asarray(first_players.tracker_id, dtype=int)
        id_to_col = {int(tid): i for i, tid in enumerate(tracker_ids)}

        # Team ID is predicted once per first-frame track (notebook: n_consecutive=1).
        team_validator = ConsecutiveValueTracker(n_consecutive=TEAM_VALIDATE_STREAK)
        first_crops = jersey_crops(first_frame, first_players)
        if first_crops:
            first_teams = np.array(self.team_classifier.predict(first_crops))
            team_validator.update(tracker_ids=tracker_ids, values=first_teams)

        self.tracker.reset()
        self.tracker.prompt_first_frame(first_frame, first_players)

        # OCR must agree 3 times before a jersey number is trusted.
        number_validator = ConsecutiveValueTracker(n_consecutive=NUMBER_VALIDATE_STREAK)
        resolved_numbers: set[int] = set()

        video_xy: list[np.ndarray] = []
        event_rows: list[dict] = []
        n_players = len(tracker_ids)

        frame_generator = sv.get_video_frames_generator(str(video_path))
        for frame_idx, frame in enumerate(tqdm(frame_generator, desc="pipeline")):
            if max_frames is not None and frame_idx >= max_frames:
                break

            # --- SAM2: propagate masks / tracker IDs ------------------------
            players = self.tracker.propagate(frame)

            # Same lists as the notebook, written to disk so RAM stays bounded.
            if history_dir is not None:
                save_history_frame(history_dir, frame_idx, frame)
                save_history_detections(history_dir, frame_idx, players)

            # --- RF-DETR: one inference, then split by class ----------------
            all_dets = self.detector.infer(frame)
            state_dets, number_dets, other_dets = self.detector.split(all_dets)
            if len(state_dets) > 0:
                # Drop duplicate player-state boxes (possession vs standing, etc.).
                state_dets = state_dets.with_nms(
                    threshold=STATE_NMS_THRESHOLD,
                    class_agnostic=True,
                )

            # --- Jersey OCR on unresolved tracks, every OCR_STRIDE frames ---
            unresolved = [
                int(tid)
                for tid in (players.tracker_id if players.tracker_id is not None else [])
                if int(tid) not in resolved_numbers
            ]
            if (
                frame_idx % self.ocr_stride == 0
                and unresolved
                and len(number_dets) > 0
                and len(players) > 0
            ):
                texts, pairs = self.ocr.recognize_and_match(frame, number_dets, players)
                if pairs:
                    player_idx, number_idx = zip(*pairs)
                    matched_ids = np.asarray(players.tracker_id)[list(player_idx)]
                    matched_texts = [texts[int(i)] for i in number_idx]
                    number_validator.update(matched_ids, values=matched_texts)
                    resolved_numbers.update(int(tid) for tid in matched_ids)

            # --- Court mapping: feet (bottom-center) -> court coordinates ---
            court_xy = self.court.map_detections(frame, players)
            aligned = np.full((n_players, 2), np.nan, dtype=float)
            if players.tracker_id is not None:
                for det_i, tid in enumerate(players.tracker_id):
                    col = id_to_col.get(int(tid))
                    if col is not None and det_i < len(court_xy):
                        aligned[col] = court_xy[det_i]
            video_xy.append(aligned)

            # --- Frame-level event flags ------------------------------------
            # Possession / jump shot / layup are True if any SAM2 track
            # overlaps an RF-DETR box of that class. Ball-in-basket is a
            # scene-level class (not attached to a player).
            state_matches = get_state_matches(
                players,
                state_dets,
                iou_threshold=STATE_IOU_THRESHOLD,
            )
            all_classes = set().union(*state_matches.values()) if state_matches else set()
            event_rows.append(
                {
                    "frame_idx": frame_idx,
                    "possession": DetectionClass.PLAYER_IN_POSSESSION in all_classes,
                    "layup_dunk": DetectionClass.PLAYER_LAYUP_DUNK in all_classes,
                    "jumpshot": DetectionClass.PLAYER_JUMP_SHOT in all_classes,
                    "ball_in_basket": bool(
                        other_dets.class_id is not None
                        and np.any(other_dets.class_id == DetectionClass.BALL_IN_BASKET)
                    ),
                }
            )

        event_df = pd.DataFrame(event_rows)

        if history_dir is not None:
            save_history_meta(
                history_dir,
                source_video=str(video_path),
                video_name=video_path.stem,
                fps=float(video_info.fps),
                width=int(video_info.width),
                height=int(video_info.height),
                n_frames=len(event_rows),
            )

        # Smooth court trajectories (notebook: jump filter + Savitzky-Golay).
        raw_xy = np.stack(video_xy, axis=0) if video_xy else np.zeros((0, n_players, 2))
        cleaned_xy = self._clean_xy(raw_xy)

        identity_df = self._build_identity_df(
            tracker_ids=tracker_ids,
            team_validator=team_validator,
            number_validator=number_validator,
        )
        player_df = self._build_player_df(
            cleaned_xy=cleaned_xy,
            tracker_ids=tracker_ids,
            identity_df=identity_df,
        )
        return PipelineResult(
            player_df=player_df,
            event_df=event_df,
            identity_df=identity_df,
            history_dir=history_dir,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fit_team_classifier(
        self,
        video_path: Path,
        max_frames: Optional[int] = None,
    ) -> None:
        """Sample central jersey crops and fit SigLIP + k-means (k=2)."""
        crops: list[np.ndarray] = []
        frames = sv.get_video_frames_generator(
            str(video_path),
            stride=self.team_crop_stride,
        )
        sampled = 0
        for frame in frames:
            if max_frames is not None and sampled * self.team_crop_stride >= max_frames:
                break
            players = self.detector.detect_players(frame)
            crops.extend(jersey_crops(frame, players))
            sampled += 1

        if not crops:
            raise RuntimeError("No jersey crops found to fit TeamClassifier.")
        self.team_classifier.fit(crops)

    @staticmethod
    def _first_frame(video_path: Path) -> np.ndarray:
        return next(sv.get_video_frames_generator(str(video_path)))

    @staticmethod
    def _clean_xy(video_xy: np.ndarray) -> np.ndarray:
        """Run ``sports.clean_paths``; keep raw xy if smoothing fails."""
        if video_xy.size == 0:
            return video_xy
        try:
            cleaned_xy, _edited_mask = clean_court_paths(video_xy)
            return cleaned_xy
        except Exception:
            return video_xy

    def _build_identity_df(
        self,
        tracker_ids: np.ndarray,
        team_validator: ConsecutiveValueTracker,
        number_validator: ConsecutiveValueTracker,
    ) -> pd.DataFrame:
        """Map each tracker_id to team, jersey number, and roster name."""
        teams = team_validator.get_validated(tracker_ids=tracker_ids)
        numbers = number_validator.get_validated(tracker_ids=tracker_ids)
        rows = []
        for tracker_id, team_value, number_value in zip(tracker_ids, teams, numbers):
            team_id = _as_int(team_value)
            team_name = self.team_names.get(team_id) if team_id is not None else None
            jersey_number = _as_jersey(number_value)
            player_name = _roster_name(self.team_rosters, team_name, jersey_number)
            rows.append(
                {
                    "tracker_id": int(tracker_id),
                    "team_id": team_id,
                    "team": team_name,
                    "number": jersey_number,
                    "name": player_name,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_player_df(
        cleaned_xy: np.ndarray,
        tracker_ids: np.ndarray,
        identity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Expand ``cleaned_xy`` (frames, players, 2) into the player table."""
        identity = identity_df.set_index("tracker_id")
        rows = []
        for frame_idx, frame_xy in enumerate(cleaned_xy):
            for col, tracker_id in enumerate(tracker_ids):
                tracker_id = int(tracker_id)
                info = identity.loc[tracker_id] if tracker_id in identity.index else None
                court_x, court_y = frame_xy[col]
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "tracker_id": tracker_id,
                        "name": None if info is None else info.get("name"),
                        "number": None if info is None else info.get("number"),
                        "court_x": float(court_x) if np.isfinite(court_x) else np.nan,
                        "court_y": float(court_y) if np.isfinite(court_y) else np.nan,
                    }
                )
        player_df = pd.DataFrame(rows)
        if len(player_df) == 0:
            return pd.DataFrame(
                columns=[
                    "frame_idx",
                    "tracker_id",
                    "name",
                    "number",
                    "court_x",
                    "court_y",
                ]
            )
        return player_df.sort_values(["frame_idx", "tracker_id"]).reset_index(drop=True)


def _as_int(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_jersey(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text in {"", "None", "nan"}:
        return None
    return text


def _roster_name(
    rosters: dict[str, dict[str, str]],
    team_name: str | None,
    jersey_number: str | None,
) -> Optional[str]:
    if team_name is None or jersey_number is None:
        return None
    roster = rosters.get(team_name, {})
    name = roster.get(jersey_number)
    if name is None and jersey_number.isdigit():
        name = roster.get(str(int(jersey_number)))
    return name
