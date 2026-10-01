"""End-to-end basketball CV pipeline.

Assembles the notebook into one pass over a video:

1. Roboflow RF-DETR object detection (players, jersey numbers, events)
   Event boxes are cleaned with class-agnostic NMS, top-1 per class, and
   per-class confidence floors (same chain as ``notebooks/testing.ipynb``).
2. SAM2 tracking (stable ``tracker_id`` + masks). If frame 0 has fewer than
   10 players, unmatched RF-DETR boxes are added every 5 seconds until 10.
3. Jersey OCR + team clustering (name / number from the roster)
4. Court keypoints + homography, then ``clean_paths`` smoothing
5. Hard-cut segmentation: on every camera cut, SAM2 is re-prompted and
   smoothing / shots / validators restart (team classifier is fitted once).

Outputs
-------
player_df
    One row per tracked player per frame:
    ``frame_idx, tracker_id, team, name, number, court_x, court_y,
    segment_id, track_uid``
    ``tracker_id`` restarts at every segment; ``track_uid`` is unique.
    ``court_x`` / ``court_y`` come from ``cleaned_xy``.

event_df
    One row per frame:
    ``frame_idx, segment_id, possession, layup_dunk, jumpshot, ball_in_basket``
    (``segment_id == -1``: untracked frame, e.g. replay after a cut)

shots_df
    One row per shot event:
    ``shot_id, start_frame, end_frame, outcome, shot_type, tracker_id,
    team, name, number, court_x, court_y``

history (``outputs/<video_name>/history``)
    ``frame_history/`` — JPEG per frame
    ``detections_history/`` — SAM2 ``sv.Detections`` (boxes, masks, tracker_id)
"""

from __future__ import annotations

import contextlib

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
    clean_event_detections,
    jersey_crops,
)
from src.pipeline.render import PipelineRenderer
from src.pipeline.history import (
    prepare_history_dir,
    save_history_detections,
    save_history_frame,
    save_history_meta,
    save_history_states,
)
from src.pipeline.rosters import DEFAULT_TEAM_NAMES, TEAM_ROSTERS, TEAM_COLORS
from src.pipeline.shots import ShotCollector, attach_identity_and_court, empty_shots_df
from src.tracking import (
    SAM2Tracker,
    concat_player_detections,
    get_state_matches,
    select_new_player_prompts,
)
from src.tracking.cuts import(
    HardCutDetector
)

# --- notebook constants -------------------------------------------------------
TEAM_CROP_STRIDE = 30          # ~1 FPS at 30 FPS, used to fit TeamClassifier
OCR_STRIDE = 5                 # run jersey OCR every N frames
NUMBER_VALIDATE_STREAK = 3     # ConsecutiveValueTracker for OCR
TEAM_VALIDATE_STREAK = 1       # teams are assigned once on the first frame
STATE_NMS_THRESHOLD = 0.5      # class-agnostic NMS on player-state boxes
STATE_IOU_THRESHOLD = 0.3      # IoU to attach RF-DETR states onto SAM2 tracks
DEFAULT_TARGET_FPS = 10.0      # subsample the source video to this rate
TARGET_PLAYERS = 10            # stop adding SAM2 tracks once this many exist
REPROMPT_SECONDS = 5.0         # if under TARGET_PLAYERS, try again this often
MIN_PLAYERS_TO_PROMPT = 6      # don't open a segment on a close-up / replay shot
ID_STRIDE = 1000               # track_uid = segment_id * ID_STRIDE + tracker_id

@dataclass
class SegmentState:
    """Everything that is reset on a hard cut (one continuous camera shot)."""

    segment_id: int
    start_frame: int
    tracker_ids: np.ndarray
    id_to_col: dict[int, int]
    team_validator: ConsecutiveValueTracker
    number_validator: ConsecutiveValueTracker
    shot_collector: ShotCollector
    resolved_numbers: set[int] = field(default_factory=set)
    video_xy: list[np.ndarray] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)  # global idx per video_xy row

    @property
    def n_players(self) -> int:
        return len(self.tracker_ids)

    @property
    def last_frame(self) -> int:
        return self.frame_indices[-1] if self.frame_indices else self.start_frame


@dataclass
class SegmentResult:
    player_df: pd.DataFrame
    identity_df: pd.DataFrame
    shots_df: pd.DataFrame


def frame_stride(source_fps: float, target_fps: float | None) -> int:
    """How many source frames to skip so processing runs near ``target_fps``."""
    if target_fps is None or target_fps <= 0:
        return 1
    source_fps = float(source_fps)
    if source_fps <= 0 or target_fps >= source_fps:
        return 1
    return max(1, int(round(source_fps / float(target_fps))))


@dataclass
class PipelineResult:
    """Tables produced by ``BasketballPipeline.run``."""

    player_df: pd.DataFrame
    event_df: pd.DataFrame
    identity_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shots_df: pd.DataFrame = field(default_factory=empty_shots_df)
    history_dir: Path | None = None


class BasketballPipeline:
    """Load models once, then run them on a video."""

    def __init__(
        self,
        team_names: dict[int, str] | None = None,
        team_rosters: dict[str, dict[str, str]] | None = None,
        ocr_stride: int = OCR_STRIDE,
        team_crop_stride: int = TEAM_CROP_STRIDE,
        use_ocr: bool = False,
    ) -> None:
        # Cluster id -> official team name. Flip this if names look swapped.
        self.team_names = team_names if team_names is not None else dict(DEFAULT_TEAM_NAMES)
        self.team_rosters = team_rosters if team_rosters is not None else TEAM_ROSTERS
        self.ocr_stride = ocr_stride
        self.team_crop_stride = team_crop_stride
        self.use_ocr = use_ocr

        # 1) Roboflow RF-DETR: players, numbers, ball-in-basket, shot actions.
        self.detector = BasketballDetector()

        # 2) SmolVLM2 jersey OCR (skipped unless --ocr).
        self.ocr = JerseyOCR() if use_ocr else None

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
        target_fps: float | None = DEFAULT_TARGET_FPS,
        output_video: str | Path | None = None,
        renderer: PipelineRenderer | None = None,
    ) -> PipelineResult:
        """Process ``video_path`` segment by segment.

        A segment is one continuous camera shot. On every hard cut the current
        segment is finalized and SAM2 is re-prompted on the first trackable
        frame after the cut. The team classifier is fitted once per video so
        team ids stay consistent across segments.

        ``target_fps`` subsamples the source video (default 10). Pass ``None``
        to process every frame. ``max_frames`` counts processed frames.

        ``output_video``: if set, an annotated MP4 (masks, identities, events,
        cuts, minimap) is written there at the processing frame rate.
        ``renderer`` lets you pass a configured ``PipelineRenderer``.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_info = sv.VideoInfo.from_video_path(str(video_path))
        stride = frame_stride(float(video_info.fps), target_fps)
        process_fps = float(video_info.fps) / stride
        print(
            f"tracking at {process_fps:.1f} fps "
            f"(stride={stride}, source={float(video_info.fps):.1f} fps)"
        )
        if history_dir is not None:
            history_dir = prepare_history_dir(history_dir)

        # Once per VIDEO (not per segment): keeps team 0 / 1 stable across cuts.
        source_limit = None if max_frames is None else max_frames * stride
        self._fit_team_classifier(video_path, max_frames=source_limit)

        reprompt_interval = max(1, int(round(process_fps * REPROMPT_SECONDS)))
        cut_detector = HardCutDetector(min_gap=0.5 * process_fps)

        total = None
        if video_info.total_frames:
            total = int(video_info.total_frames) // stride
            if max_frames is not None:
                total = min(total, max_frames)

        event_rows: list[dict] = []
        results: list[SegmentResult] = []
        seg: Optional[SegmentState] = None
        next_segment_id = 0

        sink = None
        if output_video is not None:
            renderer = renderer or PipelineRenderer()
            output_video = Path(output_video)
            output_video.parent.mkdir(parents=True, exist_ok=True)
        else:
            renderer = None

        with contextlib.ExitStack() as stack:
            if output_video is not None:
                sink = stack.enter_context(
                    sv.VideoSink(
                        str(output_video),
                        video_info=sv.VideoInfo(
                            width=int(video_info.width),
                            height=int(video_info.height),
                            fps=max(1, int(round(process_fps))),
                        ),
                        codec="mp4v",
                    )
                )
            frame_generator = sv.get_video_frames_generator(str(video_path), stride=stride)
            for frame_idx, frame in enumerate(tqdm(frame_generator, desc="pipeline", total=total)):
                if max_frames is not None and frame_idx >= max_frames:
                    break

                # Update on every frame, including untracked ones.
                hard_cut = cut_detector.update(frame, frame_idx)
                if hard_cut and renderer is not None:
                    renderer.notify_cut()

                if hard_cut and seg is not None:
                    print(f"frame {frame_idx}: HARD CUT -> closing segment {seg.segment_id}")
                    results.append(self._finalize_segment(seg))
                    seg = None

                if seg is None:
                    seg = self._init_segment(frame, frame_idx, next_segment_id, process_fps)
                    if seg is None:
                        # Replay / close-up / crowd: not trackable, retry next frame.
                        if history_dir is not None:
                            save_history_frame(history_dir, frame_idx, frame)
                            save_history_detections(history_dir, frame_idx, sv.Detections.empty())
                        event_rows.append(_untracked_event_row(frame_idx))
                        if sink is not None:
                            sink.write_frame(renderer.render_untracked(frame, frame_idx))
                        continue
                    print(
                        f"frame {frame_idx}: segment {seg.segment_id} started "
                        f"with {seg.n_players} tracks"
                    )
                    next_segment_id += 1

                # The prompt frame is also propagated (same as the old frame 0).
                event_rows.append(
                    self._process_frame(
                        frame, frame_idx, seg, history_dir, reprompt_interval,
                        renderer=renderer, sink=sink,
                    )
                )

            if seg is not None:
                results.append(self._finalize_segment(seg))

        if history_dir is not None:
            save_history_meta(
                history_dir,
                source_video=str(video_path),
                video_name=video_path.stem,
                fps=process_fps,
                width=int(video_info.width),
                height=int(video_info.height),
                n_frames=len(event_rows),
            )

        shots_df = _concat([r.shots_df for r in results])
        return PipelineResult(
            player_df=_concat([r.player_df for r in results]),
            event_df=pd.DataFrame(event_rows),
            identity_df=link_identities(_concat([r.identity_df for r in results])),
            shots_df=shots_df if not shots_df.empty else empty_shots_df(),
            history_dir=history_dir,
        )

    # ------------------------------------------------------------------
    # Segments
    # ------------------------------------------------------------------

    def _init_segment(
        self,
        frame: np.ndarray,
        frame_idx: int,
        segment_id: int,
        process_fps: float,
    ) -> Optional[SegmentState]:
        """Prompt SAM2 on ``frame``. Returns None if too few players are visible."""
        players = self.detector.detect_players(
            frame,
            assign_tracker_ids=True,
            nms_threshold=STATE_NMS_THRESHOLD,
        )
        if len(players) < MIN_PLAYERS_TO_PROMPT:
            return None
        if len(players) > TARGET_PLAYERS:
            print(
                f"frame {frame_idx}: warning, prompting SAM2 with {len(players)} "
                f"player boxes (expected {TARGET_PLAYERS})."
            )

        tracker_ids = np.asarray(players.tracker_id, dtype=int)
        id_to_col = {int(tid): i for i, tid in enumerate(tracker_ids)}

        team_validator = ConsecutiveValueTracker(n_consecutive=TEAM_VALIDATE_STREAK)
        crops = jersey_crops(frame, players)
        if crops:
            teams = np.array(self.team_classifier.predict(crops))
            team_validator.update(tracker_ids=tracker_ids, values=teams)

        # Fresh SAM2 memory: nothing from the previous shot leaks in.
        self.tracker.reset()
        self.tracker.prompt_first_frame(frame, players)

        return SegmentState(
            segment_id=segment_id,
            start_frame=frame_idx,
            tracker_ids=tracker_ids,
            id_to_col=id_to_col,
            team_validator=team_validator,
            number_validator=ConsecutiveValueTracker(n_consecutive=NUMBER_VALIDATE_STREAK),
            shot_collector=ShotCollector(fps=process_fps),
        )

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        seg: SegmentState,
        history_dir: Path | None,
        reprompt_interval: int,
        renderer: PipelineRenderer | None = None,
        sink: sv.VideoSink | None = None,
    ) -> dict:
        """One frame of tracking / detection / OCR / court / events. Returns the event row."""
        # --- SAM2: propagate masks / tracker IDs
        players = self.tracker.propagate(frame)

        # --- RF-DETR: one inference, then event cleaning
        all_dets = self.detector.infer(frame)
        local_idx = frame_idx - seg.start_frame  # re-prompt clock restarts per segment
        if (
            seg.n_players < TARGET_PLAYERS
            and local_idx > 0
            and local_idx % reprompt_interval == 0
        ):
            players, seg.tracker_ids, seg.id_to_col, n_added = self._add_missing_players(
                frame=frame,
                all_dets=all_dets,
                players=players,
                tracker_ids=seg.tracker_ids,
                id_to_col=seg.id_to_col,
                team_validator=seg.team_validator,
            )
            if n_added:
                _pad_video_xy(seg.video_xy, extra=n_added)
                print(
                    f"frame {frame_idx} (segment {seg.segment_id}): added {n_added} "
                    f"SAM2 track(s) ({seg.n_players}/{TARGET_PLAYERS})"
                )

        if history_dir is not None:
            save_history_frame(history_dir, frame_idx, frame)
            save_history_detections(history_dir, frame_idx, players)

        event_dets = clean_event_detections(all_dets, nms_iou=STATE_NMS_THRESHOLD)
        state_dets, _, other_dets = self.detector.split(event_dets)
        _, number_dets, _ = self.detector.split(all_dets)

        if history_dir is not None:
            save_history_states(history_dir, frame_idx, state_dets)

        # --- Jersey OCR (optional)
        if self.use_ocr and self.ocr is not None:
            unresolved = [
                int(tid)
                for tid in (players.tracker_id if players.tracker_id is not None else [])
                if int(tid) not in seg.resolved_numbers
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
                    seg.number_validator.update(matched_ids, values=matched_texts)
                    seg.resolved_numbers.update(int(tid) for tid in matched_ids)

        # --- Court mapping: feet -> court coordinates
        court_xy = self.court.map_detections(frame, players)
        aligned = np.full((seg.n_players, 2), np.nan, dtype=float)
        if players.tracker_id is not None:
            for det_i, tid in enumerate(players.tracker_id):
                col = seg.id_to_col.get(int(tid))
                if col is not None and det_i < len(court_xy):
                    aligned[col] = court_xy[det_i]
        seg.video_xy.append(aligned)
        seg.frame_indices.append(frame_idx)

        # --- Frame-level event flags
        state_matches = get_state_matches(
            players,
            state_dets,
            iou_threshold=STATE_IOU_THRESHOLD,
        )
        all_classes = set().union(*state_matches.values()) if state_matches else set()
        has_ball_in_basket = bool(
            other_dets.class_id is not None
            and np.any(other_dets.class_id == DetectionClass.BALL_IN_BASKET)
        )
        if players.tracker_id is not None and len(players) > 0:
            validated = seg.team_validator.get_validated(
                tracker_ids=[int(tid) for tid in players.tracker_id]
            )
            player_teams = np.array(
                [-1 if v is None else int(v) for v in validated],
                dtype=int,
            )
        else:
            player_teams = None
        possession_tracker_ids = [
            tid
            for tid, classes in state_matches.items()
            if DetectionClass.PLAYER_IN_POSSESSION in classes
        ]
        seg.shot_collector.update(
            frame_idx=frame_idx,
            all_dets=event_dets,
            players=players,
            court_xy=court_xy,
            court=self.court,
            frame=frame,
            has_ball_in_basket=has_ball_in_basket,
            player_teams=player_teams,
            possession_tracker_ids=possession_tracker_ids,
        )
        row = {
            "frame_idx": frame_idx,
            "segment_id": seg.segment_id,
            "possession": DetectionClass.PLAYER_IN_POSSESSION in all_classes,
            "layup_dunk": DetectionClass.PLAYER_LAYUP_DUNK in all_classes,
            "jumpshot": DetectionClass.PLAYER_JUMP_SHOT in all_classes,
            "ball_in_basket": has_ball_in_basket,
        }
        if sink is not None and renderer is not None:
            sink.write_frame(
                renderer.render(
                    frame=frame,
                    frame_idx=frame_idx,
                    segment_id=seg.segment_id,
                    players=players,
                    teams=player_teams,
                    labels=self._live_labels(seg, players),
                    possession_ids=possession_tracker_ids,
                    state_dets=state_dets,
                    court_xy=court_xy,
                    event_row=row,
                )
            )
        return row

    def _live_labels(self, seg: SegmentState, players: sv.Detections) -> list[str]:
        """Current best identity per visible track: '#23 Name', '#23' or 'id 4'."""
        if players.tracker_id is None or len(players) == 0:
            return []
        tids = [int(t) for t in players.tracker_id]
        teams = seg.team_validator.get_validated(tracker_ids=tids)
        numbers = seg.number_validator.get_validated(tracker_ids=tids)
        labels = []
        for tid, team_value, number_value in zip(tids, teams, numbers):
            number = _as_jersey(number_value)
            team_id = _as_int(team_value)
            team_name = self.team_names.get(team_id) if team_id is not None else None
            name = _roster_name(self.team_rosters, team_name, number)
            if number and name:
                labels.append(f"#{number} {name}")
            elif number:
                labels.append(f"#{number}")
            else:
                labels.append(f"id {tid}")
        return labels

    def _finalize_segment(self, seg: SegmentState) -> SegmentResult:
        """Smooth, build identity / player / shot tables for one segment."""
        raw_xy = (
            np.stack(seg.video_xy, axis=0)
            if seg.video_xy
            else np.zeros((0, seg.n_players, 2))
        )
        # Per segment: smoothing never crosses a cut.
        cleaned_xy = self._clean_xy(raw_xy)

        identity_df = self._build_identity_df(
            tracker_ids=seg.tracker_ids,
            team_validator=seg.team_validator,
            number_validator=seg.number_validator,
        )
        player_df = self._build_player_df(
            cleaned_xy=cleaned_xy,
            tracker_ids=seg.tracker_ids,
            identity_df=identity_df,
        )
        # _build_player_df numbers frames 0..T-1; ShotCollector uses global
        # frame_idx, so remap BEFORE attaching identity / court to shots.
        player_df = _to_global_frames(player_df, seg.frame_indices)

        shots_df = attach_identity_and_court(
            seg.shot_collector.finalize(last_frame=seg.last_frame),
            player_df=player_df,
            identity_df=identity_df,
        )
        return SegmentResult(
            player_df=_tag_segment(player_df, seg.segment_id),
            identity_df=_tag_segment(identity_df, seg.segment_id),
            shots_df=_tag_segment(shots_df, seg.segment_id),
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

    def _add_missing_players(
        self,
        frame: np.ndarray,
        all_dets: sv.Detections,
        players: sv.Detections,
        tracker_ids: np.ndarray,
        id_to_col: dict[int, int],
        team_validator,
    ) -> tuple[sv.Detections, np.ndarray, dict[int, int], int]:
        """Prompt SAM2 with unmatched RF-DETR boxes until TARGET_PLAYERS."""
        slots_left = TARGET_PLAYERS - len(tracker_ids)
        if slots_left <= 0:
            return players, tracker_ids, id_to_col, 0

        candidates = self.detector.filter_players(
            all_dets,
            nms_threshold=STATE_NMS_THRESHOLD,
        )
        new_dets = select_new_player_prompts(
            candidates,
            players,
            slots_left=slots_left,
            iou_threshold=STATE_IOU_THRESHOLD,
        )
        if len(new_dets) == 0:
            return players, tracker_ids, id_to_col, 0

        next_id = int(np.max(tracker_ids)) + 1
        new_ids = np.arange(next_id, next_id + len(new_dets), dtype=int)
        new_dets.tracker_id = new_ids
        self.tracker.add_prompts(frame,players,new_dets)

        crops = jersey_crops(frame, new_dets)
        if crops:
            teams = np.array(self.team_classifier.predict(crops))
            team_validator.update(tracker_ids=new_ids, values=teams)

        for col_offset, tid in enumerate(new_ids):
            id_to_col[int(tid)] = len(tracker_ids) + col_offset
        tracker_ids = np.concatenate([tracker_ids, new_ids])
        players = concat_player_detections(players, new_dets)
        return players, tracker_ids, id_to_col, len(new_ids)

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
                        "team": None if info is None else info.get("team"),
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
                    "team",
                    "name",
                    "number",
                    "court_x",
                    "court_y",
                ]
            )
        return player_df.sort_values(["frame_idx", "tracker_id"]).reset_index(drop=True)


def _pad_video_xy(video_xy: list[np.ndarray], extra: int) -> None:
    """Add NaN player columns to frames recorded before a new SAM2 track."""
    if extra <= 0:
        return
    for i, arr in enumerate(video_xy):
        pad = np.full((extra, 2), np.nan, dtype=float)
        video_xy[i] = np.concatenate([arr, pad], axis=0)


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


def _untracked_event_row(frame_idx: int) -> dict:
    """Frame of a shot SAM2 could not be prompted on (replay, close-up...)."""
    return {
        "frame_idx": frame_idx,
        "segment_id": -1,
        "possession": False,
        "layup_dunk": False,
        "jumpshot": False,
        "ball_in_basket": False,
    }


def _to_global_frames(df: pd.DataFrame, frame_indices: list[int]) -> pd.DataFrame:
    """Map local frame positions (0..T-1) to global ``frame_idx``."""
    if df.empty or "frame_idx" not in df.columns:
        return df
    df = df.copy()
    lut = np.asarray(frame_indices, dtype=int)
    df["frame_idx"] = lut[df["frame_idx"].to_numpy(dtype=int)]
    return df


def _tag_segment(df: pd.DataFrame, segment_id: int) -> pd.DataFrame:
    """SAM2 ids restart at every prompt: add ``segment_id`` and a unique ``track_uid``."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["segment_id"] = segment_id
    if "tracker_id" in df.columns:
        valid = df["tracker_id"].notna()
        df["track_uid"] = pd.NA
        df.loc[valid, "track_uid"] = (
            segment_id * ID_STRIDE + df.loc[valid, "tracker_id"].astype(int)
        )
    return df


def _concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if d is not None and not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def link_identities(identity_df: pd.DataFrame) -> pd.DataFrame:
    """Link tracks across cuts with (team_id, number), the only identity that
    survives a cut. Tracks without a validated number stay segment-local."""
    if identity_df.empty:
        return identity_df
    df = identity_df.copy()
    has_id = df["team_id"].notna() & df["number"].notna()
    df["player_key"] = [
        f"T{int(t)}_#{n}" if ok else f"uid_{uid}"
        for t, n, uid, ok in zip(df["team_id"], df["number"], df["track_uid"], has_id)
    ]
    return df