"""Shot-event collection and court plotting.

``ShotEventTracker`` (same settings as the testing notebook) turns per-frame
jump-shot / layup / ball-in-basket flags into START / MADE / MISSED. On START
the jump-shot or layup box is IoU-matched to a SAM2 ``tracker_id`` and the
shooter's feet are mapped to court coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from sports.basketball import ShotEventTracker, draw_made_and_miss_on_court

from src.court import CourtKeypointDetector, default_court_config
from src.detection.object_detection import DetectionClass, filter_class_ids
from src.tracking import match_detector_to_tracker_id

SHOT_COLUMNS = [
    "shot_id",
    "start_frame",
    "end_frame",
    "outcome",
    "shot_type",
    "tracker_id",
    "team",
    "name",
    "number",
    "court_x",
    "court_y",
]


def empty_shots_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SHOT_COLUMNS)


def shot_event_tracker_from_fps(fps: float) -> ShotEventTracker:
    """Notebook defaults: 1.7s miss timeout, 0.5s start/made cooldowns."""
    fps = max(float(fps), 1.0)
    return ShotEventTracker(
        reset_time_frames=int(fps * 1.7),
        minimum_frames_between_starts=int(fps * 0.5),
        cooldown_frames_after_made=int(fps * 0.5),
    )


class ShotCollector:
    """Consume detector flags each frame and emit completed shot rows."""

    def __init__(self, fps: float) -> None:
        self.tracker = shot_event_tracker_from_fps(fps)
        self.pending: dict | None = None
        self.rows: list[dict] = []

    def update(
        self,
        frame_idx: int,
        all_dets: sv.Detections,
        players: sv.Detections,
        court_xy: np.ndarray,
        court: CourtKeypointDetector,
        frame: np.ndarray,
        has_ball_in_basket: bool,
    ) -> None:
        jump_dets = filter_class_ids(all_dets, DetectionClass.PLAYER_JUMP_SHOT)
        layup_dets = filter_class_ids(all_dets, DetectionClass.PLAYER_LAYUP_DUNK)
        events = self.tracker.update(
            frame_index=frame_idx,
            has_jump_shot=len(jump_dets) > 0,
            has_layup_dunk=len(layup_dets) > 0,
            has_ball_in_basket=has_ball_in_basket,
        )
        for event in events:
            kind = event["event"]
            if kind == "START":
                event_dets = jump_dets if event["type"] == "JUMP" else layup_dets
                self.pending = _open_shot(
                    event=event,
                    event_dets=event_dets,
                    players=players,
                    court_xy=court_xy,
                    court=court,
                    frame=frame,
                )
            elif kind in {"MADE", "MISSED"}:
                self.rows.append(
                    _close_shot(
                        pending=self.pending,
                        event=event,
                    )
                )
                self.pending = None

    def finalize(self, last_frame: int) -> pd.DataFrame:
        if self.pending is not None:
            self.rows.append(
                _close_shot(
                    pending=self.pending,
                    event={
                        "event": "MISSED",
                        "frame": last_frame,
                        "type": str(self.pending.get("shot_type", "none")).upper(),
                    },
                )
            )
            self.pending = None
        shots_df = pd.DataFrame(self.rows, columns=SHOT_COLUMNS)
        if len(shots_df) == 0:
            return empty_shots_df()
        shots_df["shot_id"] = np.arange(1, len(shots_df) + 1)
        return shots_df[SHOT_COLUMNS]


def attach_identity_and_court(
    shots_df: pd.DataFrame,
    player_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer smoothed player_df court xy; copy team/name/number from identity."""
    if len(shots_df) == 0:
        return empty_shots_df()

    shots_df = shots_df.copy()
    if len(player_df) > 0:
        lookup = player_df.set_index(["frame_idx", "tracker_id"])
        for i, row in shots_df.iterrows():
            if pd.isna(row["tracker_id"]):
                continue
            key = (int(row["start_frame"]), int(row["tracker_id"]))
            if key not in lookup.index:
                continue
            loc = lookup.loc[key]
            cx, cy = loc["court_x"], loc["court_y"]
            if np.isfinite(cx) and np.isfinite(cy):
                shots_df.at[i, "court_x"] = float(cx)
                shots_df.at[i, "court_y"] = float(cy)

    if len(identity_df) > 0:
        ident = identity_df.set_index("tracker_id")
        for i, row in shots_df.iterrows():
            if pd.isna(row["tracker_id"]):
                continue
            tid = int(row["tracker_id"])
            if tid not in ident.index:
                continue
            info = ident.loc[tid]
            shots_df.at[i, "team"] = info.get("team")
            shots_df.at[i, "name"] = info.get("name")
            shots_df.at[i, "number"] = info.get("number")

    return shots_df[SHOT_COLUMNS]


def filter_shots(
    shots_df: pd.DataFrame,
    tracker_id: int | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    subset = shots_df
    if tracker_id is not None:
        ids = pd.to_numeric(subset["tracker_id"], errors="coerce")
        subset = subset[ids == tracker_id]
    if team:
        subset = subset[subset["team"].fillna("").str.lower() == team.lower()]
    return subset.reset_index(drop=True)


def plot_shot_chart(
    shots_df: pd.DataFrame,
    output_path: str | Path,
    tracker_id: int | None = None,
    team: str | None = None,
) -> Path:
    """Draw made (circle) / miss (X) markers from ``shots_df`` onto an NBA court."""
    subset = filter_shots(shots_df, tracker_id=tracker_id, team=team)
    made_xy = _xy(subset[subset["outcome"] == "made"])
    miss_xy = _xy(subset[subset["outcome"] == "missed"])
    court = draw_made_and_miss_on_court(
        config=default_court_config(),
        made_xy=made_xy,
        miss_xy=miss_xy,
        miss_color=sv.Color.from_hex("#850101"),
        made_color=sv.Color.from_hex("#007A33"),
        miss_size=10,
        made_size=25,
        made_thickness=6,
        miss_thickness=6,
        line_thickness=4,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), court)
    return output_path


def _xy(shots_df: pd.DataFrame) -> Optional[np.ndarray]:
    if len(shots_df) == 0:
        return None
    coords = shots_df[["court_x", "court_y"]].to_numpy(dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    coords = coords[finite]
    if len(coords) == 0:
        return None
    return coords


def _open_shot(
    event: dict,
    event_dets: sv.Detections,
    players: sv.Detections,
    court_xy: np.ndarray,
    court: CourtKeypointDetector,
    frame: np.ndarray,
) -> dict:
    tracker_id = match_detector_to_tracker_id(event_dets, players)
    court_x, court_y = _court_xy_for_tracker(players, court_xy, tracker_id)
    if (not np.isfinite(court_x) or not np.isfinite(court_y)) and len(event_dets) > 0:
        event_court = court.map_detections(frame, event_dets)
        if len(event_court) > 0:
            court_x, court_y = event_court[0]
    return {
        "shot_id": None,
        "start_frame": int(event["frame"]),
        "end_frame": None,
        "outcome": None,
        "shot_type": str(event["type"]).lower(),
        "tracker_id": tracker_id,
        "team": None,
        "name": None,
        "number": None,
        "court_x": float(court_x) if np.isfinite(court_x) else np.nan,
        "court_y": float(court_y) if np.isfinite(court_y) else np.nan,
    }


def _close_shot(pending: dict | None, event: dict) -> dict:
    row = pending if pending is not None else {
        "shot_id": None,
        "start_frame": int(event["frame"]),
        "end_frame": None,
        "outcome": None,
        "shot_type": str(event["type"]).lower(),
        "tracker_id": None,
        "team": None,
        "name": None,
        "number": None,
        "court_x": np.nan,
        "court_y": np.nan,
    }
    row["end_frame"] = int(event["frame"])
    row["outcome"] = str(event["event"]).lower()
    if not row.get("shot_type"):
        row["shot_type"] = str(event["type"]).lower()
    return row


def _court_xy_for_tracker(
    players: sv.Detections,
    court_xy: np.ndarray,
    tracker_id: int | None,
) -> tuple[float, float]:
    if tracker_id is None or players.tracker_id is None:
        return np.nan, np.nan
    for i, tid in enumerate(players.tracker_id):
        if int(tid) == tracker_id and i < len(court_xy):
            x, y = court_xy[i]
            return float(x), float(y)
    return np.nan, np.nan
