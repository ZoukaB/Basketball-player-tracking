"""Shot detection brick.

Notebook cells reused: 91 (ShotEventTracker.update with per-frame
has_jump_shot / has_layup_dunk / has_ball_in_basket), 87/89 (jump-shot class +
draw_made_and_miss_on_court).

Shot location = the shooter's bottom-center projected onto the court.
Timings are built from the *analysis* fps so the windows stay in seconds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv
from sports.basketball import (
    ShotEventTracker,
    draw_court,
    draw_made_and_miss_on_court,
)

LEFT_BASKET = (5.25, 25.0)
RIGHT_BASKET = (88.75, 25.0)


def basket_side(court_x: float) -> str:
    return "left" if court_x < 47.0 else "right"


def basket_xy(side: str) -> tuple[float, float]:
    return LEFT_BASKET if side == "left" else RIGHT_BASKET


class ShotBrick:
    def __init__(self, cfg: dict, analysis_fps: float) -> None:
        scfg = cfg["shots"]
        scale = float(analysis_fps)
        self.tracker = ShotEventTracker(
            reset_time_frames=int(round(scale * scfg.get("reset_time_frames_s", 1.7))),
            minimum_frames_between_starts=int(round(scale * scfg.get("min_frames_between_starts_s", 0.5))),
            cooldown_frames_after_made=int(round(scale * scfg.get("cooldown_frames_after_made_s", 0.5))),
        )
        classes = cfg["classes"]
        self.ball_in_basket_id = int(classes["ball_in_basket"])
        self.jump_shot_id = int(classes["player_jump_shot"])
        self.layup_dunk_id = int(classes["player_layup_dunk"])
        self.shot_class_ids = {self.jump_shot_id, self.layup_dunk_id}

        # Offense guard: the team with the player-in-possession is on offense, and
        # only the offense may shoot (one basket per possession).
        self.enforce_offense_team = bool(scfg.get("enforce_offense_team", True))
        self.possession_class_id = int(scfg.get("possession_class_id", 4))
        self.current_offense_team: Optional[int] = None
        self.current_basket: Optional[str] = None

        self.shots: list[dict] = []
        self.dropped: list[dict] = []
        self._pending: Optional[dict] = None

    # ------------------------------------------------------------------- main
    def update(
        self,
        frame: np.ndarray,
        index: int,
        detections: sv.Detections,
        transformer=None,
        team_brick=None,
        tracker_ids: Optional[np.ndarray] = None,
    ) -> list:
        """Feed one analysis frame; returns ShotEventRecord list."""
        self._update_offense(frame, index, detections, transformer, team_brick)

        has_jump = bool(np.any(detections.class_id == self.jump_shot_id)) if len(detections) else False
        has_layup = bool(np.any(detections.class_id == self.layup_dunk_id)) if len(detections) else False
        has_basket = bool(np.any(detections.class_id == self.ball_in_basket_id)) if len(detections) else False

        events = self.tracker.update(index, has_jump, has_layup, has_basket)

        for event in events:
            if event["event"] == "START":
                self._pending = self._capture_shooter(
                    frame, index, event["type"], detections, transformer, team_brick, tracker_ids
                )
            else:
                self._finalize(index, event)

        return events

    def _update_offense(self, frame, index, detections, transformer, team_brick) -> None:
        """Track the current offense team from the player-in-possession detection."""
        if team_brick is None or detections is None or len(detections) == 0:
            return
        poss = detections[detections.class_id == self.possession_class_id]
        if len(poss) == 0:
            return

        teams = team_brick.update(frame, poss, index=index)
        if teams is not None and len(teams) == len(poss):
            if poss.confidence is not None:
                best = int(np.argmax(poss.confidence))
            else:
                best = 0
            if int(teams[best]) >= 0:
                self.current_offense_team = int(teams[best])

        if transformer is not None:
            point = poss.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            court_xy = np.asarray(transformer.transform_points(points=np.asarray(point)))
            if len(court_xy):
                j = int(np.argmax(poss.confidence)) if poss.confidence is not None else 0
                self.current_basket = basket_side(float(court_xy[j, 0]))

    # --------------------------------------------------------------- internal
    def _capture_shooter(
        self, frame, index, shot_type, detections, transformer, team_brick, tracker_ids
    ) -> dict:
        class_id = self.jump_shot_id if shot_type == "JUMP" else self.layup_dunk_id
        pending = {"start_frame": index, "type": shot_type, "court_x": None, "court_y": None,
                   "team": None, "tracker_id": None,
                   "offense_team": self.current_offense_team,
                   "attacking_basket": self.current_basket}
        subset = detections[np.isin(detections.class_id, [class_id])]
        if len(subset) == 0:
            subset = detections[np.isin(detections.class_id, list(self.shot_class_ids))]
        if len(subset) == 0:
            return pending

        best = int(np.argmax(subset.confidence)) if subset.confidence is not None else 0

        if transformer is not None:
            point = subset.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[best : best + 1]
            court_xy = transformer.transform_points(points=np.asarray(point))
            pending["court_x"] = float(court_xy[0, 0])
            pending["court_y"] = float(court_xy[0, 1])

        if tracker_ids is not None and len(tracker_ids) == len(detections):
            ious = sv.box_iou_batch(detections.xyxy, subset.xyxy[best : best + 1]).reshape(-1)
            match = int(np.argmax(ious))
            if ious[match] >= 0.3:
                pending["tracker_id"] = int(tracker_ids[match])

        if team_brick is not None:
            teams = team_brick.update(frame, subset, index=index)
            if teams is not None and len(teams) > best and int(teams[best]) >= 0:
                pending["team"] = int(teams[best])

        return pending

    def _finalize(self, index: int, event) -> None:
        record = dict(self._pending or {"start_frame": None, "type": event["type"],
                                        "court_x": None, "court_y": None, "team": None,
                                        "tracker_id": None, "offense_team": None,
                                        "attacking_basket": None})
        record.update(
            {
                "frame": index,
                "made": event["event"] == "MADE",
                "outcome": "made" if event["event"] == "MADE" else "missed",
                "type": event["type"],
            }
        )
        shot_basket = basket_side(record["court_x"]) if record["court_x"] is not None else None
        record["shot_basket"] = shot_basket
        self.shots.append(record)

        if event["event"] in {"MADE", "MISSED"}:
            self._pending = None

    # ---------------------------------------------------------------- outputs
    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.shots, handle, indent=2)

    def draw_shot_map(self, config, teams_cfg: dict) -> np.ndarray:
        """Half/full court with made (o) / missed (x) per team."""
        court = draw_court(config=config)
        team_names = list(teams_cfg.get("team_names", {}).values())
        team_colors = list(teams_cfg.get("team_colors", {}).values())
        for team_id, hex_color in enumerate(team_colors):
            made = np.array([[s["court_x"], s["court_y"]] for s in self.shots
                             if s["team"] == team_id and s["made"] and s["court_x"] is not None], dtype=np.float32)
            missed = np.array([[s["court_x"], s["court_y"]] for s in self.shots
                               if s["team"] == team_id and not s["made"] and s["court_x"] is not None], dtype=np.float32)
            if len(made) == 0 and len(missed) == 0:
                continue
            court = draw_made_and_miss_on_court(
                config=config,
                made_xy=made if len(made) else None,
                miss_xy=missed if len(missed) else None,
                made_color=sv.Color.from_hex(hex_color),
                miss_color=sv.Color.from_hex(hex_color),
                made_size=18,
                miss_size=18,
                line_thickness=4,
                court=court,
            )
        return court
