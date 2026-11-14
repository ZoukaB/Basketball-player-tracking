"""Shot detection brick.

Notebook cells reused: 91 (ShotEventTracker.update with per-frame
has_jump_shot / has_layup_dunk / has_ball_in_basket), 87/89 (jump-shot class +
draw_made_and_miss_on_court).

Shot location = the shooter's bottom-center projected onto the court.
When `tracked` is passed, the shooter/offense team is read from the player's
SAM2 track (majority team), otherwise it falls back to per-crop. collect_shots.py
and run.py both use per-crop (no `tracked`) so their shots match; run.py uses
SAM2 only to identify the shooter (track -> jersey number).
Timings are built from the *analysis* fps so the windows stay in seconds.
"""
from __future__ import annotations

import json
from collections import Counter
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
MATCH_IOU = 0.3


def basket_side(court_x: float) -> str:
    return "left" if court_x < 47.0 else "right"


def basket_xy(side: str) -> tuple[float, float]:
    return LEFT_BASKET if side == "left" else RIGHT_BASKET


def shot_distance(shot: dict) -> Optional[float]:
    """Feet from the shot location to the attacked basket (nearest basket if unknown)."""
    cx, cy = shot.get("court_x"), shot.get("court_y")
    if cx is None or cy is None:
        return None
    side = shot.get("attacking_basket") or basket_side(cx)
    bx, by = basket_xy(side)
    return float(np.hypot(cx - bx, cy - by))


def shot_priority(shot: dict):
    """Dedup priority: shooter team matches offense, then made, then earliest."""
    offense = shot.get("offense_team")
    team = shot.get("team")
    off_match = 1 if (offense is not None and team is not None and team == offense) else 0
    made = 1 if shot.get("made") else 0
    return (off_match, made, -int(shot["frame"]))


def basket_side_mapping(rows: list[dict], min_share: float = 0.6, min_shots: int = 3):
    """Learn which team attacks which basket from the shots' own team guesses.

    Within a half each team attacks one basket, so the per-shot (crop) team
    votes, grouped by the side the shot was taken on, give a basket -> team map
    even when single guesses are noisy. Rows need `court_x` and `team` (names).
    Returns (mapping {"left"/"right": team} or None, votes per side, reason):
    None when a side has < min_shots, a majority < min_share, or both sides
    point to the same team (e.g. a video spanning halftime).
    """
    votes = {"left": Counter(), "right": Counter()}
    for row in rows:
        x, team = row.get("court_x"), row.get("team")
        if x in ("", None) or not team:
            continue
        votes[basket_side(float(x))][team] += 1
    mapping = {}
    for side, counter in votes.items():
        n = sum(counter.values())
        if n < min_shots:
            return None, votes, f"{side} basket: only {n} shots"
        team, k = counter.most_common(1)[0]
        if k / n < min_share:
            return None, votes, f"{side} basket: majority {k}/{n} < {min_share}"
        mapping[side] = team
    if mapping["left"] == mapping["right"]:
        return None, votes, f"both baskets -> {mapping['left']}"
    return mapping, votes, "ok"


def apply_basket_side_teams(rows: list[dict], mapping: dict[str, str]) -> int:
    """team = offense_team = team attacking the shot's basket; the previous
    (crop) team is kept as shooter_team. Rows without court_x are unchanged.
    Returns the number of rows whose team changed."""
    changed = 0
    for row in rows:
        row.setdefault("shooter_team", row.get("team", ""))
        if row.get("court_x") in ("", None):
            continue
        team = mapping[basket_side(float(row["court_x"]))]
        changed += team != row.get("team")
        row["team"] = row["offense_team"] = team
    return changed


def _is_duplicate(a: dict, b: dict, window_frames: int, dist_ft: float) -> bool:
    if abs(int(a["frame"]) - int(b["frame"])) > window_frames:
        return False
    ax, ay = a.get("court_x"), a.get("court_y")
    bx, by = b.get("court_x"), b.get("court_y")
    if ax is None or ay is None or bx is None or by is None:
        return True  # time-only fallback
    return float(np.hypot(ax - bx, ay - by)) <= dist_ft


def dedup_clip_shots(shots: list[dict], window_frames: int, dist_ft: float):
    """Group near-simultaneous shots; keep one per group (prefer offense team, then made)."""
    groups: list[list[dict]] = []
    for shot in sorted(shots, key=lambda s: int(s["frame"])):
        for group in groups:
            if any(_is_duplicate(shot, other, window_frames, dist_ft) for other in group):
                group.append(shot)
                break
        else:
            groups.append([shot])

    kept: list[dict] = []
    dropped: list[dict] = []
    for group in groups:
        best = max(group, key=shot_priority)
        kept.append(best)
        dropped.extend([s for s in group if s is not best])
    return kept, dropped


# ---------------------------------------------------------------------------
# Segment / video close-out, shared by collect_shots.py and run.py so both give
# the same shots. A segment is one possession (a clip, or the frames between
# two hard cuts).

SHOT_ROW_FIELDS = ["video", "segment", "frame", "segment_frame", "shot_type", "result", "team",
                   "shooter_team", "number", "offense_team", "attacking_basket", "court_x", "court_y",
                   "distance_ft", "player_id"]


def close_segment_shots(shots: "ShotBrick", analysis_fps: float, dedup_window_s: float,
                        dedup_dist_ft: float, label: str, team_name) -> tuple[list[dict], list[dict]]:
    """End of one segment: offense rule (team = possession majority), then dedup.

    `team_name(team_id) -> str`. Returns (kept, dropped) shot dicts.
    """
    clip_team = shots.apply_clip_offense()
    print(f"  {label}: offense (possession majority) = "
          f"{team_name(clip_team) if clip_team is not None else '?'} "
          f"votes={ {team_name(t): n for t, n in shots.possession_votes.items()} }")

    # De-duplicate: drop only near-simultaneous events (same shot detected twice).
    window_frames = max(1, int(round(analysis_fps * dedup_window_s)))
    kept, dropped = dedup_clip_shots(shots.shots, window_frames, dedup_dist_ft)
    for shot in dropped:
        print(f"  [DEDUP-DROP] {label} f{shot['frame']} {shot['outcome']} "
              f"{shot['type']} loc=({shot.get('court_x')},{shot.get('court_y')}) "
              f"team={shot.get('team')} offense={shot.get('offense_team')}")
    return kept, dropped


def shot_row(video_name: str, segment: int, segment_start: int, shot: dict, team_name,
             number: Optional[str] = None, player_id: str = "") -> dict:
    """One output row (SHOT_ROW_FIELDS, minus shooter_team: set by apply_video_team_rule)."""
    team_id = shot.get("team")
    offense_id = shot.get("offense_team")
    cx, cy = shot.get("court_x"), shot.get("court_y")
    distance = shot_distance(shot)
    return {
        "video": video_name,
        "segment": segment,
        "frame": shot["frame"],
        "segment_frame": shot["frame"] - segment_start,
        "shot_type": shot.get("type", ""),
        "result": shot.get("outcome", ""),
        "team": team_name(team_id) if team_id is not None else "",
        "number": number or "",
        "offense_team": team_name(offense_id) if offense_id is not None else "",
        "attacking_basket": shot.get("attacking_basket") or "",
        "court_x": "" if cx is None else round(cx, 2),
        "court_y": "" if cy is None else round(cy, 2),
        "distance_ft": "" if distance is None else round(distance, 2),
        "player_id": player_id,
    }


def player_label(number: Optional[str], name: Optional[str]) -> str:
    if number and name:
        return f"{number} {name}"
    return str(number) if number else ""


def apply_video_team_rule(video_rows: list[dict], shots_cfg: dict, video_name: str) -> None:
    """After all segments of a video: team = team attacking the shot's basket,
    learned over the whole video (shots.team_by_basket_side)."""
    for r in video_rows:
        r.setdefault("shooter_team", r["team"])
    if not shots_cfg.get("team_by_basket_side", False):
        return
    mapping, votes, reason = basket_side_mapping(
        video_rows, min_share=float(shots_cfg.get("basket_side_min_share", 0.6)),
        min_shots=int(shots_cfg.get("basket_side_min_shots", 3)))
    summary = {side: dict(c) for side, c in votes.items()}
    if mapping:
        changed = apply_basket_side_teams(video_rows, mapping)
        print(f"  {video_name}: team by basket side {mapping} (votes {summary}), "
              f"{changed} shot team(s) changed")
    else:
        print(f"  [WARN] {video_name}: basket-side teams not applied ({reason}; "
              f"votes {summary}); keeping per-shot teams")


def draw_rows_shot_map(rows: list[dict], court_config, team_colors: dict) -> np.ndarray:
    """Full court with made (o) / missed (x) per offense team, plus the shooter's
    number/name next to each shot when known. `team_colors`: {team name: hex}."""
    import cv2

    court = draw_court(config=court_config)
    for name, hex_color in team_colors.items():
        made = [[float(r["court_x"]), float(r["court_y"])] for r in rows
                if r["offense_team"] == name and r["result"] == "made" and r["court_x"] != ""]
        missed = [[float(r["court_x"]), float(r["court_y"])] for r in rows
                  if r["offense_team"] == name and r["result"] == "missed" and r["court_x"] != ""]
        if not made and not missed:
            continue
        court = draw_made_and_miss_on_court(
            config=court_config,
            made_xy=np.array(made, dtype=np.float32) if made else None,
            miss_xy=np.array(missed, dtype=np.float32) if missed else None,
            made_color=sv.Color.from_hex(hex_color),
            miss_color=sv.Color.from_hex(hex_color),
            made_size=18,
            miss_size=18,
            line_thickness=4,
            court=court,
        )

    scale, padding = 20, 50
    for r in rows:
        if r["court_x"] == "" or not (r["number"] or r["player_id"]):
            continue
        px = int(round(float(r["court_x"]) * scale + padding))
        py = int(round(float(r["court_y"]) * scale + padding))
        text = str(r["number"] or r["player_id"])
        cv2.putText(court, text, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(court, text, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return court


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
        self.ball_id = int(classes["ball"])
        self.rim_id = int(classes["rim"])
        self.ball_in_basket_id = int(classes["ball_in_basket"])
        self.jump_shot_id = int(classes["player_jump_shot"])
        self.layup_dunk_id = int(classes["player_layup_dunk"])
        self.shot_class_ids = {self.jump_shot_id, self.layup_dunk_id}

        # A tracker restart (MISSED + START in one frame) this soon after the
        # previous start and this close on court is the same shot whose shot-class
        # detection flickered (e.g. f92-98, f102-105, f107-109): merge, no phantom miss.
        self.restart_merge_frames = int(round(scale * scfg.get("restart_merge_s", 1.0)))
        self.restart_merge_ft = float(scfg.get("restart_merge_ft", 5.0))

        # Make gate: a basket only counts if the ball was seen above a rim
        # shortly before (a real make goes over then into the rim). Without it,
        # ball-in-basket boxes near the rim during rebounds/drives gave false makes.
        self.make_require_ball_above = bool(scfg.get("make_require_ball_above", False))
        self.make_ball_above_frames = int(round(scale * scfg.get("make_ball_above_window_s", 1.0)))
        self._last_ball_above: Optional[int] = None

        # Rebound filter: a layup starting within rebound_window_s after a missed
        # shot was decided, that ends missed without the ball ever above a rim, is
        # a player going up for the rebound, not a shot (Q1_part1: 6 false starts,
        # no real shot lost). Putbacks that score have the ball above the rim.
        self.drop_rebound_layups = bool(scfg.get("drop_rebound_layups", False))
        self.rebound_window_frames = int(round(scale * scfg.get("rebound_window_s", 1.0)))
        self._last_decision: Optional[tuple[int, bool]] = None  # (frame, made)
        self.dropped_rebounds: list[dict] = []

        self.enforce_offense_team = bool(scfg.get("enforce_offense_team", True))
        self.possession_class_id = int(scfg.get("possession_class_id", 4))
        self.current_offense_team: Optional[int] = None
        # One team vote per frame with a player-in-possession box (whole clip).
        self.possession_votes: Counter = Counter()
        self.current_basket: Optional[str] = None

        self.shots: list[dict] = []
        self._pending: Optional[dict] = None

    @property
    def pending(self) -> Optional[dict]:
        """The shot in progress (started, not yet decided), if any."""
        return self._pending

    # ------------------------------------------------------------------- main
    def update(
        self,
        frame: np.ndarray,
        index: int,
        detections: sv.Detections,
        transformer=None,
        team_brick=None,
        tracker_ids: Optional[np.ndarray] = None,
        tracked: Optional[sv.Detections] = None,
    ) -> list:
        """Feed one analysis frame; returns ShotEventRecord list."""
        self._update_offense(frame, index, detections, transformer, team_brick, tracked)

        has_jump = bool(np.any(detections.class_id == self.jump_shot_id)) if len(detections) else False
        has_layup = bool(np.any(detections.class_id == self.layup_dunk_id)) if len(detections) else False
        has_basket = bool(np.any(detections.class_id == self.ball_in_basket_id)) if len(detections) else False
        if self.make_require_ball_above or self.drop_rebound_layups:
            if self._ball_above_rim(detections):
                self._last_ball_above = index
        if self.make_require_ball_above:
            recent = self._last_ball_above is not None and index - self._last_ball_above <= self.make_ball_above_frames
            has_basket = has_basket and recent

        events = self.tracker.update(index, has_jump, has_layup, has_basket)

        # Restart: the tracker closed the shot in progress as MISSED because a new
        # START fired; both arrive in this update, MISSED first.
        if len(events) == 2 and events[0]["event"] == "MISSED" and events[1]["event"] == "START":
            capture = self._capture_shooter(
                frame, index, events[1]["type"], detections, transformer, team_brick, tracked
            )
            previous = self._pending
            if previous is not None and self._same_shot(previous, capture):
                # Same shot: drop the phantom miss and keep the FIRST start's
                # capture (where the shot began; a later box can be the defender).
                previous["restart_frame"] = index
                events = events[1:]
            else:
                self._finalize(index, events[0])
                self._pending = capture
            return events

        for event in events:
            if event["event"] == "START":
                self._pending = self._capture_shooter(
                    frame, index, event["type"], detections, transformer, team_brick, tracked
                )
            else:
                self._finalize(index, event)

        return events

    def _ball_above_rim(self, detections: sv.Detections) -> bool:
        """A ball box horizontally over a rim, its bottom above the rim's middle."""
        if detections is None or len(detections) == 0:
            return False
        balls = detections.xyxy[detections.class_id == self.ball_id]
        rims = detections.xyxy[detections.class_id == self.rim_id]
        for b in balls:
            cx = 0.5 * (b[0] + b[2])
            for r in rims:
                if r[0] <= cx <= r[2] and b[3] <= r[1] + 0.5 * (r[3] - r[1]):
                    return True
        return False

    def _same_shot(self, previous: dict, capture: dict) -> bool:
        """Restart within restart_merge_s of the previous start and <= restart_merge_ft away."""
        start = previous.get("start_frame")
        if start is None or capture["start_frame"] - start > self.restart_merge_frames:
            return False
        px, py = previous.get("court_x"), previous.get("court_y")
        cx, cy = capture.get("court_x"), capture.get("court_y")
        if None in (px, py, cx, cy):
            return True  # no location: time-only, like dedup_clip_shots
        return float(np.hypot(px - cx, py - cy)) <= self.restart_merge_ft

    # ------------------------------------------------------- track helpers
    @staticmethod
    def _team_from_track(team_brick, tracked, box: np.ndarray):
        """Return (tracker_id, team_id) of the tracked player best matching `box`."""
        if team_brick is None or tracked is None or len(tracked) == 0 or tracked.tracker_id is None:
            return None, None
        ious = sv.box_iou_batch(tracked.xyxy, np.asarray(box, dtype=np.float32)[None, :]).reshape(-1)
        best = int(np.argmax(ious))
        if ious[best] < MATCH_IOU:
            return None, None
        tid = int(tracked.tracker_id[best])
        return tid, team_brick.team_of(tid)

    def _update_offense(self, frame, index, detections, transformer, team_brick, tracked) -> None:
        """Track the current offense team from the player-in-possession detection."""
        if team_brick is None or detections is None or len(detections) == 0:
            return
        poss = detections[detections.class_id == self.possession_class_id]
        if len(poss) == 0:
            return

        best = int(np.argmax(poss.confidence)) if poss.confidence is not None else 0
        _, team = self._team_from_track(team_brick, tracked, poss.xyxy[best])
        if team is None:
            teams = team_brick.update(frame, poss, index=index)
            if teams is not None and len(teams) == len(poss) and int(teams[best]) >= 0:
                team = int(teams[best])
        if team is not None:
            self.current_offense_team = int(team)
            self.possession_votes[int(team)] += 1

        if transformer is not None:
            point = poss.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            court_xy = np.asarray(transformer.transform_points(points=np.asarray(point)))
            if len(court_xy):
                self.current_basket = basket_side(float(court_xy[best, 0]))

    # --------------------------------------------------------------- internal
    def _capture_shooter(self, frame, index, shot_type, detections, transformer, team_brick, tracked) -> dict:
        class_id = self.jump_shot_id if shot_type == "JUMP" else self.layup_dunk_id
        pending = {"start_frame": index, "type": shot_type, "court_x": None, "court_y": None,
                   "team": None, "tracker_id": None, "shooter_box": None,
                   "offense_team": self.current_offense_team,
                   "attacking_basket": self.current_basket}
        subset = detections[np.isin(detections.class_id, [class_id])]
        if len(subset) == 0:
            subset = detections[np.isin(detections.class_id, list(self.shot_class_ids))]
        if len(subset) == 0:
            return pending

        best = int(np.argmax(subset.confidence)) if subset.confidence is not None else 0
        pending["shooter_box"] = [float(v) for v in subset.xyxy[best]]

        if transformer is not None:
            point = subset.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[best : best + 1]
            court_xy = transformer.transform_points(points=np.asarray(point))
            pending["court_x"] = float(court_xy[0, 0])
            pending["court_y"] = float(court_xy[0, 1])

        # Prefer team from the shooter's SAM2 track.
        tid, team = self._team_from_track(team_brick, tracked, subset.xyxy[best])
        if tid is not None:
            pending["tracker_id"] = tid
        if team is not None:
            pending["team"] = int(team)
        elif team_brick is not None:
            teams = team_brick.update(frame, subset, index=index)
            if teams is not None and len(teams) > best and int(teams[best]) >= 0:
                pending["team"] = int(teams[best])

        return pending

    def _finalize(self, index: int, event) -> None:
        record = dict(self._pending or {"start_frame": None, "type": event["type"],
                                        "court_x": None, "court_y": None, "team": None,
                                        "tracker_id": None, "offense_team": None,
                                        "shooter_box": None, "attacking_basket": None})
        record.update(
            {
                "frame": index,
                "made": event["event"] == "MADE",
                "outcome": "made" if event["event"] == "MADE" else "missed",
                "type": event["type"],
            }
        )
        record["shot_basket"] = basket_side(record["court_x"]) if record["court_x"] is not None else None

        previous = self._last_decision
        self._last_decision = (index, record["made"])
        if self._is_rebound_layup(record, previous):
            self.dropped_rebounds.append(record)
        else:
            self.shots.append(record)

        if event["event"] in {"MADE", "MISSED"}:
            self._pending = None

    def _is_rebound_layup(self, record: dict, previous: Optional[tuple[int, bool]]) -> bool:
        """Missed layup starting right after a missed shot, ball never above a rim."""
        start = record.get("start_frame")
        if not self.drop_rebound_layups or record["made"] or record["type"] != "LAYUP" or start is None:
            return False
        if previous is None or previous[1] or start - previous[0] > self.rebound_window_frames:
            return False
        ball_above = self._last_ball_above is not None and self._last_ball_above >= start
        return not ball_above

    def apply_clip_offense(self) -> Optional[int]:
        """Call once at the end of a clip: one team per shot, = the offense team.

        A clip is one possession, so with `enforce_offense_team` every shot gets
        the clip's possession-majority team as both `team` and `offense_team`.
        The per-crop shooter guess is kept in `shooter_team` (a layup box often
        holds the defender too). Without possession votes, the shooter guess is
        used for both. Returns the clip's offense team (or None).
        """
        if not self.enforce_offense_team:
            return None
        clip_team = self.possession_votes.most_common(1)[0][0] if self.possession_votes else None
        for shot in self.shots:
            shot["shooter_team"] = shot.get("team")
            team = clip_team if clip_team is not None else shot.get("team")
            shot["team"] = shot["offense_team"] = team
        return clip_team

    # ---------------------------------------------------------------- outputs
    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.shots, handle, indent=2)

    def draw_shot_map(self, config, teams_cfg: dict) -> np.ndarray:
        """Half/full court with made (o) / missed (x), coloured by offense team."""
        court = draw_court(config=config)
        team_colors = list(teams_cfg.get("team_colors", {}).values())
        for team_id, hex_color in enumerate(team_colors):
            made = np.array([[s["court_x"], s["court_y"]] for s in self.shots
                             if s.get("offense_team") == team_id and s["made"] and s["court_x"] is not None],
                            dtype=np.float32)
            missed = np.array([[s["court_x"], s["court_y"]] for s in self.shots
                               if s.get("offense_team") == team_id and not s["made"] and s["court_x"] is not None],
                              dtype=np.float32)
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
