"""Annotated video output for ``BasketballPipeline`` (debug + demo).

Drawn on every processed frame:
- SAM2 masks + ellipse under the feet, coloured by team (grey = team unknown)
- label per track: ``#23 Name`` once OCR / roster resolved, else ``id 4``
- red ellipse on the player(s) in possession
- raw RF-DETR state boxes (jumpshot, layup...) with confidence, thin cyan
- banner: frame, segment, active event flags
- red border for a few frames after a hard cut
- minimap with the (unsmoothed) court positions of this frame
Untracked frames (replay / close-up after a cut) are written with a banner.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

# Palette index: 0 = team 0, 1 = team 1, 2 = unknown team
TEAM_PALETTE = sv.ColorPalette.from_hex(["#1D8CF8", "#FDB927", "#9E9E9E"])
UNKNOWN_TEAM = 2
POSSESSION_COLOR = sv.Color.from_hex("#FF3B30")
STATE_COLOR = sv.Color.from_hex("#00E5FF")

# Court frame used by CourtKeypointDetector.map_detections (NBA, feet).
# Adjust if your court coordinates use another origin / unit.
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0

CUT_FLASH_FRAMES = 5
EVENT_FLAGS = [
    ("possession", "POSSESSION"),
    ("jumpshot", "JUMP SHOT"),
    ("layup_dunk", "LAYUP/DUNK"),
    ("ball_in_basket", "BASKET"),
]


class PipelineRenderer:
    def __init__(
        self,
        show_masks: bool = True,
        show_states: bool = True,
        show_minimap: bool = True,
        minimap_width: int = 320,
    ) -> None:
        self.show_masks = show_masks
        self.show_states = show_states
        self.show_minimap = show_minimap
        self.minimap_width = minimap_width

        self.mask = sv.MaskAnnotator(color=TEAM_PALETTE, opacity=0.35)
        self.ellipse = sv.EllipseAnnotator(color=TEAM_PALETTE, thickness=2)
        self.possession = sv.EllipseAnnotator(
            color=POSSESSION_COLOR, thickness=5, color_lookup=sv.ColorLookup.INDEX
        )
        self.label = sv.LabelAnnotator(
            color=TEAM_PALETTE,
            text_color=sv.Color.BLACK,
            text_scale=0.5,
            text_padding=4,
            text_position=sv.Position.TOP_CENTER,
            smart_position=True,
        )
        self.state_box = sv.BoxAnnotator(
            color=STATE_COLOR, thickness=1, color_lookup=sv.ColorLookup.INDEX
        )
        self.state_label = sv.LabelAnnotator(
            color=STATE_COLOR,
            text_color=sv.Color.BLACK,
            text_scale=0.4,
            text_padding=2,
            text_position=sv.Position.BOTTOM_LEFT,
            color_lookup=sv.ColorLookup.INDEX,
        )
        self._frames_since_cut = 10**9

    # ------------------------------------------------------------------ API
    def notify_cut(self) -> None:
        self._frames_since_cut = 0

    def render(
        self,
        frame: np.ndarray,
        frame_idx: int,
        segment_id: int,
        players: sv.Detections,
        teams: np.ndarray | None,
        labels: list[str],
        possession_ids: list[int],
        state_dets: sv.Detections,
        court_xy: np.ndarray | None,
        event_row: dict,
    ) -> np.ndarray:
        scene = frame.copy()
        lookup = None
        if len(players) > 0:
            if teams is None:
                teams = np.full(len(players), -1)
            lookup = np.where(np.asarray(teams) < 0, UNKNOWN_TEAM, teams).astype(np.int64)
            if self.show_masks and players.mask is not None:
                scene = self.mask.annotate(scene, players, custom_color_lookup=lookup)
            scene = self.ellipse.annotate(scene, players, custom_color_lookup=lookup)
            if possession_ids and players.tracker_id is not None:
                has_ball = np.isin(players.tracker_id, np.asarray(possession_ids, dtype=int))
                if has_ball.any():
                    scene = self.possession.annotate(scene, players[has_ball])
            scene = self.label.annotate(
                scene, players, labels=labels, custom_color_lookup=lookup
            )

        if self.show_states and state_dets is not None and len(state_dets) > 0:
            scene = self.state_box.annotate(scene, state_dets)
            scene = self.state_label.annotate(
                scene, state_dets, labels=_state_labels(state_dets)
            )

        if self.show_minimap and court_xy is not None and lookup is not None:
            scene = self._minimap(scene, np.asarray(court_xy, dtype=float), lookup)

        scene = self._banner(scene, frame_idx, f"segment {segment_id}", event_row)
        return self._cut_overlay(scene)

    def render_untracked(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        scene = self._banner(frame.copy(), frame_idx, "NOT TRACKED (replay / close-up)", None)
        return self._cut_overlay(scene)

    # ------------------------------------------------------------ drawing
    def _banner(self, scene, frame_idx, title, event_row) -> np.ndarray:
        x0, y0, h = 10, 10, 30
        _panel(scene, x0, y0, 560, h + (h if event_row is not None else 0) + 10)
        _text(scene, f"frame {frame_idx} | {title}", (x0 + 10, y0 + 22), (255, 255, 255))
        if event_row is not None:
            x = x0 + 10
            for key, name in EVENT_FLAGS:
                active = bool(event_row.get(key, False))
                color = (48, 59, 255) if active else (110, 110, 110)   # BGR
                (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(scene, (x - 4, y0 + h + 4), (x + tw + 4, y0 + 2 * h), color, -1)
                _text(scene, name, (x, y0 + 2 * h - 8), (255, 255, 255))
                x += tw + 16
        return scene

    def _cut_overlay(self, scene: np.ndarray) -> np.ndarray:
        if self._frames_since_cut < CUT_FLASH_FRAMES:
            h, w = scene.shape[:2]
            cv2.rectangle(scene, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            _text(scene, "HARD CUT", (w - 160, 40), (0, 0, 255), scale=0.9, thickness=2)
        self._frames_since_cut += 1
        return scene

    def _minimap(self, scene, court_xy, lookup) -> np.ndarray:
        h, w = scene.shape[:2]
        mw = self.minimap_width
        mh = int(mw * COURT_WIDTH / COURT_LENGTH)
        pad = 10
        x0, y0 = w - mw - pad, h - mh - pad
        roi = scene[y0:y0 + mh, x0:x0 + mw]
        overlay = np.full_like(roi, (40, 70, 110))       # dark wood, BGR
        s = mw / COURT_LENGTH

        def pt(x, y):
            return int(round(x * s)), int(round(y * s))

        line = (230, 230, 230)
        cv2.rectangle(overlay, (0, 0), (mw - 1, mh - 1), line, 1)
        cv2.line(overlay, pt(47, 0), pt(47, 50), line, 1)
        cv2.circle(overlay, pt(47, 25), int(6 * s), line, 1)
        for hoop_x, key_x in ((5.25, 0), (88.75, 94 - 19)):
            cv2.circle(overlay, pt(hoop_x, 25), 3, line, -1)
            cv2.rectangle(overlay, pt(key_x, 17), pt(key_x + 19, 33), line, 1)
            cv2.ellipse(overlay, pt(hoop_x, 25), (int(23.75 * s),) * 2, 0,
                        -90 if hoop_x < 47 else 90, 90 if hoop_x < 47 else 270, line, 1)

        for (cx, cy), team in zip(court_xy[: len(lookup)], lookup):
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            if not (-5 <= cx <= COURT_LENGTH + 5 and -5 <= cy <= COURT_WIDTH + 5):
                continue   # homography outlier
            color = TEAM_PALETTE.by_idx(int(team)).as_bgr()
            cv2.circle(overlay, pt(cx, cy), 6, color, -1)
            cv2.circle(overlay, pt(cx, cy), 6, (0, 0, 0), 1)

        scene[y0:y0 + mh, x0:x0 + mw] = cv2.addWeighted(overlay, 0.85, roi, 0.15, 0)
        return scene


# ------------------------------------------------------------------ helpers
def _state_labels(dets: sv.Detections) -> list[str]:
    names = dets.data.get("class_name") if dets.data else None
    labels = []
    for i in range(len(dets)):
        if names is not None:
            name = str(names[i])
        else:
            name = _class_name(None if dets.class_id is None else dets.class_id[i])
        conf = "" if dets.confidence is None else f" {float(dets.confidence[i]):.2f}"
        labels.append(name + conf)
    return labels


def _class_name(class_id) -> str:
    if class_id is None:
        return "state"
    try:
        from src.detection import DetectionClass
        return DetectionClass(int(class_id)).name.lower()
    except Exception:
        return str(int(class_id))


def _panel(scene, x, y, w, h, alpha: float = 0.6) -> None:
    roi = scene[y:y + h, x:x + w]
    scene[y:y + h, x:x + w] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def _text(scene, text, org, color, scale: float = 0.55, thickness: int = 1) -> None:
    cv2.putText(scene, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
