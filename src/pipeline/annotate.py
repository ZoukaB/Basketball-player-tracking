"""Draw pipeline output back onto the processed frames.

Reads ``outputs/<video_name>/history`` (frames, SAM2 player detections, and the
possession / jump-shot / layup boxes saved during the run) and writes an
annotated MP4. No models run here, so a clip can be re-rendered as often as
needed once ``BasketballPipeline.run`` has filled the history directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from src.detection.object_detection import CLASS_NAMES, DetectionClass
from src.pipeline.history import FRAME_DIR, iter_history, load_history_meta
from src.pipeline.rosters import TEAM_COLORS

UNKNOWN_TEAM_COLOR = "#FFC300"   # tracks whose team never resolved
NEUTRAL_CLASS_COLOR = "#9E9E9E"  # classes that are never drawn
FALLBACK_FPS = 10

# Only these three classes are saved to states_history, so only these are drawn.
STATE_COLORS: dict[int, str] = {
    DetectionClass.PLAYER_IN_POSSESSION: "#FFD100",
    DetectionClass.PLAYER_JUMP_SHOT: "#FF4C4C",
    DetectionClass.PLAYER_LAYUP_DUNK: "#00C4FF",
}

SHOT_TYPE_LABELS = {"jump": "JUMP SHOT", "layup": "LAYUP / DUNK"}
OUTCOME_COLORS = {"made": "#007A33", "missed": "#850101"}

BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_SCALE = 1.0       # at BANNER_REFERENCE_WIDTH; scaled up on larger frames
BANNER_REFERENCE_WIDTH = 1280
BANNER_THICKNESS = 2
BANNER_MARGIN = 24


def render_annotated_video(
    history_dir: str | Path,
    output_path: str | Path,
    identity_df: Optional[pd.DataFrame] = None,
    shots_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Write masks, tracker IDs, state boxes, and shot banners to an MP4."""
    history_dir = Path(history_dir)
    if not history_dir.is_dir():
        raise FileNotFoundError(f"History directory not found: {history_dir}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = load_history_meta(history_dir)
    video_info = _video_info(meta, history_dir)

    player_palette = _player_palette(identity_df)
    mask_annotator = sv.MaskAnnotator(
        color=player_palette,
        color_lookup=sv.ColorLookup.TRACK,
        opacity=0.4,
    )
    player_box_annotator = sv.BoxAnnotator(
        color=player_palette,
        color_lookup=sv.ColorLookup.TRACK,
        thickness=2,
    )
    player_label_annotator = sv.LabelAnnotator(
        color=player_palette,
        color_lookup=sv.ColorLookup.TRACK,
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
    )
    state_palette = _state_palette()
    state_box_annotator = sv.BoxAnnotator(
        color=state_palette,
        color_lookup=sv.ColorLookup.CLASS,
        thickness=4,
    )
    state_label_annotator = sv.LabelAnnotator(
        color=state_palette,
        color_lookup=sv.ColorLookup.CLASS,
        text_color=sv.Color.BLACK,
        text_scale=0.6,
        text_thickness=2,
        text_padding=6,
        text_position=sv.Position.BOTTOM_LEFT,
    )

    labels_by_id = _player_labels(identity_df)
    shots_by_frame = _shots_by_frame(shots_df, labels_by_id)

    with sv.VideoSink(str(output_path), video_info=video_info) as sink:
        for frame_idx, frame, players, states in tqdm(
            iter_history(history_dir),
            desc="annotate",
            total=video_info.total_frames,
        ):
            annotated = frame.copy()

            if len(players) > 0 and players.tracker_id is not None:
                if players.mask is not None:
                    annotated = mask_annotator.annotate(annotated, players)
                annotated = player_box_annotator.annotate(annotated, players)
                annotated = player_label_annotator.annotate(
                    annotated,
                    players,
                    labels=[
                        labels_by_id.get(int(tid), f"#{int(tid)}")
                        for tid in players.tracker_id
                    ],
                )

            if len(states) > 0 and states.class_id is not None:
                annotated = state_box_annotator.annotate(annotated, states)
                annotated = state_label_annotator.annotate(
                    annotated,
                    states,
                    labels=_state_labels(states),
                )

            shot = shots_by_frame.get(frame_idx)
            if shot is not None:
                annotated = _draw_shot(annotated, players, shot)

            sink.write_frame(annotated)

    return output_path


def _video_info(meta: dict, history_dir: Path) -> sv.VideoInfo:
    """Size and rate of the processed frames, from meta.json when available."""
    n_frames = len(list((history_dir / FRAME_DIR).glob("*.jpg")))
    width, height = meta.get("width"), meta.get("height")
    if not width or not height:
        first = next(iter_history(history_dir), None)
        if first is None:
            raise RuntimeError(f"No history frames found in {history_dir}")
        height, width = first[1].shape[:2]
    fps = int(round(float(meta.get("fps") or FALLBACK_FPS)))
    return sv.VideoInfo(
        width=int(width),
        height=int(height),
        fps=max(fps, 1),
        total_frames=n_frames,
    )


def _player_palette(identity_df: Optional[pd.DataFrame]) -> sv.ColorPalette:
    """Palette indexed by tracker_id, so each track takes its team colour."""
    default = sv.Color.from_hex(UNKNOWN_TEAM_COLOR)
    teams: dict[int, str | None] = {}
    if identity_df is not None and len(identity_df) > 0:
        for row in identity_df.itertuples():
            team = getattr(row, "team", None)
            teams[int(row.tracker_id)] = team if isinstance(team, str) else None

    colors = [default] * (max(teams, default=0) + 1)
    for tracker_id, team in teams.items():
        hex_color = TEAM_COLORS.get(team) if team else None
        if hex_color:
            colors[tracker_id] = sv.Color.from_hex(hex_color)
    return sv.ColorPalette(colors=colors)


def _state_palette() -> sv.ColorPalette:
    """Palette indexed by class_id across every detector class."""
    colors = [sv.Color.from_hex(NEUTRAL_CLASS_COLOR) for _ in range(len(DetectionClass))]
    for class_id, hex_color in STATE_COLORS.items():
        colors[int(class_id)] = sv.Color.from_hex(hex_color)
    return sv.ColorPalette(colors=colors)


def _player_labels(identity_df: Optional[pd.DataFrame]) -> dict[int, str]:
    """``#7 11 Brunson`` when OCR resolved a number, else just ``#7``."""
    labels: dict[int, str] = {}
    if identity_df is None or len(identity_df) == 0:
        return labels
    for row in identity_df.itertuples():
        tracker_id = int(row.tracker_id)
        parts = [f"#{tracker_id}"]
        for field in ("number", "name"):
            value = getattr(row, field, None)
            if isinstance(value, str) and value:
                parts.append(value)
        labels[tracker_id] = " ".join(parts)
    return labels


def _state_labels(states: sv.Detections) -> list[str]:
    labels = []
    for i, class_id in enumerate(states.class_id):
        name = CLASS_NAMES.get(DetectionClass(int(class_id)), str(class_id))
        if states.confidence is not None:
            labels.append(f"{name} {float(states.confidence[i]):.2f}")
        else:
            labels.append(name)
    return labels


def _shots_by_frame(
    shots_df: Optional[pd.DataFrame],
    labels_by_id: dict[int, str],
) -> dict[int, dict]:
    """Expand each shot row into the frames it spans."""
    frames: dict[int, dict] = {}
    if shots_df is None or len(shots_df) == 0:
        return frames

    for row in shots_df.itertuples():
        start = _as_int(getattr(row, "start_frame", None))
        end = _as_int(getattr(row, "end_frame", None))
        if start is None:
            continue
        outcome = str(getattr(row, "outcome", "") or "").lower()
        shot_type = str(getattr(row, "shot_type", "") or "").lower()
        tracker_id = _as_int(getattr(row, "tracker_id", None))

        text = SHOT_TYPE_LABELS.get(shot_type, shot_type.upper() or "SHOT")
        if outcome:
            text = f"{text} - {outcome.upper()}"
        if tracker_id is not None:
            text = f"{text} - {labels_by_id.get(tracker_id, f'#{tracker_id}')}"

        overlay = {
            "text": text,
            "color": sv.Color.from_hex(OUTCOME_COLORS.get(outcome, UNKNOWN_TEAM_COLOR)),
            "tracker_id": tracker_id,
        }
        for frame_idx in range(start, (end if end is not None else start) + 1):
            frames[frame_idx] = overlay
    return frames


def _draw_shot(
    frame: np.ndarray,
    players: sv.Detections,
    shot: dict,
) -> np.ndarray:
    """Banner across the top plus a highlight on the shooter's box."""
    bgr = shot["color"].as_bgr()
    tracker_id = shot["tracker_id"]
    if tracker_id is not None and len(players) > 0 and players.tracker_id is not None:
        for det_idx, tid in enumerate(players.tracker_id):
            if int(tid) != tracker_id:
                continue
            x1, y1, x2, y2 = players.xyxy[det_idx].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)

    return _draw_banner(frame, shot["text"], bgr)


def _draw_banner(frame: np.ndarray, text: str, bgr: tuple[int, int, int]) -> np.ndarray:
    """Top-left caption, sized to the frame and outlined for legibility."""
    width = frame.shape[1]
    scale = BANNER_SCALE * max(width / BANNER_REFERENCE_WIDTH, 0.5)
    thickness = max(1, int(round(BANNER_THICKNESS * scale)))
    (text_w, text_h), _ = cv2.getTextSize(text, BANNER_FONT, scale, thickness)

    max_w = width - 2 * BANNER_MARGIN
    if text_w > max_w:
        scale *= max_w / text_w
        thickness = max(1, int(round(BANNER_THICKNESS * scale)))
        (text_w, text_h), _ = cv2.getTextSize(text, BANNER_FONT, scale, thickness)

    origin = (BANNER_MARGIN, BANNER_MARGIN + text_h + thickness)
    cv2.putText(frame, text, origin, BANNER_FONT, scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, BANNER_FONT, scale, bgr, thickness, cv2.LINE_AA)
    return frame


def _as_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
