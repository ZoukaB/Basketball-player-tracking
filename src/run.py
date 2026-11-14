"""Main pipeline: read a video once, run the bricks, optionally write outputs.

Shots are computed exactly as in collect_shots.py (detection + homography on
shot frames, per-crop teams, segment offense rule + dedup, basket-side team
rule), so both scripts give the same shots. SAM2 is used for one thing only:
identifying the shooter. Each shot's shooter box is linked to a SAM2 track, and
OCR reads gathered over the whole segment validate that track's jersey number.

The video may be an edited broadcast: at every hard cut (FrameCutDetector) the
current segment (= possession) is closed, SAM2 is reset and re-seeded on the next
frame with players, and the per-segment state (track teams, OCR) is cleared.
Track ids restart at each segment, so (segment, track_id) identifies a track.

`run_video(video_path, cfg, outputs)` returns the shot rows and only writes the
outputs present in the `outputs` dict:
    {"annotated_video", "tracks_csv", "shots_json", "shots_csv", "shot_map"} -> paths.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collect_shots import is_shot_relevant
from src.detection import Detector
from src.keypoints import KeypointBrick
from src.ocr import OCRBrick
from src.shots import (
    MATCH_IOU,
    SHOT_ROW_FIELDS,
    ShotBrick,
    apply_video_team_rule,
    close_segment_shots,
    draw_rows_shot_map,
    player_label,
    shot_row,
)
from src.teams import TeamBrick, resolve_team_mapping
from src.tracking import SAM2Tracker
from src.utils import FrameCutDetector, analysis_stride, load_config, resolve_path, setup_env

import supervision as sv

# Shot rows = collect_shots.py columns + the SAM2 identity of the shooter.
RUN_SHOT_FIELDS = SHOT_ROW_FIELDS + ["tracker_id", "number_track", "number_frame"]
# One row per SAM2 track per segment (identity only, no positions).
TRACK_FIELDS = ["video", "segment", "track_id", "team", "number", "name", "first_frame", "last_frame"]


def match_track(tracked: sv.Detections, box) -> Optional[int]:
    """Tracker id of the SAM2 track best overlapping `box` (IoU >= MATCH_IOU)."""
    if box is None or len(tracked) == 0 or tracked.tracker_id is None:
        return None
    ious = sv.box_iou_batch(tracked.xyxy, np.asarray(box, dtype=np.float32)[None, :]).reshape(-1)
    best = int(np.argmax(ious))
    return int(tracked.tracker_id[best]) if ious[best] >= MATCH_IOU else None


def run_video(
    video_path,
    cfg: dict,
    outputs: Optional[dict] = None,
    shared: Optional[dict] = None,
    max_seconds: float = 0.0,
    dedup_window_s: float = 0.5,
    dedup_dist_ft: float = 5.0,
) -> list[dict]:
    """Run the full pipeline on one video; return the shot rows (RUN_SHOT_FIELDS).

    If `shared` is given, its pre-built components are reused (models loaded once
    across videos); per-segment state is reset here.
    """
    video_path = Path(video_path)
    assert video_path.exists(), f"Missing video: {video_path}"

    info = sv.VideoInfo.from_video_path(str(video_path))
    stride = analysis_stride(cfg, info.fps)
    analysis_fps = info.fps / stride
    max_frames = int(round(max_seconds * info.fps / stride)) if max_seconds else 0

    render_video = bool(outputs and outputs.get("annotated_video"))
    render_keypoints = bool(cfg["render"].get("court_keypoints", True))

    print(f"Video: {video_path.name}")
    print(f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride} (analysis {analysis_fps:.1f} fps)")

    if shared:
        detector = shared["detector"]
        tracker = shared["tracker"]
        teams = shared["teams"]
        keypoints = shared["keypoints"]
        ocr = shared["ocr"]
    else:
        detector = Detector(cfg)
        tracker = SAM2Tracker(cfg)
        teams = TeamBrick(cfg)
        keypoints = KeypointBrick(cfg)
        ocr = OCRBrick(cfg)
    print(f"  OCR enabled: {ocr.enabled}")
    if not ocr.enabled:
        print("  [WARN] OCR is off: shooters get a track id but no jersey number (use --ocr)")

    cuts_cfg = cfg.get("cuts", {})
    cut_detector = (
        FrameCutDetector(cuts_cfg["z_meandiff_threshold"], cuts_cfg.get("min_gap_frames", 0))
        if cuts_cfg.get("split_segments", True) else None
    )

    hex_colors = list(cfg["teams"]["team_colors"].values())
    team_palette = sv.ColorPalette.from_hex(hex_colors)
    mask_annotator = sv.MaskAnnotator(color=team_palette, opacity=0.5, color_lookup=sv.ColorLookup.INDEX)
    box_annotator = sv.BoxAnnotator(color=team_palette, color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(
        color=team_palette, text_color=sv.Color.BLACK, color_lookup=sv.ColorLookup.INDEX
    )
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=5)

    # ------------------------------------------------------ per-segment state
    segment, segment_start = 0, 0
    shots = ShotBrick(cfg, analysis_fps=analysis_fps)
    numbers_by_frame: dict[int, list] = {}   # single-frame OCR reads on shot frames
    transformer = None
    seeded = False
    team_by_tid: dict[int, int] = {}
    seg_tracks: dict[int, list[int]] = {}    # tid -> [first video frame, last video frame]
    teams.reset_tracks()
    ocr.reset()

    shot_rows: list[dict] = []
    track_rows: list[dict] = []
    cut_frames: list[int] = []
    dropped_count = 0

    def close_segment() -> None:
        """Offense rule + dedup (as collect_shots), then shooter identity from SAM2 + OCR."""
        nonlocal dropped_count
        label = f"{video_path.name} [segment {segment}]"
        kept, dropped = close_segment_shots(
            shots, analysis_fps, dedup_window_s, dedup_dist_ft, label, teams.team_name)
        dropped_count += len(dropped)
        by_track = ocr.validated_by_track()

        n_linked = n_track_number = 0
        for shot in kept:
            number_frame = ocr.number_for_shot(shot, numbers_by_frame)
            tid = shot.get("tracker_id")
            number_track = by_track.get(int(tid)) if tid is not None else None
            number = number_track or number_frame
            team_id = shot.get("team")
            identity_team = team_id if team_id is not None else shot.get("offense_team")
            name = ocr.resolve_name(identity_team, number) if number else None
            row = shot_row(video_path.name, segment, segment_start, shot, teams.team_name,
                           number, player_label(number, name))
            row["tracker_id"] = "" if tid is None else int(tid)
            row["number_track"] = number_track or ""
            row["number_frame"] = number_frame or ""
            shot_rows.append(row)
            n_linked += tid is not None
            n_track_number += bool(number_track)

        for tid, (first, last) in sorted(seg_tracks.items()):
            team_id = team_by_tid.get(tid)
            number = by_track.get(tid)
            track_rows.append({
                "video": video_path.name,
                "segment": segment,
                "track_id": tid,
                "team": teams.team_name(team_id) if team_id is not None else "",
                "number": number or "",
                "name": (ocr.resolve_name(team_id, number) or "") if number else "",
                "first_frame": first,
                "last_frame": last,
            })
        print(f"  {label}: kept {len(kept)}, deduped {len(dropped)}, shooters linked to a track "
              f"{n_linked}/{len(kept)}, with a track number {n_track_number}/{len(kept)}, "
              f"tracks {len(seg_tracks)}")

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    video_info.fps = analysis_fps
    sink = None
    if render_video:
        out_path = Path(outputs["annotated_video"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sink = sv.VideoSink(str(out_path), video_info)
        sink.__enter__()

    timings = {k: [] for k in ("cuts", "detect", "keypoints", "ocr", "shots", "track", "teams")}
    kpts = None
    try:
        frame_generator = sv.get_video_frames_generator(source_path=str(video_path), stride=stride)
        for index, frame in enumerate(frame_generator):
            if max_frames and index >= max_frames:
                break
            video_frame = index * stride

            # ---------------------------------------------------- hard cut
            t0 = time.perf_counter()
            is_cut = cut_detector is not None and cut_detector.update(frame)[0]
            timings["cuts"].append((time.perf_counter() - t0) * 1000)
            if is_cut:
                close_segment()
                segment, segment_start = segment + 1, index
                shots = ShotBrick(cfg, analysis_fps=analysis_fps)
                numbers_by_frame = {}
                transformer = None
                seeded = False
                team_by_tid = {}
                seg_tracks = {}
                teams.reset_tracks()
                ocr.reset()
                cut_frames.append(video_frame)
                print(f"  [CUT] {video_path.name} analysis {index} (video {video_frame}) -> segment {segment}")

            # ------------------------------- shots (identical to collect_shots)
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            timings["detect"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            shot_relevant = is_shot_relevant(detections)
            if shot_relevant or transformer is None:
                kpts, mask = keypoints.landmarks(frame)
                transformer = keypoints.transformer_from(kpts, mask)
            timings["keypoints"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            if shot_relevant and ocr.enabled:
                number_dets = detections[detections.class_id == ocr.number_class_id]
                reads = ocr.read_number_boxes(frame, number_dets)
                if reads:
                    numbers_by_frame[index] = reads
            ocr_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            events = shots.update(frame, index, detections, transformer=transformer, team_brick=teams)
            timings["shots"].append((time.perf_counter() - t0) * 1000)

            # ------------------------------------- SAM2 (identity only)
            t0 = time.perf_counter()
            if not seeded:
                seed = detections[np.isin(detections.class_id, tracker.seed_class_ids)]
                if len(seed) and tracker.reprompt(frame, seed):
                    seeded = True
                    print(f"  analysis {index:4d} (video {video_frame:4d}): segment {segment} "
                          f"seeded {len(seed)} tracks")
            tracked = tracker.propagate(frame) if seeded else sv.Detections.empty()
            timings["track"].append((time.perf_counter() - t0) * 1000)

            # A shot that started on this frame: link its shooter box to a track.
            pending = shots.pending
            if (pending is not None and pending.get("start_frame") == index
                    and any(e["event"] == "START" for e in events)):
                pending["tracker_id"] = match_track(tracked, pending.get("shooter_box"))

            tracker_ids = np.asarray(tracked.tracker_id) if tracked.tracker_id is not None else np.array([], dtype=int)
            for tid in tracker_ids:
                span = seg_tracks.setdefault(int(tid), [video_frame, video_frame])
                span[1] = video_frame

            t0 = time.perf_counter()
            frame_teams = (
                teams.update(frame, tracked, tracker_ids=tracked.tracker_id, index=index)
                if len(tracked) else None
            )
            timings["teams"].append((time.perf_counter() - t0) * 1000)
            if frame_teams is not None:
                for tid, team in zip(tracker_ids, np.asarray(frame_teams)):
                    if int(team) >= 0:
                        team_by_tid[int(tid)] = int(team)

            t0 = time.perf_counter()
            if len(tracked) and ocr.should_run(index):
                number_dets = detections[detections.class_id == ocr.number_class_id]
                ocr.update(frame, number_dets, tracked)
            timings["ocr"].append(ocr_ms + (time.perf_counter() - t0) * 1000)

            # ---------------------------------------------------- render
            if sink is not None:
                annotated = frame.copy()
                if len(tracked):
                    color_idx = np.asarray(frame_teams) if frame_teams is not None else np.zeros(len(tracked), dtype=int)
                    color_idx = np.where(color_idx < 0, 0, color_idx)
                    by_track = ocr.validated_by_track()
                    labels = []
                    for tid in tracker_ids:
                        team_id = team_by_tid.get(int(tid))
                        team_txt = teams.team_name(team_id) if team_id is not None else "?"
                        number = by_track.get(int(tid))
                        if number:
                            roster_name = ocr.resolve_name(team_id, number)
                            labels.append(f"#{number} {roster_name or team_txt}")
                        else:
                            labels.append(f"ID {int(tid)} {team_txt}")
                    annotated = box_annotator.annotate(scene=annotated, detections=tracked, custom_color_lookup=color_idx)
                    annotated = mask_annotator.annotate(scene=annotated, detections=tracked, custom_color_lookup=color_idx)
                    annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels,
                                                         custom_color_lookup=color_idx)
                if render_keypoints and kpts is not None:
                    annotated = vertex_annotator.annotate(scene=annotated, key_points=kpts)
                sink.write_frame(annotated)

            if index % 30 == 0:
                print(f"  analysis {index:4d} (video {video_frame:4d}): segment={segment} "
                      f"tracks={len(tracked)} shots={len(shots.shots)}")
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)

    close_segment()
    apply_video_team_rule(shot_rows, cfg["shots"], video_path.name)

    if outputs:
        if outputs.get("shots_csv"):
            _write_csv(outputs["shots_csv"], RUN_SHOT_FIELDS, shot_rows)
        if outputs.get("shots_json"):
            path = Path(outputs["shots_json"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(shot_rows, handle, indent=2)
        if outputs.get("shot_map"):
            path = Path(outputs["shot_map"])
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), draw_rows_shot_map(shot_rows, keypoints.config, cfg["teams"]["team_colors"]))
        if outputs.get("tracks_csv"):
            _write_csv(outputs["tracks_csv"], TRACK_FIELDS, track_rows)

    print(f"  cuts: {len(cut_frames)} at video frames {cut_frames}")
    print(f"  segments: {segment + 1}, shots kept: {len(shot_rows)} (deduped {dropped_count})")
    if timings["detect"]:
        mean_ms = {k: round(float(np.mean(v)), 1) for k, v in timings.items() if v}
        print("  ms/frame:", mean_ms)
    return shot_rows


def _write_csv(path, fields: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the pipeline on one video")
    parser.add_argument("--video", default=None, help="Override config input.video")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Cap the video (0 = full)")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR (needed for jersey numbers)")
    parser.add_argument("--dedup-window-s", type=float, default=0.5, help="Duplicate window in seconds")
    parser.add_argument("--dedup-dist-ft", type=float, default=5.0, help="Duplicate court distance in feet")
    parser.add_argument("--ask-teams", action="store_true", help="Prompt to validate cluster->team mapping")
    parser.add_argument("--auto-map-color", action="store_true", help="Auto-map clusters by jersey colour")
    parser.add_argument("--team-mapping", default="outputs/team_mapping.json", help="Mapping file to load/save")
    args = parser.parse_args()

    setup_env()
    cfg = load_config()
    if args.ocr:
        cfg["ocr"]["enabled"] = True

    video_path = resolve_path(args.video or cfg["input"]["video"])
    out_cfg = cfg["output"]
    out_dir = resolve_path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build components once, fit teams offline on this video, validate the mapping.
    detector = Detector(cfg)
    keypoints = KeypointBrick(cfg)
    tracker = SAM2Tracker(cfg)
    teams = TeamBrick(cfg)
    ocr = OCRBrick(cfg)
    teams.fit_offline_from_videos(
        detector,
        [video_path],
        stride=int(cfg["teams"].get("stride", 30)),
        player_class_ids=cfg["classes"]["player_ids"],
        class_agnostic_nms=True,
    )
    resolve_team_mapping(
        teams,
        montage_path=out_dir / "team_clusters.jpg",
        mapping_path=resolve_path(args.team_mapping) if args.team_mapping else None,
        ask=args.ask_teams,
        auto_color=args.auto_map_color,
    )
    shared = {"detector": detector, "keypoints": keypoints, "tracker": tracker, "teams": teams, "ocr": ocr}

    outputs = {
        "annotated_video": str(resolve_path(out_cfg["annotated_video"])) if cfg["render"].get("annotated_video", True) else None,
        "tracks_csv": str(resolve_path(out_cfg["tracks_csv"])),
        "shots_json": str(resolve_path(out_cfg["shots_json"])),
        "shots_csv": str(resolve_path(out_cfg["shots_csv"])),
        "shot_map": str(resolve_path(out_cfg["shot_map"])),
    }
    rows = run_video(video_path, cfg, outputs, shared=shared, max_seconds=args.max_seconds,
                     dedup_window_s=args.dedup_window_s, dedup_dist_ft=args.dedup_dist_ft)

    print("\n--- summary ---")
    print(f"Shots: {len(rows)}")
    for r in rows:
        print(f"  seg{r['segment']} f{r['frame']} {r['shot_type']} {r['result']} team={r['team'] or '?'} "
              f"track={r['tracker_id'] if r['tracker_id'] != '' else '-'} "
              f"number track={r['number_track'] or '-'} frame={r['number_frame'] or '-'}")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    for key, path in outputs.items():
        if path:
            print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
