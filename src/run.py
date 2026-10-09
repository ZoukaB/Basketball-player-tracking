"""Main pipeline: read the video once, run the bricks, write outputs.

Stable baseline: continuous shot, seed SAM2 once from the first frame, then
track. No cut handling / re-prompting (kept dormant). OCR is disabled by default.

Outputs (paths from config):
    outputs/annotated.mp4   masks/boxes colored by team, track IDs, court keypoints
    outputs/tracks.csv      frame, track_id, team, x1,y1,x2,y2, court_x, court_y
    outputs/shots.json
    outputs/shot_map.png
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector
from src.keypoints import KeypointBrick
from src.ocr import OCRBrick
from src.shots import ShotBrick
from src.teams import TeamBrick
from src.tracking import SAM2Tracker
from src.utils import analysis_stride, load_config, resolve_path, setup_env

import supervision as sv

SCALE, PADDING = 20, 50


def main() -> None:
    setup_env()
    cfg = load_config()

    video_path = resolve_path(cfg["input"]["video"])
    assert video_path.exists(), f"Missing video: {video_path}"
    out_cfg = cfg["output"]
    out_dir = resolve_path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = resolve_path(out_cfg["annotated_video"])
    tracks_path = resolve_path(out_cfg["tracks_csv"])
    shots_path = resolve_path(out_cfg["shots_json"])
    shot_map_path = resolve_path(out_cfg["shot_map"])
    render_video = bool(cfg["render"].get("annotated_video", True))
    render_keypoints = bool(cfg["render"].get("court_keypoints", True))

    info = sv.VideoInfo.from_video_path(str(video_path))
    stride = analysis_stride(cfg, info.fps)
    analysis_fps = info.fps / stride

    print(f"Video: {video_path}")
    print(f"  {info.width}x{info.height} @ {info.fps:.1f} fps, stride={stride} (analysis {analysis_fps:.1f} fps)")

    detector = Detector(cfg)
    tracker = SAM2Tracker(cfg)
    teams = TeamBrick(cfg)
    keypoints = KeypointBrick(cfg)
    shots = ShotBrick(cfg, analysis_fps=analysis_fps)
    ocr = OCRBrick(cfg)
    print(f"  OCR enabled: {ocr.enabled}")

    # Team colors, one per cluster index (order = cfg teams.team_colors order).
    hex_colors = list(cfg["teams"]["team_colors"].values())
    team_palette = sv.ColorPalette.from_hex(hex_colors)
    mask_annotator = sv.MaskAnnotator(color=team_palette, opacity=0.5, color_lookup=sv.ColorLookup.INDEX)
    box_annotator = sv.BoxAnnotator(color=team_palette, color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(
        color=team_palette, text_color=sv.Color.BLACK, color_lookup=sv.ColorLookup.INDEX
    )
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=5)

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    video_info.fps = analysis_fps

    track_rows: list[dict] = []
    timings = {"detect": [], "track": [], "keypoints": [], "teams": [], "shots": [], "ocr": []}
    seeded = False
    identity: dict[int, dict] = {}
    team_by_tid: dict[int, int] = {}

    sink = sv.VideoSink(str(annotated_path), video_info) if render_video else None
    if sink is not None:
        sink.__enter__()

    try:
        frame_generator = sv.get_video_frames_generator(source_path=str(video_path), stride=stride)
        for index, frame in enumerate(frame_generator):
            video_frame = index * stride

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            timings["detect"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            kpts, mask = keypoints.landmarks(frame)
            transformer = keypoints.transformer_from(kpts, mask)
            timings["keypoints"].append((time.perf_counter() - t0) * 1000)

            if not seeded:
                seed = detections[np.isin(detections.class_id, tracker.seed_class_ids)]
                if len(seed) == 0:
                    print(f"  analysis {index:4d}: no players to seed, waiting")
                    continue
                if tracker.prompt_first_frame(frame, seed):
                    seeded = True
                    print(f"  analysis {index:4d} (video {video_frame:4d}): seeded {len(seed)} tracks")

            t0 = time.perf_counter()
            tracked = tracker.propagate(frame)
            timings["track"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            frame_teams = (
                teams.update(frame, tracked, tracker_ids=tracked.tracker_id, index=index)
                if len(tracked)
                else None
            )
            timings["teams"].append((time.perf_counter() - t0) * 1000)
            if frame_teams is not None:
                for tid, team in zip(np.asarray(tracked.tracker_id), np.asarray(frame_teams)):
                    if int(team) >= 0:
                        team_by_tid[int(tid)] = int(team)

            t0 = time.perf_counter()
            if ocr.should_run(index):
                number_dets = detections[detections.class_id == ocr.number_class_id]
                ocr.update(frame, number_dets, tracked)
                for tid, number in ocr.validated_numbers.items():
                    identity[tid] = {"number": number, "team": team_by_tid.get(tid)}
            timings["ocr"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            shots.update(
                frame,
                index,
                detections,
                transformer=transformer,
                team_brick=teams,
                tracker_ids=tracked.tracker_id,
            )
            timings["shots"].append((time.perf_counter() - t0) * 1000)

            # --- tracks rows + court coords ---
            tracker_ids = np.asarray(tracked.tracker_id) if tracked.tracker_id is not None else np.array([])
            court_xy = (
                keypoints.transform_points(transformer, tracked.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER))
                if transformer is not None and len(tracked)
                else np.full((len(tracked), 2), np.nan)
            )
            for i, tid in enumerate(tracker_ids):
                team_id = team_by_tid.get(int(tid))
                track_rows.append(
                    {
                        "frame": video_frame,
                        "track_id": int(tid),
                        "team": teams.team_name(team_id) if team_id is not None else "",
                        "x1": float(tracked.xyxy[i][0]),
                        "y1": float(tracked.xyxy[i][1]),
                        "x2": float(tracked.xyxy[i][2]),
                        "y2": float(tracked.xyxy[i][3]),
                        "court_x": float(court_xy[i][0]),
                        "court_y": float(court_xy[i][1]),
                    }
                )

            # --- annotated video ---
            if sink is not None and len(tracked):
                color_idx = (
                    np.asarray(frame_teams)
                    if frame_teams is not None
                    else np.zeros(len(tracked), dtype=int)
                )
                color_idx = np.where(color_idx < 0, 0, color_idx)
                labels = []
                for tid in tracker_ids:
                    team_id = team_by_tid.get(int(tid))
                    name = teams.team_name(team_id) if team_id is not None else "?"
                    info_txt = identity.get(int(tid))
                    if info_txt and info_txt.get("number"):
                        roster_name = ocr.resolve_name(info_txt.get("team"), info_txt["number"])
                        labels.append(f"#{info_txt['number']} {roster_name or name}")
                    else:
                        labels.append(f"ID {int(tid)} {name}")

                annotated = box_annotator.annotate(scene=frame.copy(), detections=tracked, custom_color_lookup=color_idx)
                annotated = mask_annotator.annotate(scene=annotated, detections=tracked, custom_color_lookup=color_idx)
                annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels, custom_color_lookup=color_idx)
                if render_keypoints:
                    annotated = vertex_annotator.annotate(scene=annotated, key_points=kpts)
                sink.write_frame(annotated)
            elif sink is not None:
                sink.write_frame(frame.copy())

            if index % 10 == 0:
                print(f"  analysis {index:4d} (video {video_frame:4d}): tracks={len(tracked)} "
                      f"teams_fitted={teams.fitted}")
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)

    shots.to_json(shots_path)
    court = shots.draw_shot_map(keypoints.config, cfg["teams"])
    cv2.imwrite(str(shot_map_path), court)

    with open(tracks_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "track_id", "team", "x1", "y1", "x2", "y2", "court_x", "court_y"],
        )
        writer.writeheader()
        writer.writerows(track_rows)

    print("\n--- summary ---")
    print(f"Frames processed: {len(timings['detect'])}")
    print(f"Tracks rows: {len(track_rows)}")
    print(f"Shots recorded: {len(shots.shots)}")
    print("Per-brick ms/frame (mean):")
    for name, values in timings.items():
        if values:
            print(f"  {name:9s}: {np.mean(values):8.2f}  (n={len(values)})")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Wrote {annotated_path if render_video else '(annotated video disabled)'}")
    print(f"Wrote {tracks_path}")
    print(f"Wrote {shots_path}")
    print(f"Wrote {shot_map_path}")


if __name__ == "__main__":
    main()
