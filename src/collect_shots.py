"""Offline shot collection across possession videos.

1) Fit teams the notebook way: sample crops across the videos (stride 30,
   class_agnostic_nms=True, scale 0.4) and fit once.
2) For each video, detect shots (detection + keypoints/homography; no SAM2) and
   aggregate to outputs/shots.csv + outputs/shot_map.png.

A video may hold several possessions (e.g. clips joined into one file): with
`cuts.split_segments`, it is split at hard cuts and every segment is processed
like a one-possession clip (fresh ShotBrick + homography, offense rule and dedup
per segment). Rows carry `segment` and `segment_frame`.

Usage (from repo root):
    .venv/Scripts/python.exe src/collect_shots.py
    .venv/Scripts/python.exe src/collect_shots.py --limit 3 --max-seconds 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector  # noqa: E402
from src.keypoints import KeypointBrick  # noqa: E402
from src.ocr import OCRBrick  # noqa: E402
from src.shots import (  # noqa: E402
    SHOT_ROW_FIELDS,
    ShotBrick,
    apply_video_team_rule,
    close_segment_shots,
    draw_rows_shot_map,
    player_label,
    shot_row,
)
from src.teams import TeamBrick, resolve_team_mapping  # noqa: E402
from src.utils import (  # noqa: E402
    FrameCutDetector,
    analysis_stride,
    list_videos,
    load_config,
    resolve_path,
    setup_env,
)

import cv2  # noqa: E402
import supervision as sv  # noqa: E402

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline shot collection")
    p.add_argument("--video-dir", default=None, help="Override config input.video_dir")
    p.add_argument("--limit", type=int, default=3, help="Process N videos (smallest first)")
    p.add_argument("--max-seconds", type=float, default=10.0, help="Cap per video (0 = full)")
    p.add_argument("--fit-seconds", type=float, default=0.0,
                   help="Cap per video for the team fit (0 = whole video; a 20 s cap only saw "
                        "the first possessions of a joined video)")
    p.add_argument("--fit-stride", type=int, default=30, help="Stride for the team fit")
    p.add_argument("--out-dir", default=None, help="Write shots.csv + shot_map.png here")
    p.add_argument("--videos", default="", help="Comma-separated name substrings to select specific videos")
    p.add_argument("--dedup-window-s", type=float, default=0.5, help="Duplicate window in seconds")
    p.add_argument("--dedup-dist-ft", type=float, default=5.0, help="Duplicate court distance in feet")
    p.add_argument("--ocr", action="store_true", help="Read jersey numbers to identify the shooter")
    p.add_argument("--ask-teams", action="store_true", help="Prompt to validate cluster->team mapping")
    p.add_argument("--auto-map-color", action="store_true", help="Auto-map clusters by jersey colour")
    p.add_argument("--team-mapping", default="outputs/team_mapping.json", help="Mapping file to load/save")
    return p.parse_args()


def is_shot_relevant(detections: sv.Detections) -> bool:
    """Frames with jump shot (5), layup (6) or ball-in-basket (1): only these
    recompute keypoints/homography and run the single-frame OCR."""
    return bool(np.any(np.isin(detections.class_id, [1, 5, 6]))) if len(detections) else False


def close_segment(
    video_name: str,
    segment: int,
    segment_start: int,
    shots: ShotBrick,
    numbers_by_frame: dict[int, list],
    teams: TeamBrick,
    ocr: OCRBrick,
    analysis_fps: float,
    dedup_window_s: float,
    dedup_dist_ft: float,
) -> tuple[list[dict], int]:
    """End of one possession (clip, or segment between hard cuts): offense rule,
    dedup, shot rows. Returns (rows, number of deduped shots)."""
    label = f"{video_name} [segment {segment}]"
    kept_shots, dropped_shots = close_segment_shots(
        shots, analysis_fps, dedup_window_s, dedup_dist_ft, label, teams.team_name)

    rows = []
    for shot in kept_shots:
        # Shooter number from OCR (matched to the shooter box on the shot frame).
        number = ocr.number_for_shot(shot, numbers_by_frame)
        team_id = shot.get("team")
        identity_team = team_id if team_id is not None else shot.get("offense_team")
        name = ocr.resolve_name(identity_team, number) if number else None
        rows.append(shot_row(video_name, segment, segment_start, shot, teams.team_name,
                             number, player_label(number, name)))
    print(f"  {label}: kept {len(kept_shots)}, deduped {len(dropped_shots)}")
    return rows, len(dropped_shots)


def collect_video(
    video_path: Path,
    cfg: dict,
    detector: Detector,
    teams: TeamBrick,
    keypoints: KeypointBrick,
    ocr: OCRBrick,
    max_seconds: float = 0.0,
    dedup_window_s: float = 0.5,
    dedup_dist_ft: float = 5.0,
) -> tuple[list[dict], int]:
    """Shots of one video (detection + homography, no SAM2), split at hard cuts.
    Returns (rows, number of deduped shots)."""
    cuts_cfg = cfg.get("cuts", {})
    split_segments = bool(cuts_cfg.get("split_segments", True))
    info = sv.VideoInfo.from_video_path(str(video_path))
    stride = analysis_stride(cfg, info.fps)
    analysis_fps = info.fps / stride
    max_frames = int(round(max_seconds * info.fps / stride)) if max_seconds else 0
    cut_detector = (
        FrameCutDetector(cuts_cfg["z_meandiff_threshold"], cuts_cfg.get("min_gap_frames", 0))
        if split_segments else None
    )

    rows: list[dict] = []
    dropped_count = 0
    # Per-possession state, reset at every hard cut.
    segment, segment_start = 0, 0
    shots = ShotBrick(cfg, analysis_fps=analysis_fps)
    numbers_by_frame: dict[int, list] = {}
    transformer = None

    generator = sv.get_video_frames_generator(source_path=str(video_path), stride=stride)
    for index, frame in enumerate(generator):
        if max_frames and index >= max_frames:
            break
        if cut_detector is not None and cut_detector.update(frame)[0]:
            # Hard cut = new possession: close the segment, start fresh.
            seg_rows, seg_dropped = close_segment(video_path.name, segment, segment_start, shots,
                                                  numbers_by_frame, teams, ocr, analysis_fps,
                                                  dedup_window_s, dedup_dist_ft)
            rows.extend(seg_rows)
            dropped_count += seg_dropped
            segment, segment_start = segment + 1, index
            shots = ShotBrick(cfg, analysis_fps=analysis_fps)
            numbers_by_frame = {}
            transformer = None
            print(f"  [CUT] {video_path.name} frame {index} -> segment {segment}")
        detections = detector.detect(frame)
        # Only recompute keypoints/homography on shot-relevant frames; else reuse the last one.
        shot_relevant = is_shot_relevant(detections)
        if shot_relevant or transformer is None:
            kpts, mask = keypoints.landmarks(frame)
            transformer = keypoints.transformer_from(kpts, mask)
        # OCR number reads on shot frames (no SAM2 needed).
        if shot_relevant and ocr.enabled:
            number_dets = detections[detections.class_id == ocr.number_class_id]
            reads = ocr.read_number_boxes(frame, number_dets)
            if reads:
                numbers_by_frame[index] = reads
        shots.update(frame, index, detections, transformer=transformer, team_brick=teams)

    seg_rows, seg_dropped = close_segment(video_path.name, segment, segment_start, shots,
                                          numbers_by_frame, teams, ocr, analysis_fps,
                                          dedup_window_s, dedup_dist_ft)
    rows.extend(seg_rows)
    dropped_count += seg_dropped
    apply_video_team_rule(rows, cfg["shots"], video_path.name)
    return rows, dropped_count


def main() -> None:
    args = parse_args()
    setup_env()
    cfg = load_config()
    if args.ocr:
        cfg["ocr"]["enabled"] = True

    video_dir = args.video_dir or cfg["input"]["video_dir"]
    # input.exclude only applies to the default listing: --videos picks explicitly.
    exclude = None if args.videos else cfg["input"].get("exclude")
    videos = list_videos(video_dir, args.limit, exclude=exclude)
    if args.videos:
        subs = [s.strip() for s in args.videos.split(",") if s.strip()]
        videos = [p for p in videos if any(s in p.name for s in subs)]
    assert videos, f"No videos found in {video_dir}"

    out_dir = resolve_path(args.out_dir) if args.out_dir else resolve_path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(cfg)
    teams = TeamBrick(cfg)
    keypoints = KeypointBrick(cfg)
    ocr = OCRBrick(cfg)

    print(f"Video dir: {video_dir}")
    print(f"Selected {len(videos)} videos (smallest first): {[v.name for v in videos]}")
    print(f"OCR enabled: {ocr.enabled}")

    n_crops = teams.fit_offline_from_videos(
        detector,
        videos,
        stride=args.fit_stride,
        player_class_ids=cfg["classes"]["player_ids"],
        class_agnostic_nms=True,
        max_seconds=args.fit_seconds,
    )
    print(f"Team fit: {n_crops} crops, fitted={teams.fitted}")

    resolve_team_mapping(
        teams,
        montage_path=out_dir / "team_clusters.jpg",
        mapping_path=resolve_path(args.team_mapping) if args.team_mapping else None,
        ask=args.ask_teams,
        auto_color=args.auto_map_color,
    )

    rows: list[dict] = []
    dropped_count = 0
    for video_path in videos:
        video_rows, video_dropped = collect_video(
            video_path, cfg, detector, teams, keypoints, ocr,
            max_seconds=args.max_seconds,
            dedup_window_s=args.dedup_window_s,
            dedup_dist_ft=args.dedup_dist_ft,
        )
        rows.extend(video_rows)
        dropped_count += video_dropped

    shots_csv = out_dir / "shots.csv"
    with open(shots_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHOT_ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregated shot map (made O / missed X), coloured by offense team.
    shot_map = out_dir / "shot_map.png"
    cv2.imwrite(str(shot_map), draw_rows_shot_map(rows, keypoints.config, cfg["teams"]["team_colors"]))

    print("\n--- summary ---")
    print(f"Videos: {len(videos)}")
    print(f"Shots kept: {len(rows)}  (dropped total: {dropped_count})")
    for r in rows:
        print(f"  {r['video']} seg{r['segment']} f{r['frame']} {r['shot_type']} {r['result']} "
              f"team={r['team'] or '?'} offense={r['offense_team'] or '?'} "
              f"basket={r['attacking_basket'] or '?'} loc=({r['court_x']},{r['court_y']}) dist={r['distance_ft']}")
    print(f"Wrote {shots_csv}")
    print(f"Wrote {shot_map}")


if __name__ == "__main__":
    main()
