"""Loop the full pipeline (src/run.py) over every video in a directory.

Loads models once and reuses them across clips (per-clip state is reset inside
run_video). Teams are fitted offline across all videos (notebook way). OCR is
off by default; enable with --ocr.

Mandatory outputs: a concatenated shots.csv and a combined shot_map.png.
Per-video outputs are off unless flags are given.

Usage (from repo root):
    .venv/Scripts/python.exe src/run_all.py --video-dir "<dir>" --analysis-fps 30 --ocr
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
from src.run import run_video  # noqa: E402
from src.shots import basket_side, basket_xy, dedup_clip_shots  # noqa: E402
from src.teams import TeamBrick, resolve_team_mapping  # noqa: E402
from src.tracking import SAM2Tracker  # noqa: E402
from src.utils import analysis_stride, load_config, resolve_path, setup_env  # noqa: E402

import cv2  # noqa: E402
import supervision as sv  # noqa: E402
from sports import MeasurementUnit  # noqa: E402
from sports.basketball import (  # noqa: E402
    CourtConfiguration,
    League,
    draw_court,
    draw_made_and_miss_on_court,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov"}
SHOT_FIELDS = ["video", "frame", "shot_type", "result", "team", "offense_team",
               "attacking_basket", "court_x", "court_y", "distance_ft", "player_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the pipeline over a directory of videos")
    p.add_argument("--video-dir", default=None, help="Directory of videos (default: config input.video_dir)")
    p.add_argument("--limit", type=int, default=0, help="Process N videos (0 = all)")
    p.add_argument("--videos", default="", help="Comma-separated name substrings to select specific videos")
    p.add_argument("--shots-csv", default="outputs/shots_all.csv", help="Concatenated shots CSV (mandatory)")
    p.add_argument("--shot-map", default="outputs/shot_map_all.png", help="Combined shot map output")
    p.add_argument("--out-dir", default=None, help="If set, per-video outputs go to <out-dir>/<video stem>/")
    p.add_argument("--save-annotated", action="store_true", help="Write per-video annotated.mp4")
    p.add_argument("--save-tracks", action="store_true", help="Write per-video tracks.csv")
    p.add_argument("--save-shots-json", action="store_true", help="Write per-video shots.json")
    p.add_argument("--save-per-video-map", action="store_true", help="Write per-video shot_map.png")
    p.add_argument("--analysis-fps", type=float, default=0.0, help="Override config analysis.fps (0 = config)")
    p.add_argument("--ocr", action="store_true", help="Enable OCR (runs every clip)")
    p.add_argument("--fit-stride", type=int, default=30, help="Stride for the offline team fit")
    p.add_argument("--fit-seconds", type=float, default=0.0, help="Cap per video for the team fit (0 = full)")
    p.add_argument("--dedup-window-s", type=float, default=0.5, help="Duplicate window in seconds")
    p.add_argument("--dedup-dist-ft", type=float, default=5.0, help="Duplicate court distance in feet")
    return p.parse_args()


def list_videos(video_dir: str | Path, limit: int) -> list[Path]:
    files = [p for p in Path(video_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS]
    files.sort(key=lambda p: p.name)
    return files[:limit] if limit else files


def main() -> None:
    args = parse_args()
    setup_env()
    cfg = load_config()
    if args.analysis_fps and args.analysis_fps > 0:
        cfg["analysis"]["fps"] = args.analysis_fps
    if args.ocr:
        cfg["ocr"]["enabled"] = True

    video_dir = args.video_dir or cfg["input"]["video_dir"]
    videos = list_videos(video_dir, args.limit)
    if args.videos:
        subs = [s.strip() for s in args.videos.split(",") if s.strip()]
        videos = [p for p in videos if any(s in p.name for s in subs)]
    assert videos, f"No videos found in {video_dir}"

    print(f"Video dir: {video_dir}")
    print(f"Videos: {len(videos)}  analysis fps={cfg['analysis']['fps']}  ocr={cfg['ocr']['enabled']}")

    # Build heavy components once and reuse across clips.
    detector = Detector(cfg)
    keypoints = KeypointBrick(cfg)
    tracker = SAM2Tracker(cfg)
    teams = TeamBrick(cfg)
    ocr = OCRBrick(cfg)
    shared = {"detector": detector, "keypoints": keypoints, "tracker": tracker, "teams": teams, "ocr": ocr}

    # Offline team fit across all videos (notebook way).
    n_crops = teams.fit_offline_from_videos(
        detector,
        videos,
        stride=args.fit_stride,
        player_class_ids=cfg["classes"]["player_ids"],
        class_agnostic_nms=True,
        max_seconds=args.fit_seconds,
    )
    print(f"Team fit: {n_crops} crops, fitted={teams.fitted}")

    rows: list[dict] = []
    dropped_total = 0
    for video_path in videos:
        info = sv.VideoInfo.from_video_path(str(video_path))
        stride = analysis_stride(cfg, info.fps)
        analysis_fps = info.fps / stride

        outputs = None
        if args.out_dir and (args.save_annotated or args.save_tracks or args.save_shots_json or args.save_per_video_map):
            sub = resolve_path(args.out_dir) / video_path.stem
            sub.mkdir(parents=True, exist_ok=True)
            outputs = {
                "annotated_video": str(sub / "annotated.mp4") if args.save_annotated else None,
                "tracks_csv": str(sub / "tracks.csv") if args.save_tracks else None,
                "shots_json": str(sub / "shots.json") if args.save_shots_json else None,
                "shot_map": str(sub / "shot_map.png") if args.save_per_video_map else None,
            }

        shot_list = run_video(video_path, cfg, outputs, shared=shared)

        window_frames = max(1, int(round(analysis_fps * args.dedup_window_s)))
        kept, dropped = dedup_clip_shots(shot_list, window_frames, args.dedup_dist_ft)
        dropped_total += len(dropped)
        for shot in dropped:
            print(f"  [DEDUP-DROP] {video_path.name} f{shot['frame']} {shot['outcome']} {shot['type']}")

        for shot in kept:
            team_id = shot.get("team")
            offense_id = shot.get("offense_team")
            attacking = shot.get("attacking_basket")
            cx, cy = shot.get("court_x"), shot.get("court_y")
            if cx is None:
                distance = None
            else:
                side = attacking or basket_side(cx)
                bx, by = basket_xy(side)
                distance = float(np.hypot(cx - bx, cy - by))
            rows.append(
                {
                    "video": video_path.name,
                    "frame": int(shot["frame"]) * stride,
                    "shot_type": shot.get("type", ""),
                    "result": shot.get("outcome", ""),
                    "team": cfg["teams"]["team_names"].get(team_id, "") if team_id is not None else "",
                    "offense_team": cfg["teams"]["team_names"].get(offense_id, "") if offense_id is not None else "",
                    "attacking_basket": attacking or "",
                    "court_x": "" if cx is None else round(cx, 2),
                    "court_y": "" if cy is None else round(cy, 2),
                    "distance_ft": "" if distance is None else round(distance, 2),
                    "player_id": shot.get("player_id", ""),
                }
            )
        print(f"  {video_path.name}: kept {len(kept)}, deduped {len(dropped)}")

    shots_csv = resolve_path(args.shots_csv)
    shots_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(shots_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHOT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Combined shot map across all videos.
    kcfg = cfg["keypoints"]
    court_cfg = CourtConfiguration(
        league=getattr(League, kcfg.get("league", "NBA")),
        measurement_unit=getattr(MeasurementUnit, kcfg.get("measurement_unit", "FEET")),
    )
    court = draw_court(config=court_cfg)
    team_names = cfg["teams"]["team_names"]
    team_colors = cfg["teams"]["team_colors"]
    for team_id, hex_color in enumerate(team_colors.values()):
        name = team_names.get(team_id)
        made = [[float(r["court_x"]), float(r["court_y"])] for r in rows
                if r["team"] == name and r["result"] == "made" and r["court_x"] != ""]
        missed = [[float(r["court_x"]), float(r["court_y"])] for r in rows
                  if r["team"] == name and r["result"] == "missed" and r["court_x"] != ""]
        if not made and not missed:
            continue
        court = draw_made_and_miss_on_court(
            config=court_cfg,
            made_xy=np.array(made, dtype=np.float32) if made else None,
            miss_xy=np.array(missed, dtype=np.float32) if missed else None,
            made_color=sv.Color.from_hex(hex_color),
            miss_color=sv.Color.from_hex(hex_color),
            made_size=18,
            miss_size=18,
            line_thickness=4,
            court=court,
        )
    shot_map = resolve_path(args.shot_map)
    shot_map.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(shot_map), court)

    print("\n--- summary ---")
    print(f"Videos: {len(videos)}")
    print(f"Shots kept: {len(rows)}  (deduped: {dropped_total})")
    print(f"Wrote {shots_csv}")
    print(f"Wrote {shot_map}")


if __name__ == "__main__":
    main()
