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
import fnmatch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection import Detector  # noqa: E402
from src.keypoints import KeypointBrick  # noqa: E402
from src.ocr import OCRBrick  # noqa: E402
from src.run import RUN_SHOT_FIELDS, run_video  # noqa: E402
from src.shots import draw_rows_shot_map  # noqa: E402
from src.teams import TeamBrick, resolve_team_mapping  # noqa: E402
from src.tracking import SAM2Tracker  # noqa: E402
from src.utils import load_config, resolve_path, setup_env  # noqa: E402

import cv2  # noqa: E402
from sports import MeasurementUnit  # noqa: E402
from sports.basketball import CourtConfiguration, League  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov"}


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
    p.add_argument("--ask-teams", action="store_true", help="Prompt to validate cluster->team mapping")
    p.add_argument("--auto-map-color", action="store_true", help="Auto-map clusters by jersey colour")
    p.add_argument("--team-mapping", default="outputs/team_mapping.json", help="Mapping file to load/save")
    return p.parse_args()


def list_videos(video_dir: str | Path, limit: int, exclude: list[str] | None = None) -> list[Path]:
    exclude = exclude or []
    files = [p for p in Path(video_dir).iterdir()
             if p.suffix.lower() in VIDEO_EXTS and not any(fnmatch.fnmatch(p.name, pat) for pat in exclude)]
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
    # input.exclude only applies to the default listing: --videos picks explicitly.
    exclude = None if args.videos else cfg["input"].get("exclude")
    videos = list_videos(video_dir, args.limit, exclude=exclude)
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

    resolve_team_mapping(
        teams,
        montage_path=Path(args.shots_csv).parent / "team_clusters.jpg",
        mapping_path=resolve_path(args.team_mapping) if args.team_mapping else None,
        ask=args.ask_teams,
        auto_color=args.auto_map_color,
    )

    rows: list[dict] = []
    for video_path in videos:
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

        # Segments, offense rule, dedup and basket-side teams are applied inside run_video.
        rows.extend(run_video(video_path, cfg, outputs, shared=shared,
                              dedup_window_s=args.dedup_window_s, dedup_dist_ft=args.dedup_dist_ft))

    shots_csv = resolve_path(args.shots_csv)
    shots_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(shots_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_SHOT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Combined shot map across all videos.
    kcfg = cfg["keypoints"]
    court_cfg = CourtConfiguration(
        league=getattr(League, kcfg.get("league", "NBA")),
        measurement_unit=getattr(MeasurementUnit, kcfg.get("measurement_unit", "FEET")),
    )
    shot_map = resolve_path(args.shot_map)
    shot_map.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(shot_map), draw_rows_shot_map(rows, court_cfg, cfg["teams"]["team_colors"]))

    print("\n--- summary ---")
    print(f"Videos: {len(videos)}")
    print(f"Shots kept: {len(rows)}")
    print(f"Wrote {shots_csv}")
    print(f"Wrote {shot_map}")


if __name__ == "__main__":
    main()
