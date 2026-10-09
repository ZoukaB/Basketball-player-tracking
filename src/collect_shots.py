"""Offline shot collection across possession videos.

1) Fit teams the notebook way: sample crops across the videos (stride 30,
   class_agnostic_nms=True, scale 0.4) and fit once.
2) For each video, detect shots (detection + keypoints/homography; no SAM2) and
   aggregate to outputs/shots.csv + outputs/shot_map.png.

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
from src.shots import ShotBrick, basket_xy  # noqa: E402
from src.teams import TeamBrick  # noqa: E402
from src.utils import analysis_stride, load_config, resolve_path, setup_env  # noqa: E402

import cv2  # noqa: E402
import supervision as sv  # noqa: E402
from sports.basketball import draw_court, draw_made_and_miss_on_court  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov"}
LEFT_BASKET = (5.25, 25.0)
RIGHT_BASKET = (88.75, 25.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline shot collection")
    p.add_argument("--video-dir", default=None, help="Override config input.video_dir")
    p.add_argument("--limit", type=int, default=3, help="Process N videos (smallest first)")
    p.add_argument("--max-seconds", type=float, default=10.0, help="Cap per video (0 = full)")
    p.add_argument("--fit-seconds", type=float, default=20.0, help="Cap per video for the team fit")
    p.add_argument("--fit-stride", type=int, default=30, help="Stride for the team fit")
    p.add_argument("--out-dir", default=None, help="Write shots.csv + shot_map.png here")
    p.add_argument("--videos", default="", help="Comma-separated name substrings to select specific videos")
    return p.parse_args()


def list_videos(video_dir: str | Path, limit: int | None) -> list[Path]:
    files = [p for p in Path(video_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS]
    files.sort(key=lambda p: p.stat().st_size)
    return files[:limit] if limit else files


def basket_for(court_x: float) -> tuple[float, float]:
    return LEFT_BASKET if court_x < 47.0 else RIGHT_BASKET


def main() -> None:
    args = parse_args()
    setup_env()
    cfg = load_config()

    video_dir = args.video_dir or cfg["input"]["video_dir"]
    videos = list_videos(video_dir, args.limit)
    if args.videos:
        subs = [s.strip() for s in args.videos.split(",") if s.strip()]
        videos = [p for p in videos if any(s in p.name for s in subs)]
    assert videos, f"No videos found in {video_dir}"

    detector = Detector(cfg)
    teams = TeamBrick(cfg)
    keypoints = KeypointBrick(cfg)

    print(f"Video dir: {video_dir}")
    print(f"Selected {len(videos)} videos (smallest first): {[v.name for v in videos]}")

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
    dropped_count = 0
    for video_path in videos:
        info = sv.VideoInfo.from_video_path(str(video_path))
        stride = analysis_stride(cfg, info.fps)
        analysis_fps = info.fps / stride
        shots = ShotBrick(cfg, analysis_fps=analysis_fps)
        max_frames = int(round(args.max_seconds * info.fps / stride)) if args.max_seconds else 0

        generator = sv.get_video_frames_generator(source_path=str(video_path), stride=stride)
        transformer = None
        for index, frame in enumerate(generator):
            if max_frames and index >= max_frames:
                break
            detections = detector.detect(frame)
            # Only recompute keypoints/homography when shot-relevant classes are
            # present (jump=5, layup=6, ball-in-basket=1); else reuse the last one.
            shot_relevant = (
                bool(np.any(np.isin(detections.class_id, [1, 5, 6]))) if len(detections) else False
            )
            if shot_relevant or transformer is None:
                kpts, mask = keypoints.landmarks(frame)
                transformer = keypoints.transformer_from(kpts, mask)
            shots.update(frame, index, detections, transformer=transformer, team_brick=teams)

        dropped_count += len(shots.dropped)

        # Clip-level consistency: one possession -> one offense team / one basket.
        offenses = [s["offense_team"] for s in shots.shots if s.get("offense_team") is not None]
        clip_offense = max(set(offenses), key=offenses.count) if offenses else None
        baskets = [s.get("attacking_basket") for s in shots.shots if s.get("attacking_basket")]
        clip_basket = max(set(baskets), key=baskets.count) if baskets else None

        kept = 0
        for shot in shots.shots:
            team_id = shot.get("team")
            offense_id = shot.get("offense_team")
            attacking = shot.get("attacking_basket")
            if clip_offense is not None and team_id is not None and team_id != clip_offense:
                dropped_count += 1
                print(f"  [CLIP-DROP] {video_path.name} f{shot['frame']} reason=clip_team_mismatch "
                      f"team={team_id} clip_offense={clip_offense}")
                continue
            if clip_basket is not None and attacking is not None and attacking != clip_basket:
                dropped_count += 1
                print(f"  [CLIP-DROP] {video_path.name} f{shot['frame']} reason=clip_basket_mismatch "
                      f"basket={attacking} clip_basket={clip_basket}")
                continue

            cx, cy = shot.get("court_x"), shot.get("court_y")
            if cx is None:
                distance = None
            elif attacking is not None:
                bx, by = basket_xy(attacking)
                distance = float(np.hypot(cx - bx, cy - by))
            else:
                bx, by = basket_for(cx)
                distance = float(np.hypot(cx - bx, cy - by))

            rows.append(
                {
                    "video": video_path.name,
                    "frame": shot["frame"],
                    "shot_type": shot.get("type", ""),
                    "result": shot.get("outcome", ""),
                    "team": teams.team_name(team_id) if team_id is not None else "",
                    "offense_team": teams.team_name(offense_id) if offense_id is not None else "",
                    "attacking_basket": attacking or "",
                    "court_x": "" if cx is None else round(cx, 2),
                    "court_y": "" if cy is None else round(cy, 2),
                    "distance_ft": "" if distance is None else round(distance, 2),
                    "player_id": "",
                }
            )
            kept += 1
        print(f"  {video_path.name}: kept {kept}, dropped {len(shots.dropped)} (clip_offense={clip_offense}, clip_basket={clip_basket})")

    out_dir = resolve_path(args.out_dir) if args.out_dir else resolve_path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shots_csv = out_dir / "shots.csv"
    with open(shots_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "frame", "shot_type", "result", "team", "offense_team",
                        "attacking_basket", "court_x", "court_y", "distance_ft", "player_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregated shot map (made O / missed X) per team color.
    court = draw_court(config=keypoints.config)
    team_names = cfg["teams"]["team_names"]
    team_colors = cfg["teams"]["team_colors"]
    for team_id, hex_color in enumerate(team_colors.values()):
        name = team_names.get(team_id)
        made = [
            [float(r["court_x"]), float(r["court_y"])]
            for r in rows
            if r["team"] == name and r["result"] == "made" and r["court_x"] != ""
        ]
        missed = [
            [float(r["court_x"]), float(r["court_y"])]
            for r in rows
            if r["team"] == name and r["result"] == "missed" and r["court_x"] != ""
        ]
        if not made and not missed:
            continue
        court = draw_made_and_miss_on_court(
            config=keypoints.config,
            made_xy=np.array(made, dtype=np.float32) if made else None,
            miss_xy=np.array(missed, dtype=np.float32) if missed else None,
            made_color=sv.Color.from_hex(hex_color),
            miss_color=sv.Color.from_hex(hex_color),
            made_size=18,
            miss_size=18,
            line_thickness=4,
            court=court,
        )
    shot_map = out_dir / "shot_map.png"
    cv2.imwrite(str(shot_map), court)

    print("\n--- summary ---")
    print(f"Videos: {len(videos)}")
    print(f"Shots kept: {len(rows)}  (dropped total: {dropped_count})")
    for r in rows:
        print(f"  {r['video']} f{r['frame']} {r['shot_type']} {r['result']} "
              f"team={r['team'] or '?'} offense={r['offense_team'] or '?'} "
              f"basket={r['attacking_basket'] or '?'} loc=({r['court_x']},{r['court_y']}) dist={r['distance_ft']}")
    print(f"Wrote {shots_csv}")
    print(f"Wrote {shot_map}")


if __name__ == "__main__":
    main()
