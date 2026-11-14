"""run.py on an edited (multi-possession) video: cuts, SAM2 re-seeding, and
shots identical to collect_shots.py.

Builds the models and fits teams once, then runs on the same window:
  1) collect_shots.collect_video (detection + homography, no SAM2);
  2) run.run_video (same shots + SAM2 shooter identity).
Checks:
  - at least one cut was found;
  - SAM2 was seeded in every segment (tracks.csv has tracks for each segment);
  - the shots match collect_shots field by field (frame, segment, type, result,
    team, offense, basket, location), and `number_frame` equals collect's `number`.
Writes run.py outputs to outputs/test_run_cuts/.

Usage (from repo root; SAM2 is ~2.6 s/frame, so keep the window short):
    .venv/Scripts/python.exe tests/test_run_cuts.py --seconds 20
    .venv/Scripts/python.exe tests/test_run_cuts.py --seconds 20 --ocr
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.collect_shots import collect_video  # noqa: E402
from src.detection import Detector  # noqa: E402
from src.keypoints import KeypointBrick  # noqa: E402
from src.ocr import OCRBrick  # noqa: E402
from src.run import run_video  # noqa: E402
from src.teams import TeamBrick, resolve_team_mapping  # noqa: E402
from src.tracking import SAM2Tracker  # noqa: E402
from src.utils import load_config, resolve_path, setup_env  # noqa: E402

COMPARE = ["segment", "frame", "shot_type", "result", "team", "offense_team",
           "attacking_basket", "court_x", "court_y"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run.py cut handling + parity with collect_shots")
    p.add_argument("--video", default="data/10_possessions_1video.mp4")
    p.add_argument("--seconds", type=float, default=20.0, help="Window from the start (0 = full video)")
    p.add_argument("--ocr", action="store_true", help="Enable OCR (jersey numbers)")
    p.add_argument("--out-dir", default="outputs/test_run_cuts")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_env()
    cfg = load_config()
    if args.ocr:
        cfg["ocr"]["enabled"] = True
    video = resolve_path(args.video)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(cfg)
    keypoints = KeypointBrick(cfg)
    tracker = SAM2Tracker(cfg)
    teams = TeamBrick(cfg)
    ocr = OCRBrick(cfg)
    teams.fit_offline_from_videos(detector, [video], stride=int(cfg["teams"].get("stride", 30)),
                                  player_class_ids=cfg["classes"]["player_ids"], class_agnostic_nms=True)
    resolve_team_mapping(teams, montage_path=out_dir / "team_clusters.jpg", mapping_path=None,
                         ask=False, auto_color=True)

    print("\n=== 1) collect_shots.collect_video ===")
    collect_rows, _ = collect_video(video, cfg, detector, teams, keypoints, ocr, max_seconds=args.seconds)

    print("\n=== 2) run.run_video ===")
    outputs = {
        "tracks_csv": str(out_dir / "tracks.csv"),
        "shots_csv": str(out_dir / "shots.csv"),
        "shots_json": str(out_dir / "shots.json"),
        "shot_map": str(out_dir / "shot_map.png"),
        "annotated_video": str(out_dir / "annotated.mp4"),
    }
    shared = {"detector": detector, "keypoints": keypoints, "tracker": tracker, "teams": teams, "ocr": ocr}
    run_rows = run_video(video, cfg, outputs, shared=shared, max_seconds=args.seconds)

    failures: list[str] = []

    # Segments and SAM2 seeding: every segment of the window should have tracks.
    with open(outputs["tracks_csv"], encoding="utf-8") as handle:
        track_rows = list(csv.DictReader(handle))
    seg_with_tracks = {int(r["segment"]) for r in track_rows}
    n_segments = max([int(r["segment"]) for r in run_rows] + list(seg_with_tracks) + [0]) + 1
    print(f"\nsegments seen: {n_segments}, segments with SAM2 tracks: {sorted(seg_with_tracks)}")
    if n_segments < 2:
        failures.append("no cut found in the window (try a longer --seconds)")
    shot_segments = {int(r["segment"]) for r in run_rows}
    missing = sorted(shot_segments - seg_with_tracks)
    if missing:
        failures.append(f"segments with shots but no SAM2 tracks (not re-seeded?): {missing}")

    # Shots: identical to collect_shots.
    if len(run_rows) != len(collect_rows):
        failures.append(f"shot count differs: run {len(run_rows)} vs collect {len(collect_rows)}")
    for i, (r, c) in enumerate(zip(run_rows, collect_rows)):
        diffs = [f"{k}: run={r[k]!r} collect={c[k]!r}" for k in COMPARE if str(r[k]) != str(c[k])]
        if str(r["number_frame"]) != str(c["number"]):
            diffs.append(f"number_frame={r['number_frame']!r} vs collect number={c['number']!r}")
        if diffs:
            failures.append(f"shot {i}: " + "; ".join(diffs))

    print("\nshots (run.py):")
    for r in run_rows:
        print(f"  seg{r['segment']} f{r['frame']} {r['shot_type']} {r['result']} {r['team']} "
              f"track={r['tracker_id'] if r['tracker_id'] != '' else '-'} "
              f"number track={r['number_track'] or '-'} frame={r['number_frame'] or '-'}")
    linked = sum(r["tracker_id"] != "" for r in run_rows)
    print(f"shooters linked to a SAM2 track: {linked}/{len(run_rows)}")
    if cfg["ocr"]["enabled"]:
        n_track = sum(bool(r["number_track"]) for r in run_rows)
        n_frame = sum(bool(r["number_frame"]) for r in run_rows)
        print(f"shooter number from track: {n_track}/{len(run_rows)}, "
              f"from shot frame: {n_frame}/{len(run_rows)}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
