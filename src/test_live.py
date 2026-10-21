"""Headless check of the live session on the 10 s test clip (no Streamlit).

Usage (from repo root):
    .venv/Scripts/python.exe src/test_live.py
    .venv/Scripts/python.exe src/test_live.py --source C:/path/to/clips_dir
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live import LiveShotSession  # noqa: E402
from src.utils import load_config  # noqa: E402

TEST_CLIP = "data/boston-celtics-new-york-knicks-game-1-q2-10.36-10.32.mp4"


def main() -> None:
    p = argparse.ArgumentParser(description="Headless live session test")
    p.add_argument("--source", default=TEST_CLIP)
    args = p.parse_args()

    session = LiveShotSession(load_config())
    t0 = time.perf_counter()
    session.start(args.source)
    while session.running:
        time.sleep(2.0)
        snap = session.snapshot()
        s = snap["status"]
        print(f"  [{s['state']}] {s['clip']} frame={s['frame']} fps={s['fps']:.1f} "
              f"teams_fitted={s['teams_fitted']} shots={len(snap['rows'])} "
              f"jpeg={'yes' if snap['jpeg'] else 'no'}")
    session.join()

    snap = session.snapshot()
    print(f"\nFinished in {time.perf_counter() - t0:.1f}s, state={snap['status']['state']} "
          f"{snap['status']['error']}")
    print(f"Team mapping: {snap['status']['mapping']}")
    for row in snap["rows"]:
        print(f"  {row}")
    print(f"Wrote {session.out_csv}")


if __name__ == "__main__":
    main()
