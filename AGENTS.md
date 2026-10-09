# Basketball broadcast analysis pipeline

## Context
Basketball broadcast analysis pipeline (NBA, broadcast edited so each segment is one possession).
This is a fresh restart on `main`. Goal of this iteration: the SIMPLEST working pipeline that reuses the
existing bricks from the exploration notebook (`C:\Users\zecab\.gemini\antigravity-ide\scratch\Basketball-player-tracking\notebooks\original_notebook_exploration.ipynb`).

The notebook is the source of truth for model IDs, library calls, class names and parameters.
Do NOT invent APIs or model IDs. If something isn't in the notebook, ask instead of guessing.

## Hardware constraints
- Laptop with RTX 3050 (limited VRAM).
- Team classifier (SigLIP + KMeans) runs on CPU.
- Single pass over the video: no pre-pass, no storing the frame history in memory.

## Scope of this iteration (priority order)
1. Player/ball/rim detection with the Roboflow basketball detection model from the notebook.
2. Tracking + mask propagation with SAM2, seeded by the detections.
3. Team classification: SigLIP embeddings + KMeans (k=2), fitted once on the first frame where
   10 players are detected, then only `predict` afterwards.
4. Court keypoint detection with the notebook's keypoint model + homography to court coordinates
   (reuse the notebook's view transformer logic).
5. Shot detection reusing the notebook's logic (shot events + made/missed). Shot location = shooter's
   position projected onto the court.
6. Jersey-number OCR: implemented as in the notebook but DISABLED BY DEFAULT (`ocr.enabled: false`).
   When disabled, the OCR model must not be loaded at all (no VRAM used).

Out of scope for now: player re-identification across cuts, galleries, substitution detection, dashboard.

One simple hook only: when a hard cut is detected (simple frame-difference threshold), reset the SAM2
tracker and log the frame index. Nothing more sophisticated.

## Code structure
All code lives in `src/`. Existing files there may be deleted or overwritten: this is a fresh restart.

Suggested layout (adjust if simpler):
- `src/config.yaml`: model IDs, thresholds, input/output paths, feature flags (ocr, annotated video export)
- `src/detection.py`, `src/tracking.py`, `src/teams.py`, `src/keypoints.py`, `src/shots.py`, `src/ocr.py`:
  one brick each, each a small class with `__init__(cfg)` and a per-frame method
- `src/run.py`: main loop, reads the video once, calls the bricks, writes outputs

Prefer plain functions and small classes over abstractions. No frameworks, no async, no multiprocessing.

## Outputs
- `outputs/annotated.mp4`: boxes/masks colored by team, track IDs, court keypoints (toggleable)
- `outputs/tracks.csv`: frame, track_id, team, bbox, court_x, court_y
- `outputs/shots.json`: frame, shooter track_id, team, court_x, court_y, made/missed
- `outputs/shot_map.png`: half-court diagram with made (o) / missed (x) per team

`src/run.py` prints per-brick timing (ms/frame) and peak VRAM at the end.

## How to work
- Before writing any code for a new brick, state which notebook cells you are reusing, the exact model IDs
  and key parameters, and the files you will create or change. Wait for validation.
- Implement ONE brick at a time, in the priority order above, each with a tiny test script runnable on a
  10-second clip (`C:\Users\zecab\.gemini\antigravity-ide\scratch\Basketball-player-tracking\data\boston-celtics-new-york-knicks-game-1-q2-10.36-10.32.mp4`). Stop after each brick and wait for feedback.
- If a notebook cell depends on Colab-specific features (secrets, `!pip`, inline display), replace it with
  a local equivalent and say what changed.
- Do not add features outside the current scope, even if they seem useful. Propose them instead.

