# Current stable baseline (active)

- Target input: continuous shot, no cuts, 10 detectable players on frame 0.
- Seed SAM2 once from frame-0 `player` detections (referee already suppressed),
then propagate for the whole video. No cut detection, no re-prompting.
- Verified on the continuous q2 clip: 10 tracks seeded, 10 masks every frame.

---

# Future ideas

## Team label stabilization (cluster index -> team name)

`sports.TeamClassifier` (SigLIP + UMAP + KMeans) returns arbitrary cluster indices, and UMAP may vary between runs, so `teams.team_names` (0/1) can be flipped or unstable. Options to stabilize:

- **Done:** the mapping is decided from cluster appearance. `TeamBrick.suggest_mapping`
  compares each cluster's mean jersey colour with `teams.jersey_colors` (the jerseys
  actually worn in the game, e.g. Celtics white at home, Knicks blue), falling back
  to the brand `team_colors`. Mapping on brand colours (Celtics green) swapped the
  teams for whole runs. `jersey_colors` must be set per game.
- Remaining: warn when the colour margin between the two assignments is small, and
  ask to validate against the cluster montage.
- Persist the fitted classifier + the chosen index->name mapping per game.
- Seed UMAP (random_state) and verify via the per-cluster sample montage  
produced by tests/test_teams.py.

## Speed (deprioritized)

RF-DETR and the court keypoint model now run on the GPU: `.venv` has
`onnxruntime-gpu` 1.22 (CUDA provider) instead of the CPU-only `onnxruntime`.
Measured on the RTX 3050: detection 746 -> 130 ms/frame, keypoints 91 -> 15 ms/frame;
`collect_shots.py` on one 4 s clip went from > 30 min (with OCR on) to ~70 s.
Install note: `inference-models[onnx-cu12]` from requirements.txt fails on Windows
(it builds `pycuda`, which needs a compiler + CUDA toolkit); installing
`onnxruntime-gpu>=1.17,<1.23` directly is enough for the ONNX path.

Remaining costs:
- SAM2-tiny is the biggest (~2.6 s/frame). Options: `sam2.1_hiera_t_512` config,
  smaller `image_size`, fewer seed objects.
- Team classifier: one SigLIP + UMAP predict per frame for the possession crop
  (~90 ms/frame on GPU). Could predict every N frames, or reuse SAM2 track votes.
- The first UMAP fit in a process JIT-compiles numba code (minutes with CUDA/torch
  loaded); warm it up off the critical path if latency matters.

## Better 2D homography mapping

Use skeleton keypoints on feet for better pixel to court position accuracy. 

---

## Continuous Video

### Principle: reuse the one-possession bricks, propagate them to the full video

The pipeline is built and validated on **one possession per clip**: detection,
SAM2 tracking seeded once, team votes, keypoints/homography, `ShotBrick` and the
clip-level offense rule (`ShotBrick.apply_clip_offense`). Going to a full,
continuous game must **not** redesign these bricks. A continuous video is
treated as a **sequence of possessions**, and each possession is processed exactly
like a clip today.

What changes is only a thin layer around the bricks:

1. **Segment the video into possessions, online.** One pass, no frame history
   (hardware constraint). Boundaries come from signals we already have:
   - hard cuts: `FrameCutDetector` (`src/utils.py`, z-normalized mean-diff);
   - possession change: the player-in-possession team votes flip and stay
     flipped (debounced), and/or the attacking basket switches side;
   - dead ball / replay / close-up: no usable court keypoints (< 4 anchors) for
     a while → pause the segment instead of feeding garbage frames.
2. **Run the existing bricks per segment, unchanged.** Same as `collect_shots.py`
   does per clip: a fresh `ShotBrick` per segment, SAM2 seeded at the segment
   start (the seed-once baseline), homography from that segment's frames, then
   `apply_clip_offense()` when the segment closes.
3. **Carry only game-level state across segments:**
   - the team classifier (fit once per game) and its cluster → team mapping
     (`teams.jersey_colors`);
   - game clock / quarter (see EXTRA);
   - player identity (Re-ID / OCR, below): the only part that genuinely spans
     possessions.
4. **Close a segment = finalize its outputs.** Shots, tracks and stats are written
   when the segment ends, so memory stays bounded and results stream out.

Validation path: first concatenate the 10 known possession clips into one video
and require the same shots/teams as the per-clip run (ground truth exists);
only then move to a real continuous broadcast.

### Re-ID / players entering after the first frame

Accepted limitation for now: players entering after frame 0 get a track only via  
a re-prompt (cut or `grow_on_new_max`). OCR is planned to map sparse track IDs to  
true players, which also helps reconcile IDs across re-prompts.

On a continuous video, segment boundaries make this unavoidable: SAM2 IDs restart
with each segment, so a player's identity must be re-established per segment
(OCR jersey number + team, possibly appearance) and merged into one game-level ID.

### Reprompt / consistent-tracking ideas (parked)

Everything we considered for keeping tracks consistent as the scene changes.
The machinery is kept dormant in `src/tracking.py`, `src/utils.py`, and
`src/config.yaml` for later.

1. **full_scene** — if the initial seed is under N players, re-prompt when the
  detected count reaches N. Simple but brittle when the detector merges two
   players (count never reaches N).
2. **grow_on_new_max (reset-based)** — re-prompt when the detected count reaches
  a debounced new maximum above the last prompt's count. Risk: a full reset  discards SAM2 memory and can drop an occluded player the detector can't see anymore.
3. **safety-guarded reset (implemented, dormant)** — grow_on_new_max but only
  reset when *every* tracked object is matched to a detection  (`tracks_all_matched`, IoU >= `match_iou`). If a track is unmatched (occluded but well-tracked), the reset is vetoed. Verified: it skipped growth at one  frame due to an occluded track, then applied safely once all were visible.
4. **mismatch (not implemented)** — act on detections that match no track
  (best IoU < `match_iou`) instead of raw counts; directly targets merged/missed players. See the detailed section below.
5. **ID reconciliation via OCR (planned)** — map sparse track IDs to true players
  and merge IDs across any re-prompts/cuts, so ID churn from reset-based growth does not matter downstream.

#### Reprompt policy: detection-vs-track mismatch (precise growth)

Problem with `grow_on_new_max`: it uses the raw detected player count as a proxy.
RF-DETR merges two players into one box, so the count undercounts and a new
player can enter without the count ever crossing a "new max".

Mismatch policy:

- After each prompt, each frame: compute IoU between cleaned `player` boxes and
current SAM2 track boxes (`mask_to_xyxy`, already available).
- A detection is "unmatched" if its best IoU with any track < `match_iou` (e.g. 0.3).
- If >= 1 unmatched confident detection persists >= `debounce_frames` (and the
cooldown elapsed, and track count <= `max_tracks`), re-prompt from that frame.
- Directly targets the merged/missed player instead of relying on counts.

Cost: `box_iou_batch` on ~10 detections x ~10 tracks = < 0.5 ms/frame
(< 0.02% of ~3000 ms/frame), no extra model calls or GPU work. Net wall-clock may
even improve by avoiding spurious re-prompts. Tradeoff is code complexity + a
match threshold; runtime is effectively free.

Config sketch:

```yaml
sam2:
  reprompt_policy: mismatch
  match_iou: 0.3
  debounce_frames: 3
  min_frames_between_reprompts: 15
  max_tracks: 12
```

#### Current approach: safety-guarded reset (implemented)

Reset-based `grow_on_new_max` is kept, but a reset only happens when **every**  
currently tracked object is matched to a current detection  
(`tracks_all_matched`, IoU >= `sam2.match_iou`). If any track is unmatched  
(occluded but still tracked well by SAM2), the reset is vetoed so that track is  
not lost. Verified on the clip: at one frame growth was skipped because a track  
was occluded, then applied safely once all players were visible. Default  
`reprompt_policy` is `never` as a safety net; enable via  
`tests/test_tracking.py --reprompt-policy grow_on_new_max` while it's validated.

---

## EXTRA

### Log the game clock and the detection confidence of each shot

Add both to every shot in `shots.csv` / `shots.json` (the dashboard already has a
time filter and a confidence column waiting for them).

**Game clock (quarter + time left)**

- Per-possession clips today: derived from the clip name (`...-q2-10.36-10.32.mp4`
  = Q2 starting at 10:36) minus `frame / fps`; done in `dashboard/build_data.py`
  on the dashboard branch. Move it into the pipeline output.
- Continuous video: the clock stops on dead balls, so `frame / fps` drifts. Read
  the broadcast score bug instead (quarter + game clock, OCR on a fixed screen
  region, sampled e.g. once per second), and interpolate between reads while the
  clock runs.
- Columns: `quarter`, `clock_s` (seconds left in the quarter), plus the video
  timestamp for jumping back to the footage.

**Detection confidence**

Log the evidence behind each shot, not one opaque number:

- `shot_conf`: confidence of the jump-shot / layup boxes over the frames that
  started the shot (max or mean);
- `make_conf`: ball-in-basket confidence on the frames that confirmed a make
  (empty for misses);
- `team_conf`: share of the clip's possession votes for the chosen team
  (e.g. 72/79 = 0.91), low values flag doubtful team calls;
- `location_conf`: homography quality (number of court anchors above
  `anchor_confidence`, or reprojection error).

An overall `confidence` (e.g. the minimum of the above) can drive a review flag in
the dashboard, while the separate columns explain *why* a shot is doubtful.
