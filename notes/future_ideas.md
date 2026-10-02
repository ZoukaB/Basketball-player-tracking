# Future ideas

## Current stable baseline (active)
- Target input: continuous shot, no cuts, 10 detectable players on frame 0.
- Seed SAM2 once from frame-0 `player` detections (referee already suppressed),
  then propagate for the whole video. No cut detection, no re-prompting.
- Verified on the continuous q2 clip: 10 tracks seeded, 10 masks every frame.

## Reprompt / consistent-tracking ideas (parked)
Everything we considered for keeping tracks consistent as the scene changes.
The machinery is kept dormant in `src/tracking.py`, `src/utils.py`, and
`src/config.yaml` for later.

1. **full_scene** — if the initial seed is under N players, re-prompt when the
   detected count reaches N. Simple but brittle when the detector merges two
   players (count never reaches N).
2. **grow_on_new_max (reset-based)** — re-prompt when the detected count reaches
   a debounced new maximum above the last prompt's count. Risk: a full reset
   discards SAM2 memory and can drop an occluded player the detector can't see.
3. **safety-guarded reset (implemented, dormant)** — grow_on_new_max but only
   reset when *every* tracked object is matched to a detection
   (`tracks_all_matched`, IoU >= `match_iou`). If a track is unmatched (occluded
   but well-tracked), the reset is vetoed. Verified: it skipped growth at one
   frame due to an occluded track, then applied safely once all were visible.
4. **mismatch (not implemented)** — act on detections that match no track
   (best IoU < `match_iou`) instead of raw counts; directly targets merged/missed
   players. See the detailed section below.
5. **additive add without reset (blocked)** — add a new object mid-stream. The
   fork crashes (object-count memory mismatch); needs a fork patch. See below.
6. **ID reconciliation via OCR (planned)** — map sparse track IDs to true players
   and merge IDs across any re-prompts/cuts, so ID churn from reset-based growth
   does not matter downstream.

## Reprompt policy: detection-vs-track mismatch (precise growth)

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

## Incremental object add (no reset) — BLOCKED by fork

The ideal is adding new objects mid-stream without resetting track IDs. Two
findings:

1. The fork forbids it by design: `_obj_id_to_idx` raises once
   `tracking_has_started` (line ~158), `propagate_in_video_preflight` sets that
   flag ("we don't allow adding new objects until session is reset", line ~707),
   and `add_new_prompt_during_track(if_new_target=True)` raises
   `NotImplementedError` (line ~793).
2. A spike that worked around the flag (cache the current frame's backbone
   features in `cached_features`, temporarily clear `tracking_has_started`, call
   `add_new_prompt` with a new obj_id) *added* the object fine but crashed on the
   next `track()`:
   `RuntimeError: stack expects each tensor to be equal size, but got [10,256] and [11,256]`
   in `sam2_base._prepare_memory_conditioned_features` — past memory frames were
   encoded with 10 objects, so stacking object pointers across frames fails.

   To make additive work you must retroactively pad every historical memory frame
   with a slot (zero `obj_ptr`, zero maskmem) for each new object — deep surgery
   in the fork's memory store / `_prepare_memory_conditioned_features`. Revisit
   only if ID churn becomes a real problem; `add_new_mask` was also considered and
   dropped for now.

## Current approach: safety-guarded reset (implemented)

Reset-based `grow_on_new_max` is kept, but a reset only happens when **every**
currently tracked object is matched to a current detection
(`tracks_all_matched`, IoU >= `sam2.match_iou`). If any track is unmatched
(occluded but still tracked well by SAM2), the reset is vetoed so that track is
not lost. Verified on the clip: at one frame growth was skipped because a track
was occluded, then applied safely once all players were visible. Default
`reprompt_policy` is `never` as a safety net; enable via
`tests/test_tracking.py --reprompt-policy grow_on_new_max` while it's validated.


## Speed (deprioritized)

RF-DETR runs on CPU (onnxruntime CPU in `.venv`; `CUDAExecutionProvider`
unavailable). SAM2-tiny is the bigger cost (~2.6 s/frame on the RTX 3050).
Options later: `sam2.1_hiera_t_512` config, smaller `image_size`, fewer seed
objects, GPU ONNX (`onnxruntime-gpu` / `inference-gpu`).

## Re-ID / players entering after the first frame

Accepted limitation for now: players entering after frame 0 get a track only via
a re-prompt (cut or `grow_on_new_max`). OCR is planned to map sparse track IDs to
true players, which also helps reconcile IDs across re-prompts.
