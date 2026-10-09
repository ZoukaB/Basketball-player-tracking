# Current stable baseline (active)

- Target input: continuous shot, no cuts, 10 detectable players on frame 0.
- Seed SAM2 once from frame-0 `player` detections (referee already suppressed),
then propagate for the whole video. No cut detection, no re-prompting.
- Verified on the continuous q2 clip: 10 tracks seeded, 10 masks every frame.

---

# Future ideas

## Team label stabilization (cluster index -> team name)

`sports.TeamClassifier` (SigLIP + UMAP + KMeans) returns arbitrary cluster indices, and UMAP may vary between runs, so `teams.team_names` (0/1) can be flipped or unstable. Options to stabilize:

- Decide the mapping from cluster appearance (e.g. dominant jersey colour: green vs blue) instead of a fixed index.
- Persist the fitted classifier + the chosen index->name mapping per game.
- Seed UMAP (random_state) and verify via the per-cluster sample montage  
produced by tests/test_teams.py.

## Speed (deprioritized)

RF-DETR runs on CPU (onnxruntime CPU in `.venv`; `CUDAExecutionProvider` unavailable). SAM2-tiny is the bigger cost (~2.6 s/frame on the RTX 3050). Options later: `sam2.1_hiera_t_512` config, smaller `image_size`, fewer seed objects, GPU ONNX (`onnxruntime-gpu` / `inference-gpu`).

## Re-ID / players entering after the first frame

Accepted limitation for now: players entering after frame 0 get a track only via  
a re-prompt (cut or `grow_on_new_max`). OCR is planned to map sparse track IDs to  
true players, which also helps reconcile IDs across re-prompts.

## Better 2D homography mapping

Use skeleton keypoints on feet for better pixel to court position accuracy. 

---

## Reprompt / consistent-tracking ideas (parked)

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

### Reprompt policy: detection-vs-track mismatch (precise growth)

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



## Current approach: safety-guarded reset (implemented)

Reset-based `grow_on_new_max` is kept, but a reset only happens when **every**  
currently tracked object is matched to a current detection  
(`tracks_all_matched`, IoU >= `sam2.match_iou`). If any track is unmatched  
(occluded but still tracked well by SAM2), the reset is vetoed so that track is  
not lost. Verified on the clip: at one frame growth was skipped because a track  
was occluded, then applied safely once all players were visible. Default  
`reprompt_policy` is `never` as a safety net; enable via  
`tests/test_tracking.py --reprompt-policy grow_on_new_max` while it's validated.