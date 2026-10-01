# Refactor plan: reset tracking at each hard cut

## Your role

You are refactoring an existing, **working** Python computer vision pipeline (NBA broadcast video → player tracking, court mapping, events, shots). The pipeline is correct on continuous video. The only problem: when the video contains a **hard cut** (camera change), the SAM2 tracker keeps propagating masks from the previous shot and the tracks become garbage.

Goal: split the video into **segments** (one segment = the frames between two hard cuts, called a "possession"), and fully re-initialise tracking at the start of each segment.

I will attach `pipeline.py` (containing the `run()` method) and the related modules. **Read them before writing any code.**

## Hard rules

1. **Do not change any detection, event, OCR, court-mapping, or shot logic.** Only move code and route variables. If a line of logic is not mentioned in this plan, copy it unchanged.
2. **Work one step at a time.** After each step, stop, show me the diff, and tell me exactly what to test. Do not start the next step until I confirm.
3. **Do not invent functions or signatures.** If you need a function you cannot see in the attached files, ask me for it.
4. **Never store frames in memory.** No list of frames. The only frame kept is the previous one inside `HardCutDetector`. (Writing to disk via the existing `history_dir` logic is fine and must be kept.)
5. Keep the existing stride logic: `sv.get_video_frames_generator(str(video_path), stride=stride)`. `frame_idx` = index of the processed frame (not source frame). Do not change this.
6. Keep all existing `print` messages. Keep type hints and docstrings in the same style as the existing code.
7. Target hardware is a laptop RTX 3050: call `torch.cuda.empty_cache()` at the end of each segment.

## Context: the existing `run()`

`run()` currently does, in order:

- **Before the loop:** fit team classifier on the clip; read the first frame with `self._first_frame(video_path)`; detect players; assign tracker IDs; predict teams with `team_classifier` and update `team_validator`; `self.tracker.reset()`; `self.tracker.prompt_first_frame(first_frame, first_players)`; create `number_validator`, `resolved_numbers`, `video_xy`, `event_rows`, `shot_collector`, `reprompt_interval`.
- **In the loop (per processed frame):** SAM2 propagate; RF-DETR inference; re-prompt missing players every `reprompt_interval` frames; history saving; event cleaning; optional OCR; court mapping into `video_xy`; state matches; `event_rows.append(...)`; `shot_collector.update(...)`.
- **After the loop:** build `event_df`; save history meta; `_clean_xy` (jump filter + Savitzky-Golay); `_build_identity_df`; `_build_player_df`; `shot_collector.finalize` + `attach_identity_and_court`; return `PipelineResult`.

## What resets per segment vs. what stays global

**Per segment (reset at every cut):**
`self.tracker` (reset + prompt), `tracker_ids`, `id_to_col`, `n_players`, `team_validator`, `number_validator`, `resolved_numbers`, `video_xy`, `shot_collector`, and the re-prompt timing (must be measured from the segment start frame, not from frame 0 of the video).

**Global (created once per video):**
`self.detector`, `self.team_classifier` (fitted once, so team labels stay consistent across segments), `self.court`, `event_rows`, `HardCutDetector`, history saving.

**Important pitfalls:**

- `video_xy` has one column per tracker ID. After a cut, IDs 1..N are **new, unrelated players**. Each segment needs its own `video_xy`, and `_clean_xy` must run **per segment**, never across a cut (the Savitzky-Golay filter would blend positions of different players).
- Tracker IDs restart at 1 in each segment. Every output dataframe must get a `possession_id` column. The unique key of a track is `(possession_id, track_id)`. Do not try to link IDs across segments.
- If `self.court` keeps temporal state between frames (e.g. a smoothed homography), it must also be reset at each cut. Check the court module and tell me what you find before changing anything.

## Provided code: `HardCutDetector`

Add this class in a new file `hard_cut.py`, next to the module that defines `detect_hard_cuts` (fix the import path to match the project):

```python
import numpy as np

from .cuts import detect_hard_cuts  # adapt to the real module name
from .cuts import THRESHOLD_HIGH, THRESHOLD_LOW, THRESHOLD_PIXEL


class HardCutDetector:
    """Flags hard cuts between consecutive processed frames."""

    def __init__(
        self,
        min_gap: int = 5,
        threshold_high: float = THRESHOLD_HIGH,
        threshold_low: float = THRESHOLD_LOW,
        threshold_pixel: float = THRESHOLD_PIXEL,
    ):
        self.min_gap = min_gap  # ignore cuts closer than N processed frames to the last one
        self.thresholds = dict(
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            threshold_pixel=threshold_pixel,
        )
        self.previous_frame = None
        self.last_cut = -min_gap
        self.cuts: list[int] = []

    def update(self, frame: np.ndarray, frame_idx: int) -> bool:
        """Call on every processed frame. Returns True if this frame starts a new shot."""
        is_cut = (
            self.previous_frame is not None
            and frame_idx - self.last_cut >= self.min_gap
            and detect_hard_cuts(frame_idx, frame, self.previous_frame, **self.thresholds)
        )
        self.previous_frame = frame
        if is_cut:
            self.last_cut = frame_idx
            self.cuts.append(frame_idx)
        return is_cut
```

`run()` must accept an optional `cut_thresholds: dict | None = None` argument, passed as `HardCutDetector(**(cut_thresholds or {}))`, because thresholds differ between videos.

## Target architecture

```python
@dataclass
class SegmentState:
    possession_id: int
    start_frame: int
    tracker_ids: np.ndarray
    id_to_col: dict[int, int]
    n_players: int
    team_validator: ConsecutiveValueTracker
    number_validator: ConsecutiveValueTracker
    resolved_numbers: set[int]
    video_xy: list[np.ndarray]
    shot_collector: ShotCollector
```

Three new private methods on the pipeline class:

- `_start_segment(self, frame, frame_idx, possession_id, process_fps) -> SegmentState | None`
  = the current "before the loop" code (except team classifier fitting, which stays in `run()` and runs once). Detect players on `frame`, predict teams, `self.tracker.reset()`, `self.tracker.prompt_first_frame(frame, players)`. **Returns `None` if no player is detected** (instead of raising `RuntimeError`).
- `_process_frame(self, state, frame, frame_idx, hard_cut, event_rows, history_dir) -> None`
  = the current loop body, reading and writing `state.<field>` instead of local variables. Re-prompt condition uses `frame_idx - state.start_frame`. Adds `"possession_id": state.possession_id` and `"hard_cut": hard_cut` to each `event_rows` entry.
- `_finish_segment(self, state, last_frame) -> dict[str, pd.DataFrame]`
  = the current "after the loop" code for per-segment outputs: `_clean_xy`, `_build_identity_df`, `_build_player_df`, `shot_collector.finalize(last_frame=...)`, `attach_identity_and_court`. Adds a `possession_id` column to `player_df`, `identity_df`, `shots_df`. Calls `torch.cuda.empty_cache()`. Note: check how `ShotCollector` uses frame indices (global or relative) and tell me before choosing what to pass as `last_frame`.

Target shape of `run()`:

```python
cut_detector = HardCutDetector(**(cut_thresholds or {}))
event_rows: list[dict] = []
segments: list[dict[str, pd.DataFrame]] = []
state: SegmentState | None = None
possession_id = 0

for frame_idx, frame in enumerate(tqdm(frame_generator, desc="pipeline", total=total)):
    if max_frames is not None and frame_idx >= max_frames:
        break

    hard_cut = cut_detector.update(frame, frame_idx)
    if hard_cut:
        print(f"frame {frame_idx}: HARD CUT")

    if hard_cut and state is not None:
        segments.append(self._finish_segment(state, last_frame=frame_idx - 1))
        state = None
        possession_id += 1

    if state is None:
        state = self._start_segment(frame, frame_idx, possession_id, process_fps)
        if state is None:
            continue  # no player on this frame (close-up, crowd): try the next one

    self._process_frame(state, frame, frame_idx, hard_cut, event_rows, history_dir)

if state is not None:
    segments.append(self._finish_segment(state, last_frame=frame_idx))

# concat each dataframe type across segments (ignore_index=True), build event_df
# from event_rows, save history meta, return PipelineResult.
```

`PipelineResult` gains a `cuts: list[int]` field (= `cut_detector.cuts`). If `PipelineResult` is a dataclass, add the field with a default so existing callers do not break.

## Steps

Do them in order. **Stop after each step.**

### Step 1: `HardCutDetector` + flag only

Add `hard_cut.py`. In `run()`, create the detector and call `update()` at the start of the loop, print on cut, and add `"hard_cut"` to `event_rows`. No other change.
**Test:** on a video with cuts, printed cut frames match the real cuts; all other outputs are identical to before.

### Step 2: `SegmentState` + `_finish_segment`

Create the dataclass. Move the "after the loop" per-segment code into `_finish_segment`. Still one single segment for the whole video.
**Test:** on a clip **without** cuts, `player_df`, `identity_df`, `shots_df`, `event_df` are identical to the previous run (apart from the new `possession_id` column, which is all 0).

### Step 3: `_start_segment`, prompt inside the loop

Move the "before the loop" code into `_start_segment`. Remove `self._first_frame(video_path)`. Call `_start_segment` on the first frame inside the loop (`if state is None`). Still one segment.
**Test:** same as step 2, outputs identical.

### Step 4: `_process_frame`

Move the loop body into `_process_frame`, replacing local variables by `state.<field>`. This is where variables are easily forgotten: list every variable the loop body reads or writes and confirm where each one now lives.
**Test:** same as step 2, outputs identical.

### Step 5: activate the reset

Add the `if hard_cut and state is not None:` block. Concatenate segment outputs. Add `cuts` to `PipelineResult`.
**Test:** on a video with exactly **one** cut: 2 possessions; tracker IDs restart at 1 in possession 1; tracks after the cut follow the right players; no NaN explosion in court coordinates around the cut.

### Step 6: court state check

Report whether `self.court` keeps state across frames. If yes, propose (do not apply) a reset in `_start_segment`.

## Deliverable after each step

- The diff.
- One sentence on what changed.
- The exact test I should run and what I should see.