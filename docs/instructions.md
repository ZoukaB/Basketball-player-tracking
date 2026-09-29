# Task: implement `run_possession` (NBA tracking pipeline)

## 0. Rules for you (the coding assistant)
- **Keep it simple and readable.** This is a student project. Prefer plain functions and one small dataclass over abstractions. No factories, no plugin systems, no premature optimization.
- **Do not invent APIs.** For SAM2 real-time, `inference`, `sports` and `supervision`, reuse exactly what the existing code in `src/` does. If something is unclear, ask.
- **Do not modify anything in `src/`.** It is the current working version. Read it for inspiration, copy what you need into `src2/`, and simplify there.
- Comment each step of the main loop in one short line, so the flow can be read top to bottom.
- Target: `possession.py` under ~150 lines.

## 1. Context
Computer vision pipeline on NBA broadcast video: player detection (RF-DETR), tracking (SAM2 real-time), team assignment, jersey number OCR, identification via team rosters.

The input video has been **edited so that every camera segment is exactly one possession**. Segments are separated by hard cuts. The pipeline reads the video **once, sequentially**. When a cut is detected, the current possession is saved and everything (tracker, votes) is re-initialized as if it were a new clip, starting from the cut frame.

**Scope of this task:** only the `run_possession` function, its small helpers, and a test notebook. The test video currently contains **a single possession** (no cut), so `run_possession` will run once until the end of the video.

## 2. Constraints
- GPU: **RTX 3050 laptop (~4 GB VRAM)**.
- **Never keep a history of frames or masks in memory.** Process frame by frame; only store lightweight rows (ids, boxes, labels).
- **Team classifier runs on CPU.**
- Use `torch.inference_mode()` and `torch.autocast("cuda", dtype=torch.bfloat16)` for SAM2, as in `src/`.

## 3. Existing code to reuse (read `src/`)
- **Tracking:** the SAM2 tracker class in `src/` (prompt on first frame with RF-DETR boxes, then `propagate` each frame). Keep its behavior. Add a proper `reset()` that clears the SAM2 predictor state so a new possession starts clean.
- **OCR / numbers:** the jersey number recognition and the number-box ↔ player-box matching logic in `src/`, plus the consecutive-value validation (a number is accepted after N identical reads).
- **Team classifier:** the `TeamClassifier` used in `src/` (from the `sports` package), forced to `device="cpu"`.
- **Rosters:** `TEAM_ROSTERS` and `TEAM_NAMES` as defined in `src/`.

## 4. Files to create
```
src2/
├── __init__.py
├── video.py        # iter_frames(path) -> yields (idx, frame), sequential read, no seek
├── cuts.py         # is_cut(prev, frame, frames_since_start, cfg) -> bool
├── tracking.py     # SAM2 tracker copied from src/, with reset()
├── ocr.py          # number reading + matching + validation copied from src/
├── teams.py        # fit_team_model(frame, player_detections) + predict_teams(...)
└── possession.py   # Config, PossessionResult, run_possession
notebooks/
└── test_run_possession.ipynb
```
Note: the folder is named `src2` (not `src.2`) because a dot in a folder name breaks Python imports.

## 5. Helpers

### `video.py`
```python
def iter_frames(path: str):
    """Read the video sequentially with OpenCV and yield (idx, frame). Never seeks."""
```

### `cuts.py`
Cheap hard-cut detector on CPU:
1. Downscale both frames to 160x90, convert to HSV.
2. Compute a 2D H-S histogram for each, normalize, compare with `cv2.compareHist(..., cv2.HISTCMP_CORREL)`.
3. Return `True` if correlation < `cfg.cut_threshold` (default 0.5) **and** `frames_since_start >= cfg.min_possession_frames` (default 10, debounce against flashes).

### `teams.py`
The team model is **fit once per video, on the first frame where 10 players are detected**, then reused for all later possessions.
```python
def fit_team_model(frame, player_detections) -> TeamModel:
    """Fit on the crops of the 10 players of this frame (CPU)."""

def predict_teams(team_model, frame, detections) -> np.ndarray:
    """Return a team id (0 or 1) for each detection (CPU)."""
```
- Crops: reuse the crop logic from `src/` (boxes scaled to focus on the torso).
- **Warning:** 10 crops is a very small training set. If `TeamClassifier` (SigLIP + UMAP + KMeans) fails or is unstable with 10 samples (UMAP `n_neighbors` defaults to 15), implement a simple fallback: mean HSV color of the torso crop + `KMeans(n_clusters=2)` from scikit-learn. Choose via `cfg.team_method = "siglip" | "hsv"`, default `"hsv"`.

## 6. `possession.py`

### Config
```python
@dataclass
class Config:
    player_conf: float = 0.4
    player_iou: float = 0.9
    det_stride: int = 3          # run RF-DETR every N frames (numbers, states)
    team_stride: int = 15        # re-predict team for tracks without a stable team
    ocr_stride: int = 5          # OCR only for tracks without a validated number
    n_consecutive: int = 3       # identical reads needed to validate a number
    min_box_area: int = 100
    cut_threshold: float = 0.5
    min_possession_frames: int = 10
    team_method: str = "hsv"
```

### Result
```python
@dataclass
class PossessionResult:
    possession_id: int
    start_idx: int
    end_idx: int
    tracks: pd.DataFrame        # one row per (frame, track)
    identities: pd.DataFrame    # one row per track
    team_model: object          # fitted model, passed to the next possession
    next_start: tuple[int, np.ndarray] | None   # (idx, frame) of the cut, or None at end of video
```

### Signature
```python
def run_possession(frames, first_idx, first_frame, models, cfg, possession_id=0,
                   team_model=None, out_dir=None) -> PossessionResult:
```
- `frames`: the **already-open** iterator from `iter_frames` (never reopen or seek the video).
- `models`: a simple namespace/dict holding the loaded RF-DETR model, SAM2 predictor and OCR model. Models are loaded **once** outside this function.

### Algorithm
1. **Init on `first_frame`:**
   - `tracker.reset()`, then run RF-DETR, keep player classes (`PLAYER_CLASS_IDS` from `src/`), class-agnostic NMS, assign `tracker_id = 1..N`, prompt SAM2.
   - Create empty vote containers for teams and numbers (reuse the consecutive-value tracker from `src/`).
   - If `team_model is None` and exactly 10 players are detected: fit it now.
2. **For each `(idx, frame)` in `frames`:**
   - If `is_cut(prev, frame, idx - first_idx, cfg)`: stop and set `next_start = (idx, frame)`.
   - `tracker.propagate(frame)` → boxes + tracker ids; drop boxes below `min_box_area`.
   - Every `det_stride` frames: run RF-DETR on the full frame (players + number boxes).
     - If `team_model is None` and 10 players are detected: fit the team model.
   - Every `team_stride` frames, if the team model exists: predict teams for tracks without a stable team, update votes.
   - Every `ocr_stride` frames (only on detector frames): OCR the number boxes, match them to tracks, update votes **only for tracks without a validated number**.
   - Append one lightweight row per track: `possession_id, frame_idx, track_id, x1, y1, x2, y2`.
   - `prev = frame`.
3. **Finalize:**
   - Build `identities`: `possession_id, track_id, team_id, team_name, jersey_number, player_name` (roster lookup with `TEAM_ROSTERS[team_name]`, trying both `str` and `int` keys as in `src/`).
   - If `out_dir` is set: save `tracks_{pid}.parquet` and `identities_{pid}.parquet`.
   - `torch.cuda.empty_cache()`.
   - Return the `PossessionResult`.

### Future caller (for context only, do not implement a CLI now)
```python
frames = iter_frames(video)
nxt, team_model, pid = next(frames), None, 0
while nxt is not None:
    res = run_possession(frames, *nxt, models, cfg, pid, team_model, out_dir)
    nxt, team_model, pid = res.next_start, res.team_model, pid + 1
```

## 7. Test notebook `notebooks/test_run_possession.ipynb`
The video used for the test contains **one possession only**.
1. **Setup:** paths, `sys.path` insert for the project root, imports from `src2`, `TEAM_ROSTERS` / `TEAM_NAMES`.
2. **Load models once** (same model ids and loading code as `src/`), SAM2 on CUDA, team model on CPU.
3. **Run:** `frames = iter_frames(VIDEO)`, `first_idx, first_frame = next(frames)`, call `run_possession(...)` inside a timer, with `torch.cuda.reset_peak_memory_stats()` before.
4. **Report:**
   - frames processed, FPS, peak VRAM (`torch.cuda.max_memory_allocated() / 1e9`),
   - `res.next_start is None` (expected: no cut in this video),
   - `res.team_model is not None` (a frame with 10 players was found),
   - `res.identities` displayed, and `res.tracks.head()`.
5. **Optional visual check:** a separate cell that re-reads the video and draws `res.tracks` boxes with team color + jersey number + player name into an output mp4 using `sv.VideoSink` (reading `res.tracks` per frame; still no frame history).

### Acceptance criteria
- Runs end to end on the RTX 3050 without OOM, peak VRAM < 3.5 GB.
- No false cut detected on the single-possession video.
- Team model fitted, team ids consistent with the visual check.
- Identities comparable to the results of the current code in `src/` on the same video.