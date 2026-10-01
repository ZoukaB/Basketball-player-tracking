"""One-call shot-location run on top of ``BasketballPipeline``.

Still does SAM2 player tracking and team clustering. The extra work is
writing tables plus a court shot chart so the notebook can stay thin.
``run_video`` adds an annotated MP4, rendered inline during the run by
default (no frame history on disk), or from the history with
``from_history=True``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.pipeline.pipeline import (
    DEFAULT_TARGET_FPS,
    BasketballPipeline,
    PipelineResult,
)
from src.pipeline.render import PipelineRenderer
from src.pipeline.rosters import DEFAULT_TEAM_NAMES
from src.pipeline.shots import plot_shot_chart


@dataclass
class ShotLocationRun:
    """``PipelineResult`` plus the files written for this clip."""

    result: PipelineResult
    run_dir: Path
    shots_path: Path
    player_path: Path
    identity_path: Path
    event_path: Path
    chart_path: Path | None = None
    video_path: Path | None = None


@dataclass
class VideoRun:
    """A ``ShotLocationRun`` plus the annotated clip rendered from it."""

    run: ShotLocationRun
    video_path: Path

    @property
    def result(self) -> PipelineResult:
        return self.run.result

    @property
    def run_dir(self) -> Path:
        return self.run.run_dir


def run_shot_location_pipeline(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_frames: Optional[int] = None,
    target_fps: float | None = DEFAULT_TARGET_FPS,
    team_names: dict[int, str] | None = None,
    use_ocr: bool = False,
    save_history: bool = False,
    plot: bool = True,
    pipeline: BasketballPipeline | None = None,
    output_video: str | Path | None = None,
    renderer: PipelineRenderer | None = None,
) -> ShotLocationRun:
    """Track players, assign teams, collect shot locations, write outputs.

    ``output_video``: if set, an annotated MP4 is rendered during the run
    (relative paths are resolved inside ``run_dir``).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(output_dir) if output_dir is not None else repo_root / "outputs"
    run_dir = output_root / video_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir = (run_dir / "history") if save_history else None

    if output_video is not None:
        output_video = Path(output_video)
        if not output_video.is_absolute():
            output_video = run_dir / output_video

    if pipeline is None:
        pipeline = BasketballPipeline(
            team_names=dict(team_names) if team_names is not None else dict(DEFAULT_TEAM_NAMES),
            use_ocr=use_ocr,
        )
    else:
        if team_names is not None:
            pipeline.team_names = dict(team_names)
        if use_ocr and pipeline.ocr is None:
            print("warning: use_ocr=True ignored, this pipeline was built without OCR")
        pipeline.use_ocr = use_ocr and pipeline.ocr is not None

    result = pipeline.run(
        video_path,
        max_frames=max_frames,
        history_dir=history_dir,
        target_fps=target_fps,
        output_video=output_video,
        renderer=renderer,
    )

    shots_path = run_dir / "shots_df.csv"
    player_path = run_dir / "player_df.csv"
    identity_path = run_dir / "identity_df.csv"
    event_path = run_dir / "event_df.csv"
    result.shots_df.to_csv(shots_path, index=False)
    result.player_df.to_csv(player_path, index=False)
    result.identity_df.to_csv(identity_path, index=False)
    result.event_df.to_csv(event_path, index=False)

    chart_path = None
    if plot:
        chart_path = run_dir / "shot_chart.jpg"
        plot_shot_chart(result.shots_df, chart_path)

    _print_summary(result)
    print(f"run_dir: {run_dir}")
    for path in (shots_path, player_path, identity_path, event_path, chart_path, output_video):
        if path is not None:
            print(f"wrote {path}")
    if history_dir is not None:
        print(f"history: {history_dir}")

    return ShotLocationRun(
        result=result,
        run_dir=run_dir,
        shots_path=shots_path,
        player_path=player_path,
        identity_path=identity_path,
        event_path=event_path,
        chart_path=chart_path,
        video_path=output_video,
    )


def run_video(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_frames: Optional[int] = None,
    target_fps: float | None = DEFAULT_TARGET_FPS,
    team_names: dict[int, str] | None = None,
    use_ocr: bool = False,
    plot: bool = True,
    pipeline: BasketballPipeline | None = None,
    renderer: PipelineRenderer | None = None,
    from_history: bool = False,
    keep_history: bool = True,
) -> VideoRun:
    """Run the pipeline and write ``<run_dir>/<stem>-annotated.mp4``.

    Default: rendered inline during the run (masks, identities, events, cuts,
    minimap), nothing stored on disk besides the MP4.

    ``from_history=True``: old path, saves the frame history and renders it
    afterwards with ``annotate.render_annotated_video`` (has the per-shot
    outcome banner, but its identity lookup must be segment-aware).
    """
    video_path = Path(video_path)
    video_name = f"{video_path.stem}-annotated.mp4"
    common = dict(
        output_dir=output_dir,
        max_frames=max_frames,
        target_fps=target_fps,
        team_names=team_names,
        use_ocr=use_ocr,
        plot=plot,
        pipeline=pipeline,
    )

    if not from_history:
        run = run_shot_location_pipeline(
            video_path, save_history=False, output_video=video_name, renderer=renderer, **common
        )
        return VideoRun(run=run, video_path=run.video_path)

    from src.pipeline.annotate import render_annotated_video

    run = run_shot_location_pipeline(video_path, save_history=True, **common)
    history_dir = run.result.history_dir
    if history_dir is None:
        raise RuntimeError("Pipeline did not write a history directory to render from.")

    annotated_path = render_annotated_video(
        history_dir=history_dir,
        output_path=run.run_dir / video_name,
        identity_df=run.result.identity_df,
        shots_df=run.result.shots_df,
    )
    print(f"wrote {annotated_path}")

    if not keep_history:
        shutil.rmtree(history_dir, ignore_errors=True)
        print(f"removed {history_dir}")

    return VideoRun(run=run, video_path=Path(annotated_path))


def _print_summary(result: PipelineResult) -> None:
    ev, ids = result.event_df, result.identity_df
    n_seg = int(ev.loc[ev["segment_id"] >= 0, "segment_id"].nunique()) if len(ev) else 0
    n_untracked = int((ev["segment_id"] < 0).sum()) if len(ev) else 0
    print(f"segments: {n_seg} (untracked frames: {n_untracked})")
    if len(ids):
        n_tracks = ids["track_uid"].nunique() if "track_uid" in ids else ids["tracker_id"].nunique()
        print(f"tracks (all segments): {n_tracks}")
        if "player_key" in ids:
            identified = ids["player_key"].str.startswith("T").sum()
            print(
                f"players identified by team+number: "
                f"{ids.loc[ids['player_key'].str.startswith('T'), 'player_key'].nunique()} "
                f"({identified}/{len(ids)} tracks)"
            )
    else:
        print("tracks: 0")
    print(f"player_df rows: {len(result.player_df)}")
    print(f"shots: {len(result.shots_df)}")
    if len(result.shots_df) and "outcome" in result.shots_df.columns:
        print(result.shots_df["outcome"].value_counts().to_string())