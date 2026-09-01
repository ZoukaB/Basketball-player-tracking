"""One-call shot-location run on top of ``BasketballPipeline``.

Still does SAM2 player tracking and team clustering. The extra work is
writing tables plus a court shot chart so the notebook can stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.pipeline.pipeline import (
    DEFAULT_TARGET_FPS,
    BasketballPipeline,
    PipelineResult,
)
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
) -> ShotLocationRun:
    """Track players, assign teams, collect shot locations, write outputs.

    Change ``video_path`` / ``max_frames`` / ``target_fps`` between runs.
    Pass an existing ``pipeline`` to skip reloading RF-DETR, SAM2, and SigLIP.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(output_dir) if output_dir is not None else repo_root / "outputs"
    run_dir = output_root / video_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir = (run_dir / "history") if save_history else None

    if pipeline is None:
        pipeline = BasketballPipeline(
            team_names=dict(team_names) if team_names is not None else dict(DEFAULT_TEAM_NAMES),
            use_ocr=use_ocr,
        )
    elif team_names is not None:
        pipeline.team_names = dict(team_names)
        pipeline.use_ocr = use_ocr

    result = pipeline.run(
        video_path,
        max_frames=max_frames,
        history_dir=history_dir,
        target_fps=target_fps,
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

    print(f"run_dir: {run_dir}")
    print(f"players tracked: {result.identity_df['tracker_id'].nunique() if len(result.identity_df) else 0}")
    print(f"player_df rows: {len(result.player_df)}")
    print(f"shots: {len(result.shots_df)}")
    if len(result.shots_df) and "outcome" in result.shots_df.columns:
        print(result.shots_df["outcome"].value_counts().to_string())
    print(f"wrote {shots_path}")
    print(f"wrote {player_path}")
    print(f"wrote {identity_path}")
    print(f"wrote {event_path}")
    if chart_path is not None:
        print(f"wrote {chart_path}")
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
    )
