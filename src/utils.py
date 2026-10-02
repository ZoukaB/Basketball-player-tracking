"""Shared helpers: repo paths, environment/API-key setup, config loading."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_env() -> None:
    """Load .env and configure runtime env vars.

    Must be called before importing `inference` so that the ONNX Runtime
    execution provider is picked up.
    """
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)

    roboflow_key = os.environ.get("ROBOFLOW_API_KEY")
    if roboflow_key and not os.environ.get("API_KEY"):
        os.environ["API_KEY"] = roboflow_key

    os.environ.setdefault(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider",
    )
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # SigLIP (teams) is expected to be cached locally; offline avoids a network
    # metadata check that can fail in this environment. Set these env vars
    # before huggingface_hub is imported. Unset them externally to allow downloads.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else REPO_ROOT / "src" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the repo root when not absolute."""
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path)


def analysis_stride(cfg: dict, video_fps: float) -> int:
    """Frame stride for the global analysis fps (>= 1)."""
    target_fps = float(cfg.get("analysis", {}).get("fps", video_fps))
    if target_fps <= 0 or video_fps <= 0:
        return 1
    return max(1, int(round(video_fps / target_fps)))


class FrameCutDetector:
    """Brightness-invariant hard-cut detector.

    Score = mean absolute difference between the current and previous frame after
    per-frame z-normalization of a small grayscale image. Normalization removes
    global brightness/contrast changes (camera exposure bumps) while preserving
    structural changes (real cuts).
    """

    def __init__(
        self,
        threshold: float,
        min_gap_frames: int = 0,
        size: tuple[int, int] = (160, 90),
    ) -> None:
        self.threshold = float(threshold)
        self.min_gap_frames = int(min_gap_frames)
        self.size = size
        self._prev: np.ndarray | None = None
        self._frames_since_cut = 10**9

    def _normalized_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        return (gray - gray.mean()) / (gray.std() + 1e-6)

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return (is_cut, score) for this frame; score is 0.0 on the first frame."""
        gray = self._normalized_gray(frame)
        if self._prev is None:
            self._prev = gray
            return False, 0.0

        score = float(np.mean(np.abs(gray - self._prev)))
        self._prev = gray
        self._frames_since_cut += 1

        is_cut = score > self.threshold and self._frames_since_cut >= self.min_gap_frames
        if is_cut:
            self._frames_since_cut = 0
        return is_cut, score
