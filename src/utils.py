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
    """Simple hard-cut detector based on mean absolute grayscale frame difference.

    Downscales to a small grayscale image so the score is robust to noise and
    cheap to compute. Threshold is on the 0-255 intensity scale.
    """

    def __init__(self, threshold: float, size: tuple[int, int] = (160, 90)) -> None:
        self.threshold = float(threshold)
        self.size = size
        self._prev_gray: np.ndarray | None = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return (is_cut, score) for this frame; score is 0.0 on the first frame."""
        gray = self._gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0.0
        score = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
        self._prev_gray = gray
        return score > self.threshold, score
