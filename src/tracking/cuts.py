"""Two-frame hard-cut detection from HSV histograms and pixel similarity."""

from __future__ import annotations

import cv2
import numpy as np

THRESHOLD_HIGH = 0.975
THRESHOLD_LOW = 0.70
THRESHOLD_PIXEL = 0.85


def compare_histograms(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    bins: int = 32,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return HSV Hue-Saturation histogram correlation for two BGR frames."""
    hsv_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    hsv_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
    hist_current = cv2.calcHist(
        [hsv_current], [0, 1], None, [bins, bins], [0, 180, 0, 256]
    )
    hist_previous = cv2.calcHist(
        [hsv_previous], [0, 1], None, [bins, bins], [0, 180, 0, 256]
    )
    cv2.normalize(hist_current, hist_current)
    cv2.normalize(hist_previous, hist_previous)
    correlation = float(
        cv2.compareHist(hist_current, hist_previous, cv2.HISTCMP_CORREL)
    )
    return correlation, hist_current, hist_previous


def pixel_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Return grayscale pixel similarity in ``[0, 1]`` (1 = identical)."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray1, gray2)
    return float(1 - difference.mean() / 255)


def detect_hard_cuts(
    frame_idx: int,
    frame: np.ndarray,
    previous_frame: np.ndarray,
    threshold_high: float = THRESHOLD_HIGH,
    threshold_low: float = THRESHOLD_LOW,
    threshold_pixel: float = THRESHOLD_PIXEL,
) -> bool:
    """True when the current pair looks like a hard camera cut."""
    correlation, _, _ = compare_histograms(frame, previous_frame)
    similarity = pixel_difference(frame, previous_frame)
    if (
        correlation < threshold_high
        and correlation > threshold_low
        and similarity < threshold_pixel
    ):
        print(
            f"Hard cut detected after frame {frame_idx}, "
            f"correlation: {correlation}, pixel difference: {similarity}"
        )
        return True
    return False
