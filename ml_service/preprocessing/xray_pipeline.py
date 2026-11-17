"""
OsteoGuard AI - Planar Radiography (X-ray) Preprocessing Pipeline.
Applies robust percentile intensity scaling and optional Contrast-Limited Adaptive
Histogram Equalization (CLAHE) to reveal subtle cortical thinning and trabecular patterns.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


def normalize_xray_intensity(
    arr: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """
    Robust intensity normalization for planar radiographs:
      1. Computes lower and upper percentile cutoff thresholds.
      2. Clips extreme outliers (dead pixels, direct X-ray beam collimation).
      3. Min-Max scales to [0.0, 1.0].
    """
    if lower_percentile >= upper_percentile:
        raise ValueError(f"Lower percentile ({lower_percentile}) must be strictly less than upper ({upper_percentile})")

    p_low = np.percentile(arr, lower_percentile)
    p_high = np.percentile(arr, upper_percentile)

    if p_high <= p_low:
        # Fallback to absolute min-max if distribution is degenerate
        p_low = np.min(arr)
        p_high = np.max(arr)

    if p_high <= p_low:
        # Uniform or dead image
        return np.zeros_like(arr, dtype=np.float32)

    clipped = np.clip(arr, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)

    return normalized.astype(np.float32)


def apply_clahe(
    arr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE):
      - Operates on [0.0, 1.0] float array converted to 8-bit representation.
      - Uses OpenCV createCLAHE with specified clip limit and tile grid.
      - Returns normalized float32 array in [0.0, 1.0].
    """
    if arr.ndim != 2:
        raise ValueError(f"CLAHE requires a 2D planar array, got shape {arr.shape}")

    # Scale to 8-bit integer domain
    scaled_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    equalized_uint8 = clahe.apply(scaled_uint8)

    # Return float32 in [0.0, 1.0]
    return (equalized_uint8.astype(np.float32) / 255.0)


def preprocess_xray_image(
    raw_array: np.ndarray,
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes end-to-end planar radiograph preprocessing:
      1. Robust percentile intensity normalization to [0.0, 1.0].
      2. Optional CLAHE enhancement to accentuate trabecular details.
    Returns:
      (normalized_array, provenance_metadata)
    """
    normalized = normalize_xray_intensity(
        raw_array,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    clahe_applied = False
    if use_clahe and normalized.ndim == 2:
        normalized = apply_clahe(normalized, clip_limit=clip_limit)
        clahe_applied = True

    provenance: Dict[str, Any] = {
        "pipeline": "xray_intensity_normalization",
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "clahe_applied": clahe_applied,
        "clip_limit": float(clip_limit) if clahe_applied else None,
        "output_min": float(np.min(normalized)),
        "output_max": float(np.max(normalized)),
    }

    return normalized, provenance

