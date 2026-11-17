"""
OsteoGuard AI - Computed Tomography (CT) Preprocessing Pipeline.
Converts raw CT stored attenuation values to physical Hounsfield Units (HU)
and applies standard skeletal bone-windowing (WL: +400 HU, WW: 1500 HU).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


# Standard Clinical Bone Window Parameters
DEFAULT_CT_WINDOW_CENTER: float = 400.0  # Window Level (WL) in HU
DEFAULT_CT_WINDOW_WIDTH: float = 1500.0  # Window Width (WW) in HU


def convert_to_hounsfield_units(
    raw_pixels: np.ndarray,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
) -> np.ndarray:
    """
    Applies DICOM Rescale Slope (0028,1053) and Rescale Intercept (0028,1052):
      HU = (pixel_value * rescale_slope) + rescale_intercept
    """
    hu_array = (raw_pixels.astype(np.float32) * float(rescale_slope)) + float(rescale_intercept)
    return hu_array


def apply_bone_window(
    hu_array: np.ndarray,
    window_center: float = DEFAULT_CT_WINDOW_CENTER,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> np.ndarray:
    """
    Applies linear bone-windowing:
      Lower threshold = window_center - (window_width / 2.0)  # e.g., 400 - 750 = -350 HU
      Upper threshold = window_center + (window_width / 2.0)  # e.g., 400 + 750 = +1150 HU
    Clips voxel intensities and linearly scales output to [0.0, 1.0].
    """
    if window_width <= 0:
        raise ValueError(f"Window width must be strictly positive, got {window_width}")

    lower_bound = window_center - (window_width / 2.0)
    upper_bound = window_center + (window_width / 2.0)

    clipped = np.clip(hu_array, lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / float(window_width)

    return normalized.astype(np.float32)


def preprocess_ct_image(
    raw_array: np.ndarray,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    window_center: float = DEFAULT_CT_WINDOW_CENTER,
    window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes end-to-end CT preprocessing:
      1. Converts raw pixels to Hounsfield Units.
      2. Applies skeletal bone-windowing to produce [0.0, 1.0] float32 tensor.
    Returns:
      (normalized_array, provenance_metadata)
    """
    # Defensive copy: raw input data is preserved untouched
    hu = convert_to_hounsfield_units(raw_array, rescale_slope, rescale_intercept)
    normalized = apply_bone_window(hu, window_center, window_width)

    provenance: Dict[str, Any] = {
        "pipeline": "ct_bone_window",
        "rescale_slope": float(rescale_slope),
        "rescale_intercept": float(rescale_intercept),
        "window_center": float(window_center),
        "window_width": float(window_width),
        "hu_min_observed": float(np.min(hu)),
        "hu_max_observed": float(np.max(hu)),
        "hu_mean_observed": float(np.mean(hu)),
        "output_min": float(np.min(normalized)),
        "output_max": float(np.max(normalized)),
    }

    return normalized, provenance

