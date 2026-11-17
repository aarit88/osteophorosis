"""
OsteoGuard AI - Modality Strategy and Preprocessing Engine.
Enforces strict modality separation (Rule 8: Never mix X-rays, CT scans,
nuclear bone scans and animal micro-CT unscientifically).
Provides calibration for CT Hounsfield Units and Plain Radiograph CLAHE.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import cv2

from osteoguard.constants import (
    CT_BONE_WINDOW_LEVEL,
    CT_BONE_WINDOW_WIDTH,
    SUPPORTED_MODALITIES,
    XRAY_CLAHE_CLIP_LIMIT,
    XRAY_CLAHE_TILE_GRID_SIZE,
    ModalityType,
)


class ModalityValidationError(Exception):
    """Raised when an unapproved modality or animal scan is detected."""
    pass


class ModalityStrategyEngine:
    """
    Scientifically sound radiometric calibration and validation per imaging modality.
    """

    @staticmethod
    def validate_modality(modality: str) -> str:
        mod_lower = modality.strip().lower()
        if mod_lower not in SUPPORTED_MODALITIES:
            raise ModalityValidationError(
                f"Unsupported modality '{modality}'. Supported modalities are: {SUPPORTED_MODALITIES}. "
                "Mixing unapproved modalities or animal micro-CT violates scientific integrity protocols."
            )
        return mod_lower

    @staticmethod
    def calibrate_ct_bone_window(
        raw_array: np.ndarray,
        rescale_slope: float = 1.0,
        rescale_intercept: float = -1024.0,
        window_level: float = CT_BONE_WINDOW_LEVEL,
        window_width: float = CT_BONE_WINDOW_WIDTH,
    ) -> np.ndarray:
        """
        Converts raw CT voxel attenuation values to standardized Hounsfield Units (HU)
        and applies an anatomical bone window (paper Section: Radiometric Calibration):
          HU = raw * RescaleSlope + RescaleIntercept
          I_win = clip( (HU - (WL - 0.5 * WW)) / WW, 0, 1 )
        """
        hu = raw_array.astype(np.float32) * rescale_slope + rescale_intercept
        lower_bound = window_level - 0.5 * window_width
        upper_bound = window_level + 0.5 * window_width

        clamped = np.clip((hu - lower_bound) / (window_width + 1e-8), 0.0, 1.0)
        return clamped.astype(np.float32)

    @staticmethod
    def calibrate_xray_clahe(
        image_array: np.ndarray,
        clip_limit: float = XRAY_CLAHE_CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = XRAY_CLAHE_TILE_GRID_SIZE,
    ) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to normalize contrast variability and scatter radiation in digital plain radiographs.
        """
        # Ensure 8-bit unsigned integer input for CLAHE
        if image_array.dtype != np.uint8:
            norm = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)
            img_u8 = (norm * 255.0).astype(np.uint8)
        else:
            img_u8 = image_array

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized = clahe.apply(img_u8)
        # Rescale to float [0.0, 1.0]
        return (equalized.astype(np.float32) / 255.0)

    @classmethod
    def preprocess_image(
        cls,
        image_array: np.ndarray,
        modality: str = ModalityType.XRAY.value,
        ct_metadata: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Dispatches radiometric calibration based on validated modality."""
        mod = cls.validate_modality(modality)
        if mod == ModalityType.CT.value:
            slope = ct_metadata.get("rescale_slope", 1.0) if ct_metadata else 1.0
            intercept = ct_metadata.get("rescale_intercept", -1024.0) if ct_metadata else -1024.0
            return cls.calibrate_ct_bone_window(image_array, slope, intercept)
        elif mod == ModalityType.XRAY.value:
            return cls.calibrate_xray_clahe(image_array)
        elif mod == ModalityType.BONE_SCAN.value:
            # Nuclear bone scan intensity normalization
            min_v = np.min(image_array)
            max_v = np.max(image_array)
            norm = (image_array - min_v) / (max_v - min_v + 1e-8)
            return norm.astype(np.float32)
        raise ModalityValidationError(f"Unhandled modality: {modality}")

