"""
OsteoGuard AI - Spatial Standardization & Unified Preprocessing Engine.
Performs aspect-ratio preserving resizing, symmetric padding, PyTorch 3-channel
tensor transformation with ImageNet normalization, and complete provenance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from ml_service.preprocessing.ct_pipeline import preprocess_ct_image
from ml_service.preprocessing.loader import (
    LoadedMedicalImage,
    SpatialMetadata,
    load_medical_image,
)
from ml_service.preprocessing.xray_pipeline import preprocess_xray_image
from ml_service.utils.logging import logger


# Standard ImageNet normalization parameters for pre-trained CNN backbones
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class PreprocessedStudyResult:
    """Complete container encapsulating preprocessed tensors, raw data, and full provenance."""
    tensor_224: torch.Tensor          # Shape [3, 224, 224], ImageNet standardized for CNN backbone
    normalized_2d: np.ndarray         # Shape [H_target, W_target], float32 in [0.0, 1.0]
    raw_array: np.ndarray             # Original pixel array preserved untouched
    modality: str                     # "ct" | "xray" | "bone_scan"
    format: str                       # "dicom" | "png" | "jpeg" | "tiff" | "nifti"
    spatial: SpatialMetadata          # Preserved voxel spacing, slice thickness, origin
    provenance: Dict[str, Any]        # Complete audit trail of transforms applied
    warnings: List[str] = field(default_factory=list)


def resize_preserve_aspect_ratio(
    arr: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    pad_value: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
    """
    Resizes 2D image array to target_size (target_h, target_w) preserving aspect ratio.
    Adds symmetric padding along the shorter dimension.
    Returns:
      (padded_array, scale_factor, (pad_top, pad_bottom, pad_left, pad_right))
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D planar array for resizing, got shape {arr.shape}")

    orig_h, orig_w = arr.shape[:2]
    target_h, target_w = target_size

    scale = min(target_h / orig_h, target_w / orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # Resize with specified interpolation
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)

    # Compute symmetric padding
    total_pad_h = target_h - new_h
    total_pad_w = target_w - new_w

    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded.astype(np.float32), float(scale), (pad_top, pad_bottom, pad_left, pad_right)


def to_torch_backbone_tensor(
    normalized_2d: np.ndarray,
    normalize_imagenet: bool = True,
) -> torch.Tensor:
    """
    Converts 2D [0.0, 1.0] float32 array to 3-channel PyTorch tensor [3, H, W]
    with optional standard ImageNet z-score normalization.
    """
    if normalized_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {normalized_2d.shape}")

    # Broadcast to 3 channels: [3, H, W]
    three_channel = np.stack([normalized_2d, normalized_2d, normalized_2d], axis=0)

    if normalize_imagenet:
        # (X - mean) / std
        three_channel = (three_channel - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(three_channel.astype(np.float32))
    return tensor


def preprocess_medical_study(
    file_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    use_clahe: bool = True,
) -> PreprocessedStudyResult:
    """
    Full end-to-end preprocessing pipeline for clinical imaging studies:
      1. Loads file (DICOM, NIfTI, PNG, JPG, TIFF) via load_medical_image.
      2. Validates image quality, orientation, and extracts spatial metadata.
      3. Routes to modality-specific pipeline:
         - CT: converts to Hounsfield Units, applies standard bone window (+400/1500 HU).
         - X-ray: applies percentile intensity normalization and optional CLAHE.
      4. Standardizes spatial dimensions preserving aspect ratio with zero padding.
      5. Generates standardized 3-channel PyTorch tensor for deep visual backbones.
      6. Records complete provenance and non-fatal metadata warnings.
    """
    loaded: LoadedMedicalImage = load_medical_image(file_path)
    warnings: List[str] = list(loaded.warnings)

    raw_2d = loaded.pixel_array
    # If 3D volumetric CT, extract center representative axial slice
    if raw_2d.ndim == 3:
        center_slice_idx = raw_2d.shape[2] // 2
        raw_2d = raw_2d[:, :, center_slice_idx]
        warnings.append(
            f"Volumetric input detected ({loaded.pixel_array.shape}); "
            f"extracted representative center axial slice #{center_slice_idx}."
        )

    # Route based on detected modality
    if loaded.modality == "ct":
        rescale_slope = float(loaded.metadata.get("RescaleSlope", 1.0))
        rescale_intercept = float(loaded.metadata.get("RescaleIntercept", 0.0))
        norm_2d, mod_provenance = preprocess_ct_image(
            raw_2d,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
        )
    else:
        # Default to X-ray / Planar pipeline
        norm_2d, mod_provenance = preprocess_xray_image(
            raw_2d,
            use_clahe=use_clahe,
        )

    # Spatial Standardization (aspect-ratio preserved)
    standardized_2d, scale, padding = resize_preserve_aspect_ratio(
        norm_2d,
        target_size=target_size,
    )

    # Convert to PyTorch CNN input tensor
    tensor_224 = to_torch_backbone_tensor(standardized_2d, normalize_imagenet=True)

    # Compile comprehensive provenance manifest
    provenance: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": loaded.source_path,
        "source_sha256": loaded.checksum_sha256,
        "format": loaded.format,
        "modality": loaded.modality,
        "original_shape": list(loaded.spatial.original_shape),
        "original_dtype": loaded.spatial.original_dtype,
        "voxel_spacing": list(loaded.spatial.voxel_spacing),
        "slice_thickness": loaded.spatial.slice_thickness,
        "origin": list(loaded.spatial.origin),
        "direction_cosines": list(loaded.spatial.direction_cosines),
        "modality_pipeline": mod_provenance,
        "target_size": list(target_size),
        "scale_factor": scale,
        "padding": list(padding),
        "output_tensor_shape": list(tensor_224.shape),
    }

    return PreprocessedStudyResult(
        tensor_224=tensor_224,
        normalized_2d=standardized_2d,
        raw_array=loaded.pixel_array,
        modality=loaded.modality,
        format=loaded.format,
        spatial=loaded.spatial,
        provenance=provenance,
        warnings=warnings,
    )

