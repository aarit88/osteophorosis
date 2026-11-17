"""
OsteoGuard AI Medical Preprocessing Package.
Provides format loaders, DICOM metadata readers, HIPAA PHI de-identification,
modality-specific normalization (CT bone-window & X-ray CLAHE), and spatial standardizers.
"""

from ml_service.preprocessing.ct_pipeline import (
    DEFAULT_CT_WINDOW_CENTER,
    DEFAULT_CT_WINDOW_WIDTH,
    apply_bone_window,
    convert_to_hounsfield_units,
    preprocess_ct_image,
)
from ml_service.preprocessing.deidentification import (
    generate_salted_pseudo_id,
    sanitize_dicom_metadata,
)
from ml_service.preprocessing.loader import (
    CorruptImageError,
    LoadedMedicalImage,
    SpatialMetadata,
    UnsupportedFormatError,
    load_medical_image,
    validate_image_quality,
    validate_orientation,
)
from ml_service.preprocessing.standardizer import (
    PreprocessedStudyResult,
    preprocess_medical_study,
    resize_preserve_aspect_ratio,
    to_torch_backbone_tensor,
)
from ml_service.preprocessing.xray_pipeline import (
    apply_clahe,
    normalize_xray_intensity,
    preprocess_xray_image,
)

__all__ = [
    # Loader
    "load_medical_image",
    "LoadedMedicalImage",
    "SpatialMetadata",
    "CorruptImageError",
    "UnsupportedFormatError",
    "validate_orientation",
    "validate_image_quality",
    # De-identification
    "sanitize_dicom_metadata",
    "generate_salted_pseudo_id",
    # CT Pipeline
    "convert_to_hounsfield_units",
    "apply_bone_window",
    "preprocess_ct_image",
    "DEFAULT_CT_WINDOW_CENTER",
    "DEFAULT_CT_WINDOW_WIDTH",
    # X-ray Pipeline
    "normalize_xray_intensity",
    "apply_clahe",
    "preprocess_xray_image",
    # Standardization & Pipelines
    "resize_preserve_aspect_ratio",
    "to_torch_backbone_tensor",
    "preprocess_medical_study",
    "PreprocessedStudyResult",
]
