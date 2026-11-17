"""
Automated Unit Tests for OsteoGuard AI Preprocessing Pipeline.
Uses synthetic NumPy test arrays solely for software validation.
Explicitly distinguished from clinical medical training data.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

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


def test_ct_hu_conversion_and_bone_windowing():
    """
    Verifies Hounsfield Unit conversion and linear bone-windowing (+400/1500 HU).
    Checks exact mathematical boundary values:
      -350 HU -> 0.0
      +400 HU -> 0.5
      +1150 HU -> 1.0
    """
    # Raw pixel values with slope=1.0 and intercept=-1024
    raw_pixels = np.array([0, 674, 1424, 2174, 3024], dtype=np.float32)
    slope = 1.0
    intercept = -1024.0

    hu = convert_to_hounsfield_units(raw_pixels, rescale_slope=slope, rescale_intercept=intercept)
    # Expected HU: [-1024, -350, 400, 1150, 2000]
    expected_hu = np.array([-1024.0, -350.0, 400.0, 1150.0, 2000.0], dtype=np.float32)
    np.testing.assert_allclose(hu, expected_hu, atol=1e-4)

    # Apply bone window
    windowed = apply_bone_window(hu, window_center=400.0, window_width=1500.0)

    # -1024 HU (Air) clips to 0.0
    assert windowed[0] == 0.0
    # -350 HU (Lower boundary) maps to 0.0
    assert windowed[1] == 0.0
    # +400 HU (Center / Cancellous bone) maps to 0.5
    assert windowed[2] == pytest.approx(0.5, abs=1e-4)
    # +1150 HU (Upper boundary) maps to 1.0
    assert windowed[3] == pytest.approx(1.0, abs=1e-4)
    # +2000 HU (Dense cortical bone) clips to 1.0
    assert windowed[4] == 1.0


def test_distinct_ct_and_xray_normalizations():
    """
    Strict Rule: Never apply identical normalization to X-rays and CT.
    Verifies that CT and X-ray preprocessing produce distinct numerical responses.
    """
    # Create synthetic test gradient array
    synthetic_arr = np.linspace(-500, 1500, 100, dtype=np.float32).reshape(10, 10)

    ct_norm, ct_prov = preprocess_ct_image(
        synthetic_arr,
        rescale_slope=1.0,
        rescale_intercept=0.0,
        window_center=400.0,
        window_width=1500.0,
    )
    xray_norm, xray_prov = preprocess_xray_image(
        synthetic_arr,
        use_clahe=False,
    )

    assert ct_prov["pipeline"] == "ct_bone_window"
    assert xray_prov["pipeline"] == "xray_intensity_normalization"

    # Distinct distributions
    assert not np.array_equal(ct_norm, xray_norm)
    # CT maps -500 HU to 0.0 (clipped at -350 HU)
    assert ct_norm[0, 0] == 0.0
    # X-ray maps the minimum (-500) to 0.0 via percentile scaling
    assert xray_norm[0, 0] == 0.0
    # Mid-values differ fundamentally
    assert ct_norm[5, 5] != pytest.approx(xray_norm[5, 5], abs=0.05)


def test_xray_clahe_preprocessing():
    """Verifies that CLAHE produces a valid 2D float32 array in [0.0, 1.0]."""
    # Create synthetic 2D test array
    np.random.seed(42)
    synthetic_xray = np.random.uniform(0.1, 0.9, (64, 64)).astype(np.float32)

    equalized = apply_clahe(synthetic_xray, clip_limit=2.0, tile_grid_size=(8, 8))

    assert equalized.shape == (64, 64)
    assert equalized.dtype == np.float32
    assert float(np.min(equalized)) >= 0.0
    assert float(np.max(equalized)) <= 1.0


def test_spatial_resizing_preserves_aspect_ratio():
    """
    Verifies aspect-ratio preserving resizing with symmetric zero padding.
    A non-square 100x200 array resized to 224x224 must add vertical padding.
    """
    rect_arr = np.ones((100, 200), dtype=np.float32) * 0.8
    padded, scale, padding = resize_preserve_aspect_ratio(rect_arr, target_size=(224, 224))

    assert padded.shape == (224, 224)
    # Scale: min(224/100, 224/200) = min(2.24, 1.12) = 1.12
    assert scale == pytest.approx(1.12, abs=1e-2)

    pad_top, pad_bottom, pad_left, pad_right = padding
    # Padding should be primarily vertical (top & bottom)
    assert pad_top > 0
    assert pad_bottom > 0
    # Center area retains value 0.8
    center_val = padded[112, 112]
    assert center_val == pytest.approx(0.8, abs=1e-2)
    # Edge padded area must be 0.0
    assert padded[0, 0] == 0.0


def test_to_torch_backbone_tensor():
    """Verifies conversion to 3-channel PyTorch tensor with ImageNet normalization."""
    arr = np.full((224, 224), 0.5, dtype=np.float32)
    tensor = to_torch_backbone_tensor(arr, normalize_imagenet=True)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # (0.5 - 0.485) / 0.229 ≈ 0.0655
    expected_ch0 = (0.5 - 0.485) / 0.229
    assert tensor[0, 0, 0].item() == pytest.approx(expected_ch0, abs=1e-3)


def test_phi_deidentification_sanitization():
    """Verifies HIPAA Safe Harbor redaction of all 18 PHI identifiers."""
    raw_metadata = {
        "PatientName": "Doe^John",
        "PatientID": "MRN-987654321",
        "PatientBirthDate": "19600101",
        "PatientAge": "64Y",
        "AccessionNumber": "ACC-12345",
        "InstitutionName": "Metropolitan Hospital",
        "ReferringPhysicianName": "Smith^Alice",
        "StudyDate": "20261114",
        "SliceThickness": "1.25",
        "PixelSpacing": "[0.75, 0.75]",
        "KVP": "120",
    }

    sanitized, redacted = sanitize_dicom_metadata(raw_metadata)

    # Identifiers must be removed or hashed
    assert "PatientName" not in sanitized
    assert "PatientBirthDate" not in sanitized
    assert "AccessionNumber" not in sanitized
    assert "InstitutionName" not in sanitized
    assert "ReferringPhysicianName" not in sanitized
    assert "StudyDate" not in sanitized

    # Patient ID must be pseudo-anonymized
    assert sanitized["PatientID"].startswith("PT_ANON_")
    assert "MRN" not in sanitized["PatientID"]

    # Coarsened age group
    assert sanitized["PatientAgeGroup"] == "60-69"

    # Non-identifying parameters must be preserved
    assert sanitized["SliceThickness"] == "1.25"
    assert sanitized["KVP"] == "120"
    assert "PatientName" in redacted
    assert "PatientBirthDate" in redacted


def test_orientation_validation():
    """Verifies image orientation validation with direction cosines."""
    # Standard Axial CT: row=[1, 0, 0], col=[0, 1, 0] -> Orthogonal
    valid, err = validate_orientation((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    assert valid is True
    assert err is None

    # Corrupted / Non-orthogonal direction cosines: dot product != 0
    valid, err = validate_orientation((1.0, 0.5, 0.0, 0.5, 1.0, 0.0))
    assert valid is False
    assert "non-orthogonal" in err


def test_corrupt_file_detection(tmp_path: Path):
    """Verifies that empty, corrupt, or unsupported files raise appropriate exceptions."""
    # Zero-byte file
    empty_file = tmp_path / "empty.png"
    empty_file.touch()
    with pytest.raises(CorruptImageError) as exc:
        load_medical_image(empty_file)
    assert "Zero-byte file" in str(exc.value)

    # Corrupt truncated PNG
    corrupt_png = tmp_path / "corrupt.png"
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    with pytest.raises(CorruptImageError):
        load_medical_image(corrupt_png)

    # Unsupported format
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("not an image", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        load_medical_image(txt_file)


def test_standard_formats_loading(tmp_path: Path):
    """Verifies loading of PNG, JPG, and TIFF image formats."""
    # Varied synthetic pixel array to clear dynamic range quality check
    arr = np.linspace(10, 240, 64 * 64, dtype=np.uint8).reshape(64, 64)
    img = Image.fromarray(arr)

    # PNG
    png_path = tmp_path / "test.png"
    img.save(png_path)
    loaded_png = load_medical_image(png_path)
    assert loaded_png.format == "png"
    assert loaded_png.pixel_array.shape == (64, 64)

    # JPEG
    jpg_path = tmp_path / "test.jpg"
    img.save(jpg_path)
    loaded_jpg = load_medical_image(jpg_path)
    assert loaded_jpg.format == "jpeg"

    # TIFF
    tif_path = tmp_path / "test.tiff"
    img.save(tif_path)
    loaded_tif = load_medical_image(tif_path)
    assert loaded_tif.format == "tiff"


def test_end_to_end_preprocessed_study(tmp_path: Path):
    """Verifies complete preprocess_medical_study pipeline execution and provenance."""
    # Create synthetic test radiograph
    arr = np.linspace(20, 230, 64 * 80, dtype=np.uint8).reshape(64, 80)
    img = Image.fromarray(arr)
    img_path = tmp_path / "study.png"
    img.save(img_path)

    result = preprocess_medical_study(img_path, target_size=(224, 224), use_clahe=True)

    assert isinstance(result, PreprocessedStudyResult)
    assert result.tensor_224.shape == (3, 224, 224)
    assert result.normalized_2d.shape == (224, 224)
    assert result.raw_array.shape == (64, 80)  # Raw array preserved untouched
    assert result.modality == "xray"

    # Provenance audit trail
    prov = result.provenance
    assert prov["source_file"] == str(img_path)
    assert len(prov["source_sha256"]) == 64
    assert prov["target_size"] == [224, 224]
    assert prov["modality_pipeline"]["clahe_applied"] is True
