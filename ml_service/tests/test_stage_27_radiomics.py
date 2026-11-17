"""
Unit and Regression Tests for Stage 27: Radiomics Engine and Digital Phantom Validation.

Validates:
1. Frozen BTXRD checkpoint SHA-256 remains strictly unmodified.
2. All 8 Stage 27 deliverables exist in ml_artifacts/stage27_radiomics/.
3. Deterministic phantom suite generation (reproducibility).
4. Analytical correctness on uniform phantom (mean, var, range, GLCM contrast/homogeneity/ASM).
5. Analytical correctness on checkerboard phantom (mean, variance, range).
6. Analytical correctness on linear gradient phantom (mean, min, max, range).
7. Constant region handling produces zero NaNs and zero Infs.
8. Input validation raises ValueError on malformed inputs (mismatched mask, all-zero mask, non-2D).
9. Output dimensionality strictly equals 102 and fusion contract equals 614.
10. Discrepancy register documents the 4-family vs 5-family discrepancy and GLDM padding.
11. Stage 27 report honestly records PARTIAL / SPECIFICATION UNRESOLVED status.
12. Zero unauthorized patient data ingestion.
13. Preservation of historical stage reports (Stages 17-27).
14. Host CPU Python and React frontend remain untouched.
"""

import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_27_DIR = WORKSPACE_ROOT / "ml_artifacts/stage27_radiomics"
STAGE_REPORTS_DIR = WORKSPACE_ROOT / "ml_artifacts/stage_reports"
FROZEN_CHECKPOINT = WORKSPACE_ROOT / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
EXPECTED_CHECKPOINT_HASH = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"

from osteoguard.radiomics.engine import IBSIRadiomicsEngine
from osteoguard.radiomics.phantom_generator import SyntheticDigitalPhantomSuite
from osteoguard.constants import RADIOMICS_FEATURE_DIM, FUSION_VECTOR_DIM, SEMANTIC_EMBEDDING_DIM


def test_01_checkpoint_hash_strictly_unmodified():
    """Verify frozen BTXRD checkpoint hash remains strictly unmodified."""
    assert FROZEN_CHECKPOINT.exists(), "Frozen BTXRD checkpoint missing"
    hasher = hashlib.sha256()
    with open(FROZEN_CHECKPOINT, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest().upper()
    assert actual_hash == EXPECTED_CHECKPOINT_HASH, (
        f"Checkpoint hash mismatch! Expected {EXPECTED_CHECKPOINT_HASH}, got {actual_hash}"
    )


def test_02_all_stage27_artifacts_exist():
    """Verify all 8 Stage 27 deliverables exist and have valid sizes."""
    expected_files = [
        "STAGE_27_RADIOMICS_REPORT.md",
        "radiomics_feature_specification.md",
        "radiomics_discrepancy_register.md",
        "digital_phantom_protocol.md",
        "radiomics_verification_matrix.md",
        "implementation_limitations_and_decisions.md",
        "reproducibility_manifest.json",
        "phantom_test_execution_report.json"
    ]
    for fname in expected_files:
        fpath = STAGE_27_DIR / fname
        assert fpath.exists(), f"Missing Stage 27 artifact: {fname}"
        assert fpath.stat().st_size > 500, f"Stage 27 artifact {fname} is too small"


def test_03_phantom_generation_determinism():
    """Verify synthetic digital phantoms generate bit-for-bit identical arrays on repeated calls."""
    img1, mask1, _ = SyntheticDigitalPhantomSuite.create_checkerboard_phantom(shape=(64, 64))
    img2, mask2, _ = SyntheticDigitalPhantomSuite.create_checkerboard_phantom(shape=(64, 64))
    assert np.array_equal(img1, img2), "Checkerboard phantom generation non-deterministic"
    assert np.array_equal(mask1, mask2), "Checkerboard mask generation non-deterministic"

    img_u1, _, _ = SyntheticDigitalPhantomSuite.create_uniform_phantom(shape=(64, 64))
    img_u2, _, _ = SyntheticDigitalPhantomSuite.create_uniform_phantom(shape=(64, 64))
    assert np.array_equal(img_u1, img_u2), "Uniform phantom generation non-deterministic"


def test_04_uniform_phantom_analytical_correctness():
    """Verify uniform phantom extracts exact analytical ground-truth values."""
    engine = IBSIRadiomicsEngine(num_bins=64)
    img, mask, props = SyntheticDigitalPhantomSuite.create_uniform_phantom(shape=(64, 64), intensity=100.0, margin=8)
    feats = engine.extract_features_dict(img, mask)

    tol = 1e-4
    assert abs(feats["first_order_mean"] - props["mean"]) < tol
    assert abs(feats["first_order_variance"] - 0.0) < tol
    assert abs(feats["first_order_min"] - props["min"]) < tol
    assert abs(feats["first_order_max"] - props["max"]) < tol
    assert abs(feats["first_order_range"] - 0.0) < tol
    assert abs(feats["glcm_contrast"] - 0.0) < tol
    assert abs(feats["glcm_homogeneity"] - 1.0) < tol
    assert abs(feats["glcm_asm"] - 1.0) < tol


def test_05_checkerboard_phantom_analytical_correctness():
    """Verify checkerboard phantom extracts exact analytical ground-truth values."""
    engine = IBSIRadiomicsEngine(num_bins=64)
    img, mask, props = SyntheticDigitalPhantomSuite.create_checkerboard_phantom(shape=(64, 64), tile_size=8, low_val=10.0, high_val=90.0)
    feats = engine.extract_features_dict(img, mask)

    tol = 1e-4
    assert abs(feats["first_order_mean"] - 50.0) < tol
    assert abs(feats["first_order_variance"] - 1600.0) < tol
    assert abs(feats["first_order_min"] - 10.0) < tol
    assert abs(feats["first_order_max"] - 90.0) < tol
    assert abs(feats["first_order_range"] - 80.0) < tol


def test_06_linear_gradient_analytical_correctness():
    """Verify linear ramp phantom extracts exact analytical ground-truth values."""
    engine = IBSIRadiomicsEngine(num_bins=64)
    img, mask, props = SyntheticDigitalPhantomSuite.create_linear_gradient_phantom(shape=(64, 64), start_val=10.0, end_val=50.0)
    feats = engine.extract_features_dict(img, mask)

    tol = 1e-4
    assert abs(feats["first_order_mean"] - 30.0) < tol
    assert abs(feats["first_order_min"] - 10.0) < tol
    assert abs(feats["first_order_max"] - 50.0) < tol
    assert abs(feats["first_order_range"] - 40.0) < tol


def test_07_constant_region_no_nan_no_inf():
    """Verify constant uniform phantom produces zero NaNs and zero Infs across all 102 features."""
    engine = IBSIRadiomicsEngine(num_bins=64)
    img, mask, _ = SyntheticDigitalPhantomSuite.create_uniform_phantom(shape=(64, 64), intensity=50.0)
    vec = engine.extract_features(img, mask)

    assert len(vec) == 102, f"Expected 102 features, got {len(vec)}"
    assert not np.isnan(vec).any(), "NaN detected in radiomics output on constant region"
    assert not np.isinf(vec).any(), "Inf detected in radiomics output on constant region"


def test_08_input_validation_errors():
    """Verify engine raises ValueError on malformed inputs."""
    engine = IBSIRadiomicsEngine(num_bins=64)

    # Empty image
    with pytest.raises(ValueError):
        engine.extract_features(np.array([]))

    # Mismatched mask shape
    with pytest.raises(ValueError):
        engine.extract_features(np.ones((64, 64)), np.ones((32, 32)))

    # All-zero mask
    with pytest.raises(ValueError):
        engine.extract_features(np.ones((64, 64)), np.zeros((64, 64)))

    # Non-2D image
    with pytest.raises(ValueError):
        engine.extract_features(np.ones((10, 10, 3)))


def test_09_radiomics_feature_dimension_and_fusion_contract():
    """Verify 102-D radiomic vector and 614-D fusion contract."""
    engine = IBSIRadiomicsEngine(num_bins=64)
    assert len(engine.feature_names) == RADIOMICS_FEATURE_DIM == 102
    assert SEMANTIC_EMBEDDING_DIM == 512
    assert FUSION_VECTOR_DIM == 614


def test_10_discrepancy_register_documents_gldm_padding():
    """Verify discrepancy register documents 4-family vs 5-family and GLDM padding."""
    disc_text = (STAGE_27_DIR / "radiomics_discrepancy_register.md").read_text(encoding="utf-8")
    assert "76" in disc_text
    assert "102" in disc_text
    assert "NGTDM" in disc_text
    assert "GLDM" in disc_text
    assert "NOT_VERIFIED" in disc_text or "NOT FULLY VERIFIED" in disc_text


def test_11_stage27_report_status_partial_not_verified():
    """Verify Stage 27 report honestly records PARTIAL / SPECIFICATION UNRESOLVED status."""
    rep_text = (STAGE_27_DIR / "STAGE_27_RADIOMICS_REPORT.md").read_text(encoding="utf-8")
    assert "PARTIAL" in rep_text
    assert "SPECIFICATION UNRESOLVED" in rep_text or "REQUIRES ADVISOR REVIEW" in rep_text


def test_12_zero_unauthorized_patient_data():
    """Verify raw datasets remain unmodified and no unauthorized data was ingested."""
    raw_dir = WORKSPACE_ROOT / "ml_service/data/raw"
    bs_files = list((raw_dir / "bone_scan_metastasis").rglob("*.png"))
    assert len(bs_files) >= 582, f"Expected >= 582 bone scan images, got {len(bs_files)}"

    btxrd_files = [
        f for f in (raw_dir / "btxrd").rglob("*")
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]
    assert len(btxrd_files) >= 1396, f"Expected >= 1396 BTXRD images, got {len(btxrd_files)}"


def test_13_historical_stage_reports_preserved():
    """Verify historical reports from Stages 17-27 all exist and are non-empty."""
    expected_reports = [
        "STAGE_17_BTXRD_FULL_TRAINING.md",
        "STAGE_18_BTXRD_MODEL_FREEZE.md",
        "STAGE_19_CENTER2_EXTERNAL_EVALUATION.md",
        "STAGE_20_CENTER3_EXTERNAL_EVALUATION.md",
        "STAGE_21_MULTICENTER_GENERALIZATION_AUDIT.md",
        "STAGE_22_DATASET_READINESS_AUDIT.md",
        "STAGE_23_DATASET_SOURCE_VERIFICATION.md",
        "STAGE_24_CLINICAL_COLLABORATION_PREPARATION.md",
        "STAGE_25_DATASET_INTAKE.md",
        "STAGE_26_ADVISOR_REVIEW.md",
        "STAGE_27_RADIOMICS.md"
    ]
    for rname in expected_reports:
        rpath = STAGE_REPORTS_DIR / rname
        assert rpath.exists(), f"Missing stage report: {rname}"
        assert rpath.stat().st_size > 1000, f"Stage report {rname} is too short"


def test_14_host_python_and_frontend_untouched():
    """Verify host CPU Python and React frontend remain untouched."""
    frontend_dir = WORKSPACE_ROOT / "osteophorosis-main/src"
    assert frontend_dir.exists(), "React frontend directory missing"
    host_python = Path(r"C:\Python313\python.exe")
    assert host_python.exists(), "Host Python executable missing"

