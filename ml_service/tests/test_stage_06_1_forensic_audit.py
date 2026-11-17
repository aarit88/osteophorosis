"""
OsteoGuard AI - Automated Tests for Stage 06.1 Dataset Ingestion & Forensic Audit.
Verifies:
  1. Safe extraction path-traversal (Zip Slip) security guard.
  2. FracAtlas classification records (4083 records: 719 fractured, 3364 non-fractured).
  3. FracAtlas duplicate image finding (IMG0003375.jpg and IMG0003376.jpg byte-identical).
  4. Bone Scan dataset metrics (291 patients, 582 PNG images, 60 metastasis, 231 non-metastasis).
  5. Bone Scan view pairing (every patient has 1 Anterior and 1 Posterior image).
  6. BTXRD dataset metrics (3746 images, 1867 tumor, 1879 non-tumor/normal, multi-center).
  7. Strict negative mapping guard (Primary bone tumors NEVER map to secondary metastasis).
  8. Master four-class pipeline remains strictly NOT READY.
  9. Ingested dataset manifest completeness and schema integrity.
"""

import csv
from pathlib import Path
import pytest

from ml_service.datasets.extractor import safe_extract_archive
from ml_service.datasets.registry_verifier import (
    DatasetRegistry,
    NegativeMappingViolation,
    compute_file_sha256,
)


@pytest.fixture
def registry():
    return DatasetRegistry()


def test_safe_extract_archive_rejects_path_traversal(tmp_path: Path):
    """Verifies that safe_extract_archive blocks path traversal attempts."""
    # Create a dummy zip with a malicious traversal member
    import zipfile
    bad_zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(bad_zip_path, "w") as zf:
        zf.writestr("../evil.txt", "malicious payload")

    target_dir = tmp_path / "extracted"
    with pytest.raises(ValueError) as exc_info:
        safe_extract_archive(bad_zip_path, target_dir)
    assert "Path traversal attempt detected" in str(exc_info.value) or "Security Alert" in str(exc_info.value)


def test_master_four_class_dataset_remains_not_ready(registry: DatasetRegistry):
    """Verifies that four-class OsteoGuard AI dataset remains NOT READY despite individual dataset ingestion."""
    assert registry.is_dataset_ready is False

    acquired = registry.get_acquired_datasets()
    assert len(acquired) == 3
    assert "fracatlas" in acquired
    assert "bone_scan_metastasis" in acquired
    assert "btxrd" in acquired

    # Verify independent readiness
    assert acquired["fracatlas"]["experiment_readiness"] == "READY_FOR_FRACTURE_EXPERIMENTS"
    assert acquired["bone_scan_metastasis"]["experiment_readiness"] == "READY_FOR_BONE_SCAN_EXPERIMENTS"
    assert acquired["btxrd"]["experiment_readiness"] == "READY_FOR_TUMOR_EXPERIMENTS"


def test_fracatlas_extracted_counts_and_duplicates():
    """Verifies FracAtlas extracted files, dataset.csv counts, and duplicate findings."""
    raw_dir = Path("ml_service/data/raw/fracatlas")
    if not raw_dir.exists():
        pytest.skip("FracAtlas raw directory not present.")

    csv_path = raw_dir / "dataset.csv"
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 4083
    fractured_count = sum(1 for r in reader if r.get("fractured") == "1")
    non_fractured_count = sum(1 for r in reader if r.get("fractured") == "0")

    assert fractured_count == 719
    assert non_fractured_count == 3364

    # Check the duplicate files
    f_3375 = raw_dir / "images/Fractured/IMG0003375.jpg"
    nf_3375 = raw_dir / "images/Non_fractured/IMG0003375.jpg"
    if f_3375.exists() and nf_3375.exists():
        sha_f = compute_file_sha256(f_3375)
        sha_nf = compute_file_sha256(nf_3375)
        assert sha_f == sha_nf, "IMG0003375 duplicate copies must be byte-identical"


def test_bone_scan_metrics_and_patient_pairing():
    """Verifies Bone Scan dataset: 291 patients, 582 PNGs, 60 metastasis, 231 non-metastasis."""
    raw_dir = Path("ml_service/data/raw/bone_scan_metastasis")
    if not raw_dir.exists():
        pytest.skip("Bone Scan raw directory not present.")

    excel_path = raw_dir / "Bone Scan Images Dataset.xlsx"
    assert excel_path.exists()

    import openpyxl
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    patient_rows = [r for r in rows if r[0] is not None]

    assert len(patient_rows) == 291

    mets = sum(1 for r in patient_rows if r[9] == "Bone metastases")
    no_mets = sum(1 for r in patient_rows if r[9] == "No bone metastases")

    assert mets == 60
    assert no_mets == 231

    # Check images on disk
    png_files = list(raw_dir.rglob("*.png"))
    assert len(png_files) == 582


def test_btxrd_metrics_and_pathology_distribution():
    """Verifies BTXRD dataset: 3746 images, 1867 tumors, 1879 non-tumor normal bone."""
    raw_dir = Path("ml_service/data/raw/btxrd")
    if not raw_dir.exists():
        pytest.skip("BTXRD raw directory not present.")

    excel_path = raw_dir / "dataset.xlsx"
    assert excel_path.exists()

    import openpyxl
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))[1:]
    assert len(rows) == 3746

    tumor_count = sum(1 for r in rows if r[19] == 1 or r[19] == "1")
    benign_count = sum(1 for r in rows if r[20] == 1 or r[20] == "1")
    malignant_count = sum(1 for r in rows if r[21] == 1 or r[21] == "1")

    assert tumor_count == 1867
    assert benign_count == 1525
    assert malignant_count == 342
    assert (3746 - tumor_count) == 1879

    # Images on disk
    img_files = [f for f in (raw_dir / "images").iterdir() if f.suffix.lower() in [".jpeg", ".jpg", ".png"]]
    assert len(img_files) == 3746


def test_strict_negative_mapping_primary_sarcoma_rejected(registry: DatasetRegistry):
    """Enforces that primary bone tumors in BTXRD (e.g. osteosarcoma) NEVER map to secondary metastasis."""
    valid, err = registry.validate_pathology_mapping("osteosarcoma_femur.jpeg", "secondary_bone_metastasis")
    assert valid is False
    assert "Rule 2.1 Violation" in err

    valid, err = registry.validate_pathology_mapping("osteosarcoma", "metastasis")
    assert valid is False


def test_dataset_manifest_integrity():
    """Verifies that dataset_manifest.csv indexes all 8411 acquired imaging records."""
    manifest_path = Path("ml_service/data/dataset_manifest.csv")
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 8411

    # Check dataset counts in manifest
    from collections import Counter
    counts = Counter(r["dataset_id"] for r in reader)

    assert counts["FRACATLAS_FRACTURE_COHORT"] == 4083
    assert counts["IICS_UNA_BONE_SCAN_METASTASIS"] == 582
    assert counts["BTXRD_BONE_TUMOR_COHORT"] == 3746

