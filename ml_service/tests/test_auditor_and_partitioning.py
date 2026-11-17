"""
OsteoGuard AI - Automated Tests for Stage 06: Dataset Label Audit & Patient-Level Partitioning.
Verifies:
  1. Exact duplicate detection via SHA-256.
  2. Near-duplicate perceptual hash (dHash) and Hamming distance metric.
  3. Conflicting label detection across patients.
  4. Missing annotations and uncertain diagnostic category detection.
  5. Patient-level stratification with ZERO patient leakage across train, val, and test.
  6. Patient leakage detection verification function.
  7. Multi-site / external site evaluation holdout strategy.
  8. Strict training-only fitting assertion (test data isolation).
  9. Audit artifact generation (dataset_quality_report.json, class_distribution.csv, duplicate_audit.csv, split_manifest.csv).
  10. Zero-dataset audit state correctly marked as NOT_READY.
"""

import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from ml_service.datasets.auditor import (
    DatasetAuditor,
    compute_hamming_distance,
    compute_perceptual_hash,
)
from ml_service.datasets.partitioner import (
    PatientLeakageError,
    PatientLevelPartitioner,
    SPLIT_MANIFEST_COLUMNS,
    TrainingIsolationError,
    assert_training_only_fit,
    verify_patient_leakage,
)
from ml_service.datasets.registry_verifier import compute_file_sha256
from osteoguard.constants import TARGET_CLASSES


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Creates a temporary multi-class mock image tree for testing audit utilities."""
    root = tmp_path / "data"
    for cls in ["metastasis", "paget", "oi", "healthy"]:
        (root / cls).mkdir(parents=True, exist_ok=True)
    return root


def test_exact_duplicate_detection_sha256(temp_dataset_dir: Path):
    """Verifies identical binary files are caught as exact SHA-256 duplicates."""
    cls_dir = temp_dataset_dir / "metastasis"
    img_data = np.linspace(10, 200, 64 * 64, dtype=np.uint8).reshape(64, 64)
    pil_img = Image.fromarray(img_data)

    img1_path = cls_dir / "PT_001_slice01.png"
    img2_path = cls_dir / "PT_002_slice01_copy.png"
    pil_img.save(img1_path)
    pil_img.save(img2_path)

    sha1 = compute_file_sha256(img1_path)
    sha2 = compute_file_sha256(img2_path)
    assert sha1 == sha2

    auditor = DatasetAuditor()
    report = auditor.audit_dataset_directory(temp_dataset_dir)

    assert report["exact_duplicates_count"] >= 1
    exact_dups = [d for d in report["duplicate_pairs"] if d["type"] == "EXACT_SHA256"]
    assert len(exact_dups) >= 1
    assert exact_dups[0]["sha256"] == sha1


def test_near_duplicate_detection_dhash():
    """Verifies that perceptual hashing catches visually near-identical images within Hamming threshold."""
    base_arr = np.linspace(20, 220, 64 * 64, dtype=np.uint8).reshape(64, 64)
    img_base = Image.fromarray(base_arr)

    # Slight perturbation: add +2 to center pixels (visually nearly identical)
    perturbed_arr = base_arr.copy()
    perturbed_arr[30:34, 30:34] = np.clip(perturbed_arr[30:34, 30:34] + 2, 0, 255)
    img_perturbed = Image.fromarray(perturbed_arr)

    # Completely different image: random noise
    rng = np.random.RandomState(999)
    diff_arr = rng.randint(0, 256, (64, 64), dtype=np.uint8)
    img_diff = Image.fromarray(diff_arr)

    h_base = compute_perceptual_hash(img_base)
    h_perturbed = compute_perceptual_hash(img_perturbed)
    h_diff = compute_perceptual_hash(img_diff)

    dist_near = compute_hamming_distance(h_base, h_perturbed)
    dist_diff = compute_hamming_distance(h_base, h_diff)

    assert dist_near <= 3, f"Expected near-duplicate distance <= 3, got {dist_near}"
    assert dist_diff > 3, f"Expected distinct image distance > 3, got {dist_diff}"


def test_conflicting_label_detection(temp_dataset_dir: Path):
    """Verifies that assigning the same patient identifier to multiple diagnostic classes triggers a conflict."""
    img_data = np.linspace(20, 180, 64 * 64, dtype=np.uint8).reshape(64, 64)
    pil_img = Image.fromarray(img_data)

    # Patient PT_999 placed into both metastasis and healthy
    (temp_dataset_dir / "metastasis" / "PT_999_slice01.png").parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(temp_dataset_dir / "metastasis" / "PT_999_slice01.png")
    pil_img.save(temp_dataset_dir / "healthy" / "PT_999_slice02.png")

    auditor = DatasetAuditor()
    report = auditor.audit_dataset_directory(temp_dataset_dir)

    assert report["conflicting_patients_count"] >= 1
    # Check that the conflict is reported in blockers
    assert any("Conflicting labels detected" in b for b in report["blockers"])


def test_uncertain_keywords_and_missing_masks_flagging(temp_dataset_dir: Path):
    """Verifies flagging of borderline/uncertain images and missing segmentation masks."""
    img_data = np.linspace(30, 190, 64 * 64, dtype=np.uint8).reshape(64, 64)
    pil_img = Image.fromarray(img_data)

    # Uncertain keyword in filename
    uncertain_path = temp_dataset_dir / "oi" / "PT_101_borderline_scan.png"
    pil_img.save(uncertain_path)

    auditor = DatasetAuditor()
    report = auditor.audit_dataset_directory(temp_dataset_dir)

    assert report["uncertain_cases_count"] >= 1
    assert any("PT_101_borderline_scan.png" in u for u in report["uncertain_cases"])
    # Missing masks are tracked
    assert report["missing_annotations_count"] >= 1


def test_zero_patient_leakage_partitioning():
    """Verifies that patient partitioning assigns all slices of a patient strictly to one split."""
    # Create 40 synthetic records across 20 patients (2 slices per patient)
    records = []
    classes = ["metastasis", "paget", "oi", "healthy"]

    for i in range(20):
        pid = f"PT_{i:03d}"
        target_cls = classes[i % 4]
        for slice_idx in range(2):
            records.append({
                "patient_id": pid,
                "target_class": target_cls,
                "modality": "xray",
                "file_path": f"/data/{target_cls}/{pid}_slice{slice_idx}.png",
                "sha256": f"hash_{i}_{slice_idx}",
                "site_id": "site_01",
                "source_id": "dataset_a",
            })

    partitioner = PatientLevelPartitioner(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    partitioned = partitioner.partition_records(records)

    assert len(partitioned) == len(records)

    # Check zero leakage
    is_clean, violations = verify_patient_leakage(partitioned)
    assert is_clean is True
    assert len(violations) == 0

    # Ensure every patient has all slices in exactly one split
    patient_splits = {}
    for r in partitioned:
        pid = r["patient_id"]
        split = r["split"]
        if pid in patient_splits:
            assert patient_splits[pid] == split, f"Patient {pid} leaked across {patient_splits[pid]} and {split}!"
        else:
            patient_splits[pid] = split

    # Ensure all three splits (train, val, test) have representation
    splits_present = {r["split"] for r in partitioned}
    assert "train" in splits_present
    assert "val" in splits_present
    assert "test" in splits_present


def test_patient_leakage_verifier_catches_leak():
    """Verifies that verify_patient_leakage detects and reports patient overlap."""
    leaked_records = [
        {"patient_id": "PT_001", "split": "train"},
        {"patient_id": "PT_002", "split": "val"},
        {"patient_id": "PT_001", "split": "test"},  # Leaked patient
    ]
    is_clean, violations = verify_patient_leakage(leaked_records)
    assert is_clean is False
    assert len(violations) >= 1
    assert "PT_001" in violations[0]


def test_external_site_holdout_strategy():
    """Verifies that external site records are isolated to external_test."""
    records = [
        {"patient_id": "PT_001", "target_class": "metastasis", "site_id": "site_internal_1"},
        {"patient_id": "PT_002", "target_class": "metastasis", "site_id": "site_internal_2"},
        {"patient_id": "PT_003", "target_class": "paget", "site_id": "site_internal_1"},
        {"patient_id": "PT_004", "target_class": "paget", "site_id": "site_internal_2"},
        {"patient_id": "PT_EXT_01", "target_class": "metastasis", "site_id": "site_external"},
        {"patient_id": "PT_EXT_02", "target_class": "paget", "site_id": "site_external"},
    ]

    partitioner = PatientLevelPartitioner(seed=42)
    partitioned = partitioner.partition_records(records, holdout_sites=["site_external"])

    ext_records = [r for r in partitioned if r["patient_id"].startswith("PT_EXT")]
    assert all(r["split"] == "external_test" for r in ext_records)

    int_records = [r for r in partitioned if not r["patient_id"].startswith("PT_EXT")]
    assert all(r["split"] in ["train", "val", "test"] for r in int_records)

    is_clean, violations = verify_patient_leakage(partitioned)
    assert is_clean is True


def test_training_isolation_guard():
    """Verifies that statistical normalization or feature scalers cannot be fitted on non-train data."""
    # Train is allowed
    assert_training_only_fit("train")
    assert_training_only_fit("TRAIN")

    # Non-train splits must raise TrainingIsolationError
    for forbidden_split in ["val", "test", "external_test", "evaluation"]:
        with pytest.raises(TrainingIsolationError) as exc_info:
            assert_training_only_fit(forbidden_split)
        assert "strictly prohibited" in str(exc_info.value)


def test_zero_dataset_audit_state(tmp_path: Path):
    """Verifies that auditing an empty directory marks the dataset as NOT_READY with zero samples."""
    empty_dir = tmp_path / "empty_store"
    empty_dir.mkdir()

    auditor = DatasetAuditor()
    report = auditor.audit_dataset_directory(empty_dir)

    assert report["status"] == "NOT_READY"
    assert report["total_samples"] == 0
    assert report["total_patients"] == 0
    assert len(report["missing_classes"]) == 4
    assert any("ZERO clinical image samples exist on disk" in b for b in report["blockers"])


def test_artifact_export_integrity(tmp_path: Path):
    """Verifies that exporting audit artifacts produces valid JSON and CSV files."""
    auditor = DatasetAuditor()
    report = auditor.audit_dataset_directory(tmp_path)

    out_dir = tmp_path / "splits_out"
    exported = auditor.export_audit_artifacts(report, out_dir)

    # Check existence
    assert exported["dataset_quality_report"].exists()
    assert exported["class_distribution"].exists()
    assert exported["duplicate_audit"].exists()

    # Verify JSON parsability
    with open(exported["dataset_quality_report"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["status"] == "NOT_READY"

    # Verify CSV structure
    with open(exported["class_distribution"], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["target_class", "samples", "patients", "studies"]

    # Verify split manifest export
    partitioner = PatientLevelPartitioner()
    manifest_path = partitioner.export_split_manifest([], output_path=out_dir / "split_manifest.csv")
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == SPLIT_MANIFEST_COLUMNS

