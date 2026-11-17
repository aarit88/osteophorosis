"""
Stage 22 - OsteoGuard AI Four-Class Dataset Readiness and Gap Audit Tests.

Verifies:
1. Frozen checkpoint SHA-256 hash remains invariant.
2. All 9 Stage 22 audit artifacts exist in ml_artifacts/four_class_dataset_audit.
3. existing_dataset_inventory.csv correctly reflects FracAtlas, Bone Scan, and BTXRD counts.
4. four_class_coverage_matrix.csv documents all 4 target classes and identifies missing classes.
5. dataset_provenance_audit.json issues the master decision: NOT READY — DATA ACQUISITION REQUIRED.
6. modality_compatibility_report.md documents physical incompatibilities and rejects single shared encoder.
7. data_quality_requirements.md enforces patient-level Group K-Fold and ground-truth standards.
8. dataset_acquisition_roadmap.md enforces prioritized acquisition and permanent holdout sealing.
9. Preservation of Stage 17 to 21 reports and artifacts.
10. Preservation of FracAtlas and Bone Scintigraphy checkpoints.
11. Preservation of host CPU Python and React frontend.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

from ml_service.config.settings import get_settings

EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def audit_dir(base_dir):
    d = base_dir / "ml_artifacts/four_class_dataset_audit"
    assert d.exists(), f"Stage 22 audit directory missing: {d}"
    return d


@pytest.fixture(scope="module")
def frozen_checkpoint(base_dir):
    p = base_dir / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
    assert p.exists(), f"Frozen checkpoint missing: {p}"
    return p


def test_01_checkpoint_hash_unmodified(frozen_checkpoint):
    """Verifies that the frozen checkpoint SHA-256 remains completely unaltered."""
    h = hashlib.sha256()
    with open(frozen_checkpoint, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    current_hash = h.hexdigest().upper()
    assert current_hash == EXPECTED_SHA256, f"Checkpoint hash mismatch: {current_hash}"


def test_02_all_stage22_artifacts_exist(audit_dir):
    """Verifies that all 9 required Stage 22 artifacts exist with non-zero size."""
    expected_files = [
        "existing_dataset_inventory.csv",
        "four_class_coverage_matrix.csv",
        "dataset_provenance_audit.json",
        "missing_dataset_research.md",
        "modality_compatibility_report.md",
        "data_quality_requirements.md",
        "dataset_acquisition_roadmap.md",
        "FOUR_CLASS_READINESS_DECISION.md",
        "STAGE_22_DATASET_READINESS_AUDIT.md",
    ]
    for fn in expected_files:
        fp = audit_dir / fn
        assert fp.exists(), f"Missing Stage 22 artifact: {fn}"
        assert fp.stat().st_size > 0, f"Empty Stage 22 artifact: {fn}"


def test_03_existing_dataset_inventory_integrity(audit_dir):
    """Verifies existing_dataset_inventory.csv row counts and details."""
    p = audit_dir / "existing_dataset_inventory.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3

    names = [r["dataset_name"] for r in rows]
    assert names == ["FracAtlas", "Bone Scintigraphy Dataset", "BTXRD"]

    # FracAtlas
    fa = rows[0]
    assert int(fa["total_images"]) == 4083
    assert fa["imaging_modality"] == "Planar Radiography (X-ray)"
    assert fa["patient_level_ids_available"] == "False"

    # Bone Scan
    bs = rows[1]
    assert int(bs["total_images"]) == 582
    assert int(bs["total_patients"]) == 291
    assert "True" in bs["patient_level_ids_available"]

    # BTXRD
    bt = rows[2]
    assert int(bt["total_images"]) == 3746
    assert bt["patient_level_ids_available"] == "False"


def test_04_four_class_coverage_matrix_integrity(audit_dir):
    """Verifies four_class_coverage_matrix.csv covers all 4 targets and marks Paget/OI as missing."""
    p = audit_dir / "four_class_coverage_matrix.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    targets = set(r["target_diagnosis"] for r in rows)
    assert "Secondary Bone Metastasis" in targets
    assert "Paget's Disease of Bone" in targets
    assert "Osteogenesis Imperfecta" in targets
    assert "Healthy Skeletal Control" in targets

    # Check that in-tree candidates for Paget's and OI are marked MISSING
    paget_intree = [r for r in rows if r["target_diagnosis"] == "Paget's Disease of Bone" and r["candidate_type"].startswith("In-Tree")]
    assert len(paget_intree) == 1
    assert paget_intree[0]["dataset_suitability"] == "MISSING"

    oi_intree = [r for r in rows if r["target_diagnosis"] == "Osteogenesis Imperfecta" and r["candidate_type"].startswith("In-Tree")]
    assert len(oi_intree) == 1
    assert oi_intree[0]["dataset_suitability"] == "MISSING"

    # Check BTXRD Class 2 is marked UNSUITABLE for metastasis
    btxrd_mets = [r for r in rows if r["target_diagnosis"] == "Secondary Bone Metastasis" and "BTXRD" in r["candidate_dataset"]]
    assert len(btxrd_mets) == 1
    assert btxrd_mets[0]["dataset_suitability"] == "UNSUITABLE"

    # Check FracAtlas non-fractured is marked UNSUITABLE for healthy controls
    fa_healthy = [r for r in rows if r["target_diagnosis"] == "Healthy Skeletal Control" and "FracAtlas" in r["candidate_dataset"]]
    assert len(fa_healthy) == 1
    assert fa_healthy[0]["dataset_suitability"] == "UNSUITABLE"


def test_05_readiness_decision_unanimous_not_ready(audit_dir):
    """Verifies FOUR_CLASS_READINESS_DECISION.md and JSON document master NOT READY verdict."""
    # JSON
    json_p = audit_dir / "dataset_provenance_audit.json"
    with open(json_p, encoding="utf-8") as f:
        data = json.load(f)
    assert data["overall_readiness"] == "NOT READY — DATA ACQUISITION REQUIRED"
    assert data["coverage_status"]["pagets_disease"].startswith("COMPLETELY_MISSING")
    assert data["coverage_status"]["osteogenesis_imperfecta"].startswith("COMPLETELY_MISSING")

    # Markdown
    md_p = audit_dir / "FOUR_CLASS_READINESS_DECISION.md"
    content = md_p.read_text(encoding="utf-8")
    assert "NOT READY — DATA ACQUISITION REQUIRED" in content


def test_06_modality_compatibility_report_recommends_decoupled_architecture(audit_dir):
    """Verifies modality_compatibility_report.md rejects single shared encoder."""
    p = audit_dir / "modality_compatibility_report.md"
    content = p.read_text(encoding="utf-8")
    assert "Shared Single Image Encoder" in content
    assert "REJECTED" in content
    assert "Modality-Specific Feature Encoders" in content
    assert "Late Fusion" in content


def test_07_data_quality_requirements_integrity(audit_dir):
    """Verifies data_quality_requirements.md enforces patient-level Group K-Fold and ground-truth standards."""
    p = audit_dir / "data_quality_requirements.md"
    content = p.read_text(encoding="utf-8")
    assert "Group K-Fold" in content
    assert "Cryptographic Patient Identifiers" in content
    assert "Multi-Reader Annotation" in content
    assert "Independent Hospital-Level External Validation" in content


def test_08_dataset_acquisition_roadmap_phases(audit_dir):
    """Verifies dataset_acquisition_roadmap.md documents prioritized phases and permanent holdout sealing."""
    p = audit_dir / "dataset_acquisition_roadmap.md"
    content = p.read_text(encoding="utf-8")
    assert "Phase 1: Ingestion of Missing Pathology Classes" in content or "Priority 1" in content
    assert "Stanford AIMI MURA" in content
    assert "TCIA" in content
    assert "permanently burned holdouts" in content


def test_09_stage_17_to_21_reports_and_artifacts_preserved(base_dir):
    """Verifies that Stage 17 through 21 reports and directories exist."""
    stages = ["STAGE_17_BTXRD_FULL_TRAINING.md", "STAGE_18_BTXRD_MODEL_FREEZE.md", "STAGE_19_CENTER2_EXTERNAL_EVALUATION.md", "STAGE_20_CENTER3_EXTERNAL_EVALUATION.md", "STAGE_21_MULTICENTER_GENERALIZATION_AUDIT.md"]
    for s in stages:
        assert (base_dir / f"ml_artifacts/stage_reports/{s}").exists()

    # Stage 21 artifacts
    assert (base_dir / "ml_artifacts/btxrd_multicenter_audit/multicenter_performance_comparison.csv").exists()


def test_10_fracatlas_artifacts_preservation(base_dir):
    """Verifies FracAtlas Stage 09 checkpoint and Stage 10 report remain intact."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists()
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    fracatlas_rep = base_dir / "ml_artifacts/stage_reports/STAGE_10_FRACATLAS_HOLDOUT_EVALUATION.md"
    assert fracatlas_rep.exists()


def test_11_bone_scintigraphy_artifacts_preservation(base_dir):
    """Verifies Bone Scintigraphy fold 0 checkpoint and test metrics remain intact."""
    scint_fold0 = base_dir / "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"
    assert scint_fold0.exists()

    scint_test_eval = base_dir / "ml_artifacts/bone_scan_test_evaluation/test_metrics.json"
    assert scint_test_eval.exists()


def test_12_cpu_python_and_frontend_preservation(base_dir):
    """Verifies host CPU Python and React frontend are completely intact."""
    cpu_py = Path(r"C:\Python313\python.exe")
    assert cpu_py.exists()
    cmd = [str(cpu_py), "-c", "import torch; print(torch.__version__)"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "+cpu" in res.stdout

    frontend_file = base_dir / "osteophorosis-main/src/App.tsx"
    assert frontend_file.exists()
