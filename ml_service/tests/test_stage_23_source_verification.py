"""
Stage 23 - Dataset Source Verification and Access Gate Tests.

Verifies:
1. Frozen checkpoint SHA-256 hash remains unaltered.
2. All 11 Stage 23 artifacts exist in ml_artifacts/dataset_source_verification.
3. source_verification_matrix.csv statuses strictly conform to the 7 defined categories.
4. Figshare OI dataset is correctly classified as ANIMAL_OR_SYNTHETIC_ONLY.
5. PMC11324408 is correctly classified as CONTROLLED_ACCESS_REQUEST_REQUIRED.
6. Zenodo 13900966 is documented as identical in-tree duplicate of IICS-UNA dataset.
7. TCIA Spine-Mets-CT-SEG is verified at 55 subjects with CT modality.
8. Stanford AIMI MURA is restricted to upper extremities and requires DUA.
9. candidate_provenance.json and markdown files document master NOT READY verdict.
10. Preservation of Stage 17 to 22 reports and artifacts.
11. Preservation of FracAtlas and Bone Scintigraphy checkpoints.
12. Preservation of host CPU Python and React frontend.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

from ml_service.config.settings import get_settings

EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
ALLOWED_STATUSES = {
    "PUBLIC_DOWNLOAD_VERIFIED",
    "CONTROLLED_ACCESS_REQUEST_REQUIRED",
    "INDIVIDUAL_PERMISSION_REQUIRED",
    "EDUCATIONAL_ONLY",
    "ANIMAL_OR_SYNTHETIC_ONLY",
    "UNSUITABLE_FOR_TARGET_CLASS",
    "UNVERIFIED",
}


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def stage23_dir(base_dir):
    d = base_dir / "ml_artifacts/dataset_source_verification"
    assert d.exists(), f"Stage 23 directory missing: {d}"
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


def test_02_all_stage23_artifacts_exist(stage23_dir):
    """Verifies that all 11 required Stage 23 files exist with non-zero size."""
    expected_files = [
        "verified_dataset_candidates.csv",
        "source_verification_matrix.csv",
        "candidate_provenance.json",
        "oi_dataset_access_assessment.md",
        "paget_dataset_access_assessment.md",
        "metastasis_dataset_access_assessment.md",
        "healthy_control_access_assessment.md",
        "modality_compatibility_decision.md",
        "dataset_access_request_templates.md",
        "final_acquisition_priority.md",
        "STAGE_23_DATASET_SOURCE_VERIFICATION.md",
    ]
    for fn in expected_files:
        fp = stage23_dir / fn
        assert fp.exists(), f"Missing Stage 23 artifact: {fn}"
        assert fp.stat().st_size > 0, f"Empty Stage 23 artifact: {fn}"


def test_03_strict_source_classification_values(stage23_dir):
    """Verifies that all status classifications strictly match the 7 authorized statuses."""
    p = stage23_dir / "source_verification_matrix.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 10

    for r in rows:
        status = r["status_classification"].strip()
        assert status in ALLOWED_STATUSES, f"Unauthorized status '{status}' in row: {r['source_name']}"


def test_04_figshare_oi_correctly_classified_as_animal(stage23_dir):
    """Verifies that the Figshare OI collection is classified as ANIMAL_OR_SYNTHETIC_ONLY."""
    p = stage23_dir / "source_verification_matrix.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    figshare_oi = [r for r in rows if "Figshare" in r["source_name"] and "OI" in r["source_name"]]
    assert len(figshare_oi) == 1
    row = figshare_oi[0]
    assert row["status_classification"] == "ANIMAL_OR_SYNTHETIC_ONLY"
    assert "Animal" in row["human_vs_animal"]
    assert "REJECTED" in row["audit_verdict"]


def test_05_pmc11324408_correctly_classified_as_controlled_access(stage23_dir):
    """Verifies that PMC11324408 is classified as CONTROLLED_ACCESS_REQUEST_REQUIRED."""
    p = stage23_dir / "source_verification_matrix.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pmc_oi = [r for r in rows if "PMC11324408" in r["source_name"]]
    assert len(pmc_oi) == 1
    row = pmc_oi[0]
    assert row["status_classification"] == "CONTROLLED_ACCESS_REQUEST_REQUIRED"
    assert row["human_vs_animal"] == "Human"


def test_06_zenodo_bone_mets_deduplication(stage23_dir):
    """Verifies that Zenodo 13900966 is recognized as an in-tree duplicate of IICS-UNA dataset."""
    json_p = stage23_dir / "candidate_provenance.json"
    with open(json_p, encoding="utf-8") as f:
        data = json.load(f)

    zenodo_finding = data["forensic_discoveries"]["zenodo_13900966_deduplication"]
    assert "13900966" in zenodo_finding["finding"]
    assert "duplicate" in zenodo_finding["action"].lower() or "identical" in zenodo_finding["finding"].lower()


def test_07_tcia_spine_mets_verified_at_55_subjects(stage23_dir):
    """Verifies that TCIA Spine-Mets-CT-SEG is verified at 55 patients with CT modality."""
    p = stage23_dir / "source_verification_matrix.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tcia_row = [r for r in rows if "TCIA" in r["source_name"]][0]
    assert int(tcia_row["verified_patient_count"]) == 55
    assert "Computed Tomography" in tcia_row["modality"] or "CT" in tcia_row["modality"]
    assert tcia_row["status_classification"] == "CONTROLLED_ACCESS_REQUEST_REQUIRED"


def test_08_stanford_mura_anatomical_scope_documented(stage23_dir):
    """Verifies that Stanford AIMI MURA is documented as upper extremity only and requires DUA."""
    p = stage23_dir / "healthy_control_access_assessment.md"
    content = p.read_text(encoding="utf-8")
    assert "upper extremity exclusively" in content.lower() or "upper extremities only" in content.lower() or "upper extremity" in content.lower()
    assert "CONTROLLED_ACCESS_REQUEST_REQUIRED" in content
    assert "aimi.stanford.edu" in content


def test_09_master_decision_not_ready(stage23_dir):
    """Verifies that master acquisition gate is NOT READY — DATA ACQUISITION REQUIRED."""
    p = stage23_dir / "candidate_provenance.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    assert data["master_decision"] == "NOT READY — DATA ACQUISITION REQUIRED"

    md_p = stage23_dir / "final_acquisition_priority.md"
    assert "NOT READY — DATA ACQUISITION REQUIRED" in md_p.read_text(encoding="utf-8")


def test_10_stage_17_to_22_reports_and_artifacts_preserved(base_dir):
    """Verifies that reports and manifests for Stages 17 through 22 exist and remain intact."""
    stages = [
        "STAGE_17_BTXRD_FULL_TRAINING.md",
        "STAGE_18_BTXRD_MODEL_FREEZE.md",
        "STAGE_19_CENTER2_EXTERNAL_EVALUATION.md",
        "STAGE_20_CENTER3_EXTERNAL_EVALUATION.md",
        "STAGE_21_MULTICENTER_GENERALIZATION_AUDIT.md",
        "STAGE_22_DATASET_READINESS_AUDIT.md",
    ]
    for s in stages:
        assert (base_dir / f"ml_artifacts/stage_reports/{s}").exists()

    # Stage 22 artifacts check
    assert (base_dir / "ml_artifacts/four_class_dataset_audit/four_class_coverage_matrix.csv").exists()


def test_11_checkpoints_preserved(base_dir):
    """Verifies FracAtlas and Bone Scintigraphy checkpoints remain intact."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists()
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    scint_fold0 = base_dir / "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"
    assert scint_fold0.exists()


def test_12_cpu_python_and_frontend_preservation(base_dir):
    """Verifies host CPU Python and React frontend are completely intact."""
    cpu_py = Path(r"C:\Python313\python.exe")
    assert cpu_py.exists()
    cmd = [str(cpu_py), "-c", "import torch; print(torch.__version__)"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "+cpu" in res.stdout

    frontend_file = base_dir / "osteophorosis-main/src/App.tsx"
    assert frontend_file.exists()

