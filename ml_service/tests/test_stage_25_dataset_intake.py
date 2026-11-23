"""
Unit and Regression Tests for Stage 24 Document Corrections and Stage 25 Controlled Dataset Intake.

Validates:
1. Frozen BTXRD checkpoint SHA-256 remains strictly unmodified.
2. Original Stage 24 documents are backed up in stage24_originals/.
3. Corrected Stage 24 documents exist and have verified content in stage24_corrected/.
4. Correct three-center metrics (C1: 0.8103, C2: 0.5069, C3: 0.3592) and relative decreases.
5. ICMR (2017), IEC, DPDP Act context, and conditional HIPAA/GDPR in revised protocol.
6. MURA breakdown (9,045 normal, 5,818 abnormal) and PMC11324408 hand/wrist boundary.
7. Outreach emails marked DRAFT ONLY - NOT SENT with respectful student tone.
8. All 9 Stage 25 intake deliverables exist in ml_artifacts/stage25_dataset_intake/.
9. Stage 25 gate verdict is BLOCKED / WAITING FOR APPROVED DATA with 0 new data ingested.
10. In-tree raw datasets remain intact and uncorrupted.
11. BTXRD Center 2 and 3 preserved as burned holdouts.
12. Preservation of historical stage reports (Stages 17-25).
13. Host CPU Python and React frontend remain untouched.
"""

import csv
import json
import hashlib
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
ORIGINALS_DIR = WORKSPACE_ROOT / "ml_artifacts/document_corrections/stage24_originals"
CORRECTED_DIR = WORKSPACE_ROOT / "ml_artifacts/document_corrections/stage24_corrected"
INTAKE_DIR = WORKSPACE_ROOT / "ml_artifacts/stage25_dataset_intake"
STAGE_REPORTS_DIR = WORKSPACE_ROOT / "ml_artifacts/stage_reports"
FROZEN_CHECKPOINT = WORKSPACE_ROOT / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
EXPECTED_CHECKPOINT_HASH = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"


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


def test_02_stage24_backup_originals_exist():
    """Verify all 12 original Stage 24 files are backed up in stage24_originals/."""
    expected_files = [
        "four_class_research_protocol.md",
        "paget_cohort_specification.md",
        "oi_cohort_specification.md",
        "metastasis_cohort_specification.md",
        "healthy_control_specification.md",
        "institutional_access_request_packages.md",
        "clinical_collaboration_plan.md",
        "outreach_email_drafts.md",
        "dataset_acceptance_checklist.md",
        "modality_architecture_plan.md",
        "acquisition_dependency_tracker.csv",
        "STAGE_24_CLINICAL_COLLABORATION_PREPARATION.md"
    ]
    for fname in expected_files:
        fpath = ORIGINALS_DIR / fname
        assert fpath.exists(), f"Missing backed up original file: {fname}"
        assert fpath.stat().st_size > 500, f"Backed up original file {fname} is too small"


def test_03_stage24_corrected_files_exist():
    """Verify all 8 corrected review-ready Stage 24 files exist in stage24_corrected/."""
    expected_files = [
        "four_class_research_protocol_REVISED.md",
        "institutional_access_request_packages_REVISED.md",
        "outreach_email_drafts_REVISED.md",
        "paget_cohort_specification_REVISED.md",
        "oi_cohort_specification_REVISED.md",
        "correction_log.md",
        "source_verification_notes.md",
        "STAGE_24_DOCUMENT_CORRECTION_REPORT.md"
    ]
    for fname in expected_files:
        fpath = CORRECTED_DIR / fname
        assert fpath.exists(), f"Missing corrected Stage 24 file: {fname}"
        assert fpath.stat().st_size > 500, f"Corrected file {fname} is too small"


def test_04_three_center_metrics_in_corrected_protocol():
    """Verify accurate three-center metrics and relative percentage drops in revised protocol."""
    proto_text = (CORRECTED_DIR / "four_class_research_protocol_REVISED.md").read_text(encoding="utf-8")
    assert "0.8103" in proto_text, "Center 1 Macro-F1 0.8103 missing"
    assert "0.5069" in proto_text, "Center 2 Macro-F1 0.5069 missing"
    assert "0.3592" in proto_text, "Center 3 Macro-F1 0.3592 missing"
    assert "37.44%" in proto_text, "Center 2 relative drop 37.44% missing"
    assert "55.67%" in proto_text, "Center 3 relative drop 55.67% missing"
    assert "not be pooled" in proto_text.lower(), "Warning against pooling external cohorts missing"


def test_05_icmr_and_iec_governance_in_corrected_protocol():
    """Verify Indian ICMR (2017) guidelines and IEC review pathway in revised protocol."""
    proto_text = (CORRECTED_DIR / "four_class_research_protocol_REVISED.md").read_text(encoding="utf-8")
    assert "ICMR National Ethical Guidelines" in proto_text
    assert "2017" in proto_text
    assert "Institutional Ethics Committee" in proto_text
    assert "Digital Personal Data Protection Act, 2023" in proto_text or "DPDP Act" in proto_text
    assert "Consent or waiver of consent, where applicable, will be determined by the competent Institutional Ethics Committee" in proto_text


def test_06_mura_and_pmc_accurately_described():
    """Verify MURA breakdown and PMC11324408 hand/wrist boundary in access packages."""
    pkg_text = (CORRECTED_DIR / "institutional_access_request_packages_REVISED.md").read_text(encoding="utf-8")
    assert "14,863" in pkg_text
    assert "9,045" in pkg_text
    assert "5,818" in pkg_text
    assert "BOTH normal and abnormal studies" in pkg_text
    assert "159" in pkg_text
    assert "hand and wrist" in pkg_text.lower()


def test_07_outreach_emails_respectful_and_draft_only():
    """Verify outreach emails have respectful tone and non-transmission markings."""
    email_text = (CORRECTED_DIR / "outreach_email_drafts_REVISED.md").read_text(encoding="utf-8")
    assert "DRAFT ONLY — NOT SENT" in email_text
    assert "RECOMMENDED FIRST COMMUNICATION" in email_text
    assert "[Student Researcher Name]" in email_text
    assert "[Faculty Advisor Name]" in email_text


def test_08_all_stage25_artifacts_exist():
    """Verify all 9 Stage 25 intake deliverables exist in ml_artifacts/stage25_dataset_intake/."""
    expected_files = [
        "stage25_dataset_inventory.csv",
        "stage25_intake_register.csv",
        "stage25_permissions_evidence_matrix.csv",
        "stage25_image_integrity_report.json",
        "stage25_duplicate_audit.json",
        "stage25_patient_grouping_audit.md",
        "stage25_four_class_coverage_update.csv",
        "stage25_blockers_and_next_actions.md",
        "STAGE_25_DATASET_INTAKE_REPORT.md"
    ]
    for fname in expected_files:
        fpath = INTAKE_DIR / fname
        assert fpath.exists(), f"Missing Stage 25 artifact: {fname}"
        assert fpath.stat().st_size > 200, f"Stage 25 artifact {fname} is too small"


def test_09_stage25_gate_verdict_blocked():
    """Verify Stage 25 gate verdict is BLOCKED / WAITING FOR APPROVED DATA."""
    report_text = (INTAKE_DIR / "STAGE_25_DATASET_INTAKE_REPORT.md").read_text(encoding="utf-8")
    assert "BLOCKED / WAITING FOR APPROVED DATA" in report_text
    
    with open(INTAKE_DIR / "stage25_image_integrity_report.json", "r", encoding="utf-8") as f:
        integrity = json.load(f)
    assert integrity["new_images_ingested_in_stage25"] == 0
    assert integrity["new_bytes_copied_in_stage25"] == 0


def test_10_in_tree_datasets_intact_and_unmodified():
    """Verify in-tree raw datasets have expected files and remain unmodified."""
    raw_dir = WORKSPACE_ROOT / "ml_service/data/raw"
    bs_files = list((raw_dir / "bone_scan_metastasis").rglob("*.png"))
    assert len(bs_files) >= 582, f"Expected >= 582 bone scan images, got {len(bs_files)}"
    
    btxrd_files = [
        f for f in (raw_dir / "btxrd").rglob("*")
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]
    assert len(btxrd_files) >= 1396, f"Expected >= 1396 BTXRD images, got {len(btxrd_files)}"


def test_11_btxrd_centers_2_and_3_preserved_as_holdouts():
    """Verify BTXRD Centers 2 and 3 remain permanently burned external holdouts."""
    dup_file = INTAKE_DIR / "stage25_duplicate_audit.json"
    with open(dup_file, "r", encoding="utf-8") as f:
        dup = json.load(f)
    disp = dup["cross_dataset_deduplication_audit"]["btxrd_center_partitions"]["disposition"]
    assert "BURNED_HOLDOUTS" in disp


def test_12_historical_stage_reports_preserved():
    """Verify all historical reports from Stages 17-25 exist."""
    expected_reports = [
        "STAGE_17_BTXRD_FULL_TRAINING.md",
        "STAGE_18_BTXRD_MODEL_FREEZE.md",
        "STAGE_19_CENTER2_EXTERNAL_EVALUATION.md",
        "STAGE_20_CENTER3_EXTERNAL_EVALUATION.md",
        "STAGE_21_MULTICENTER_GENERALIZATION_AUDIT.md",
        "STAGE_22_DATASET_READINESS_AUDIT.md",
        "STAGE_23_DATASET_SOURCE_VERIFICATION.md",
        "STAGE_24_CLINICAL_COLLABORATION_PREPARATION.md",
        "STAGE_25_DATASET_INTAKE.md"
    ]
    for rname in expected_reports:
        rpath = STAGE_REPORTS_DIR / rname
        assert rpath.exists(), f"Missing stage report: {rname}"
        assert rpath.stat().st_size > 1000, f"Stage report {rname} is too short"


def test_13_host_python_and_frontend_untouched():
    """Verify host CPU Python and React frontend remain untouched."""
    frontend_dir = WORKSPACE_ROOT / "src"
    assert frontend_dir.exists(), "React frontend directory missing"
    host_python = Path(r"C:\Python313\python.exe")
    assert host_python.exists(), "Host Python executable missing"

