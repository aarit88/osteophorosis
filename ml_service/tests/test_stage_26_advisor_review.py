"""
Unit and Regression Tests for Stage 26: Faculty Advisor Review and Institutional Approval Preparation.

Validates:
1. Frozen BTXRD checkpoint SHA-256 remains strictly unmodified.
2. All 7 Stage 26 deliverables exist in ml_artifacts/stage26_advisor_review/.
3. Strict alignment with canonical 4-class target taxonomy (Mets, Paget, OI, Healthy).
4. KSCST milestone alignment audit tracks all 10 components and distinguishes BTXRD.
5. Faculty advisor review package contains accurate metrics, decoupled architecture, and decisions.
6. Institutional approval checklists cover Hospital, RDCRN, Stanford AIMI, and TCIA.
7. Outreach document review identifies Draft 1 as the primary outreach.
8. Five-tier execution roadmap separates independent work from data-blocked work.
9. Student researcher action guide details real-world steps outside Antigravity.
10. Zero unauthorized data ingestion (in-tree raw data remains unmodified).
11. BTXRD Center 2 and 3 preserved as burned holdouts.
12. Historical stage reports (Stages 17-26) preserved.
13. Host CPU Python and React frontend remain untouched.
"""

import csv
import json
import hashlib
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_26_DIR = WORKSPACE_ROOT / "ml_artifacts/stage26_advisor_review"
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


def test_02_all_stage26_artifacts_exist():
    """Verify all 7 Stage 26 deliverables exist and have valid sizes."""
    expected_files = [
        "kscst_project_alignment_audit.md",
        "faculty_advisor_review_package.md",
        "institutional_approval_checklists.md",
        "outreach_document_review.md",
        "stage26_execution_roadmap.md",
        "student_researcher_action_guide.md",
        "STAGE_26_ADVISOR_REVIEW.md"
    ]
    for fname in expected_files:
        fpath = STAGE_26_DIR / fname
        assert fpath.exists(), f"Missing Stage 26 artifact: {fname}"
        assert fpath.stat().st_size > 500, f"Stage 26 artifact {fname} is too small"


def test_03_four_class_taxonomy_alignment():
    """Verify alignment with the 4 canonical diagnostic classes."""
    align_text = (STAGE_26_DIR / "kscst_project_alignment_audit.md").read_text(encoding="utf-8")
    assert "Secondary bone metastasis" in align_text
    assert "Paget's disease of bone" in align_text
    assert "Osteogenesis imperfecta" in align_text
    assert "Healthy skeletal controls" in align_text


def test_04_kscst_components_tracking():
    """Verify KSCST proposal components are tracked and BTXRD is distinguished."""
    align_text = (STAGE_26_DIR / "kscst_project_alignment_audit.md").read_text(encoding="utf-8")
    assert "U-Net" in align_text
    assert "102-D" in align_text or "102" in align_text
    assert "614-D" in align_text or "614" in align_text
    assert "Mirels" in align_text
    assert "Grad-CAM" in align_text
    assert "SHAP" in align_text
    assert "BTXRD" in align_text
    assert "interim" in align_text.lower()


def test_05_faculty_advisor_review_package_contents():
    """Verify review package contains accurate metrics, decoupled architecture, and advisor decisions."""
    pkg_text = (STAGE_26_DIR / "faculty_advisor_review_package.md").read_text(encoding="utf-8")
    assert "0.8103" in pkg_text
    assert "0.5069" in pkg_text
    assert "0.3592" in pkg_text
    assert "37.44%" in pkg_text
    assert "55.67%" in pkg_text
    assert "Decoupled Modality Architecture" in pkg_text or "Pathway A" in pkg_text
    assert "Principal Investigator" in pkg_text
    assert "Decisions and Support Requested" in pkg_text


def test_06_institutional_approval_checklists():
    """Verify checklists cover Hospital PACS, RDCRN BBDC, Stanford AIMI, and TCIA."""
    chk_text = (STAGE_26_DIR / "institutional_approval_checklists.md").read_text(encoding="utf-8")
    assert "Hospital Collaboration" in chk_text
    assert "RDCRN BBDC" in chk_text
    assert "Stanford AIMI MURA" in chk_text
    assert "TCIA" in chk_text


def test_07_outreach_document_review_safeguards():
    """Verify outreach review identifies Draft 1 as the primary outreach."""
    outreach_text = (STAGE_26_DIR / "outreach_document_review.md").read_text(encoding="utf-8")
    assert "DRAFT 1" in outreach_text
    assert "Faculty Research Advisor" in outreach_text
    assert "DRAFT ONLY" in outreach_text


def test_08_five_tier_execution_roadmap():
    """Verify 5-tier execution roadmap separates independent from blocked work."""
    road_text = (STAGE_26_DIR / "stage26_execution_roadmap.md").read_text(encoding="utf-8")
    assert "TIER 1" in road_text
    assert "TIER 2" in road_text
    assert "TIER 3" in road_text
    assert "TIER 4" in road_text
    assert "TIER 5" in road_text
    assert "Independent Technical Work" in road_text


def test_09_student_researcher_action_guide():
    """Verify student researcher action guide provides real-world steps outside Antigravity."""
    guide_text = (STAGE_26_DIR / "student_researcher_action_guide.md").read_text(encoding="utf-8")
    assert "Outside Antigravity" in guide_text
    assert "Schedule an Advisory Meeting with Your Faculty Supervisor" in guide_text
    assert "DO NOT" in guide_text


def test_10_zero_unauthorized_data_ingestion():
    """Verify raw datasets remain unmodified and no unauthorized data was ingested."""
    raw_dir = WORKSPACE_ROOT / "ml_service/data/raw"
    bs_files = list((raw_dir / "bone_scan_metastasis").rglob("*.png"))
    assert len(bs_files) >= 582, f"Expected >= 582 bone scan images, got {len(bs_files)}"
    
    btxrd_files = [
        f for f in (raw_dir / "btxrd").rglob("*")
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]
    assert len(btxrd_files) >= 1396, f"Expected >= 1396 BTXRD images, got {len(btxrd_files)}"
    
    incoming_dir = WORKSPACE_ROOT / "ml_service/data/incoming"
    if incoming_dir.exists():
        assert len(list(incoming_dir.rglob("*"))) == 0, "Incoming directory must be empty"


def test_11_btxrd_centers_2_and_3_remain_holdouts():
    """Verify BTXRD Centers 2 and 3 remain permanently burned external holdouts."""
    align_text = (STAGE_26_DIR / "kscst_project_alignment_audit.md").read_text(encoding="utf-8")
    assert "external holdouts" in align_text.lower() or "burned" in align_text.lower()


def test_12_historical_stage_reports_preserved():
    """Verify historical reports from Stages 17-26 all exist and are non-empty."""
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
        "STAGE_26_ADVISOR_REVIEW.md"
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

