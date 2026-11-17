"""
Unit and Regression Tests for Stage 24: Clinical Collaboration and Data Access Preparation.

Validates:
1. Frozen BTXRD checkpoint SHA-256 remains unmodified.
2. All 12 Stage 24 artifacts exist and have valid structure/content.
3. Research protocol contains required ethical governance markers and safety disclaimers.
4. Paget's disease specification enforces the radiographic triad and rejects web harvesting.
5. Osteogenesis imperfecta specification enforces human diagnosis and rejects animal micro-CT.
6. Secondary metastasis specification enforces modality decoupling and rejects sarcoma conflation.
7. Healthy control specification enforces anatomical matching and documents trauma/MURA boundaries.
8. Institutional access packages cover RDCRN BBDC, Hospital PACS, TCIA, and Stanford AIMI.
9. Outreach email drafts contain explicit placeholders and non-transmission disclaimers.
10. Data acceptance checklist defines the 10-point gate and official disposition tiers.
11. Modality architecture plan specifies decoupled pathways and clinical late fusion.
12. Acquisition dependency tracker prioritizes Paget and human OI as Priority 1 critical paths.
13. Preservation of all prior stage reports (Stages 17-23).
14. Preservation of host Python and React frontend.
"""

import csv
import hashlib
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_24_DIR = WORKSPACE_ROOT / "ml_artifacts/clinical_collaboration_preparation"
STAGE_REPORTS_DIR = WORKSPACE_ROOT / "ml_artifacts/stage_reports"
FROZEN_CHECKPOINT = WORKSPACE_ROOT / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
EXPECTED_CHECKPOINT_HASH = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"


def test_01_checkpoint_hash_unmodified():
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


def test_02_all_stage24_artifacts_exist():
    """Verify that all 12 required Stage 24 artifacts exist in ml_artifacts/clinical_collaboration_preparation."""
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
        fpath = STAGE_24_DIR / fname
        assert fpath.exists(), f"Missing Stage 24 artifact: {fname}"
        assert fpath.stat().st_size > 500, f"Artifact {fname} is unexpectedly small: {fpath.stat().st_size} bytes"


def test_03_research_protocol_governance_placeholders():
    """Verify research protocol contains required ethical markers and disclaimers."""
    protocol_text = (STAGE_24_DIR / "four_class_research_protocol.md").read_text(encoding="utf-8")
    assert "PENDING ETHICS COMMITTEE APPROVAL" in protocol_text
    assert "REQUIRING CLINICIAN INPUT" in protocol_text
    assert "INSTITUTIONAL REVIEW REQUIRED" in protocol_text
    assert "OG-PROT-2026-04C-DRAFT" in protocol_text
    assert "HIPAA" in protocol_text
    assert "GDPR" in protocol_text
    assert "INVESTIGATIONAL RESEARCH USE ONLY" in protocol_text


def test_04_paget_specification_integrity():
    """Verify Paget's specification defines radiographic triad and bans educational scraping."""
    paget_text = (STAGE_24_DIR / "paget_cohort_specification.md").read_text(encoding="utf-8")
    assert "Cortical Thickening" in paget_text
    assert "Trabeculae" in paget_text
    assert "Expansion" in paget_text or "Osteomegaly" in paget_text
    assert "alkaline phosphatase" in paget_text.lower()
    assert "Radiopaedia" in paget_text
    assert "STRICT PROHIBITION ON EDUCATIONAL WEB HARVESTING" in paget_text


def test_05_oi_specification_integrity():
    """Verify OI specification enforces human diagnosis and rejects animal micro-CT."""
    oi_text = (STAGE_24_DIR / "oi_cohort_specification.md").read_text(encoding="utf-8")
    assert "Sillence" in oi_text
    assert "COL1A1" in oi_text
    assert "ABSOLUTE DISQUALIFICATION OF ANIMAL MODELS" in oi_text
    assert "micro-CT" in oi_text or "μCT" in oi_text
    assert "oim mouse" in oi_text or "murine" in oi_text


def test_06_metastasis_specification_integrity():
    """Verify metastasis specification enforces modality decoupling and rejects sarcoma conflation."""
    mets_text = (STAGE_24_DIR / "metastasis_cohort_specification.md").read_text(encoding="utf-8")
    assert "Pathway A: Radiography" in mets_text
    assert "Pathway B: Scintigraphy" in mets_text
    assert "Pathway C: Volumetric CT" in mets_text
    assert "STRICT REJECTION OF PRIMARY SARCOMA CONFLATION" in mets_text
    assert "osteosarcoma" in mets_text.lower()
    assert "IICS-UNA" in mets_text


def test_07_healthy_control_specification_integrity():
    """Verify healthy control specification defines trauma and MURA boundaries."""
    control_text = (STAGE_24_DIR / "healthy_control_specification.md").read_text(encoding="utf-8")
    assert "FracAtlas" in control_text
    assert "CANNOT BE UNCRITICALLY EQUATED" in control_text
    assert "Stanford AIMI MURA" in control_text
    assert "upper-extremity" in control_text.lower()
    assert "ANATOMICAL MATCHING STRATEGY" in control_text


def test_08_access_request_packages_completeness():
    """Verify access packages cover all four target repositories."""
    pkg_text = (STAGE_24_DIR / "institutional_access_request_packages.md").read_text(encoding="utf-8")
    assert "RDCRN" in pkg_text
    assert "Brittle Bone Disorders Consortium" in pkg_text
    assert "Paget's Disease" in pkg_text
    assert "TCIA" in pkg_text
    assert "Spine-Mets-CT-SEG" in pkg_text
    assert "Stanford Center for Artificial Intelligence in Medicine & Imaging" in pkg_text or "AIMI" in pkg_text


def test_09_outreach_emails_placeholders_and_disclaimer():
    """Verify outreach email drafts have placeholders and explicit non-transmission warnings."""
    email_text = (STAGE_24_DIR / "outreach_email_drafts.md").read_text(encoding="utf-8")
    assert "[Researcher Name]" in email_text
    assert "[Institution Name]" in email_text
    assert "[Faculty Advisor Name]" in email_text
    assert "DRAFT ONLY" in email_text
    assert "STRICTLY PROHIBITED FROM TRANSMISSION" in email_text or "NO EMAILS HAVE BEEN AUTOMATICALLY TRANSMITTED" in email_text


def test_10_dataset_acceptance_checklist_10_points():
    """Verify 10-point gate and official disposition tiers."""
    chk_text = (STAGE_24_DIR / "dataset_acceptance_checklist.md").read_text(encoding="utf-8")
    assert "10-POINT DATA ACCEPTANCE GATE" in chk_text
    assert "ACCEPTED FOR RESEARCH INGESTION" in chk_text
    assert "CONDITIONALLY ACCEPTED" in chk_text
    assert "AWAITING PERMISSION" in chk_text
    assert "REJECTED" in chk_text


def test_11_modality_architecture_plan_decoupling():
    """Verify decoupled multi-pathway architecture and clinical late fusion."""
    arch_text = (STAGE_24_DIR / "modality_architecture_plan.md").read_text(encoding="utf-8")
    assert "Pathway A: Plain Radiography" in arch_text or "PATHWAY A: X-RAY" in arch_text
    assert "Pathway B: Planar Bone Scintigraphy" in arch_text or "PATHWAY B: SCINTIGR." in arch_text
    assert "Pathway C: Volumetric CT" in arch_text or "PATHWAY C: CT" in arch_text
    assert "CLINICAL LATE-FUSION LAYER" in arch_text
    assert "STRICTLY REJECTED" in arch_text


def test_12_acquisition_dependency_tracker_contents():
    """Verify dependency tracker tracks cohorts and prioritizes Paget and OI as Priority 1."""
    tracker_path = STAGE_24_DIR / "acquisition_dependency_tracker.csv"
    with open(tracker_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) >= 8, f"Expected >= 8 dependency rows, got {len(reader)}"
    
    classes_tracked = {r["cohort_class"] for r in reader}
    assert "Paget's Disease of Bone" in classes_tracked
    assert "Osteogenesis Imperfecta" in classes_tracked
    assert "Secondary Bone Metastasis" in classes_tracked
    assert "Healthy Skeletal Control" in classes_tracked

    priority_1_classes = {r["cohort_class"] for r in reader if "PRIORITY_1" in r["priority_rank"]}
    assert "Paget's Disease of Bone" in priority_1_classes
    assert "Osteogenesis Imperfecta" in priority_1_classes


def test_13_preservation_of_all_prior_stage_reports():
    """Verify stage reports for Stages 17-24 all exist and are non-empty."""
    expected_reports = [
        "STAGE_17_BTXRD_FULL_TRAINING.md",
        "STAGE_18_BTXRD_MODEL_FREEZE.md",
        "STAGE_19_CENTER2_EXTERNAL_EVALUATION.md",
        "STAGE_20_CENTER3_EXTERNAL_EVALUATION.md",
        "STAGE_21_MULTICENTER_GENERALIZATION_AUDIT.md",
        "STAGE_22_DATASET_READINESS_AUDIT.md",
        "STAGE_23_DATASET_SOURCE_VERIFICATION.md",
        "STAGE_24_CLINICAL_COLLABORATION_PREPARATION.md"
    ]
    for rname in expected_reports:
        rpath = STAGE_REPORTS_DIR / rname
        assert rpath.exists(), f"Missing stage report: {rname}"
        assert rpath.stat().st_size > 1000, f"Stage report {rname} is suspiciously short"


def test_14_preservation_of_host_python_and_frontend():
    """Verify host CPU Python and React frontend remain intact."""
    frontend_dir = WORKSPACE_ROOT / "osteophorosis-main/src"
    assert frontend_dir.exists(), "React frontend directory missing"
    host_python = Path(r"C:\Python313\python.exe")
    assert host_python.exists(), "Host Python executable missing"

