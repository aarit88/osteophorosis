from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ml_service.app.api.v1.research_routes import get_authenticated_reviewer, qualification_router, _require_qualification_role
from ml_service.app.services.source_qualification import (
    AcquisitionPlan, Evidence, QualificationError, QualificationReview, SourceAssessmentUpdate, Stage38Handoff,
    SourceQualification, SourceQualificationService,
)


def source(source_id="candidate_001", **kwargs):
    return SourceQualification(
        source_id=source_id, source_name="Synthetic qualification fixture",
        original_url="https://example.invalid/synthetic-fixture", repository_or_institution="Synthetic test fixture",
        modalities=["ct"], available_labels=["fixture label"], label_mapping={"SECONDARY_BONE_METASTASIS": ["fixture label"]},
        target_class_interest=["SECONDARY_BONE_METASTASIS"],
        known_limitations=["Synthetic test fixture; not a real dataset."], **kwargs,
    )


def ev(kind="PUBLISHED_CLAIM", evidence_id="ev-test-001"):
    return Evidence(evidence_id=evidence_id, kind=kind, statement="Synthetic fixture supports a test-only claim.", reference_url="https://example.invalid/test")


def test_registration_evidence_review_and_persistence_are_audited(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    row = svc.create_source(source(), "reviewer-pseudo-fixture")
    assert row["qualification_status"] == "DISCOVERED"
    svc.add_evidence(row["source_id"], ev(), "reviewer-pseudo-fixture")
    decision = QualificationReview(status="INFORMATION_VERIFIED", evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="Synthetic test review only.")
    reviewed = svc.review(row["source_id"], decision, "reviewer-pseudo-fixture")
    assert reviewed["responsible_reviewer_reference"] == "reviewer-pseudo-fixture"
    assert SourceQualificationService(tmp_path / "registry.json").get_source(row["source_id"])["version"] == 3


def test_candidate_registration_cannot_self_assert_reviewer_or_access_approval(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    with pytest.raises(QualificationError, match="reviewer review"):
        svc.create_source(source(responsible_reviewer_reference="invented-reviewer"), "creator")
    with pytest.raises(QualificationError, match="evidence-linked reviewer"):
        svc.create_source(source(access_request_status="APPROVED"), "creator")


def test_counts_require_evidence_and_assumptions_cannot_qualify(tmp_path):
    with pytest.raises(ValidationError, match="claims must reference evidence"):
        source(verified_patient_count=20)
    svc = SourceQualificationService(tmp_path / "registry.json")
    svc.create_source(source(), "r")
    svc.add_evidence("candidate_001", ev("UNVERIFIED_ASSUMPTION"), "r")
    with pytest.raises(QualificationError, match="assumptions"):
        svc.review("candidate_001", QualificationReview(status="INFORMATION_VERIFIED", evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="Not actually verified."), "r")


def test_review_requires_existing_evidence_and_explicit_confirmation(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    svc.create_source(source(), "r")
    with pytest.raises(QualificationError, match="explicitly confirm"):
        svc.review("candidate_001", QualificationReview(status="INFORMATION_VERIFIED", evidence_ids=["unknown"], explicit_confirmation=False, notes="No."), "r")
    with pytest.raises(QualificationError, match="reference evidence"):
        svc.review("candidate_001", QualificationReview(status="INFORMATION_VERIFIED", evidence_ids=["unknown"], explicit_confirmation=True, notes="No evidence."), "r")


def test_license_and_access_assessments_require_cited_evidence_and_external_references(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    svc.create_source(source(), "r"); svc.add_evidence("candidate_001", ev(), "r")
    with pytest.raises(QualificationError, match="authorization reference"):
        svc.update_assessment("candidate_001", SourceAssessmentUpdate(evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="No permission fabrication.", access_request_status="APPROVED"), "r")
    updated = svc.update_assessment("candidate_001", SourceAssessmentUpdate(evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="Synthetic test assessment.", license_verification_status="PENDING", access_request_status="APPROVED", access_authorization_reference="external-reference"), "r")
    assert updated["access_request_status"] == "APPROVED"
    assert updated["assessment_evidence_ids"] == ["ev-test-001"]
    assert updated["assessment_references"]["access_authorization_reference"] == "external-reference"


def test_coverage_excludes_unsubstantiated_candidate_and_reports_no_patient_counts(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    svc.create_source(source(), "r")
    matrix = svc.coverage()
    assert matrix["actual_patient_counts"] is None
    row = next(row for row in matrix["rows"] if row["diagnostic_class"] == "SECONDARY_BONE_METASTASIS" and row["modality"] == "ct")
    assert "candidate_001" not in row["candidate_source_ids"]
    svc.add_evidence("candidate_001", ev(), "r")
    matrix = svc.coverage()
    row = next(row for row in matrix["rows"] if row["diagnostic_class"] == "SECONDARY_BONE_METASTASIS" and row["modality"] == "ct")
    assert "candidate_001" in row["candidate_source_ids"]
    assert row["ontology_mapping_source_ids"] == ["candidate_001"]
    assert row["prepared"] is False


def test_acquisition_plan_does_not_grant_access_and_requires_real_handoff(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    svc.create_source(source(), "r")
    plan = AcquisitionPlan(source_id="candidate_001", method="Research request", access_request_instructions="Submit form", transfer_method="Approved institutional secure transfer", storage_location_reference="protected storage reference", expected_package_structure="versioned dataset folder", expected_modalities=["ct"], expected_labels=["source labels"], expected_manifest="Stage33 manifest", expected_provenance=["source citation"], verification_requirements=["Stage38 package verification"], responsible_researcher_reference="researcher reference")
    saved = svc.create_plan(plan, "r")
    assert saved["status"] == "DRAFT"
    assert saved["blockers"]
    assert svc.get_source("candidate_001")["qualification_status"] == "DISCOVERED"
    with pytest.raises(QualificationError, match="handed to Stage38"):
        svc.handoff_stage38("candidate_001", Stage38Handoff(stage38_source_id="missing", explicit_confirmation=True, notes="Test only."), "r", type("Onboarding", (), {"list_sources": lambda self: []})())


def test_request_plan_needs_qualified_source_and_external_approval_reference(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    svc.create_source(source(), "r"); svc.add_evidence("candidate_001", ev(), "r")
    svc.update_assessment("candidate_001", SourceAssessmentUpdate(evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="Synthetic reviewed route.", license_verification_status="PENDING", access_request_status="REQUIRED"), "r")
    for target in ["INFORMATION_VERIFIED", "ACCESS_REQUEST_REQUIRED", "QUALIFIED_FOR_REQUEST"]:
        svc.review("candidate_001", QualificationReview(status=target, evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="Evidence-linked synthetic review."), "r")
    plan = AcquisitionPlan(source_id="candidate_001", method="Request", access_request_instructions="Official steps", transfer_method="Approved transfer", storage_location_reference="storage reference", expected_package_structure="package structure", expected_modalities=["ct"], expected_labels=["original labels"], expected_manifest="manifest reference", expected_provenance=["source reference"], verification_requirements=["Stage38 verification"], responsible_researcher_reference="external researcher reference")
    svc.create_plan(plan, "r")
    svc.transition_plan(plan.plan_id, "REVIEW_REQUIRED", "r", "Submitted for review.")
    ready = svc.transition_plan(plan.plan_id, "READY_FOR_EXTERNAL_REQUEST", "r", "Review completed.")
    assert ready["status"] == "READY_FOR_EXTERNAL_REQUEST"
    svc.transition_plan(plan.plan_id, "REQUEST_SUBMITTED", "r", "Operator submitted outside the application.")
    svc.transition_plan(plan.plan_id, "ACCESS_DECISION_PENDING", "r", "External decision is pending.")
    with pytest.raises(QualificationError, match="authorization reference"):
        svc.transition_plan(plan.plan_id, "APPROVED_EXTERNALLY", "r", "No fabricated approval.")
    approved = svc.transition_plan(plan.plan_id, "APPROVED_EXTERNALLY", "r", "External authorization was checked.", "externally-issued-reference")
    assert approved["external_authorization_reference"] == "externally-issued-reference"


def test_stage38_handoff_requires_exact_source_reference_and_modality(tmp_path):
    svc = SourceQualificationService(tmp_path / "registry.json")
    svc.create_source(source(), "r"); svc.add_evidence("candidate_001", ev(), "r")
    svc.update_assessment("candidate_001", SourceAssessmentUpdate(evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="Synthetic governance assessment.", license_verification_status="VERIFIED", access_request_status="REQUIRED", data_use_agreement_status="REQUIRED", access_requirements=["Institutional authorization"], ethics_requirements=["Ethics review requirement"], governance_requirements_reviewed=True), "r")
    for target in ["INFORMATION_VERIFIED", "GOVERNANCE_REVIEW_REQUIRED", "QUALIFIED_FOR_ACQUISITION"]:
        svc.review("candidate_001", QualificationReview(status=target, evidence_ids=["ev-test-001"], explicit_confirmation=True, notes="Synthetic test review."), "r")
    onboarding = type("Onboarding", (), {"list_sources": lambda self: [{"source_id": "stage38_fixture", "source_reference": "https://different.example/source", "modalities": ["ct"]}]})()
    with pytest.raises(QualificationError, match="does not match"):
        svc.handoff_stage38("candidate_001", Stage38Handoff(stage38_source_id="stage38_fixture", explicit_confirmation=True, notes="Test mismatch."), "r", onboarding)


def test_ledger_detects_tampering_and_symlink_is_rejected(tmp_path):
    path = tmp_path / "registry.json"
    svc = SourceQualificationService(path)
    svc.create_source(source(), "r")
    record = json.loads(path.read_text())
    record["sources"]["candidate_001"]["source_name"] = "Tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(QualificationError, match="integrity"):
        svc.list_sources()
    if hasattr(path, "symlink_to"):
        path.unlink()
        other = tmp_path / "other.json"; other.write_text("{}")
        try: path.symlink_to(other)
        except OSError: pytest.skip("Symlink creation is unavailable on this platform.")
        with pytest.raises(QualificationError, match="symlink"):
            svc._save(svc._empty())


def test_api_contract_and_role_gate():
    paths = {route.path for route in qualification_router.routes}
    assert "/api/v1/dataset-qualification/sources" in paths
    assert "/api/v1/dataset-qualification/coverage" in paths
    assert "/api/v1/dataset-qualification/feasibility" in paths
    route = next(route for route in qualification_router.routes if route.path.endswith("/sources") and "GET" in route.methods)
    assert route.dependant.dependencies[0].call is get_authenticated_reviewer
    with pytest.raises(Exception) as exc:
        _require_qualification_role({"roles": frozenset({"CLINICAL_ADJUDICATOR"})}, reviewers=True)
    assert getattr(exc.value, "status_code", None) == 403


def test_api_authentication_requires_configured_reviewer_and_forbidden_role():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as auth_error:
        get_authenticated_reviewer(authorization="")
    assert auth_error.value.status_code in {401, 503}
    with pytest.raises(HTTPException) as denied:
        _require_qualification_role({"roles": frozenset({"CLINICAL_ADJUDICATOR"})}, reviewers=True)
    assert denied.value.status_code == 403


def test_curated_catalog_stays_discovered_and_does_not_map_classes():
    sources = SourceQualificationService().list_sources()
    assert {row["source_id"] for row in sources} >= {"TCIA_SPINE_METS_CT_SEG", "TCIA_HEALTHY_TOTAL_BODY_CT", "EURREB_OI_REGISTRY"}
    assert all(row["qualification_status"] == "DISCOVERED" for row in sources)
    assert all(not row["label_mapping"] for row in sources)
    assert all(row["reviewed_at"] is None for row in sources)
    healthy_ct = next(row for row in sources if row["source_id"] == "TCIA_HEALTHY_TOTAL_BODY_CT")
    assert healthy_ct["source_reported_patient_count"] == 30
    assert healthy_ct["verified_patient_count"] is None
