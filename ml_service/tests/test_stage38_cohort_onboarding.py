from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from ml_service.app.api.v1.research_routes import get_authenticated_reviewer, _require_onboarding_role, onboarding_router
from ml_service.app.services.cohort_onboarding import (
    AcquisitionEvent, CohortOnboardingService, OnboardingStateError, SourceRegistration,
)
from ml_service.tests.test_track_d_intake import _valid_record


def source_payload(source_id="SRC_LOCAL_01"):
    return SourceRegistration(
        source_id=source_id, name="Fixture cohort", source_type="INSTITUTIONAL",
        source_reference="institutional-reference-001", description="Isolated nonclinical validation fixture.",
        modalities=["xray"], target_categories=["PAGETS_DISEASE"], acquisition_method="Authorized secure transfer",
        license_or_terms="Research terms reference", data_use_reference="DUA reference recorded, approval not asserted",
        ethics_reference="Ethics reference recorded, approval not asserted", version="v1",
        access_restriction="RESTRICTED", intended_use="Governed research intake only",
    )


def event(status, auth=None):
    return AcquisitionEvent(status=status, reason="Test lifecycle transition", authorization_reference=auth,
                            evidence_reference="evidence-reference")


def make_package(root: Path, source_id="SRC_LOCAL_01"):
    package = root / "packages" / source_id / "v1"
    record = _valid_record(package, "SMP_ONBOARD_01", "PT_ONBOARD_01", "paget", "xray")
    (package / "manifest.json").write_text(json.dumps({
        "manifest_version": "2.0.0", "track": "track_d",
        "ontology_version": "OSTEO_ONTOLOGY_V1.0_STAGE34", "records": [record],
    }), encoding="utf-8")
    return package


def test_registry_schema_persistence_duplicate_and_role_api(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    source = service.register(source_payload(), "REV_STEWARD_01")
    assert source["acquisition_status"] == "REGISTERED"
    assert CohortOnboardingService(tmp_path / "registry.json").source("SRC_LOCAL_01")["version"] == "v1"
    with pytest.raises(FileExistsError):
        service.register(source_payload(), "REV_STEWARD_01")

    paths = {route.path for route in onboarding_router.routes}
    assert "/api/v1/dataset-onboarding/sources" in paths
    sources_route = next(route for route in onboarding_router.routes if route.path == "/api/v1/dataset-onboarding/sources" and "GET" in route.methods)
    assert sources_route.dependant.dependencies[0].call is get_authenticated_reviewer
    with pytest.raises(HTTPException) as denied:
        _require_onboarding_role({"roles": frozenset({"CLINICAL_ADJUDICATOR"})})
    assert denied.value.status_code == 403


def test_access_approval_requires_authorization_reference_and_transition_order(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    service.transition("SRC_LOCAL_01", event("ACCESS_PENDING"), "REV_STEWARD_01")
    with pytest.raises(OnboardingStateError, match="Invalid acquisition transition"):
        service.transition("SRC_LOCAL_01", event("ACQUIRED"), "REV_STEWARD_01")
    with pytest.raises(OnboardingStateError, match="authorization reference"):
        service.transition("SRC_LOCAL_01", event("ACCESS_APPROVED"), "REV_PI_01")
    assert service.source("SRC_LOCAL_01")["governance_approved"] is False


def test_missing_provenance_blocks_verification(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    payload = source_payload().model_dump()
    payload["license_or_terms"] = None
    payload["ethics_reference"] = None
    payload["data_use_reference"] = None
    service.register(SourceRegistration.model_validate(payload), "REV_STEWARD_01")
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    report = service.verify_package("SRC_LOCAL_01")
    assert report["valid"] is False
    assert "Required license/data-use and ethics references are not documented." in report["errors"]


def test_expected_source_fingerprint_mismatch_is_rejected(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    payload = source_payload().model_dump()
    payload["fingerprint_sha256"] = "0" * 64
    service.register(SourceRegistration.model_validate(payload), "REV_STEWARD_01")
    make_package(tmp_path)
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    report = service.verify_package("SRC_LOCAL_01")
    assert report["valid"] is False
    assert any("expected fingerprint" in error for error in report["errors"])


def test_manifest_package_verification_atomic_stage_and_change_detection(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    package = make_package(tmp_path)
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    verification = service.verify_package("SRC_LOCAL_01")
    assert verification["valid"] is True
    assert verification["clinical_validity"] == "NOT_ESTABLISHED"
    assert verification["training_authorized"] is False
    assert service.transition("SRC_LOCAL_01", event("VERIFIED"), "REV_STEWARD_01")["verification_status"] == "VERIFIED"
    assert "inventory" not in service.list_sources()[0]["package"]
    staged = service.stage("SRC_LOCAL_01", "REV_STEWARD_01")
    assert staged["status"] == "STAGED_FOR_STAGE33_INTAKE"
    assert staged["training_authorized"] is False
    with pytest.raises(FileExistsError):
        service.stage("SRC_LOCAL_01", "REV_STEWARD_01")
    assert (tmp_path / "staged" / "SRC_LOCAL_01" / "v1" / "manifest.json").is_file()
    provenance = json.loads((tmp_path / "staged" / "SRC_LOCAL_01" / "v1" / "provenance.json").read_text())
    assert provenance["inventory_sha256"] == service.source("SRC_LOCAL_01")["package"]["inventory_sha256"]
    assert provenance["event_chain"][-1]["action"] == "PACKAGE_STAGED_FOR_INTAKE"
    assert provenance["training_authorized"] is False


def test_integrity_report_is_deterministic_and_detects_post_verification_change(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    package = make_package(tmp_path)
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    first = service.verify_package("SRC_LOCAL_01")
    second = service.verify_package("SRC_LOCAL_01")
    assert first["inventory_sha256"] == second["inventory_sha256"]
    (package / "unexpected.exe").write_bytes(b"no")
    changed = service.verify_package("SRC_LOCAL_01")
    assert changed["valid"] is False
    assert any("Unsupported file extension" in reason for reason in changed["errors"])


def test_empty_files_and_symlinks_fail_package_verification(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    package = make_package(tmp_path)
    (package / "empty.csv").touch()
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    result = service.verify_package("SRC_LOCAL_01")
    assert result["valid"] is False
    assert any("Empty file" in item for item in result["errors"])


def test_package_symlink_is_rejected(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    package = make_package(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("not in package")
    try:
        os.symlink(outside, package / "linked.csv")
    except OSError:
        pytest.skip("Symlink creation is not available to this test account.")
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    result = service.verify_package("SRC_LOCAL_01")
    assert result["valid"] is False
    assert any("Symbolic-link files" in item for item in result["errors"])


def test_direct_identifier_filename_is_rejected_without_echoing_value(tmp_path):
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    package = make_package(tmp_path)
    (package / "MRN_987654.csv").write_text("placeholder")
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    report = service.verify_package("SRC_LOCAL_01")
    serialized = json.dumps(report)
    assert report["valid"] is False
    assert "Potential direct identifier" in serialized
    assert "987654" not in serialized


def test_tampered_ledger_is_rejected(tmp_path):
    state_path = tmp_path / "registry.json"
    service = CohortOnboardingService(state_path, tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    state = json.loads(state_path.read_text())
    state["events"][0]["details"]["version"] = "altered"
    state_path.write_text(json.dumps(state))
    with pytest.raises(OnboardingStateError, match="integrity verification failed"):
        service.list_sources()


def test_rejects_unsafe_version_and_prevents_staging_a_changed_package(tmp_path):
    with pytest.raises(ValidationError):
        SourceRegistration.model_validate({**source_payload().model_dump(), "version": "../outside"})
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged")
    service.register(source_payload(), "REV_STEWARD_01")
    package = make_package(tmp_path)
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    service.transition("SRC_LOCAL_01", event("VERIFIED"), "REV_STEWARD_01")
    image = next(package.rglob("*.png"))
    image.write_bytes(image.read_bytes() + b"modified after verification")
    with pytest.raises(OnboardingStateError, match="changed after verification"):
        service.stage("SRC_LOCAL_01", "REV_STEWARD_01")
    assert not (tmp_path / "staged" / "SRC_LOCAL_01" / "v1").exists()
    assert service.source("SRC_LOCAL_01")["acquisition_status"] == "RESTRICTED"
    assert service.events("SRC_LOCAL_01")[-1]["action"] == "POST_VERIFICATION_INTEGRITY_FAILURE"


def test_package_verification_uses_central_protected_holdout_registry(tmp_path):
    project = tmp_path / "project"
    manifest_dir = project / "ml_service" / "data" / "splits"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "btxrd_split_manifest.csv").write_text(
        "image_id,split\nSMP_ONBOARD_01,center2\n", encoding="utf-8"
    )
    service = CohortOnboardingService(tmp_path / "registry.json", tmp_path / "packages", tmp_path / "staged", project)
    service.register(source_payload(), "REV_STEWARD_01")
    make_package(tmp_path)
    for status, auth in [("ACCESS_PENDING", None), ("ACCESS_APPROVED", "external-auth-ref"),
                         ("ACQUIRED", None), ("VERIFICATION_PENDING", None)]:
        service.transition("SRC_LOCAL_01", event(status, auth), "REV_STEWARD_01")
    result = service.verify_package("SRC_LOCAL_01")
    assert result["valid"] is False
    assert result["intake_validation"]["protected_holdout_registry"]["sample_id_count"] == 1
    assert result["intake_validation"]["rejection_reason_counts"]["PROTECTED_EXTERNAL_HOLDOUT_SAMPLE_ID"] >= 1
