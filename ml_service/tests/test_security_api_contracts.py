from __future__ import annotations

import hashlib
import hmac
import time
import json
from fastapi import HTTPException

from ml_service.app.api.v1.ml_routes import _verify_gateway_hmac
from ml_service.app.main import create_app
from ml_service.app.schemas.research import WorkflowTransitionRequest
from ml_service.app.services.research_service import ResearchService
from ml_service.app.services.readiness_service import MLReadinessService
from ml_service.config.settings import MLSettings
from ml_service.app.api.v1.research_routes import get_authenticated_clinical_reviewer
from ml_service.app import main as main_module


def test_gateway_hmac_is_mandatory_in_production(monkeypatch):
    monkeypatch.setenv("OSTEO_ENV", "production")
    monkeypatch.delenv("OSTEO_GATEWAY_HMAC_SECRET", raising=False)
    try:
        _verify_gateway_hmac("POST", "/api/v1/analyze", b"body", None, None)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 503
    else:
        raise AssertionError("Production accepted an unsigned gateway request")


def test_gateway_hmac_accepts_valid_signed_payload(monkeypatch):
    secret, body, path = "a" * 32, b"payload", "/api/v1/analyze"
    monkeypatch.setenv("OSTEO_ENV", "test")
    monkeypatch.setenv("OSTEO_GATEWAY_HMAC_SECRET", secret)
    timestamp = str(int(time.time()))
    payload = f"{timestamp}:POST:{path}:{hashlib.sha256(body).hexdigest()}".encode()
    signature = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    _verify_gateway_hmac("POST", path, body, timestamp, signature)


def test_readiness_governance_and_track_d_route_contracts():
    app = create_app()
    paths = app.openapi()["paths"]
    assert "get" in paths["/api/v1/datasets/readiness"]
    assert "get" in paths["/api/v1/datasets/governance-status"]
    assert "post" in paths["/api/v1/diagnostic/four-class"]
    assert "503" in paths["/api/v1/diagnostic/four-class"]["post"]["responses"]
    adjudication = paths["/api/v1/datasets/adjudication/decision"]["post"]
    assert "authorization" in {param["name"].lower() for param in adjudication.get("parameters", [])}
    assert "requestBody" in adjudication
    readiness = ResearchService().get_track_d_dataset_readiness().model_dump()
    governance = ResearchService().get_dataset_governance_status().model_dump()
    assert {"readiness_status", "ready_for_training", "manifest_schema_validation", "protected_holdouts"} <= readiness.keys()
    assert {"governance_gate_status", "pending_ethics_approvals_count", "pending_duas_count"} <= governance.keys()


def test_workflow_transition_rejects_self_reported_reviewer_identity_or_role():
    service = ResearchService()
    request = WorkflowTransitionRequest(
        source_id="unrelated", target_state="SOURCE_VERIFIED", reviewer_pseudo_id="RAD_01",
        reviewer_role="PRINCIPAL_INVESTIGATOR", reason="test", evidence_reference="test",
    )
    try:
        service.validate_workflow_transition(request, {"reviewer_pseudo_id": "RAD_01", "roles": {"RADIOLOGIST"}})
    except PermissionError:
        pass
    else:
        raise AssertionError("A self-reported unauthorized role was accepted")


def test_track_d_checkpoint_must_be_explicitly_sha_pinned(tmp_path, monkeypatch):
    artifact = tmp_path / "candidate.pt"
    artifact.write_bytes(b"fixture bytes only; not model weights")
    settings = MLSettings(active_checkpoint_path=artifact, checkpoints_dir=tmp_path)
    service = MLReadinessService(settings=settings)
    monkeypatch.delenv("OSTEO_TRACK_D_CHECKPOINT_SHA256", raising=False)
    assert service.get_active_checkpoint() is None
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setenv("OSTEO_TRACK_D_CHECKPOINT_SHA256", expected)
    assert service.get_active_checkpoint() == artifact
    monkeypatch.setenv("OSTEO_TRACK_D_CHECKPOINT_SHA256", "0" * 64)
    assert service.get_active_checkpoint() is None


def test_track_a_b_or_c_checkpoints_are_not_discovered_as_track_d(tmp_path, monkeypatch):
    (tmp_path / "track-c-frozen.pt").write_bytes(b"preserved other-track artifact")
    settings = MLSettings(active_checkpoint_path=None, checkpoints_dir=tmp_path)
    monkeypatch.setenv("OSTEO_TRACK_D_CHECKPOINT_SHA256", "a" * 64)
    assert MLReadinessService(settings=settings).get_active_checkpoint() is None


def test_fastapi_cors_origins_accept_exact_comma_separated_allowlist(monkeypatch):
    monkeypatch.setenv("OSTEO_CORS_ORIGINS", "https://clinic.example,https://review.example")
    settings = MLSettings(_env_file=None)
    assert settings.cors_origins == ["https://clinic.example", "https://review.example"]


def test_production_startup_requires_institutional_reviewer_registry_and_ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("OSTEO_ENV", "production")
    monkeypatch.setenv("OSTEO_GATEWAY_HMAC_SECRET", "g" * 64)
    monkeypatch.setenv("OSTEO_AUDIT_LEDGER_HMAC_KEY", "l" * 64)
    monkeypatch.setenv("OSTEO_ONBOARDING_LEDGER_HMAC_KEY", "o" * 64)
    monkeypatch.setenv("OSTEO_SOURCE_QUALIFICATION_HMAC_KEY", "q" * 64)
    monkeypatch.setenv("OSTEO_SOURCE_QUALIFICATION_FILE", str(tmp_path / "qualification.json"))
    monkeypatch.setenv("OSTEO_ADJUDICATION_LEDGER_FILE", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("OSTEO_CORS_ORIGINS", "https://research.example")
    monkeypatch.setenv("OSTEO_REVIEWER_AUTHORIZATION_FILE", str(tmp_path / "missing-registry.json"))
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: MLSettings(env="production", cors_origins=["https://research.example"], _env_file=None),
    )

    try:
        main_module.create_app()
    except RuntimeError as exc:
        assert "institution-provisioned reviewer authorization registry" in str(exc)
    else:
        raise AssertionError("Production startup accepted missing reviewer authorization configuration")


def test_clinical_adjudication_dependency_rejects_provisioned_nonclinical_role(tmp_path, monkeypatch):
    token = "synthetic-test-token-only"
    registry = tmp_path / "test-reviewers.json"
    registry.write_text(json.dumps({"registry_version": "1.0", "reviewers": [{
        "reviewer_pseudo_id": "REV_TEST_ONLY", "roles": ["PRINCIPAL_INVESTIGATOR"],
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(), "active": True,
    }]}), encoding="utf-8")
    monkeypatch.setenv("OSTEO_REVIEWER_AUTHORIZATION_FILE", str(registry))
    try:
        get_authenticated_clinical_reviewer(f"Bearer {token}")
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("A non-clinical reviewer role was allowed to adjudicate")

