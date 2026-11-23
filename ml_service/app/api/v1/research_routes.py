"""
OsteoGuard AI - Research & Integrated Prototype API Endpoints (v1).

Exposes:
- Research Tracks (A, B, C, D)
- Model Evidence Registry
- Architecture Readiness & Governance Checklist
- Radiomics, Biomechanics, Explainability, and Uncertainty Demonstrations
- End-to-End Synthetic Research Workbench
- Safe Gating: Four-Class Diagnostic Inference strictly BLOCKED
"""

from fastapi import APIRouter, Depends, Header, HTTPException, status
import os
from pathlib import Path

from ml_service.app.schemas.research import (
    ResearchTracksResponse,
    EvidenceRegistryResponse,
    ArchitectureReadinessResponse,
    DatasetGovernanceResponse,
    RadiomicsDemoRequest,
    RadiomicsDemoResponse,
    BiomechanicsGeometryRequest,
    BiomechanicsGeometryResponse,
    BiomechanicsMirelsRequest,
    BiomechanicsMirelsResponse,
    ExplainabilityDemoRequest,
    ExplainabilityDemoResponse,
    UncertaintyDemoRequest,
    UncertaintyDemoResponse,
    SyntheticWorkbenchRunRequest,
    SyntheticWorkbenchRunResponse,
    TrackDDatasetReadinessResponse,
    DatasetSourcesResponse,
    DatasetAcquisitionStatusResponse,
    DatasetGovernanceStatusResponse,
    CohortPlanningResponse,
    WorkflowTransitionRequest,
    ReviewerAdjudicationDecisionRequest,
)
from ml_service.app.services.research_service import ResearchService
from ml_service.app.services.reviewer_authorization import (
    ReviewerAuthorizationConfigurationError, ReviewerAuthorizationError,
    ReviewerAuthorizationService,
)
from ml_service.app.services.cohort_onboarding import (
    AcquisitionEvent, CohortOnboardingService, OnboardingStateError, SourceRegistration,
)

router = APIRouter(prefix="/api/v1", tags=["Research Operations & Prototype Workbench"])
onboarding_router = APIRouter(prefix="/api/v1/dataset-onboarding", tags=["Dataset Acquisition & Provenance"])

ONBOARDING_ROLES = {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR", "DATA_PROTECTION_OFFICER", "IRB_OFFICER", "LEGAL_COUNSEL"}


def get_authenticated_reviewer(authorization: str = Header(default="")):
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    try:
        return ReviewerAuthorizationService().authenticate_bearer(token)
    except ReviewerAuthorizationConfigurationError as exc:
        raise HTTPException(status_code=503, detail="Reviewer authorization is not configured.") from exc
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=401, detail="Valid reviewer credentials are required.") from exc


def _onboarding_service() -> CohortOnboardingService:
    return CohortOnboardingService()


def _require_onboarding_role(principal, allowed_roles=ONBOARDING_ROLES):
    if not principal["roles"].intersection(allowed_roles):
        raise HTTPException(status_code=403, detail="Dataset onboarding role is required.")
    return principal


@onboarding_router.get("/sources")
def onboarding_sources(principal=Depends(get_authenticated_reviewer)):
    _require_onboarding_role(principal)
    return {"schema_version": CohortOnboardingService.SCHEMA_VERSION, "sources": _onboarding_service().list_sources()}


@onboarding_router.post("/sources", status_code=201)
def register_onboarding_source(payload: SourceRegistration, principal=Depends(get_authenticated_reviewer)):
    _require_onboarding_role(principal, {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"})
    try:
        return _onboarding_service().register(payload, principal["reviewer_pseudo_id"])
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OnboardingStateError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@onboarding_router.get("/sources/{source_id}")
def onboarding_source(source_id: str, principal=Depends(get_authenticated_reviewer)):
    _require_onboarding_role(principal)
    try:
        return _onboarding_service().source(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OnboardingStateError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@onboarding_router.post("/sources/{source_id}/events")
def onboarding_event(source_id: str, payload: AcquisitionEvent, principal=Depends(get_authenticated_reviewer)):
    role_map = {
        "ACCESS_PENDING": {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"},
        "ACCESS_APPROVED": {"PRINCIPAL_INVESTIGATOR", "DATA_PROTECTION_OFFICER", "IRB_OFFICER", "LEGAL_COUNSEL"},
        "ACQUIRED": {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"},
        "VERIFICATION_PENDING": {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"},
        "VERIFIED": {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"},
        "REJECTED": {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR", "DATA_PROTECTION_OFFICER", "IRB_OFFICER", "LEGAL_COUNSEL"},
        "RESTRICTED": {"DATA_PROTECTION_OFFICER", "IRB_OFFICER", "LEGAL_COUNSEL"},
        "ARCHIVED": {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"},
    }
    _require_onboarding_role(principal, role_map[payload.status])
    try:
        return _onboarding_service().transition(source_id, payload, principal["reviewer_pseudo_id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OnboardingStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@onboarding_router.get("/sources/{source_id}/provenance")
def onboarding_provenance(source_id: str, principal=Depends(get_authenticated_reviewer)):
    _require_onboarding_role(principal)
    service = _onboarding_service()
    try:
        source = service.source(source_id)
        return {"source_id": source_id, "source": source, "events": service.events(source_id),
                "integrity_is_authenticity": False, "clinical_validity": "NOT_ESTABLISHED", "training_authorized": False}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OnboardingStateError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@onboarding_router.post("/sources/{source_id}/verify")
def verify_onboarding_package(source_id: str, principal=Depends(get_authenticated_reviewer)):
    _require_onboarding_role(principal, {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"})
    try:
        return _onboarding_service().verify_package(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OnboardingStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@onboarding_router.post("/sources/{source_id}/stage", status_code=201)
def stage_onboarding_package(source_id: str, principal=Depends(get_authenticated_reviewer)):
    _require_onboarding_role(principal, {"DATA_STEWARD", "PRINCIPAL_INVESTIGATOR"})
    try:
        return _onboarding_service().stage(source_id, principal["reviewer_pseudo_id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OnboardingStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

# Dependency singleton
_service_instance = None


def get_research_service() -> ResearchService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ResearchService()
    return _service_instance


def get_authenticated_clinical_reviewer(authorization: str = Header(default="")):
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    try:
        principal = ReviewerAuthorizationService().authenticate_bearer(token)
    except ReviewerAuthorizationConfigurationError as exc:
        raise HTTPException(status_code=503, detail="Reviewer authorization is not configured.") from exc
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=401, detail="Valid reviewer credentials are required.") from exc
    if not principal["roles"].intersection({"CLINICAL_ADJUDICATOR", "RADIOLOGIST"}):
        raise HTTPException(status_code=403, detail="Clinical adjudication role is required.")
    return principal


@router.get(
    "/research/status",
    summary="Research Subsystem Overall Status",
    description="Returns aggregate research prototype status and tracks breakdown.",
)
def get_research_status(service: ResearchService = Depends(get_research_service)):
    tracks_res = service.get_research_tracks()
    arch_res = service.get_architecture_readiness()
    return {
        "status": "RESEARCH_PROTOTYPE_ONLINE",
        "prototype_version": "1.0.0",
        "four_class_inference_status": "BLOCKED — APPROVED CLINICAL DATA AND GOVERNANCE REQUIRED",
        "active_tracks_count": tracks_res.active_tracks_count,
        "tracks": [t.title for t in tracks_res.tracks],
        "architecture_readiness": arch_res.system_readiness,
        "frozen_checkpoint_status": next(
            (item["status"] for item in service.get_checkpoint_inventory()["tracks"] if item.get("track") == "C"),
            "MISSING",
        ),
        "synthetic_workbench_available": True,
        "disclaimer": "RESEARCH PROTOTYPE ONLY: Demonstrates verified engineering components on synthetic phantoms.",
    }


@router.get(
    "/research/tracks",
    response_model=ResearchTracksResponse,
    summary="Four Discrete Research Tracks",
    description="Returns metadata, historical metrics, and limitations for Tracks A, B, C, and D.",
)
def get_research_tracks(service: ResearchService = Depends(get_research_service)) -> ResearchTracksResponse:
    return service.get_research_tracks()


@router.get(
    "/evidence/registry",
    response_model=EvidenceRegistryResponse,
    summary="Model Evidence Registry",
    description="Central registry documenting historical model training, validation evidence, and holdout limitations.",
)
def get_evidence_registry(service: ResearchService = Depends(get_research_service)) -> EvidenceRegistryResponse:
    return service.get_evidence_registry()


@router.get("/artifacts/checkpoints", summary="Read-only checkpoint inventory and SHA-256 verification")
def get_checkpoint_inventory(service: ResearchService = Depends(get_research_service)):
    return service.get_checkpoint_inventory()


@router.get(
    "/architecture/readiness",
    response_model=ArchitectureReadinessResponse,
    summary="Four-Class Architecture Component Readiness",
    description="Detailed component-by-component implementation and verification audit.",
)
def get_architecture_readiness(service: ResearchService = Depends(get_research_service)) -> ArchitectureReadinessResponse:
    return service.get_architecture_readiness()


@router.get(
    "/datasets/status",
    response_model=DatasetGovernanceResponse,
    summary="Dataset Readiness & Governance Status",
    description="Audits clinical dataset availability, acquisition blockers, and required ethical clearances.",
)
def get_dataset_governance_status(service: ResearchService = Depends(get_research_service)) -> DatasetGovernanceResponse:
    return service.get_dataset_governance()


@router.get(
    "/datasets/readiness",
    response_model=TrackDDatasetReadinessResponse,
    summary="Stage 33 Track D Dataset Intake & Leakage-Safe Readiness",
    description="Reports modality-separated counts, metadata/quarantine audit, split distribution, and 7 readiness gate criteria.",
)
def get_track_d_dataset_readiness(service: ResearchService = Depends(get_research_service)) -> TrackDDatasetReadinessResponse:
    return service.get_track_d_dataset_readiness()


@router.get(
    "/datasets/sources",
    response_model=DatasetSourcesResponse,
    summary="Stage 35 Institutional Dataset Source Registry",
    description="Lists proposed and verified clinical cohort sources across all 4 diagnostic classes.",
)
def get_dataset_sources(service: ResearchService = Depends(get_research_service)) -> DatasetSourcesResponse:
    return service.get_dataset_sources()


@router.get(
    "/datasets/acquisition-status",
    response_model=DatasetAcquisitionStatusResponse,
    summary="Stage 35 Cohort Acquisition Workflow Status",
    description="Tracks sequential workflow stages, received scans, validated patient counts, and outstanding blockers per cohort.",
)
def get_dataset_acquisition_status(
    service: ResearchService = Depends(get_research_service),
) -> DatasetAcquisitionStatusResponse:
    return service.get_dataset_acquisition_status()


@router.get(
    "/datasets/governance-status",
    response_model=DatasetGovernanceStatusResponse,
    summary="Stage 35 Institutional Ethics, DUA & Governance Status",
    description="Reports pending ethics approvals, pending DUAs, expired clearances, evidence completeness, and draft institutional templates.",
)
def get_dataset_governance_status_v2(
    service: ResearchService = Depends(get_research_service),
) -> DatasetGovernanceStatusResponse:
    return service.get_dataset_governance_status()


@router.get(
    "/datasets/cohort-planning",
    response_model=CohortPlanningResponse,
    summary="Stage 35 Cohort Planning vs Actual Reconciled Readiness",
    description="Separates planned patient/image counts from verified actual counts and documents pending biostatistical sample-size justification.",
)
def get_cohort_planning(service: ResearchService = Depends(get_research_service)) -> CohortPlanningResponse:
    return service.get_cohort_planning()


@router.post(
    "/datasets/workflow/transition",
    summary="Stage 35 Audited Workflow Transition Validation",
    description="Validates reviewer role, pseudonymous ID, evidence reference, approval expiry, and transition sequence.",
)
def post_workflow_transition(
    request: WorkflowTransitionRequest,
    reviewer=Depends(get_authenticated_reviewer),
    service: ResearchService = Depends(get_research_service),
):
    try:
        return service.validate_workflow_transition(request, reviewer)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.post("/datasets/adjudication/decision", summary="Submit an authenticated clinical adjudication decision")
def post_adjudication_decision(
    request: ReviewerAdjudicationDecisionRequest,
    authorization: str = Header(default=""),
    reviewer=Depends(get_authenticated_clinical_reviewer),
):
    ledger_name = os.environ.get("OSTEO_ADJUDICATION_LEDGER_FILE", "").strip()
    if not ledger_name or len(os.environ.get("OSTEO_AUDIT_LEDGER_HMAC_KEY", "").encode("utf-8")) < 32:
        raise HTTPException(status_code=503, detail="Signed adjudication ledger is not configured.")
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    try:
        from ml_service.app.services.adjudication_engine import AdjudicationEngine
        engine = AdjudicationEngine(ledger_path=Path(ledger_name))
        result = engine.submit_reviewer_decision(
            sample_id=request.sample_id,
            decision=request.decision,
            reviewer_pseudo_id=reviewer["reviewer_pseudo_id"],
            reviewer_token=token,
            rationale=request.rationale,
            resolved_target_class=request.resolved_target_class,
            source_report_ref=request.source_report_ref,
            diagnostic_evidence_category=request.diagnostic_evidence_category,
            diagnostic_evidence=request.diagnostic_evidence,
            radiology_report_confirmation_status=request.radiology_report_confirmation_status,
            histopathology_confirmation_status=request.histopathology_confirmation_status,
            review_date=request.review_date,
        )
        engine.save_ledger()
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Sample is not present in the adjudication ledger.") from exc
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=403, detail="Reviewer is not authorized for this decision.") from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Signed adjudication ledger is unavailable.") from exc


@router.post(
    "/radiomics/demo",
    response_model=RadiomicsDemoResponse,
    summary="Radiomics Feature Extraction Demonstration",
    description="Extracts 102 radiomics features from synthetic digital phantoms (81 verified / 21 unverified).",
)
def run_radiomics_demo(
    request: RadiomicsDemoRequest,
    service: ResearchService = Depends(get_research_service),
) -> RadiomicsDemoResponse:
    return service.run_radiomics_demo(request)


@router.post(
    "/biomechanics/geometry",
    response_model=BiomechanicsGeometryResponse,
    summary="Cross-Sectional Biomechanics Geometry Demonstration",
    description="Computes second moments of area and polar moments on 2D synthetic bone phantoms.",
)
def run_biomechanics_geometry(
    request: BiomechanicsGeometryRequest,
    service: ResearchService = Depends(get_research_service),
) -> BiomechanicsGeometryResponse:
    return service.run_biomechanics_geometry(request)


@router.post(
    "/biomechanics/mirels",
    response_model=BiomechanicsMirelsResponse,
    summary="Conventional Mirels Fracture Risk Scoring",
    description="Computes classical 4-factor Mirels score and prophylactic fixation recommendations.",
)
def run_biomechanics_mirels(
    request: BiomechanicsMirelsRequest,
    service: ResearchService = Depends(get_research_service),
) -> BiomechanicsMirelsResponse:
    return service.run_biomechanics_mirels(request)


@router.post(
    "/explainability/demo",
    response_model=ExplainabilityDemoResponse,
    summary="Grad-CAM Visual Explainability Demonstration",
    description="Generates synthetic Grad-CAM heatmap on controlled spatial CNN fixture.",
)
def run_explainability_demo(
    request: ExplainabilityDemoRequest,
    service: ResearchService = Depends(get_research_service),
) -> ExplainabilityDemoResponse:
    return service.run_explainability_demo(request)


@router.post(
    "/uncertainty/demo",
    response_model=UncertaintyDemoResponse,
    summary="Predictive Uncertainty & Selective Prediction Demonstration",
    description="Evaluates predictive entropy, top-two margin, and tri-state triage (H_norm > 0.60 candidate threshold).",
)
def run_uncertainty_demo(
    request: UncertaintyDemoRequest,
    service: ResearchService = Depends(get_research_service),
) -> UncertaintyDemoResponse:
    return service.run_uncertainty_demo(request)


@router.post(
    "/workbench/run-demo",
    response_model=SyntheticWorkbenchRunResponse,
    summary="End-to-End Synthetic Research Workbench Execution",
    description="Executes a coordinated demonstration across all validated engineering modules.",
)
def run_synthetic_workbench(
    request: SyntheticWorkbenchRunRequest,
    service: ResearchService = Depends(get_research_service),
) -> SyntheticWorkbenchRunResponse:
    return service.run_synthetic_workbench(request)
