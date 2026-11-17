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

from fastapi import APIRouter, Depends, HTTPException, status

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
    BlockedInferenceResponse,
)
from ml_service.app.services.research_service import ResearchService

router = APIRouter(prefix="/api/v1", tags=["Research Operations & Prototype Workbench"])

# Dependency singleton
_service_instance = None


def get_research_service() -> ResearchService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ResearchService()
    return _service_instance


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
        "frozen_checkpoint_status": "LOCKED (SHA-256 verified)",
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


@router.post(
    "/diagnostic/four-class",
    response_model=BlockedInferenceResponse,
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    summary="Four-Class Diagnostic Inference (Strictly Blocked)",
    description="Diagnostic prediction endpoint. Fails with HTTP 503 explaining missing clinical datasets.",
)
def diagnostic_four_class_blocked(
    service: ResearchService = Depends(get_research_service),
) -> BlockedInferenceResponse:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=service.get_blocked_inference().model_dump(),
    )

