"""
OsteoGuard AI - ML Infrastructure API Endpoints (v1).
Exposes health, status, dataset readiness, and model readiness status.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ml_service.app.schemas.ml_status import (
    DatasetStatusResponse,
    MLHealthResponse,
    MLStatusResponse,
    ModelStatusResponse,
)
from ml_service.app.services.readiness_service import MLReadinessService
from ml_service.config.settings import MLSettings, get_settings
from ml_service.utils.logging import logger

router = APIRouter(prefix="/api", tags=["ML Operations"])


def get_readiness_service(settings: MLSettings = Depends(get_settings)) -> MLReadinessService:
    return MLReadinessService(settings=settings)


@router.get(
    "/ml/health",
    response_model=MLHealthResponse,
    summary="ML Service Health Check",
    description="Returns service uptime, runtime environment, version, and compute device telemetry.",
)
def get_ml_health(service: MLReadinessService = Depends(get_readiness_service)) -> MLHealthResponse:
    logger.debug("Received GET /api/ml/health request")
    return service.get_health_status()


@router.get(
    "/health",
    response_model=MLHealthResponse,
    summary="Legacy Compatibility Health Check",
    description="Frontend LiveEngine compatibility alias for /api/ml/health.",
    include_in_schema=True,
)
def get_legacy_health(service: MLReadinessService = Depends(get_readiness_service)) -> MLHealthResponse:
    return service.get_health_status()


@router.get(
    "/ml/status",
    response_model=MLStatusResponse,
    summary="ML Subsystem Overall Status",
    description="Returns aggregate state of ML pipeline, model lifecycle, and dataset readiness.",
)
def get_ml_status(service: MLReadinessService = Depends(get_readiness_service)) -> MLStatusResponse:
    logger.info("Evaluating aggregate ML system status")
    return service.get_overall_status()


@router.get(
    "/ml/datasets/status",
    response_model=DatasetStatusResponse,
    summary="Dataset Ingestion & Readiness Status",
    description="Audits clinical imaging dataset on disk, verifying patient counts and class balances.",
)
def get_dataset_status(service: MLReadinessService = Depends(get_readiness_service)) -> DatasetStatusResponse:
    logger.info("Auditing clinical dataset readiness on disk")
    return service.get_dataset_status()


@router.get(
    "/ml/models/status",
    response_model=ModelStatusResponse,
    summary="ML Model Lifecycle & Checkpoint Status",
    description="Reports the finite state machine status of the model, checkpoint verification, and inference eligibility.",
)
def get_model_status(service: MLReadinessService = Depends(get_readiness_service)) -> ModelStatusResponse:
    logger.info("Evaluating model readiness and checkpoint validity")
    return service.get_model_status()


@router.post(
    "/v1/analyze",
    summary="Differential Diagnostic Inference (Strictly Gated)",
    description="Performs inference. Fails with HTTP 503 if genuine trained weights are not loaded.",
)
async def analyze_image_gated(
    image: UploadFile = File(...),
    modality: str = Form("xray"),
    service: MLReadinessService = Depends(get_readiness_service),
):
    """
    Diagnostic analysis gate.
    Strict Rule 6 & 7 enforcement: Never returns fake predictions or phantom confidence percentages.
    """
    logger.warning("Inference requested on gated endpoint /api/v1/analyze")
    # Will raise ModelNotReadyException (HTTP 503) when in zero-dataset / zero-weights state
    service.ensure_inference_ready()

    # In subsequent stages (Stage 07+), inference will run here once weights are trained.
    return {"message": "Inference passed readiness gate"}
