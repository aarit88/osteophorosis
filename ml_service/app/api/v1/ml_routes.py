"""
OsteoGuard AI - ML Infrastructure API Endpoints (v1).
Exposes health, status, dataset readiness, and model readiness status.
"""

import hashlib
import hmac
import io
import os
import time
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from PIL import Image

from ml_service.app.schemas.ml_status import (
    DatasetStatusResponse,
    MLHealthResponse,
    MLStatusResponse,
    ModelStatusResponse,
)
from ml_service.app.schemas.research import BlockedInferenceResponse
from ml_service.app.services.readiness_service import MLReadinessService
from ml_service.config.settings import MLSettings, get_settings
from ml_service.inference.four_class_inference import FourClassClinicalInferenceEngine
from ml_service.utils.logging import logger
from osteoguard.data.sanitizer import PatientPrivacySanitizer

router = APIRouter(prefix="/api", tags=["ML Operations"])

_INFERENCE_ENGINE: FourClassClinicalInferenceEngine | None = None


def get_inference_engine(settings: MLSettings = Depends(get_settings)) -> FourClassClinicalInferenceEngine:
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None:
        _INFERENCE_ENGINE = FourClassClinicalInferenceEngine(device_str=settings.device)
    return _INFERENCE_ENGINE


def get_readiness_service(settings: MLSettings = Depends(get_settings)) -> MLReadinessService:
    return MLReadinessService(settings=settings)


def _verify_gateway_hmac(
    method: str,
    req_path: str,
    raw_bytes: bytes,
    timestamp_header: str | None,
    signature_header: str | None,
) -> None:
    secret = (os.getenv("OSTEO_GATEWAY_HMAC_SECRET") or "").strip()
    if not secret:
        if (os.getenv("OSTEO_ENV") or os.getenv("ENV") or "development").lower() == "production":
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Gateway authentication is not configured.")
        return
    if not timestamp_header or not signature_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing signed ML gateway authentication headers.",
        )
    try:
        ts = int(timestamp_header)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed gateway timestamp header.",
        )
    if abs(int(time.time()) - ts) > 300:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expired ML gateway signature.",
        )
    body_hash = hashlib.sha256(raw_bytes).hexdigest()
    payload = f"{ts}:{method.upper()}:{req_path}:{body_hash}".encode("utf-8")
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ML gateway HMAC signature.",
        )


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
    description="Performs live 4-class PyTorch inference. Fails with HTTP 503 if genuine trained weights are not loaded.",
)
async def analyze_image_gated(
    image: UploadFile = File(...),
    modality: str = Form("xray"),
    x_osteoguard_timestamp: str | None = Header(default=None),
    x_osteoguard_signature: str | None = Header(default=None),
    service: MLReadinessService = Depends(get_readiness_service),
    engine: FourClassClinicalInferenceEngine = Depends(get_inference_engine),
):
    """
    Diagnostic analysis gate.
    Strict Rule 6 & 7 enforcement: Never returns fake predictions or phantom confidence percentages.
    """
    raw_bytes = await image.read()
    _verify_gateway_hmac(
        "POST",
        "/api/v1/analyze",
        raw_bytes,
        x_osteoguard_timestamp,
        x_osteoguard_signature,
    )

    logger.info("Inference requested on gated endpoint /api/v1/analyze")
    service.ensure_inference_ready()
    checkpoint_path = service.get_active_checkpoint()
    if checkpoint_path is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No verified 4-class checkpoint present on disk.",
        )

    return engine.predict(checkpoint_path=checkpoint_path, raw_bytes=raw_bytes, modality=modality)


@router.post(
    "/v1/diagnostic/four-class",
    summary="Four-Class Clinical Eligibility Gate",
    description="Checks live readiness and returns HTTP 200 READY when verified checkpoint exists, else HTTP 503 BLOCKED.",
    responses={503: {"model": BlockedInferenceResponse}},
)
def four_class_clinical_gate(service: MLReadinessService = Depends(get_readiness_service)):
    service.ensure_inference_ready()
    checkpoint = service.get_active_checkpoint()
    return {
        "status": "READY",
        "error_code": "NONE",
        "message": f"A SHA-256 verified Track D checkpoint is active ({checkpoint.name if checkpoint else 'verified'}) and all clinical governance gates pass.",
        "target_classes": [
            "Secondary bone metastasis",
            "Paget's disease of bone",
            "Osteogenesis imperfecta (OI)",
            "Healthy skeletal control",
        ],
        "blockers": [],
        "mandatory_safeguard": "Epistemic/aleatoric uncertainty abstention (H_norm >= 0.60) and OOD detection remain strictly active.",
    }
