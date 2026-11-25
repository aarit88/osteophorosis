"""
OsteoGuard AI - ML Infrastructure API Endpoints (v1).
Exposes health, status, dataset readiness, and model readiness status.
"""

import hashlib
import hmac
import io
import os
import time
from pathlib import Path
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
from ml_service.inference.fracatlas_inference import (
    FracAtlasFractureClassifier,
    TRACK_A_CLASSES,
    TRACK_A_OPERATING_THRESHOLDS,
    preprocess_fracatlas_bytes,
)
from ml_service.inference.four_class_inference import FourClassClinicalInferenceEngine
from ml_service.utils.logging import logger
from osteoguard.data.sanitizer import PatientPrivacySanitizer

router = APIRouter(prefix="/api", tags=["ML Operations"])

_INFERENCE_ENGINE: FourClassClinicalInferenceEngine | None = None
_TRACK_A_ENGINE: FracAtlasFractureClassifier | None = None
_TRACK_A_ENGINE_FINGERPRINT: tuple[str, int, int] | None = None


def _track_a_checkpoint(settings: MLSettings) -> tuple[Path | None, str | None]:
    """Resolve the configured preliminary checkpoint inside the artifact root only."""
    artifact_root = settings.resolve_path(settings.artifacts_dir).resolve()
    configured = settings.track_a_checkpoint_path
    if not configured.is_absolute():
        configured = artifact_root / configured
    try:
        if configured.is_symlink():
            return None, "TRACK_A_CHECKPOINT_PATH_INVALID"
        resolved = configured.resolve(strict=True)
        resolved.relative_to(artifact_root)
        if not resolved.is_file() or resolved.suffix.lower() not in {".pt", ".pth"}:
            return None, "TRACK_A_CHECKPOINT_PATH_INVALID"
        return resolved, None
    except FileNotFoundError:
        return None, "TRACK_A_CHECKPOINT_NOT_AVAILABLE"
    except (OSError, ValueError):
        return None, "TRACK_A_CHECKPOINT_PATH_INVALID"


def _load_track_a_engine(checkpoint: Path) -> FracAtlasFractureClassifier:
    """Load and cache the actual Track A model; never substitute demo weights."""
    global _TRACK_A_ENGINE, _TRACK_A_ENGINE_FINGERPRINT
    stat = checkpoint.stat()
    fingerprint = (str(checkpoint), stat.st_size, stat.st_mtime_ns)
    if _TRACK_A_ENGINE is None or _TRACK_A_ENGINE_FINGERPRINT != fingerprint:
        _TRACK_A_ENGINE = FracAtlasFractureClassifier(checkpoint_path=checkpoint)
        _TRACK_A_ENGINE_FINGERPRINT = fingerprint
    return _TRACK_A_ENGINE


def _predict_track_a_bytes(raw_bytes: bytes, operating_point: str, settings: MLSettings) -> dict:
    """Validate and run one in-memory Track A image; shared by the route and focused tests."""
    if len(raw_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"error_code": "IMAGE_TOO_LARGE", "detail": "Image exceeds the 15 MB limit."})
    if operating_point not in TRACK_A_OPERATING_THRESHOLDS:
        raise HTTPException(status_code=422, detail={"error_code": "INVALID_OPERATING_POINT", "detail": "Select a documented Track A operating point."})

    checkpoint, error_code = _track_a_checkpoint(settings)
    if checkpoint is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": error_code or "TRACK_A_CHECKPOINT_NOT_AVAILABLE",
                "detail": "The Track A checkpoint is missing or its configured path is not allowed. No prediction was generated.",
            },
        )

    try:
        image_tensor = preprocess_fracatlas_bytes(raw_bytes)
    except (OSError, ValueError, Image.UnidentifiedImageError):
        raise HTTPException(status_code=415, detail={"error_code": "INVALID_XRAY_IMAGE", "detail": "Upload a valid PNG, JPEG, or WebP radiograph image."})

    try:
        engine = _load_track_a_engine(checkpoint)
        result = engine.predict_tensor(image_tensor, threshold=TRACK_A_OPERATING_THRESHOLDS[operating_point])
    except Exception:
        logger.error("Track A inference failed while loading or running the configured checkpoint")
        raise HTTPException(
            status_code=503,
            detail={"error_code": "TRACK_A_CHECKPOINT_INVALID", "detail": "The configured Track A checkpoint could not be loaded. No prediction was generated."},
        )

    return {
        **result,
        "track_id": "track_a",
        "modality": "xray",
        "operating_point": operating_point,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "disclaimer": "Preliminary binary fracture research output only. Not a four-class differential diagnosis or clinical assessment. Per-image output does not measure accuracy.",
    }


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


@router.get(
    "/v1/preliminary/track-a/status",
    summary="Track A FracAtlas X-ray checkpoint availability",
)
def get_track_a_status(settings: MLSettings = Depends(get_settings)):
    checkpoint, error_code = _track_a_checkpoint(settings)
    return {
        "track_id": "track_a",
        "model_name": "FracAtlas ResNet-101 binary fracture research classifier",
        "modality": "xray",
        "input_size": [224, 224],
        "classes": [TRACK_A_CLASSES[index] for index in sorted(TRACK_A_CLASSES)],
        "checkpoint_available": checkpoint is not None,
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint is not None else None,
        "status": "CHECKPOINT_PRESENT_LOAD_ON_RUN" if checkpoint is not None else "CHECKPOINT_UNAVAILABLE",
        "error_code": error_code,
        "clinical_use": False,
    }


@router.post(
    "/v1/preliminary/track-a/analyze",
    summary="Run the existing Track A binary fracture research classifier",
)
async def analyze_track_a_radiograph(
    image: UploadFile = File(...),
    operating_point: str = Form("high_sensitivity_screening"),
    x_osteoguard_timestamp: str | None = Header(default=None),
    x_osteoguard_signature: str | None = Header(default=None),
    settings: MLSettings = Depends(get_settings),
):
    """Run actual Track A inference only; this endpoint is separate from the gated Track D API."""
    raw_bytes = await image.read(15 * 1024 * 1024 + 1)
    _verify_gateway_hmac(
        "POST",
        f"/api/v1/preliminary/track-a/analyze?operating_point={operating_point}",
        raw_bytes,
        x_osteoguard_timestamp,
        x_osteoguard_signature,
    )
    return _predict_track_a_bytes(raw_bytes, operating_point, settings)


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
