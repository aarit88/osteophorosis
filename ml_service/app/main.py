"""
OsteoGuard AI - FastAPI Application Factory.
Provides life-cycle management, CORS configuration, structured exception handlers,
and registered route handlers for the ML subsystem.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from urllib.parse import urlsplit
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml_service.app.api.v1.ml_routes import router as ml_router
from ml_service.app.api.v1.research_routes import router as research_router, onboarding_router, preparation_router, qualification_router
from ml_service.app.schemas.ml_status import ErrorDetailResponse
from ml_service.config.settings import get_settings
from ml_service.utils.exceptions import OsteoGuardException
from ml_service.utils.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and graceful shutdown."""
    settings = get_settings()
    logger.info(
        f"Starting {settings.service_name} v{settings.service_version} "
        f"[env={settings.env}, backbone={settings.active_backbone}, device={settings.device}]"
    )
    yield
    logger.info(f"Shutting down {settings.service_name}")


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    settings = get_settings()
    if settings.env.lower() == "production":
        gateway_secret = os.environ.get("OSTEO_GATEWAY_HMAC_SECRET", "").strip()
        if len(gateway_secret) < 32:
            raise RuntimeError("Production requires OSTEO_GATEWAY_HMAC_SECRET of at least 32 characters.")
        if not os.environ.get("OSTEO_CORS_ORIGINS", "").strip():
            raise RuntimeError("Production requires explicitly configured OSTEO_CORS_ORIGINS.")
        if not settings.cors_origins or "*" in settings.cors_origins:
            raise RuntimeError("Production requires explicit CORS origins without wildcard access.")
        for origin in settings.cors_origins:
            parsed = urlsplit(origin)
            local = parsed.hostname in {"localhost", "127.0.0.1"}
            if (not parsed.hostname or parsed.path or parsed.query or parsed.fragment
                    or (parsed.scheme != "https" and not (local and parsed.scheme == "http"))):
                raise RuntimeError("Production CORS origins must be exact HTTPS origins (localhost may use HTTP).")
        ledger_key = os.environ.get("OSTEO_AUDIT_LEDGER_HMAC_KEY", "").strip()
        if len(ledger_key) < 32:
            raise RuntimeError("Production requires an externally supplied adjudication ledger HMAC key.")
        if not os.environ.get("OSTEO_ADJUDICATION_LEDGER_FILE", "").strip():
            raise RuntimeError("Production requires an adjudication ledger persistence path.")
        if len(os.environ.get("OSTEO_ONBOARDING_LEDGER_HMAC_KEY", "").strip()) < 32:
            raise RuntimeError("Production requires an external dataset onboarding ledger HMAC key.")
        if len(os.environ.get("OSTEO_SOURCE_QUALIFICATION_HMAC_KEY", "").strip()) < 32:
            raise RuntimeError("Production requires an external dataset source qualification HMAC key.")
        if not os.environ.get("OSTEO_SOURCE_QUALIFICATION_FILE", "").strip():
            raise RuntimeError("Production requires a dataset source qualification persistence path.")
        reviewer_registry = os.environ.get("OSTEO_REVIEWER_AUTHORIZATION_FILE", "").strip()
        if not reviewer_registry or not Path(reviewer_registry).is_file():
            raise RuntimeError("Production requires an institution-provisioned reviewer authorization registry.")

    application = FastAPI(
        title=settings.service_name,
        description=(
            "OsteoGuard AI — A Unified Deep Learning and Radiomic Framework for Differential "
            "Diagnosis (Metastasis, Paget's Disease, Osteogenesis Imperfecta, Healthy Bone) "
            "and Biomechanical Fracture Risk Stratification."
        ),
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-OsteoGuard-Timestamp", "X-OsteoGuard-Signature"],
    )

    # 2. Global Exception Handlers
    @application.exception_handler(OsteoGuardException)
    async def osteoguard_exception_handler(request: Request, exc: OsteoGuardException) -> JSONResponse:
        logger.warning(
            f"Handled OsteoGuardException [{exc.error_code}] on {request.url.path}: {exc.message}"
        )
        payload = ErrorDetailResponse(
            error_code=exc.error_code,
            detail=exc.message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=str(request.url.path),
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(f"Request validation failure on {request.url.path}: {exc.errors()}")
        payload = ErrorDetailResponse(
            error_code="VALIDATION_ERROR",
            detail="Request schema validation failed.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=str(request.url.path),
            details={"errors": exc.errors()},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.info(f"HTTPException [{exc.status_code}] on {request.url.path}: {exc.detail}")
        if isinstance(exc.detail, dict):
            detail_msg = exc.detail.get("message", "Service blocked or unavailable.")
            details_obj = exc.detail
            err_code = exc.detail.get("error_code", f"HTTP_{exc.status_code}")
        else:
            detail_msg = str(exc.detail)
            details_obj = {}
            err_code = f"HTTP_{exc.status_code}"

        payload = ErrorDetailResponse(
            error_code=err_code,
            detail=detail_msg,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=str(request.url.path),
            details=details_obj,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
        payload = ErrorDetailResponse(
            error_code="INTERNAL_SERVER_ERROR",
            detail="An unexpected internal server error occurred.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=str(request.url.path),
            details={"type": type(exc).__name__},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )

    # 3. Root & Health Endpoints
    @application.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.service_name,
            "version": settings.service_version,
            "status": "online",
            "docs": "/docs",
            "endpoints": {
                "health": "/api/ml/health",
                "api_health": "/api/ml/health",
                "status": "/api/ml/status",
                "datasets": "/api/ml/datasets/status",
                "models": "/api/ml/models/status",
                "research_status": "/api/v1/research/status",
                "research_tracks": "/api/v1/research/tracks",
            },
        }

    @application.get("/health", tags=["Root"])
    async def direct_health():
        from ml_service.app.services.readiness_service import MLReadinessService
        svc = MLReadinessService(settings=settings)
        return svc.get_health_status()

    @application.get("/health/live", tags=["Operations"])
    async def liveness():
        """Process liveness only; this deliberately does not imply model readiness."""
        return {"status": "ok", "service": "ml_api", "check": "liveness"}

    @application.get("/health/ready", tags=["Operations"])
    async def operational_readiness():
        """Readiness to serve API requests; clinical readiness is reported separately."""
        return {
            "status": "ready",
            "service": "ml_api",
            "check": "api_readiness",
            "clinical_readiness_endpoint": "/api/ml/models/status",
        }

    @application.get("/health/model", tags=["Operations"])
    async def clinical_model_readiness():
        """Clinical inference readiness; unlike process readiness, blocked models return 503."""
        from ml_service.app.services.readiness_service import MLReadinessService

        service = MLReadinessService(settings=settings)
        readiness = service.get_model_status()
        if not readiness.can_infer:
            error_code = "CLINICAL_MODEL_NOT_READY"
            configured_checkpoint = settings.active_checkpoint_path
            if configured_checkpoint is not None:
                checkpoint_path = settings.resolve_path(configured_checkpoint)
                error_code = (
                    "TRACK_D_CHECKPOINT_HASH_INVALID"
                    if checkpoint_path.is_file() and service.get_active_checkpoint() is None
                    else "TRACK_D_CHECKPOINT_MISSING"
                )
            elif readiness.model_state.value == "AWAITING_DATA":
                error_code = "CLINICAL_DATA_UNAVAILABLE"
            elif readiness.model_state.value == "DATA_VALIDATION_REQUIRED":
                error_code = "DATA_GOVERNANCE_INCOMPLETE"
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "model_state": readiness.model_state.value,
                         "error_code": error_code},
            )
        return {"status": "ready", "model_state": readiness.model_state.value}

    # 4. Include API Routers
    application.include_router(ml_router)
    application.include_router(research_router)
    application.include_router(onboarding_router)
    application.include_router(preparation_router)
    application.include_router(qualification_router)

    return application


app = create_app()
