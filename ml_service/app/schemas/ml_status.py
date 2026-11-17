"""
OsteoGuard AI - API Request & Response Schemas for ML Infrastructure.
Provides typed data contracts for health, status, dataset readiness, and model readiness.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from ml_service.app.schemas.model_state import ModelState


class MLHealthResponse(BaseModel):
    """Payload for GET /api/ml/health and GET /api/health."""
    model_config = ConfigDict(extra="ignore")

    status: Literal["ok", "error", "degraded"] = Field(..., description="Operational status of service")
    service: str = Field(..., description="Service identifier name")
    version: str = Field(..., description="Semantic version string")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of request response")
    environment: str = Field(default="development", description="Runtime environment: development | staging | production")
    device: str = Field(default="cpu", description="Compute device detected by PyTorch")
    uptime_seconds: float = Field(default=0.0, description="Service process uptime in seconds")


class DatasetStatusResponse(BaseModel):
    """Payload for GET /api/ml/datasets/status."""
    model_config = ConfigDict(extra="ignore")

    status: Literal["EMPTY", "INVALID", "READY", "BLOCKED"] = Field(..., description="Cohort availability summary")
    dataset_path: str = Field(..., description="Target dataset root directory audited")
    sample_count: int = Field(default=0, ge=0, description="Total verified clinical image samples detected")
    patient_count: int = Field(default=0, ge=0, description="Total verified unique patient identifiers detected")
    classes_present: Dict[str, int] = Field(default_factory=dict, description="Image count per target pathology class")
    missing_classes: List[str] = Field(default_factory=list, description="Mandatory classes with zero samples")
    is_ready: bool = Field(..., description="True only if verified clinical data clears all pre-training criteria")
    validation_passed: bool = Field(default=False, description="True if patient-level validation report passed")
    blockers: List[str] = Field(default_factory=list, description="List of concrete reasons preventing training clearance")
    action_required: List[str] = Field(default_factory=list, description="Prescribed steps to unblock dataset readiness")


class ModelArchitectureInfo(BaseModel):
    """Structural metadata about the active ML model."""
    model_config = ConfigDict(extra="ignore")

    backbone: str = Field(..., description="Deep semantic backbone (resnet101 | swin_transformer)")
    radiomics_features: int = Field(default=102, description="IBSI radiomic descriptor vector size")
    semantic_features: int = Field(default=512, description="Projected deep visual representation size")
    total_fusion_dim: int = Field(default=614, description="Combined late feature fusion dimension (512 + 102)")
    num_classes: int = Field(default=4, description="Target differential classes count")
    target_classes: List[str] = Field(
        default_factory=lambda: ["metastasis", "paget", "oi", "healthy"],
        description="Target pathology classifications"
    )


class ModelStatusResponse(BaseModel):
    """Payload for GET /api/ml/models/status."""
    model_config = ConfigDict(extra="ignore")

    model_state: ModelState = Field(..., description="Active model lifecycle state from finite state machine")
    active_backbone: str = Field(..., description="Active convolutional or transformer backbone")
    weights_loaded: bool = Field(..., description="Whether authenticated model weights are loaded into memory")
    checkpoint_path: Optional[str] = Field(None, description="Absolute or relative path to active weights checkpoint")
    checkpoint_verified: bool = Field(default=False, description="Whether checkpoint has passed cryptographic/integrity checks")
    architecture: ModelArchitectureInfo = Field(..., description="Architectural specifications")
    can_infer: bool = Field(..., description="True strictly when authentic weights exist and passed validation")
    can_train: bool = Field(..., description="True strictly when dataset readiness gate is cleared")
    training_history_available: bool = Field(default=False, description="Whether authentic training logs exist on disk")
    message: str = Field(..., description="Diagnostic assessment of model status")
    blockers: List[str] = Field(default_factory=list, description="Blockers preventing inference or training")


class MLStatusResponse(BaseModel):
    """Payload for GET /api/ml/status."""
    model_config = ConfigDict(extra="ignore")

    status: Literal["operational", "awaiting_data", "unconfigured", "training", "error"] = Field(..., description="High-level subsystem status")
    service: str = Field(..., description="Service title")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    model_state: ModelState = Field(..., description="Active finite state of the model")
    dataset_state: Literal["EMPTY", "INVALID", "READY", "BLOCKED"] = Field(..., description="Dataset status summary")
    active_backbone: str = Field(..., description="Active deep learning backbone")
    device: str = Field(..., description="Target PyTorch device")
    sample_count: int = Field(default=0, ge=0, description="Total verified clinical images available")
    patient_count: int = Field(default=0, ge=0, description="Total verified unique patients available")
    weights_loaded: bool = Field(default=False, description="Weights loaded into runtime")
    can_train: bool = Field(default=False, description="Whether training can be started")
    can_infer: bool = Field(default=False, description="Whether inference is permitted")
    message: str = Field(..., description="Comprehensive status narrative")


class ErrorDetailResponse(BaseModel):
    """Structured error payload returned on API exceptions."""
    model_config = ConfigDict(extra="ignore")

    error_code: str = Field(..., description="Standardized application error identifier")
    detail: str = Field(..., description="Human-readable description of error")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    path: Optional[str] = Field(None, description="Request URL path where exception occurred")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Diagnostic contextual metadata")
