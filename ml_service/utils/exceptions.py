"""
OsteoGuard AI - Domain Exception Hierarchy.
Strict exception classes mapped to standardized HTTP status codes and error responses.
"""

from typing import Any, Dict, Optional


class OsteoGuardException(Exception):
    """Base domain exception for OsteoGuard ML subsystem."""
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ML_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ModelNotConfiguredException(OsteoGuardException):
    """Raised when ML model architecture or runtime paths are misconfigured."""
    def __init__(self, message: str = "Model architecture is not configured.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MODEL_NOT_CONFIGURED",
            status_code=503,
            details=details,
        )


class AwaitingDataException(OsteoGuardException):
    """Raised when operation requires clinical dataset cohorts that have not yet been downloaded."""
    def __init__(self, message: str = "No clinical imaging datasets are present on disk.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AWAITING_DATA",
            status_code=400,
            details=details,
        )


class DatasetNotReadyException(OsteoGuardException):
    """Raised when dataset exists but fails validation criteria or patient-level stratification."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATASET_VALIDATION_FAILED",
            status_code=422,
            details=details,
        )


class ModelNotReadyException(OsteoGuardException):
    """
    Raised when model inference is attempted but no authentic, verified weights exist.
    Strictly prevents fake predictions in compliance with Rules 6 and 7.
    """
    def __init__(
        self,
        message: str = "Model inference is unavailable. No authenticated model checkpoint exists. Strict Rule 6 prohibits phantom predictions.",
        current_state: str = "AWAITING_DATA",
        details: Optional[Dict[str, Any]] = None,
    ):
        merged_details = details or {}
        merged_details["current_state"] = current_state
        super().__init__(
            message=message,
            error_code="FOUR_CLASS_MODEL_NOT_READY",
            status_code=503,
            details=merged_details,
        )


class InvalidModalityException(OsteoGuardException):
    """Raised when an unapproved imaging modality is submitted."""
    def __init__(self, modality: str, allowed: list):
        super().__init__(
            message=f"Modality '{modality}' is not supported. Permitted modalities: {allowed}",
            error_code="INVALID_MODALITY",
            status_code=400,
            details={"submitted_modality": modality, "allowed_modalities": allowed},
        )


class CheckpointNotFoundException(OsteoGuardException):
    """Raised when specified checkpoint file does not exist on disk."""
    def __init__(self, checkpoint_path: str):
        super().__init__(
            message=f"Model checkpoint not found at: {checkpoint_path}",
            error_code="CHECKPOINT_NOT_FOUND",
            status_code=404,
            details={"checkpoint_path": checkpoint_path},
        )
