"""
Tests for Structured Logging and Global Exception Handlers.
"""

import json
import logging
from fastapi.testclient import TestClient

from ml_service.app.schemas.ml_status import ErrorDetailResponse
from ml_service.utils.exceptions import (
    AwaitingDataException,
    InvalidModalityException,
    OsteoGuardException,
)
from ml_service.utils.logging import JSONFormatter


def test_json_formatter_serializes_fields():
    """Verifies that JSONFormatter produces valid JSON with required metadata keys."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="osteoguard.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="Test message for structured logging",
        args=(),
        exc_info=None,
    )
    record.extra_data = {"patient_id": "PT999", "modality": "ct"}

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "osteoguard.test"
    assert parsed["message"] == "Test message for structured logging"
    assert parsed["data"]["patient_id"] == "PT999"
    assert parsed["data"]["modality"] == "ct"
    assert "timestamp" in parsed


def test_custom_exception_hierarchy():
    """Verifies that custom OsteoGuard exceptions hold correct codes and status codes."""
    exc1 = AwaitingDataException()
    assert exc1.error_code == "AWAITING_DATA"
    assert exc1.status_code == 400

    exc2 = InvalidModalityException("mri", ["xray", "ct", "bone_scan"])
    assert exc2.error_code == "INVALID_MODALITY"
    assert exc2.status_code == 400
    assert exc2.details["submitted_modality"] == "mri"


def test_validation_error_handler(client: TestClient):
    """Verifies that invalid requests return standard ErrorDetailResponse with 422."""
    # POST to analyze endpoint without the required 'image' file
    response = client.post("/api/v1/analyze", data={"modality": "xray"})
    assert response.status_code == 422
    data = response.json()

    assert data["error_code"] == "VALIDATION_ERROR"
    assert "errors" in data["details"]
    assert "timestamp" in data
