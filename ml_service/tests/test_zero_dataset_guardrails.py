"""
Tests for Strict Zero-Dataset Medical AI Guardrails.
Enforces:
  - Rule 4: Never invent dataset labels or patient records.
  - Rule 5: Never use generated synthetic images as real training data.
  - Rule 6: Never claim that a model has been trained unless actual training logs exist.
  - Rule 7: Never fabricate evaluation metrics or fake predictions.
"""

import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ml_service.app.services.readiness_service import MLReadinessService
from ml_service.utils.exceptions import ModelNotReadyException


def test_inference_endpoint_blocked_in_zero_dataset_state(client: TestClient):
    """
    Verifies that calling POST /api/v1/analyze without trained weights fails with HTTP 503
    and returns a structured MODEL_NOT_READY error instead of a fake prediction.
    """
    # Create test image in-memory
    img = Image.new("L", (128, 128), color=120)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    response = client.post(
        "/api/v1/analyze",
        files={"image": ("xray.png", img_byte_arr, "image/png")},
        data={"modality": "xray"},
    )

    assert response.status_code == 503
    data = response.json()

    assert data["error_code"] == "MODEL_NOT_READY"
    assert "Rule 6 & 7 violation avoided" in data["detail"] or "AWAITING_DATA" in data["detail"]
    assert "details" in data
    assert "current_state" in data["details"]
    assert data["details"]["current_state"] == "AWAITING_DATA"

    # Confirm NO fake prediction fields exist in response
    assert "prediction" not in data
    assert "confidence" not in data
    assert "probabilities" not in data


def test_ensure_inference_ready_raises_exception():
    """Verifies that MLReadinessService.ensure_inference_ready() raises ModelNotReadyException."""
    service = MLReadinessService()
    with pytest.raises(ModelNotReadyException) as exc_info:
        service.ensure_inference_ready()

    assert exc_info.value.error_code == "MODEL_NOT_READY"
    assert exc_info.value.status_code == 503
    assert exc_info.value.details["current_state"] == "AWAITING_DATA"
