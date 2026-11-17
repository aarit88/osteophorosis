"""
Unit and API integration tests for FastAPI Endpoints and OpenAPI Pydantic Schemas.
Verifies contract compatibility with the frontend TypeScript interfaces.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from osteoguard.api.app import app
from osteoguard.api.schemas import AnalyzeResponse, HealthResponse
from osteoguard.constants import TARGET_CLASSES


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /api/health endpoint structure."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] in ("production", "simulation")

    # Validate against Pydantic schema
    validated = HealthResponse(**data)
    assert validated.status == "ok"


def test_analyze_endpoint_contract(client):
    """Verify /api/v1/analyze returns valid AnalyzeResponse conforming to OpenAPI spec."""
    # Create test in-memory image
    img = Image.new("L", (128, 128), color=150)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("test_femur.png", buf, "image/png")}
    data = {"modality": "xray"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()

    # Schema validation
    validated = AnalyzeResponse(**res_json)
    assert validated.requestId.startswith("req_")
    assert validated.modality == "xray"
    assert validated.prediction.topClass in TARGET_CLASSES
    assert 0.0 <= validated.prediction.confidence <= 1.0

    # Probabilities
    probs = validated.prediction.probabilities
    total_p = probs.metastasis + probs.paget + probs.oi + probs.healthy
    assert abs(total_p - 1.0) < 0.05

    # Biomechanical fracture risk
    assert 0.0 <= validated.fractureRisk.percent <= 100.0
    assert validated.fractureRisk.level in ("low", "medium", "high")

    # Explainability
    assert len(validated.shap) == 6
    assert len(validated.counterfactuals) >= 1
    assert "investigational" in validated.disclaimer.lower()

