"""
Tests for OsteoGuard ML Health and Base Endpoints.
"""

from fastapi.testclient import TestClient


def test_get_ml_health(client: TestClient):
    """Verifies GET /api/ml/health contract and attributes."""
    response = client.get("/api/ml/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "OsteoGuard AI ML Service"
    assert "version" in data
    assert "timestamp" in data
    assert "environment" in data
    assert "device" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0


def test_get_legacy_health_alias(client: TestClient):
    """Verifies GET /api/health compatibility alias for React LiveEngine."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "service" in data


def test_get_root_endpoint(client: TestClient):
    """Verifies root GET / returns metadata and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "online"
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/api/ml/health"
    assert data["endpoints"]["status"] == "/api/ml/status"
    assert data["endpoints"]["datasets"] == "/api/ml/datasets/status"
    assert data["endpoints"]["models"] == "/api/ml/models/status"
