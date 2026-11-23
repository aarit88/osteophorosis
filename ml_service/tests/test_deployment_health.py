from fastapi.testclient import TestClient

from ml_service.app.main import create_app


def test_liveness_and_api_readiness_are_separate_from_clinical_readiness():
    with TestClient(create_app()) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        clinical = client.get("/health/model")

    assert live.status_code == 200
    assert live.json() == {"status": "ok", "service": "ml_api", "check": "liveness"}
    assert ready.status_code == 200
    assert ready.json()["check"] == "api_readiness"
    assert ready.json()["clinical_readiness_endpoint"] == "/api/ml/models/status"
    assert "path" not in ready.json()
    assert clinical.status_code == 503
    assert clinical.json()["status"] == "not_ready"
    assert clinical.json()["model_state"] == "AWAITING_DATA"
    assert clinical.json()["error_code"] == "CLINICAL_DATA_UNAVAILABLE"


def test_health_endpoints_do_not_expose_tracebacks_or_internal_filesystem_paths():
    with TestClient(create_app()) as client:
        for endpoint in ("/health/live", "/health/ready", "/health/model"):
            response = client.get(endpoint)
            body = response.text.lower()
            assert "traceback" not in body
            assert "c:\\users\\" not in body
            assert "/var/lib/" not in body
