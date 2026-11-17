"""
Tests for ML Status, Dataset Status, and Model Status Endpoints in Zero-Dataset State.
Strictly verifies that no phantom training or fake predictions are reported.
"""

from fastapi.testclient import TestClient


def test_get_ml_status_zero_dataset(client: TestClient):
    """Verifies GET /api/ml/status correctly reports AWAITING_DATA and zero samples."""
    response = client.get("/api/ml/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "awaiting_data"
    assert data["model_state"] == "AWAITING_DATA"
    assert data["dataset_state"] in ["EMPTY", "BLOCKED"]
    assert data["sample_count"] == 0
    assert data["patient_count"] == 0
    assert data["weights_loaded"] is False
    assert data["can_train"] is False
    assert data["can_infer"] is False
    assert "AWAITING_DATA" in data["message"]


def test_get_datasets_status_zero_dataset(client: TestClient):
    """Verifies GET /api/ml/datasets/status confirms zero samples on disk with blockers."""
    response = client.get("/api/ml/datasets/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ["EMPTY", "BLOCKED"]
    assert data["sample_count"] == 0
    assert data["patient_count"] == 0
    assert data["is_ready"] is False
    assert data["validation_passed"] is False
    assert len(data["blockers"]) > 0
    assert len(data["missing_classes"]) == 4
    assert set(data["missing_classes"]) == {"metastasis", "paget", "oi", "healthy"}


def test_get_models_status_zero_dataset(client: TestClient):
    """Verifies GET /api/ml/models/status reports uninitialized weights and inference blocked."""
    response = client.get("/api/ml/models/status")
    assert response.status_code == 200
    data = response.json()

    assert data["model_state"] == "AWAITING_DATA"
    assert data["active_backbone"] == "resnet101"
    assert data["weights_loaded"] is False
    assert data["checkpoint_path"] is None
    assert data["checkpoint_verified"] is False
    assert data["can_infer"] is False
    assert data["can_train"] is False
    assert data["training_history_available"] is False
    assert len(data["blockers"]) > 0
    assert data["architecture"]["radiomics_features"] == 102
    assert data["architecture"]["semantic_features"] == 512
    assert data["architecture"]["total_fusion_dim"] == 614
    assert data["architecture"]["num_classes"] == 4
