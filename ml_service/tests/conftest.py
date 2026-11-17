"""
Pytest configuration and fixtures for ml_service test suite.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from ml_service.app.main import create_app
from ml_service.app.services.readiness_service import MLReadinessService
from ml_service.config.settings import MLSettings


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_settings(tmp_path: Path) -> MLSettings:
    """Returns an isolated MLSettings instance with temporary directories."""
    return MLSettings(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data" / "raw",
        interim_data_dir=tmp_path / "data" / "interim",
        processed_data_dir=tmp_path / "data" / "processed",
        splits_data_dir=tmp_path / "data" / "splits",
        artifacts_dir=tmp_path / "artifacts",
        checkpoints_dir=tmp_path / "artifacts" / "checkpoints",
        metrics_dir=tmp_path / "artifacts" / "metrics",
        logs_dir=tmp_path / "artifacts" / "logs",
        stage_reports_dir=tmp_path / "artifacts" / "stage_reports",
        min_patients_per_class=3,
    )


@pytest.fixture
def isolated_service(sample_settings: MLSettings) -> MLReadinessService:
    """Returns an isolated MLReadinessService instance operating on tmp_path."""
    return MLReadinessService(settings=sample_settings)
