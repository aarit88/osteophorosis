"""Track A X-ray route contracts; fixtures are not trained model weights."""

from pathlib import Path

import pytest
from fastapi import HTTPException

from ml_service.app.api.v1.ml_routes import _predict_track_a_bytes, get_track_a_status
from ml_service.config.settings import MLSettings


def test_track_a_status_reports_checkpoint_absence_without_substituting_a_model(tmp_path):
    settings = MLSettings(
        base_dir=tmp_path,
        artifacts_dir=Path("ml_artifacts"),
        track_a_checkpoint_path=Path("fracatlas_full_training/best_checkpoint.pt"),
    )
    body = get_track_a_status(settings)
    assert body["track_id"] == "track_a"
    assert body["modality"] == "xray"
    assert body["checkpoint_available"] is False
    assert body["error_code"] == "TRACK_A_CHECKPOINT_NOT_AVAILABLE"
    assert body["clinical_use"] is False


def test_track_a_scan_returns_503_without_checkpoint_and_never_generates_prediction(tmp_path):
    settings = MLSettings(
        base_dir=tmp_path,
        artifacts_dir=Path("ml_artifacts"),
        track_a_checkpoint_path=Path("fracatlas_full_training/best_checkpoint.pt"),
    )
    with pytest.raises(HTTPException) as raised:
        _predict_track_a_bytes(b"synthetic test fixture only", "high_sensitivity_screening", settings)
    assert raised.value.status_code == 503
    body = raised.value.detail
    assert body["error_code"] == "TRACK_A_CHECKPOINT_NOT_AVAILABLE"
    assert "No prediction was generated" in body["detail"]


def test_track_a_checkpoint_must_resolve_inside_artifacts_root(tmp_path):
    outside = tmp_path / "outside-model.txt"
    outside.write_text("test path fixture; not model weights", encoding="utf-8")
    settings = MLSettings(
        base_dir=tmp_path,
        artifacts_dir=Path("ml_artifacts"),
        track_a_checkpoint_path=outside,
    )
    body = get_track_a_status(settings)
    assert body["checkpoint_available"] is False
    assert body["error_code"] == "TRACK_A_CHECKPOINT_PATH_INVALID"
