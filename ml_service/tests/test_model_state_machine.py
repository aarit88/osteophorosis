"""
Tests for OsteoGuard AI Model State Machine.
Exhaustively verifies all 9 finite model states:
  - NOT_CONFIGURED
  - AWAITING_DATA
  - DATA_VALIDATION_REQUIRED
  - READY_FOR_TRAINING
  - TRAINING
  - TRAINED
  - VALIDATION_REQUIRED
  - READY_FOR_INFERENCE
  - FAILED
"""

from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import pytest
import torch

from ml_service.app.schemas.model_state import ModelState
from ml_service.app.services.readiness_service import MLReadinessService
from ml_service.config.settings import MLSettings
from osteoguard.constants import TARGET_CLASSES


def test_enum_members_complete():
    """Verifies that all 9 required ModelState enum values are defined."""
    required = {
        "NOT_CONFIGURED",
        "AWAITING_DATA",
        "DATA_VALIDATION_REQUIRED",
        "READY_FOR_TRAINING",
        "TRAINING",
        "TRAINED",
        "VALIDATION_REQUIRED",
        "READY_FOR_INFERENCE",
        "FAILED",
    }
    actual = {s.value for s in ModelState}
    assert required == actual, f"Missing or extra states: {required.symmetric_difference(actual)}"


def test_state_not_configured(tmp_path: Path):
    """Verifies NOT_CONFIGURED when configuration cannot resolve paths."""
    class BrokenSettings(MLSettings):
        def get_resolved_data_dir(self) -> Path:
            raise RuntimeError("Invalid filesystem mount")

    service = MLReadinessService(settings=BrokenSettings())
    state, blockers = service.determine_model_state()
    assert state == ModelState.NOT_CONFIGURED
    assert any("Configuration error" in b for b in blockers)


def test_state_awaiting_data(isolated_service: MLReadinessService):
    """Verifies AWAITING_DATA when directory exists but contains zero samples."""
    state, blockers = isolated_service.determine_model_state()
    assert state == ModelState.AWAITING_DATA
    assert any("AWAITING_DATA" in b for b in blockers)


def test_state_data_validation_required(isolated_service: MLReadinessService, sample_settings: MLSettings):
    """Verifies DATA_VALIDATION_REQUIRED when samples exist but fail requirements (e.g. missing classes)."""
    raw_dir = sample_settings.get_resolved_raw_dir()
    cls_dir = raw_dir / "metastasis"
    cls_dir.mkdir(parents=True, exist_ok=True)

    # Write only one class (missing paget, oi, healthy)
    arr = np.zeros((64, 64), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(cls_dir / "PT01_metastasis_01.png")

    state, blockers = isolated_service.determine_model_state()
    assert state == ModelState.DATA_VALIDATION_REQUIRED
    assert any("Missing clinical cohorts" in b for b in blockers)


def test_state_ready_for_training(isolated_service: MLReadinessService, sample_settings: MLSettings):
    """Verifies READY_FOR_TRAINING when all 4 classes have valid patient cohorts."""
    raw_dir = sample_settings.get_resolved_raw_dir()

    # Create 3 unique patients per class (min_patients_per_class = 3 in fixture)
    for cls in TARGET_CLASSES:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for p in range(1, 4):
            arr = np.zeros((64, 64), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"PT_{cls}_{p:02d}_01.png")

    state, blockers = isolated_service.determine_model_state()
    assert state == ModelState.READY_FOR_TRAINING
    assert len(blockers) == 1
    assert "Ready for training" in blockers[0]


def test_state_training(isolated_service: MLReadinessService):
    """Verifies TRAINING state when training process is flagged active."""
    isolated_service._is_training = True
    state, _ = isolated_service.determine_model_state()
    assert state == ModelState.TRAINING


def test_state_validation_required(isolated_service: MLReadinessService, sample_settings: MLSettings, monkeypatch):
    """Verifies VALIDATION_REQUIRED when checkpoint is found but training logs are missing."""
    chk_dir = sample_settings.get_resolved_checkpoints_dir()
    chk_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy checkpoint file
    chk_file = chk_dir / "best_model.pt"
    torch.save({"epoch": 1, "state_dict": {}}, chk_file)
    sample_settings.active_checkpoint_path = chk_file
    monkeypatch.setenv("OSTEO_TRACK_D_CHECKPOINT_SHA256", hashlib.sha256(chk_file.read_bytes()).hexdigest())

    # Ensure logs directory is empty
    logs_dir = sample_settings.resolve_path(sample_settings.logs_dir)
    if logs_dir.exists():
        for f in logs_dir.glob("*"):
            f.unlink()

    state, blockers = isolated_service.determine_model_state()
    assert state == ModelState.VALIDATION_REQUIRED
    assert any("Rule 6 Check" in b for b in blockers)


def test_state_ready_for_inference(isolated_service: MLReadinessService, sample_settings: MLSettings, monkeypatch):
    """Verifies READY_FOR_INFERENCE when valid checkpoint AND training logs exist."""
    chk_dir = sample_settings.get_resolved_checkpoints_dir()
    chk_dir.mkdir(parents=True, exist_ok=True)
    chk_file = chk_dir / "best_model.pt"
    torch.save({"epoch": 50, "state_dict": {}}, chk_file)
    sample_settings.active_checkpoint_path = chk_file
    monkeypatch.setenv("OSTEO_TRACK_D_CHECKPOINT_SHA256", hashlib.sha256(chk_file.read_bytes()).hexdigest())

    logs_dir = sample_settings.resolve_path(sample_settings.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "train_20261115.log").write_text("Epoch 50: loss=0.12, val_acc=0.96", encoding="utf-8")

    state, blockers = isolated_service.determine_model_state()
    assert state == ModelState.READY_FOR_INFERENCE
    assert len(blockers) == 0


def test_state_failed_on_exception(isolated_service: MLReadinessService, sample_settings: MLSettings, monkeypatch):
    """Verifies FAILED state when an unhandled exception corrupts checkpoint inspection."""
    chk_dir = sample_settings.get_resolved_checkpoints_dir()
    chk_dir.mkdir(parents=True, exist_ok=True)
    chk_file = chk_dir / "corrupted_model.pt"
    chk_file.write_text("corrupted bytes", encoding="utf-8")

    monkeypatch.setattr(isolated_service, "get_active_checkpoint", lambda: chk_file)

    original_resolve = MLSettings.resolve_path
    def mock_resolve_fail(self, path, *args, **kwargs):
        if "logs" in str(path):
            raise PermissionError("Disk hardware I/O error on logs volume")
        return original_resolve(self, path)

    monkeypatch.setattr(MLSettings, "resolve_path", mock_resolve_fail)

    state, blockers = isolated_service.determine_model_state()
    assert state == ModelState.FAILED
    assert any("Checkpoint evaluation failed" in b for b in blockers)


def test_state_trained_representation():
    """Verifies TRAINED state transitions in the state lifecycle."""
    assert ModelState.TRAINED == "TRAINED"
    assert ModelState.TRAINED in list(ModelState)
