"""
OsteoGuard AI - Stage 08B FracAtlas ResNet-101 Pilot Test Suite.

Verifies:
  1. Stratified pilot subset sampling (1,024 images, 180 fractured, 844 non-fractured).
  2. Pretrained weights provenance (TorchVision IMAGENET1K_V2, SHA-256).
  3. Model layer freezing integrity (layer4 & fc trainable, conv1-layer3 frozen).
  4. Training logs and validation metrics consistency.
  5. Checkpoint reloading and CUDA inference integrity.
  6. Generated visual artifacts (loss curves, confusion matrix).
"""

import csv
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import yaml

from ml_service.config.settings import get_settings
from ml_service.models.resnet_classifier import build_resnet101_classifier


@pytest.fixture(scope="module")
def pilot_dir() -> Path:
    settings = get_settings()
    d = settings.base_dir / "ml_artifacts/fracatlas_pilot"
    assert d.exists(), f"Pilot directory missing: {d}"
    return d


def test_pilot_sample_manifest(pilot_dir: Path):
    """Verify pilot sample manifest exists, has 1,024 images, and preserves stratification."""
    sample_csv = pilot_dir / "pilot_sample_manifest.csv"
    assert sample_csv.exists(), "pilot_sample_manifest.csv missing"

    with open(sample_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    assert len(records) == 1024, f"Expected 1,024 records, found {len(records)}"

    n_frac = sum(1 for r in records if r["fractured"] == "1")
    n_non_frac = sum(1 for r in records if r["fractured"] == "0")
    assert n_frac == 180, f"Expected 180 fractured, found {n_frac}"
    assert n_non_frac == 844, f"Expected 844 non-fractured, found {n_non_frac}"

    # Verify no missing image paths
    for r in records:
        assert Path(r["file_path"]).exists(), f"Missing image: {r['file_path']}"


def test_pretrained_weights_info(pilot_dir: Path):
    """Verify pretrained weights metadata and architecture freezing."""
    info_json = pilot_dir / "pretrained_weights_info.json"
    assert info_json.exists(), "pretrained_weights_info.json missing"

    with open(info_json, "r", encoding="utf-8") as f:
        info = json.load(f)

    assert info["architecture"] == "ResNet101"
    assert "IMAGENET1K_V2" in info["weights_enum"]
    assert info["total_parameters"] == 42504258
    assert info["trainable_parameters"] == 14968834
    assert info["frozen_parameters"] == 27535424
    assert info["trainable_percentage"] == 35.22
    assert Path(info["local_cache_path"]).exists()
    assert len(info["cache_sha256"]) == 64


def test_training_log_and_progression(pilot_dir: Path):
    """Verify training log contains 3 epochs with valid loss progression and memory bounds."""
    log_csv = pilot_dir / "training_log.csv"
    assert log_csv.exists(), "training_log.csv missing"

    with open(log_csv, "r", encoding="utf-8") as f:
        logs = list(csv.DictReader(f))

    assert len(logs) == 3, f"Expected 3 epochs, found {len(logs)}"

    # Loss progression: train loss should decrease over epochs
    train_losses = [float(l["train_loss"]) for l in logs]
    val_losses = [float(l["val_loss"]) for l in logs]
    roc_aucs = [float(l["val_roc_auc"]) for l in logs]

    assert train_losses[2] < train_losses[0], "Train loss did not decrease across pilot epochs"
    assert val_losses[2] < val_losses[0], "Val loss did not decrease across pilot epochs"
    assert roc_aucs[2] > roc_aucs[0], "Validation ROC-AUC did not improve across pilot epochs"

    # Peak VRAM safety: must be comfortably below 4 GB VRAM budget (< 1500 MB)
    for l in logs:
        peak_mb = float(l["peak_vram_mb"])
        assert peak_mb < 1500.0, f"Peak VRAM {peak_mb} MiB exceeds safety ceiling"


def test_validation_metrics_summary(pilot_dir: Path):
    """Verify validation metrics JSON contains expected structure and strong AUC."""
    metrics_json = pilot_dir / "validation_metrics.json"
    assert metrics_json.exists(), "validation_metrics.json missing"

    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["evaluation_split"] == "validation"
    assert data["sample_count"] == 612
    assert data["best_epoch"] in [1, 2, 3]

    final_m = data["final_epoch_metrics"]
    assert final_m["roc_auc"] >= 0.70, f"Expected ROC-AUC >= 0.70, got {final_m['roc_auc']}"
    assert final_m["accuracy"] > 0.75
    assert final_m["confusion_matrix"]["true_negatives"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_reload_and_cuda_inference(pilot_dir: Path):
    """Verify best_checkpoint.pt can be loaded into fresh architecture and perform CUDA inference."""
    best_ckpt = pilot_dir / "best_checkpoint.pt"
    assert best_ckpt.exists(), "best_checkpoint.pt missing"
    assert best_ckpt.stat().st_size > 100 * (1024 ** 2), "Checkpoint unexpectedly small"

    device = torch.device("cuda:0")
    model = build_resnet101_classifier(pretrained=False, num_classes=2, in_channels=3)
    ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)

    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(dummy_input)
            probs = torch.softmax(logits, dim=1)

    assert logits.shape == (2, 2)
    assert probs.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=device), atol=1e-3)


def test_visual_artifacts_exist(pilot_dir: Path):
    """Verify confusion matrix and loss curve plots exist and are non-empty."""
    cm_png = pilot_dir / "confusion_matrix.png"
    loss_png = pilot_dir / "loss_curves.png"

    assert cm_png.exists(), "confusion_matrix.png missing"
    assert cm_png.stat().st_size > 10000, "confusion_matrix.png unexpectedly small"

    assert loss_png.exists(), "loss_curves.png missing"
    assert loss_png.stat().st_size > 10000, "loss_curves.png unexpectedly small"

