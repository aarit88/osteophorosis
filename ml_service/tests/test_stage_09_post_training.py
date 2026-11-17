"""
Stage 09 Post-Training Verification Tests.
Verifies best and last checkpoint reloading on CUDA, artifact completeness,
validation prediction consistency, quarantined test isolation,
and non-interference with CPU environment and frontend.
"""

import csv
import json
import math
from pathlib import Path
import pytest
import torch

from ml_service.config.settings import get_settings
from ml_service.training.fracatlas_full_trainer import build_full_resnet101_model


@pytest.fixture(scope="module")
def artifacts_dir():
    settings = get_settings()
    d = settings.base_dir / "ml_artifacts/fracatlas_full_training"
    assert d.exists(), f"Artifacts directory missing: {d}"
    return d


def test_artifacts_completeness(artifacts_dir):
    """Verifies all required Stage 09 artifacts exist and have non-zero size."""
    expected_files = [
        "training_config.yaml",
        "training_log.csv",
        "validation_metrics.json",
        "validation_predictions.csv",
        "threshold_analysis.csv",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "confusion_matrix.png",
        "loss_curves.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "training_summary.json",
    ]
    for filename in expected_files:
        p = artifacts_dir / filename
        assert p.exists(), f"Required artifact {filename} is missing"
        assert p.stat().st_size > 0, f"Artifact {filename} is empty"


def test_best_checkpoint_reload_on_cuda(artifacts_dir):
    """Verifies best_checkpoint.pt can be reloaded on CUDA and has valid keys."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ckpt_path = artifacts_dir / "best_checkpoint.pt"
    device = torch.device("cuda:0")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert "epoch" in ckpt
    assert "model_state_dict" in ckpt
    assert "val_pr_auc" in ckpt
    assert "val_roc_auc" in ckpt
    assert "config" in ckpt

    assert ckpt["epoch"] == 9
    assert ckpt["val_pr_auc"] == pytest.approx(0.7556, rel=1e-3)
    assert ckpt["val_roc_auc"] == pytest.approx(0.8871, rel=1e-3)

    # Instantiate model and load state_dict
    model, _ = build_full_resnet101_model(device, pi_prior=0.1757)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Test forward pass with dummy input
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 2)
    assert not torch.isnan(out).any()


def test_last_checkpoint_reload(artifacts_dir):
    """Verifies last_checkpoint.pt is reloadable and matches epoch 16."""
    ckpt_path = artifacts_dir / "last_checkpoint.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert ckpt["epoch"] == 16


def test_validation_predictions_integrity(artifacts_dir):
    """Verifies that validation_predictions.csv has exactly 612 records with valid probabilities."""
    val_preds_path = artifacts_dir / "validation_predictions.csv"
    records = []
    with open(val_preds_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    assert len(records) == 612
    n_fractured = sum(1 for r in records if int(r["true_label"]) == 1)
    n_non_fractured = sum(1 for r in records if int(r["true_label"]) == 0)
    assert n_fractured == 107
    assert n_non_fractured == 505

    for r in records:
        p_f = float(r["prob_fractured"])
        p_n = float(r["prob_non_fractured"])
        assert 0.0 <= p_f <= 1.0
        assert 0.0 <= p_n <= 1.0
        assert math.isclose(p_f + p_n, 1.0, abs_tol=1e-4)


def test_threshold_analysis_validity(artifacts_dir):
    """Verifies threshold_analysis.csv contains 91 rows (0.05 to 0.95)."""
    thresh_path = artifacts_dir / "threshold_analysis.csv"
    with open(thresh_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 91
    assert float(rows[0]["threshold"]) == 0.05
    assert float(rows[-1]["threshold"]) == 0.95

    # Check conservation of total samples: tp + fp + tn + fn == 612
    for r in rows:
        total = int(r["tp"]) + int(r["fp"]) + int(r["tn"]) + int(r["fn"])
        assert total == 612
        assert int(r["tp"]) + int(r["fn"]) == 107
        assert int(r["tn"]) + int(r["fp"]) == 505


def test_holdout_test_set_strictly_quarantined(artifacts_dir):
    """Verifies that none of the holdout test images appear in validation_predictions.csv."""
    settings = get_settings()
    manifest_path = settings.base_dir / "ml_service/data/splits/fracatlas_classification_split.csv"

    test_image_ids = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"].lower() == "test":
                test_image_ids.add(r["image_id"])

    assert len(test_image_ids) == 614

    val_preds_path = artifacts_dir / "validation_predictions.csv"
    val_image_ids = set()
    with open(val_preds_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val_image_ids.add(r["image_id"])

    overlap = test_image_ids.intersection(val_image_ids)
    assert len(overlap) == 0, f"Quarantine breach! Overlapping images: {overlap}"
