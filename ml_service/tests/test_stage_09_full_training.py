"""
Stage 09 Full-Scale Training Pipeline Verification Tests.
Verifies dataset partitions, class prior initialization, layer freezing,
Focal Loss integration, BatchNorm eval preservation, and test partition isolation.
"""

import csv
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ml_service.config.settings import get_settings
from ml_service.training.focal_loss import FocalLoss
from ml_service.training.fracatlas_full_trainer import (
    build_full_resnet101_model,
    set_train_mode_with_eval_bn,
    get_training_transforms,
    get_validation_transforms,
    FracAtlasFullDataset,
)


@pytest.fixture(scope="module")
def manifest_path():
    settings = get_settings()
    path = settings.base_dir / "ml_service/data/splits/fracatlas_classification_split.csv"
    assert path.exists(), f"Split manifest missing at {path}"
    return path


def test_dataset_partitions_exact_counts(manifest_path):
    """Verifies train=2857, val=612, test=614 and exact class balances."""
    counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["split"].lower()
            lbl = int(row["fractured"])
            counts[sp][lbl] += 1

    # Train
    assert counts["train"][1] == 502, f"Expected 502 train fractured, got {counts['train'][1]}"
    assert counts["train"][0] == 2355, f"Expected 2355 train non-fractured, got {counts['train'][0]}"
    assert (counts["train"][0] + counts["train"][1]) == 2857

    # Val
    assert counts["val"][1] == 107, f"Expected 107 val fractured, got {counts['val'][1]}"
    assert counts["val"][0] == 505, f"Expected 505 val non-fractured, got {counts['val'][0]}"
    assert (counts["val"][0] + counts["val"][1]) == 612

    # Test (Quarantined)
    assert (counts["test"][0] + counts["test"][1]) == 614


def test_class_prior_initialization():
    """Verifies class prior logit bias computation."""
    n_pos = 502
    n_neg = 2355
    n_total = 2857

    pi = n_pos / n_total
    expected_b1 = math.log(pi / (1.0 - pi))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, model_info = build_full_resnet101_model(device, pi_prior=pi)

    bias = model.fc.bias.detach().cpu()
    assert bias[0].item() == pytest.approx(0.0, abs=1e-6)
    assert bias[1].item() == pytest.approx(expected_b1, abs=1e-5)
    assert bias[1].item() < -1.54  # Should be approximately -1.545388


def test_model_initial_freezing():
    """Verifies initial layer freezing: conv1-layer3 frozen; layer4 and fc trainable."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = build_full_resnet101_model(device, pi_prior=0.1757)

    # Frozen
    for name, p in model.conv1.named_parameters():
        assert not p.requires_grad, f"conv1.{name} should be frozen"
    for name, p in model.bn1.named_parameters():
        assert not p.requires_grad, f"bn1.{name} should be frozen"
    for name, p in model.layer1.named_parameters():
        assert not p.requires_grad, f"layer1.{name} should be frozen"
    for name, p in model.layer2.named_parameters():
        assert not p.requires_grad, f"layer2.{name} should be frozen"
    for name, p in model.layer3.named_parameters():
        assert not p.requires_grad, f"layer3.{name} should be frozen initially"

    # Trainable
    for name, p in model.layer4.named_parameters():
        assert p.requires_grad, f"layer4.{name} should be trainable"
    for name, p in model.fc.named_parameters():
        assert p.requires_grad, f"fc.{name} should be trainable"


def test_batch_norm_eval_mode_retention():
    """Verifies that set_train_mode_with_eval_bn keeps all BatchNorm modules in eval mode."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = build_full_resnet101_model(device, pi_prior=0.1757)

    set_train_mode_with_eval_bn(model)

    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            assert not m.training, f"BatchNorm module {name} should be in eval mode!"


def test_focal_loss_forward_with_model():
    """Verifies that FocalLoss works with model outputs."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss(gamma=2.0, alpha=(0.25, 0.75), reduction="mean").to(device)

    logits = torch.randn(4, 2, device=device)
    targets = torch.tensor([0, 1, 0, 1], device=device)

    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0


def test_test_partition_isolation(manifest_path):
    """Verifies that test partition hashes and paths are strictly isolated."""
    splits = {"train": set(), "val": set(), "test": set()}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits[row["split"].lower()].add(row["sha256_hash"])

    # No hash overlap
    train_test_overlap = splits["train"].intersection(splits["test"])
    val_test_overlap = splits["val"].intersection(splits["test"])
    train_val_overlap = splits["train"].intersection(splits["val"])

    assert len(train_test_overlap) == 0, f"Data leakage: {len(train_test_overlap)} hashes overlap train & test"
    assert len(val_test_overlap) == 0, f"Data leakage: {len(val_test_overlap)} hashes overlap val & test"
    assert len(train_val_overlap) == 0, f"Data leakage: {len(train_val_overlap)} hashes overlap train & val"

