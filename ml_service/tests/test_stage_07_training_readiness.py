"""
OsteoGuard AI - Stage 07 Training Readiness Test Suite.

Verifies:
  1. FracAtlas Dataset & DataLoader operations (RGB conversion, 224x224 shape, label mapping).
  2. Transform distinction (train augmentation vs deterministic val/test).
  3. ResNet-101 binary model architecture and feature hook.
  4. 11-step training pipeline smoke test (forward, loss, backward, grads, optimizer update, checkpoint save/reload).
  5. Training YAML configuration completeness.
"""

from pathlib import Path
import pytest
import torch
import yaml

from ml_service.config.settings import get_settings
from ml_service.datasets.fracatlas_dataset import (
    FracAtlasClassificationDataset,
    create_fracatlas_dataloaders,
    get_fracatlas_transforms,
)
from ml_service.models.resnet_classifier import (
    ResNet101BinaryClassifier,
    build_resnet101_classifier,
)
from ml_service.training.smoke_test import run_training_smoke_test


@pytest.fixture(scope="module")
def manifest_path() -> Path:
    settings = get_settings()
    return settings.base_dir / "ml_service/data/splits/fracatlas_classification_split.csv"


def test_fracatlas_dataset_partitions(manifest_path: Path):
    """Verify Dataset initialization across train, val, and test splits."""
    ds_train = FracAtlasClassificationDataset(manifest_path=manifest_path, split="train")
    ds_val = FracAtlasClassificationDataset(manifest_path=manifest_path, split="val")
    ds_test = FracAtlasClassificationDataset(manifest_path=manifest_path, split="test")

    assert len(ds_train) == 2857, f"Expected 2857 train, got {len(ds_train)}"
    assert len(ds_val) == 612, f"Expected 612 val, got {len(ds_val)}"
    assert len(ds_test) == 614, f"Expected 614 test, got {len(ds_test)}"
    assert len(ds_train) + len(ds_val) + len(ds_test) == 4083

    # Check sample item
    img, lbl = ds_train[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(lbl, torch.Tensor)
    assert lbl.item() in (0, 1)

    # Check metadata retrieval
    ds_meta = FracAtlasClassificationDataset(
        manifest_path=manifest_path,
        split="train",
        return_metadata=True,
    )
    img, lbl, meta = ds_meta[0]
    assert "image_id" in meta
    assert "anatomic_region" in meta
    assert "sha256_hash" in meta


def test_fracatlas_transform_distinction():
    """Verify augmentations are restricted to train split."""
    tf_train = get_fracatlas_transforms("train")
    tf_val = get_fracatlas_transforms("val")
    tf_test = get_fracatlas_transforms("test")

    train_names = [t.__class__.__name__ for t in tf_train.transforms]
    val_names = [t.__class__.__name__ for t in tf_val.transforms]
    test_names = [t.__class__.__name__ for t in tf_test.transforms]

    assert "RandomHorizontalFlip" in train_names
    assert "RandomRotation" in train_names
    assert "RandomHorizontalFlip" not in val_names
    assert "RandomHorizontalFlip" not in test_names


def test_resnet101_binary_classifier_architecture():
    """Verify ResNet-101 binary model tensor propagation and feature hooks."""
    model = build_resnet101_classifier(pretrained=False, num_classes=2, in_channels=3)
    x = torch.randn(2, 3, 224, 224)

    # Forward features (Grad-CAM hook)
    feats = model.forward_features(x)
    assert feats.shape == (2, 2048, 7, 7)

    # Forward logits
    logits = model(x)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()

    # Probabilities
    probs = model.predict_proba(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_training_smoke_test_pipeline():
    """Verify full 11-step training smoke test."""
    results = run_training_smoke_test(batch_size=4)
    assert results["status"] == "PASS"

    steps = results["steps"]
    assert steps["1_train_batch_loaded"] is True
    assert steps["2_val_batch_loaded"] is True
    assert steps["5_finite_inputs_verified"] is True
    assert steps["8_backward_pass_verified"] is True
    assert steps["9_gradients_verified"]["parameters_with_grad"] > 0
    assert steps["10_optimizer_update_verified"]["fc_weight_difference_l1"] > 0.0
    assert steps["11_checkpoint_save_reload_verified"] is True


def test_training_yaml_configuration():
    """Verify training YAML configuration format and mandatory fields."""
    settings = get_settings()
    yaml_path = settings.base_dir / "ml_service/config/training_fracatlas_binary.yaml"
    assert yaml_path.exists(), f"Configuration file missing: {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Mandatory fields
    assert "model" in cfg
    assert cfg["model"]["architecture"] == "resnet101"
    assert cfg["model"]["num_classes"] == 2

    assert "dataset" in cfg
    assert cfg["dataset"]["name"] == "fracatlas"
    assert cfg["dataset"]["input_dimensions"]["height"] == 224
    assert cfg["dataset"]["input_dimensions"]["width"] == 224

    assert "dataloader" in cfg
    assert cfg["dataloader"]["batch_size"] > 0
    assert cfg["dataloader"]["num_workers"] == 0  # Windows safe

    assert "training" in cfg
    assert cfg["training"]["epochs"] > 0
    assert cfg["training"]["optimizer"]["name"] == "AdamW"
    assert cfg["training"]["optimizer"]["learning_rate"] > 0
    assert cfg["training"]["loss"]["name"] == "CrossEntropyLoss"
    assert cfg["training"]["random_seed"] == 42
    assert cfg["training"]["early_stopping"]["enabled"] is True

    assert "checkpoints" in cfg
    assert "checkpoint_dir" in cfg["checkpoints"]

    assert "hardware" in cfg
    assert "device" in cfg["hardware"]

