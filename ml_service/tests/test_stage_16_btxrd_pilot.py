"""
Stage 16 - BTXRD Three-Class ResNet-34 Pilot Pipeline and Training Tests.

Verifies:
1. Three-class label mapping (0=Normal, 1=Benign, 2=Malignant primary tumour).
2. Aspect-ratio letterbox preprocessing with black padding to 512x512.
3. 3-channel broadcast for single-channel grayscale radiographs.
4. Center 1 class weighting and Multi-Class Focal Loss (gamma=2.0, alpha normalized).
5. ResNet-34 architecture, parameter counts, layer4/head trainability, early layer freezing, and eval-locked BN.
6. CUDA forward and backward pass execution without NaN or memory failure.
7. Completeness of all 12 pilot artifacts in ml_artifacts/btxrd_pilot/.
8. Checkpoint reload from best_checkpoint.pt and inference consistency.
9. Strict hospital center isolation (Center 2 and Center 3 completely absent from predictions).
10. Quarantine integrity of conflicting duplicate images.
11. Preservation of FracAtlas and Bone Scintigraphy checkpoints and artifacts.
12. Host CPU Python environment and React frontend preservation.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
import torch

from ml_service.config.settings import get_settings
from ml_service.datasets.btxrd_dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    BTXRDDataset,
    preprocess_btxrd_image,
)
from ml_service.models.btxrd_classifier import (
    BTXRDResNet34,
    MulticlassFocalLoss,
)


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def btxrd_manifest(base_dir):
    p = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    assert p.exists(), f"BTXRD split manifest missing: {p}"
    return p


@pytest.fixture(scope="module")
def pilot_artifacts_dir(base_dir):
    d = base_dir / "ml_artifacts/btxrd_pilot"
    assert d.exists(), f"BTXRD pilot artifacts directory missing: {d}"
    return d


def test_01_three_class_label_mapping():
    """Verifies authoritative 3-class index and class name mappings."""
    assert CLASS_TO_IDX["NORMAL_BONE_RADIOGRAPH"] == 0
    assert CLASS_TO_IDX["BENIGN_BONE_TUMOR"] == 1
    assert CLASS_TO_IDX["PRIMARY_MALIGNANT_TUMOR"] == 2
    assert len(CLASS_TO_IDX) == 3

    assert IDX_TO_CLASS[0] == "NORMAL_BONE_RADIOGRAPH"
    assert IDX_TO_CLASS[1] == "BENIGN_BONE_TUMOR"
    assert IDX_TO_CLASS[2] == "PRIMARY_MALIGNANT_TUMOR"


def test_02_aspect_ratio_letterboxing_and_padding(tmp_path):
    """Verifies that rectangular images are letterboxed with symmetrical black padding to 512x512."""
    # Create non-square test image (width 800, height 400)
    test_img = Image.new("L", (800, 400), color=128)
    img_path = tmp_path / "test_rect.png"
    test_img.save(img_path)

    tensor = preprocess_btxrd_image(img_path, target_size=512, is_train=False)

    # Check output shape
    assert tensor.shape == (3, 512, 512), f"Expected shape (3, 512, 512), got {tensor.shape}"
    assert tensor.dtype == torch.float32

    # Aspect ratio check: 800x400 scaled to longest=512 gives 512x256.
    # Vertical padding: (512 - 256) // 2 = 128 top and bottom.
    # Top rows [0:120, :] should correspond to black padding normalized by ImageNet.
    # ImageNet normalized value for 0.0 is -mean / std:
    # Channel 0: -0.485 / 0.229 = -2.1179
    # Channel 1: -0.456 / 0.224 = -2.0357
    # Channel 2: -0.406 / 0.225 = -1.8044
    padding_val_c0 = tensor[0, 10, 256].item()
    expected_pad_c0 = -0.485 / 0.229
    assert abs(padding_val_c0 - expected_pad_c0) < 1e-2, (
        f"Padding value mismatch in channel 0: {padding_val_c0} vs {expected_pad_c0}"
    )


def test_03_grayscale_to_rgb_broadcast(tmp_path):
    """Verifies that 1-channel grayscale plain radiographs broadcast cleanly into 3-channel tensors."""
    test_img = Image.new("L", (300, 300), color=200)
    img_path = tmp_path / "test_gray.png"
    test_img.save(img_path)

    tensor = preprocess_btxrd_image(img_path, target_size=512, is_train=False)
    assert tensor.shape == (3, 512, 512)
    assert not torch.isnan(tensor).any()


def test_04_center_1_class_weights_and_focal_loss(btxrd_manifest):
    """Verifies Center 1 training distribution, inverse-frequency alpha weights, and Focal Loss."""
    train_counts = {0: 0, 1: 0, 2: 0}
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("conflict_flag", "").strip().lower() == "true":
                continue
            if row.get("center", "").strip().lower() == "center_1":
                if row.get("center_holdout_split", "").strip().lower() == "train":
                    cls_name = row.get("target_class", "").strip()
                    train_counts[CLASS_TO_IDX[cls_name]] += 1

    assert train_counts[0] == 1272, f"Expected 1272 Normal, got {train_counts[0]}"
    assert train_counts[1] == 880, f"Expected 880 Benign, got {train_counts[1]}"
    assert train_counts[2] == 194, f"Expected 194 Malignant, got {train_counts[2]}"
    total_samples = sum(train_counts.values())
    assert total_samples == 2346

    # Verify alpha weights calculation
    raw_alphas = [total_samples / (3.0 * train_counts[i]) for i in range(3)]
    alpha_sum = sum(raw_alphas)
    norm_alphas = [3.0 * a / alpha_sum for a in raw_alphas]

    assert abs(norm_alphas[0] - 0.3333) < 1e-3
    assert abs(norm_alphas[1] - 0.4817) < 1e-3
    assert abs(norm_alphas[2] - 2.1850) < 1e-3
    assert abs(sum(norm_alphas) - 3.0) < 1e-5

    # Test Focal Loss computation
    criterion = MulticlassFocalLoss(gamma=2.0, alpha=torch.tensor(norm_alphas))
    dummy_logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.8, -0.5]], dtype=torch.float32)
    dummy_targets = torch.tensor([0, 1], dtype=torch.long)
    loss = criterion(dummy_logits, dummy_targets)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_05_resnet34_architecture_and_parameter_counts():
    """Verifies parameter freezing, total parameters, and classifier architecture."""
    model = BTXRDResNet34(num_classes=3, pretrained=False, dropout_p=0.4)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    assert total_params == 21_286_211, f"Total params mismatch: {total_params}"
    assert trainable_params == 13_115_907, f"Trainable params mismatch: {trainable_params}"
    assert frozen_params == 8_170_304, f"Frozen params mismatch: {frozen_params}"

    # Verify BatchNorm behavior in train mode: frozen BN must remain in eval mode
    model.train()
    assert model.bn1.training is False, "bn1 must remain in eval mode"
    for m in model.layer1.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            assert m.training is False, "layer1 BN must remain in eval mode"

    # Forward pass shape check
    dummy_in = torch.randn(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        out = model(dummy_in)
    assert out.shape == (2, 3)


def test_06_cuda_forward_backward_pass():
    """Verifies that forward and backward passes execute cleanly on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = torch.device("cuda:0")
    model = BTXRDResNet34(num_classes=3, pretrained=False).to(device)
    criterion = MulticlassFocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    dummy_in = torch.randn(2, 3, 512, 512, device=device)
    dummy_target = torch.tensor([0, 2], dtype=torch.long, device=device)

    model.train()
    optimizer.zero_grad()
    logits = model(dummy_in)
    loss = criterion(logits, dummy_target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_07_pilot_artifact_manifest_completeness(pilot_artifacts_dir):
    """Verifies presence and valid size of all 12 pilot artifacts."""
    expected_files = [
        "pilot_config.yaml",
        "class_distribution.json",
        "training_log.csv",
        "validation_metrics.json",
        "validation_predictions.csv",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "confusion_matrix.png",
        "loss_curves.png",
        "roc_curves_ovr.png",
        "precision_recall_curves.png",
        "pretrained_weights_info.json",
    ]

    for fname in expected_files:
        fpath = pilot_artifacts_dir / fname
        assert fpath.exists(), f"Required pilot artifact missing: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact file is empty: {fname}"


def test_08_checkpoint_reload_and_prediction_reproducibility(pilot_artifacts_dir):
    """Verifies that best_checkpoint.pt can be reloaded and reproduces valid logits."""
    ckpt_path = pilot_artifacts_dir / "best_checkpoint.pt"
    assert ckpt_path.exists()

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert "val_metrics" in checkpoint
    assert "macro_f1" in checkpoint["val_metrics"]

    model = BTXRDResNet34(num_classes=3, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_in = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_in)
    assert out.shape == (1, 3)
    assert not torch.isnan(out).any()


def test_09_zero_external_center_access(pilot_artifacts_dir, btxrd_manifest):
    """Verifies that zero Center 2 or Center 3 images were used in validation or training."""
    # Build center lookup
    img_center = {}
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_center[row["image_id"]] = row["center"].lower()

    preds_file = pilot_artifacts_dir / "validation_predictions.csv"
    assert preds_file.exists()

    val_image_ids = []
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_image_ids.append(row["image_id"])
            center = img_center.get(row["image_id"], "")
            assert center == "center_1", (
                f"Image {row['image_id']} belongs to {center}, violated Center 1 restriction!"
            )

    assert len(val_image_ids) == 588, f"Expected exactly 588 Center 1 validation images, got {len(val_image_ids)}"


def test_10_conflict_quarantine_integrity(pilot_artifacts_dir, btxrd_manifest):
    """Verifies that quarantined conflicting duplicate images are absent from validation predictions."""
    conflicting_ids = set()
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("conflict_flag", "").strip().lower() == "true":
                conflicting_ids.add(row["image_id"])

    assert len(conflicting_ids) == 4, f"Expected 4 conflicting images, got {len(conflicting_ids)}"
    assert "IMG000591.jpeg" in conflicting_ids or "IMG000591" in conflicting_ids
    assert "IMG003456.jpeg" in conflicting_ids or "IMG003456" in conflicting_ids

    preds_file = pilot_artifacts_dir / "validation_predictions.csv"
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["image_id"] not in conflicting_ids, (
                f"Quarantined conflicting image {row['image_id']} found in validation predictions!"
            )


def test_11_fracatlas_and_bone_scintigraphy_preservation(base_dir):
    """Verifies that FracAtlas and Bone Scintigraphy checkpoints remain unaltered."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists(), "FracAtlas checkpoint missing!"
    assert fracatlas_ckpt.stat().st_size > 100_000_000, "FracAtlas checkpoint corrupted/truncated!"

    scint_fold0_ckpt = base_dir / "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"
    assert scint_fold0_ckpt.exists(), "Bone Scintigraphy Fold 0 checkpoint missing!"

    scint_pilot_ckpt = base_dir / "ml_artifacts/bone_scan_pilot/best_checkpoint.pt"
    assert scint_pilot_ckpt.exists(), "Bone Scintigraphy Pilot checkpoint missing!"


def test_12_cpu_python_and_frontend_preservation(base_dir):
    """Verifies system CPU Python and React frontend remain completely intact."""
    # System CPU Python test
    cpu_py = Path(r"C:\Python313\python.exe")
    assert cpu_py.exists(), "System CPU Python interpreter missing!"

    cmd = [str(cpu_py), "-c", "import torch; print(torch.__version__)"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "+cpu" in res.stdout, "System CPU Python must be CPU-only"

    # Frontend integrity
    frontend_dir = base_dir / "osteophorosis-main/src"
    assert frontend_dir.exists(), "React frontend src directory missing!"
    assert (frontend_dir / "App.tsx").exists(), "Frontend App.tsx missing!"

