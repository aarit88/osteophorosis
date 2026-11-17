"""
Stage 17 - Full-Scale BTXRD Three-Class ResNet-34 Training Tests.

Verifies:
1. Three-class label mapping (0=Normal, 1=Benign, 2=Malignant primary bone tumour).
2. Data loader integrity (Center 1 clean: 2,346 train, 588 val).
3. Aspect-ratio letterbox preprocessing with black padding to 512x512.
4. Radiograph label consistency and valid ranges.
5. Multi-Class Focal Loss formulation and numerical stability.
6. Center 1 class weighting strictly computed from training partition.
7. Model output shape (B, 3) and progressive unfreezing readiness.
8. CUDA forward and backward pass execution without memory failure or NaN.
9. Checkpoint reload from best_checkpoint.pt into BTXRDResNet34.
10. Completeness of all 12 full-training artifacts in ml_artifacts/btxrd_full_training/.
11. Strict hospital center isolation (Center 2 and Center 3 absent from predictions).
12. Conflicting duplicate quarantine enforcement (4 conflicting images absent).
13. Preservation of FracAtlas checkpoints and artifacts.
14. Preservation of Bone Scintigraphy checkpoints and artifacts.
15. Preservation of Stage 16 pilot artifacts, host CPU Python environment, and React frontend.
"""

import csv
import json
from pathlib import Path
import subprocess
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
def full_artifacts_dir(base_dir):
    d = base_dir / "ml_artifacts/btxrd_full_training"
    assert d.exists(), f"BTXRD full training artifacts directory missing: {d}"
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


def test_02_dataloader_integrity_and_sample_counts(btxrd_manifest, base_dir):
    """Verifies that clean Center 1 partitions contain exactly 2,346 train and 588 val images."""
    train_dataset = BTXRDDataset(
        manifest_path=btxrd_manifest,
        center="center_1",
        split="train",
        split_protocol="center_holdout_split",
        target_size=512,
        is_train=False,
        augment=False,
        base_dir=base_dir,
    )
    val_dataset = BTXRDDataset(
        manifest_path=btxrd_manifest,
        center="center_1",
        split="val",
        split_protocol="center_holdout_split",
        target_size=512,
        is_train=False,
        augment=False,
        base_dir=base_dir,
    )

    assert len(train_dataset) == 2346, f"Expected 2,346 train images, got {len(train_dataset)}"
    assert len(val_dataset) == 588, f"Expected 588 val images, got {len(val_dataset)}"

    train_counts = train_dataset.get_class_counts()
    assert train_counts["NORMAL_BONE_RADIOGRAPH"] == 1272
    assert train_counts["BENIGN_BONE_TUMOR"] == 880
    assert train_counts["PRIMARY_MALIGNANT_TUMOR"] == 194

    val_counts = val_dataset.get_class_counts()
    assert val_counts["NORMAL_BONE_RADIOGRAPH"] == 319
    assert val_counts["BENIGN_BONE_TUMOR"] == 228
    assert val_counts["PRIMARY_MALIGNANT_TUMOR"] == 41


def test_03_aspect_ratio_letterboxing_and_padding(tmp_path):
    """Verifies that rectangular X-rays are letterboxed with symmetrical black padding to 512x512."""
    test_img = Image.new("L", (1024, 512), color=150)
    img_path = tmp_path / "test_rect_large.png"
    test_img.save(img_path)

    tensor = preprocess_btxrd_image(img_path, target_size=512, is_train=False)

    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == torch.float32

    # Vertical padding verification (scaled to 512x256, padding 128 top/bottom)
    expected_pad_c0 = -0.485 / 0.229
    assert abs(tensor[0, 10, 256].item() - expected_pad_c0) < 1e-2


def test_04_label_consistency(btxrd_manifest):
    """Verifies that all target classes in manifest belong strictly to the 3 authorized classes."""
    valid_classes = {"NORMAL_BONE_RADIOGRAPH", "BENIGN_BONE_TUMOR", "PRIMARY_MALIGNANT_TUMOR"}
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["target_class"] in valid_classes, f"Invalid class: {row['target_class']}"


def test_05_focal_loss_numerical_stability():
    """Verifies that MulticlassFocalLoss handles extreme probabilities without NaN or Inf."""
    loss_fn = MulticlassFocalLoss(gamma=2.0, alpha=torch.tensor([0.3333, 0.4817, 2.1850]))

    # Confident correct predictions
    confident_logits = torch.tensor([[50.0, -50.0, -50.0], [-50.0, 50.0, -50.0]], dtype=torch.float32)
    confident_targets = torch.tensor([0, 1], dtype=torch.long)
    loss_conf = loss_fn(confident_logits, confident_targets)
    assert not torch.isnan(loss_conf)
    assert not torch.isinf(loss_conf)
    assert loss_conf.item() < 1e-4

    # Highly erroneous predictions
    wrong_logits = torch.tensor([[-50.0, 50.0, -50.0]], dtype=torch.float32)
    wrong_targets = torch.tensor([0], dtype=torch.long)
    loss_wrong = loss_fn(wrong_logits, wrong_targets)
    assert not torch.isnan(loss_wrong)
    assert not torch.isinf(loss_wrong)
    assert loss_wrong.item() > 0.0


def test_06_center_1_class_weights_calculation(btxrd_manifest):
    """Verifies that normalized alpha weights sum to 3.0 and match Center 1 train frequencies."""
    train_counts = {0: 1272, 1: 880, 2: 194}
    total = sum(train_counts.values())

    raw_weights = [total / (3.0 * train_counts[c]) for c in range(3)]
    sum_raw = sum(raw_weights)
    norm_weights = [round(float(3.0 * w / sum_raw), 4) for w in raw_weights]

    assert norm_weights[0] == 0.3333
    assert norm_weights[1] == 0.4817
    assert norm_weights[2] == 2.1850
    assert abs(sum(norm_weights) - 3.0000) < 1e-3


def test_07_model_output_shape_and_progressive_unfreezing():
    """Verifies parameter structure and unfreeze_layer3() capability."""
    model = BTXRDResNet34(num_classes=3, pretrained=False, dropout_p=0.4, freeze_early=True)

    # Initial frozen state
    meta_initial = model.get_model_metadata()
    assert meta_initial["trainable_parameters"] == 13_115_907
    assert meta_initial["frozen_parameters"] == 8_170_304

    # Unfreeze layer3
    model.unfreeze_layer3()
    meta_unfrozen = model.get_model_metadata()
    assert meta_unfrozen["trainable_parameters"] > meta_initial["trainable_parameters"]
    assert meta_unfrozen["frozen_parameters"] < meta_initial["frozen_parameters"]

    # Forward pass shape check
    dummy_in = torch.randn(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        out = model(dummy_in)
    assert out.shape == (2, 3)


def test_08_cuda_forward_backward_pass():
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


def test_09_checkpoint_reload_and_prediction_reproducibility(full_artifacts_dir):
    """Verifies that best_checkpoint.pt can be reloaded and reproduces valid logits."""
    ckpt_path = full_artifacts_dir / "best_checkpoint.pt"
    assert ckpt_path.exists(), "best_checkpoint.pt missing!"

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert "val_metrics" in checkpoint
    assert "macro_f1" in checkpoint["val_metrics"]

    model = BTXRDResNet34(num_classes=3, pretrained=False)
    if "layer3.0.conv1.weight" in checkpoint["model_state_dict"] and checkpoint["epoch"] >= 6:
        model.unfreeze_layer3()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_in = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_in)
    assert out.shape == (1, 3)
    assert not torch.isnan(out).any()


def test_10_artifact_manifest_completeness(full_artifacts_dir):
    """Verifies presence and valid size of all 12 full training deliverables."""
    expected_files = [
        "training_config.yaml",
        "training_log.csv",
        "validation_metrics.json",
        "validation_predictions.csv",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "class_distribution.json",
        "confusion_matrix.png",
        "loss_curves.png",
        "roc_curves_ovr.png",
        "precision_recall_curves.png",
        "training_summary.json",
    ]

    for fname in expected_files:
        fpath = full_artifacts_dir / fname
        assert fpath.exists(), f"Required artifact missing: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"


def test_11_zero_external_center_access(full_artifacts_dir, btxrd_manifest):
    """Verifies that zero Center 2 or Center 3 images were evaluated in validation predictions."""
    img_center = {}
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_center[row["image_id"]] = row["center"].lower()

    preds_file = full_artifacts_dir / "validation_predictions.csv"
    assert preds_file.exists()

    val_image_ids = []
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_image_ids.append(row["image_id"])
            center = img_center.get(row["image_id"], "")
            assert center == "center_1", (
                f"Image {row['image_id']} belongs to {center}, violated Center 1 boundary!"
            )

    assert len(val_image_ids) == 588, f"Expected 588 validation predictions, got {len(val_image_ids)}"


def test_12_conflict_quarantine_integrity(full_artifacts_dir, btxrd_manifest):
    """Verifies that quarantined conflicting duplicate images are absent from validation predictions."""
    conflicting_ids = set()
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("conflict_flag", "").strip().lower() == "true":
                conflicting_ids.add(row["image_id"])

    assert len(conflicting_ids) == 4
    preds_file = full_artifacts_dir / "validation_predictions.csv"
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["image_id"] not in conflicting_ids, (
                f"Quarantined conflicting image {row['image_id']} leaked into validation predictions!"
            )


def test_13_fracatlas_artifacts_preservation(base_dir):
    """Verifies FracAtlas Stage 09 checkpoint and holdout test report remain untouched."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists(), "FracAtlas checkpoint missing!"
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    fracatlas_rep = base_dir / "ml_artifacts/stage_reports/STAGE_10_FRACATLAS_HOLDOUT_EVALUATION.md"
    assert fracatlas_rep.exists()


def test_14_bone_scintigraphy_artifacts_preservation(base_dir):
    """Verifies Bone Scintigraphy checkpoints and holdout test evaluation remain untouched."""
    scint_fold0 = base_dir / "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"
    assert scint_fold0.exists(), "Bone scan fold 0 checkpoint missing!"

    scint_test_eval = base_dir / "ml_artifacts/bone_scan_test_evaluation/test_metrics.json"
    assert scint_test_eval.exists(), "Bone scan test metrics missing!"


def test_15_stage_16_pilot_and_frontend_preservation(base_dir):
    """Verifies Stage 16 pilot checkpoint, host CPU Python, and React frontend are completely intact."""
    pilot_ckpt = base_dir / "ml_artifacts/btxrd_pilot/best_checkpoint.pt"
    assert pilot_ckpt.exists(), "Stage 16 pilot checkpoint missing!"
    assert pilot_ckpt.stat().st_size > 100_000_000

    # CPU Python environment
    cpu_py = Path(r"C:\Python313\python.exe")
    assert cpu_py.exists()
    cmd = [str(cpu_py), "-c", "import torch; print(torch.__version__)"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "+cpu" in res.stdout

    # React frontend integrity
    frontend_dir = base_dir / "osteophorosis-main/src"
    assert frontend_dir.exists()
    assert (frontend_dir / "App.tsx").exists()

