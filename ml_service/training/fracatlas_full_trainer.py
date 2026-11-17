"""
OsteoGuard AI - Full-Scale FracAtlas ResNet-101 Training Pipeline.

Stage 09:
- Full dataset partitions (2,857 train, 612 validation; 614 test quarantined).
- Official TorchVision ResNet-101 ImageNet-1K V2 pretrained weights.
- Classifier head prior-matching logit bias initialization:
    pi = 502 / 2857 ~ 0.1757088
    b0 = 0.0
    b1 = log(pi / (1 - pi)) = log(502 / 2355) ~ -1.545388
- Numerically stable Focal Loss (gamma=2.0, alpha=[0.25, 0.75]).
- Conservative radiograph data augmentation.
- Progressive backbone unfreezing:
    Epochs 1-5: layer4 (lr=1e-5) + fc (lr=1e-4) trainable.
    Epochs 6-30: layer3 progressively unfrozen with lr=3e-6.
- BatchNorm statistics kept in eval mode across all backbone layers.
- Effective batch size 16 (micro-batch size 2, gradient accumulation 8).
- FP16 mixed precision via torch.amp with GradScaler.
- Primary checkpoint selection metric: Validation PR-AUC.
- Early stopping with patience 7.
- Complete validation threshold sweep (0.05 - 0.95) on best checkpoint.
- Output artifacts saved to ml_artifacts/fracatlas_full_training/.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is first in sys.path to prevent module shadowing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ResNet101_Weights
import torchvision.transforms as transforms
import yaml

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from ml_service.training.focal_loss import FocalLoss


# ------------------------------------------------------------------------------
# 1. Dataset & Transform Definitions
# ------------------------------------------------------------------------------

def get_training_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Conservative radiograph augmentation for training partition."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def get_validation_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Deterministic preprocessing for validation partition."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


class FracAtlasFullDataset(Dataset):
    """
    Dataset wrapper for FracAtlas full-scale training and validation.
    Reads records directly from the manifest split.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        return_metadata: bool = False,
    ):
        self.records = records
        self.split = split
        self.return_metadata = return_metadata
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_training_transforms()
        else:
            self.transform = get_validation_transforms()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        file_path = Path(row["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        with Image.open(file_path) as img:
            image_rgb = img.convert("RGB")

        tensor = self.transform(image_rgb)
        label = torch.tensor(int(row["fractured"]), dtype=torch.long)

        if self.return_metadata:
            meta = {
                "image_id": row["image_id"],
                "file_path": str(file_path),
                "anatomic_region": row.get("anatomic_region", "unknown"),
                "view_projection": row.get("view_projection", "unknown"),
                "sha256_hash": row.get("sha256_hash", ""),
                "split": row.get("split", self.split),
            }
            return tensor, label, meta

        return tensor, label


# ------------------------------------------------------------------------------
# 2. Model Initialization & Prior Logit Bias
# ------------------------------------------------------------------------------

def build_full_resnet101_model(
    device: torch.device,
    pi_prior: float,
    dropout_rate: float = 0.2,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Builds ResNet-101 with ImageNet-1K V2 pretrained weights.
    Sets prior-matching logit bias for the classification head.
    Initializes freezing strategy: conv1, bn1, layer1-3 frozen; layer4 + fc trainable.
    """
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)

    # Classification head: 2048 -> 2
    model.fc = nn.Linear(2048, 2)

    # Initialize classifier head bias with prior log-odds
    # pi = P(fracture); b0 = 0.0, b1 = log(pi / (1 - pi))
    b0 = 0.0
    b1 = math.log(pi_prior / (1.0 - pi_prior))
    with torch.no_grad():
        model.fc.bias[0] = b0
        model.fc.bias[1] = b1

    model.to(device)

    # Freeze conv1, bn1, layer1, layer2, layer3 initially
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = False

    # layer4 and fc are trainable initially
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_init = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_init = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    ckpt_cache_path = Path.home() / ".cache/torch/hub/checkpoints/resnet101-cd907fc2.pth"
    sha256_hash = ""
    if ckpt_cache_path.exists():
        h = hashlib.sha256()
        with open(ckpt_cache_path, "rb") as f:
            while chunk := f.read(8192 * 1024):
                h.update(chunk)
        sha256_hash = h.hexdigest()

    model_info = {
        "architecture": "ResNet101",
        "weights_enum": str(weights),
        "weights_url": weights.url,
        "local_cache_path": str(ckpt_cache_path),
        "cache_sha256": sha256_hash,
        "total_parameters": total_params,
        "initial_trainable_parameters": trainable_params_init,
        "initial_frozen_parameters": frozen_params_init,
        "initial_trainable_percentage": round((trainable_params_init / total_params) * 100, 2),
        "classifier_head": "Linear(in_features=2048, out_features=2, bias=True)",
        "prior_pi": pi_prior,
        "init_bias_b0": b0,
        "init_bias_b1": b1,
        "output_mapping": {0: "candidate_control_radiograph", 1: "acute_bone_fracture"},
    }
    return model, model_info


def set_train_mode_with_eval_bn(model: nn.Module):
    """
    Sets model to train mode, but strictly keeps all BatchNorm modules in eval mode
    to prevent running statistics drift caused by micro-batch size 2.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


# ------------------------------------------------------------------------------
# 3. Validation Evaluation Function
# ------------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates validation partition and calculates all diagnostic metrics."""
    model.eval()
    total_val_loss = 0.0
    val_samples = 0
    all_targets = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            with autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)

            total_val_loss += loss.item() * batch_size
            val_samples += batch_size

            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fractured)
            all_targets.extend(y.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().float().numpy().tolist())

    duration = time.time() - start_time
    avg_loss = total_val_loss / val_samples if val_samples > 0 else 0.0

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred_05 = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred_05))
    prec = float(precision_score(y_true, y_pred_05, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred_05, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred_05, zero_division=0.0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred_05))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    return {
        "val_loss": round(avg_loss, 6),
        "val_duration_seconds": round(duration, 2),
        "accuracy_05": round(acc, 4),
        "precision_05": round(prec, 4),
        "recall_sensitivity_05": round(rec, 4),
        "specificity_05": round(spec, 4),
        "f1_score_05": round(f1, 4),
        "balanced_accuracy_05": round(bal_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "true_positives_05": int(tp),
        "false_positives_05": int(fp),
        "true_negatives_05": int(tn),
        "false_negatives_05": int(fn),
        "y_true": y_true,
        "y_prob": y_prob,
    }


# ------------------------------------------------------------------------------
# 4. Threshold Sweep & Calibration
# ------------------------------------------------------------------------------

def perform_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_csv_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Performs full threshold sweep from 0.05 to 0.95 (step 0.01).
    Identifies Best F1, Best Youden's J, and High Sensitivity (>=85%) operating points.
    """
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]
    records = []

    best_f1_val = -1.0
    best_f1_op = None

    best_youden_val = -2.0
    best_youden_op = None

    high_sens_candidates = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
        bal_acc = float((sens + spec) / 2.0)
        youden = float(sens + spec - 1.0)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        rec = {
            "threshold": t,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "youden_j": round(youden, 4),
            "false_positive_rate": round(fpr, 4),
        }
        records.append(rec)

        if f1 > best_f1_val:
            best_f1_val = f1
            best_f1_op = rec

        if youden > best_youden_val:
            best_youden_val = youden
            best_youden_op = rec

        if sens >= 0.85:
            high_sens_candidates.append(rec)

    # Highest specificity among sensitivity >= 85%
    if high_sens_candidates:
        high_sens_candidates.sort(key=lambda r: (r["specificity"], r["precision"]), reverse=True)
        best_high_sens_op = high_sens_candidates[0]
    else:
        # Fallback to operating point with highest sensitivity
        records_sorted_sens = sorted(records, key=lambda r: r["sensitivity"], reverse=True)
        best_high_sens_op = records_sorted_sens[0]

    # Save to CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold", "tp", "fp", "tn", "fn", "sensitivity", "specificity",
        "precision", "f1_score", "balanced_accuracy", "youden_j", "false_positive_rate"
    ]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    operating_points = {
        "best_f1_operating_point": best_f1_op,
        "best_youden_j_operating_point": best_youden_op,
        "high_sensitivity_operating_point": best_high_sens_op,
    }
    return records, operating_points


# ------------------------------------------------------------------------------
# 5. Visualizations Generator
# ------------------------------------------------------------------------------

def generate_visual_artifacts(
    history: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    operating_points: Dict[str, Any],
    output_dir: Path,
):
    """Generates loss_curves.png, roc_curve.png, precision_recall_curve.png, and confusion_matrix.png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    # 1. Loss & Metric Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Focal Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Focal Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss (Focal Loss)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(epochs, [h["train_roc_auc"] for h in history], label="Train ROC-AUC", color="#2ca02c", lw=2)
    axes[1].plot(epochs, [h["val_roc_auc"] for h in history], label="Val ROC-AUC", color="#9467bd", lw=2)
    axes[1].plot(epochs, [h["val_pr_auc"] for h in history], label="Val PR-AUC", color="#d62728", lw=2, linestyle="--")
    axes[1].set_title("Discriminative Metrics Across Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=200)
    plt.close()

    # 2. ROC Curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_arr, tpr_arr, color="#1f77b4", lw=2.5, label=f"Val ROC (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")

    # Mark operating points
    best_f1 = operating_points["best_f1_operating_point"]
    best_youden = operating_points["best_youden_j_operating_point"]
    high_sens = operating_points["high_sensitivity_operating_point"]

    plt.scatter([best_f1["false_positive_rate"]], [best_f1["sensitivity"]], color="#d62728", s=80, zorder=5,
                label=f"Max F1 (th={best_f1['threshold']:.2f}, Sens={best_f1['sensitivity']:.2f})")
    plt.scatter([best_youden["false_positive_rate"]], [best_youden["sensitivity"]], color="#2ca02c", s=80, zorder=5, marker="^",
                label=f"Max Youden J (th={best_youden['threshold']:.2f}, Sens={best_youden['sensitivity']:.2f})")
    plt.scatter([high_sens["false_positive_rate"]], [high_sens["sensitivity"]], color="#ff7f0e", s=80, zorder=5, marker="s",
                label=f"High Sens (th={high_sens['threshold']:.2f}, Sens={high_sens['sensitivity']:.2f})")

    plt.title("FracAtlas ResNet-101 Validation ROC Curve", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    # 3. Precision-Recall Curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    prevalence = np.mean(y_true)

    plt.figure(figsize=(7, 6))
    plt.plot(rec_arr, prec_arr, color="#d62728", lw=2.5, label=f"Val PR (AUC = {pr_auc_val:.4f})")
    plt.axhline(prevalence, color="grey", linestyle="--", lw=1.5, label=f"Prevalence Baseline ({prevalence:.4f})")

    plt.scatter([best_f1["sensitivity"]], [best_f1["precision"]], color="#1f77b4", s=80, zorder=5,
                label=f"Max F1 (th={best_f1['threshold']:.2f}, Prec={best_f1['precision']:.2f})")
    plt.scatter([best_youden["sensitivity"]], [best_youden["precision"]], color="#2ca02c", s=80, zorder=5, marker="^",
                label=f"Max Youden J (th={best_youden['threshold']:.2f}, Prec={best_youden['precision']:.2f})")

    plt.title("FracAtlas ResNet-101 Validation Precision-Recall Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()

    # 4. Confusion Matrix for Chosen Operating Point (Max Youden / Balanced)
    chosen_op = best_youden
    cm = np.array([
        [chosen_op["tn"], chosen_op["fp"]],
        [chosen_op["fn"], chosen_op["tp"]]
    ])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Development Threshold = {chosen_op['threshold']:.2f})\n"
              f"Sens: {chosen_op['sensitivity']:.2%}, Spec: {chosen_op['specificity']:.2%}",
              fontsize=11, fontweight="bold")
    plt.colorbar(im)
    classes = ["Non-fractured (0)", "Fractured (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------------------
# 6. Main Full Training Orchestration
# ------------------------------------------------------------------------------

def run_full_training(
    manifest_path: Path,
    output_dir: Path,
    max_epochs: int = 30,
    warmup_epochs: int = 5,
    patience: int = 7,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 8,
    fc_lr: float = 1e-4,
    layer4_lr: float = 1e-5,
    layer3_lr: float = 3e-6,
    weight_decay: float = 0.01,
    random_seed: int = 42,
):
    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 09 FULL-SCALE FRACATLAS RESNET-101 TRAINING", flush=True)
    print("=" * 80, flush=True)

    # 1. Device check
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available. GPU environment required.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    print(f"[Device] CUDA Device: {gpu_name} (Total VRAM: {total_vram_mb:.1f} MiB)", flush=True)

    # 2. Seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. Load & verify partitions
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    train_records = []
    val_records = []
    test_count = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["split"].lower()
            if sp == "train":
                train_records.append(row)
            elif sp == "val":
                val_records.append(row)
            elif sp == "test":
                test_count += 1

    train_frac = sum(1 for r in train_records if r["fractured"] == "1")
    train_non_frac = sum(1 for r in train_records if r["fractured"] == "0")
    val_frac = sum(1 for r in val_records if r["fractured"] == "1")
    val_non_frac = sum(1 for r in val_records if r["fractured"] == "0")

    print(f"[Manifest] Loaded split manifest: {manifest_path}", flush=True)
    print(f"[Manifest] Train records: {len(train_records)} (Fractured: {train_frac}, Non-fractured: {train_non_frac})", flush=True)
    print(f"[Manifest] Val records:   {len(val_records)} (Fractured: {val_frac}, Non-fractured: {val_non_frac})", flush=True)
    print(f"[Manifest] Test records:  {test_count} (QUARANTINED - strictly untouched)", flush=True)

    assert len(train_records) == 2857, f"Expected 2857 train records, got {len(train_records)}"
    assert train_frac == 502, f"Expected 502 train fractured, got {train_frac}"
    assert train_non_frac == 2355, f"Expected 2355 train non-fractured, got {train_non_frac}"
    assert len(val_records) == 612, f"Expected 612 val records, got {len(val_records)}"
    assert val_frac == 107, f"Expected 107 val fractured, got {val_frac}"
    assert val_non_frac == 505, f"Expected 505 val non-fractured, got {val_non_frac}"
    assert test_count == 614, f"Expected 614 test records, got {test_count}"

    # 4. Class prior
    pi = train_frac / len(train_records)
    print(f"[Class Prior] Training fracture prevalence: pi = {pi:.7f} ({train_frac}/{len(train_records)})", flush=True)
    expected_b1 = math.log(pi / (1.0 - pi))
    print(f"[Class Prior] Logit bias: b0 = 0.0, b1 = {expected_b1:.6f}", flush=True)

    # 5. Datasets & Loaders
    train_dataset = FracAtlasFullDataset(train_records, split="train")
    val_dataset = FracAtlasFullDataset(val_records, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # Safe batch size for val forward pass
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # 6. Model initialization
    model, model_info = build_full_resnet101_model(device, pi_prior=pi, dropout_rate=0.2)
    print(f"[Model] Pretrained ResNet-101 built. Trainable params: {model_info['initial_trainable_parameters']:,} / {model_info['total_parameters']:,}", flush=True)

    # 7. Criterion (Focal Loss)
    criterion = FocalLoss(gamma=2.0, alpha=(0.25, 0.75), reduction="mean").to(device)
    print("[Criterion] Focal Loss initialized (gamma=2.0, alpha=[0.25, 0.75])", flush=True)

    # 8. Optimizer & Scheduler
    optimizer = AdamW([
        {"params": model.fc.parameters(), "lr": fc_lr, "name": "fc", "initial_lr": fc_lr},
        {"params": model.layer4.parameters(), "lr": layer4_lr, "name": "layer4", "initial_lr": layer4_lr},
    ], weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    # 9. Output setup
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = output_dir / "training_log.csv"
    best_ckpt_path = output_dir / "best_checkpoint.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"

    log_fields = [
        "epoch", "train_loss", "val_loss", "train_roc_auc", "val_roc_auc", "val_pr_auc",
        "precision_05", "recall_sensitivity_05", "specificity_05", "f1_score_05",
        "balanced_accuracy_05", "fc_lr", "layer4_lr", "layer3_lr", "epoch_duration_sec",
        "peak_vram_mb", "layer3_unfrozen", "patience_counter", "is_best"
    ]
    with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    # Save training config
    config_dict = {
        "dataset": "FracAtlas Full Classification",
        "manifest_path": str(manifest_path),
        "total_train_samples": len(train_records),
        "train_fractured": train_frac,
        "train_non_fractured": train_non_frac,
        "total_val_samples": len(val_records),
        "val_fractured": val_frac,
        "val_non_fractured": val_non_frac,
        "total_test_samples": test_count,
        "test_status": "quarantined_untouched",
        "architecture": "ResNet101",
        "weights": "ResNet101_Weights.IMAGENET1K_V2",
        "cached_weights_sha256": model_info["cache_sha256"],
        "class_prior_pi": pi,
        "fc_bias_init": [0.0, expected_b1],
        "loss": "FocalLoss",
        "focal_gamma": 2.0,
        "focal_alpha": [0.25, 0.75],
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": micro_batch_size * grad_accum_steps,
        "mixed_precision": "FP16 (torch.amp)",
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "scheduler": "CosineAnnealingLR",
        "max_epochs": max_epochs,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "fc_lr_init": fc_lr,
        "layer4_lr_init": layer4_lr,
        "layer3_lr_init": layer3_lr,
        "random_seed": random_seed,
        "gpu_name": gpu_name,
        "total_vram_mb": round(total_vram_mb, 1),
    }
    with open(output_dir / "training_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    print(f"[Config] Saved training configuration to {output_dir / 'training_config.yaml'}", flush=True)

    # 10. Training Loop
    history = []
    best_val_pr_auc = -1.0
    best_epoch = -1
    best_val_metrics = None
    patience_counter = 0
    layer3_unfrozen = False

    training_start_time = time.time()
    print("\n" + "=" * 80, flush=True)
    print("BEGINNING TRAINING EXECUTION", flush=True)
    print("=" * 80, flush=True)

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        # Check progressive unfreezing at epoch warmup_epochs + 1 (i.e. epoch 6)
        if epoch > warmup_epochs and not layer3_unfrozen:
            print(f"\n>>> [Epoch {epoch}] Unfreezing layer3 parameters for progressive backbone fine-tuning!", flush=True)
            for p in model.layer3.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": model.layer3.parameters(),
                "lr": layer3_lr,
                "name": "layer3",
                "initial_lr": layer3_lr,
            })
            scheduler.base_lrs.append(layer3_lr)
            layer3_unfrozen = True

            total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f">>> [Epoch {epoch}] Trainable parameters updated: {total_trainable:,} / {model_info['total_parameters']:,}", flush=True)

        # Set model train mode with eval BN
        set_train_mode_with_eval_bn(model)
        optimizer.zero_grad()

        total_train_loss = 0.0
        train_samples = 0
        train_preds_list = []
        train_targets_list = []

        torch.cuda.reset_peak_memory_stats(device)

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            batch_sz = x.size(0)

            with autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            total_train_loss += loss.item() * batch_sz
            train_samples += batch_sz

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)[:, 1]
                train_preds_list.extend(probs.cpu().float().numpy().tolist())
                train_targets_list.extend(y.cpu().numpy().tolist())

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Step scheduler
        scheduler.step()

        # Compute train metrics
        epoch_train_loss = total_train_loss / train_samples if train_samples > 0 else 0.0
        try:
            train_roc_auc = float(roc_auc_score(train_targets_list, train_preds_list))
        except Exception:
            train_roc_auc = 0.5

        # Evaluate validation set
        val_res = evaluate_model(model, val_loader, criterion, device)
        epoch_val_loss = val_res["val_loss"]
        val_roc_auc = val_res["roc_auc"]
        val_pr_auc = val_res["pr_auc"]

        epoch_duration = time.time() - epoch_start_time
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        current_fc_lr = optimizer.param_groups[0]["lr"]
        current_layer4_lr = optimizer.param_groups[1]["lr"]
        current_layer3_lr = optimizer.param_groups[2]["lr"] if layer3_unfrozen else 0.0

        is_best = False
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_val_metrics = val_res
            is_best = True
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_metrics": {k: v for k, v in val_res.items() if k not in ("y_true", "y_prob")},
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "config": config_dict,
                "layer3_unfrozen": layer3_unfrozen,
            }, best_ckpt_path)
        else:
            patience_counter += 1

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": {k: v for k, v in val_res.items() if k not in ("y_true", "y_prob")},
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
            "config": config_dict,
            "layer3_unfrozen": layer3_unfrozen,
        }, last_ckpt_path)

        # Log epoch record
        log_row = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(epoch_val_loss, 6),
            "train_roc_auc": round(train_roc_auc, 4),
            "val_roc_auc": round(val_roc_auc, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "precision_05": val_res["precision_05"],
            "recall_sensitivity_05": val_res["recall_sensitivity_05"],
            "specificity_05": val_res["specificity_05"],
            "f1_score_05": val_res["f1_score_05"],
            "balanced_accuracy_05": val_res["balanced_accuracy_05"],
            "fc_lr": f"{current_fc_lr:.3e}",
            "layer4_lr": f"{current_layer4_lr:.3e}",
            "layer3_lr": f"{current_layer3_lr:.3e}" if layer3_unfrozen else "frozen",
            "epoch_duration_sec": round(epoch_duration, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "layer3_unfrozen": layer3_unfrozen,
            "patience_counter": patience_counter,
            "is_best": is_best,
        }
        history.append(log_row)

        with open(log_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_row)

        star = " ★ BEST" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} (AUC: {train_roc_auc:.4f}) | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"VRAM: {peak_vram:.1f} MiB | "
            f"Time: {epoch_duration:.1f}s{star}",
            flush=True
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement in validation PR-AUC for {patience} consecutive epochs. Stopping at epoch {epoch}.", flush=True)
            break

    total_training_duration = time.time() - training_start_time
    print("\n" + "=" * 80, flush=True)
    print(f"TRAINING COMPLETE in {total_training_duration / 60:.2f} minutes", flush=True)
    print(f"Best Checkpoint: Epoch {best_epoch} with Val PR-AUC = {best_val_pr_auc:.4f}", flush=True)
    print("=" * 80, flush=True)

    # 11. Final Validation Threshold Analysis on Best Checkpoint
    print("\n[Threshold Calibration] Loading best checkpoint for validation threshold sweep...", flush=True)
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset_meta = FracAtlasFullDataset(val_records, split="val", return_metadata=True)
    val_meta_loader = DataLoader(val_dataset_meta, batch_size=4, shuffle=False, num_workers=0)

    val_preds_records = []
    val_true_list = []
    val_prob_list = []

    with torch.no_grad():
        for x, y, meta in val_meta_loader:
            x = x.to(device)
            with autocast("cuda", dtype=torch.float16):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

            p_frac = probs[:, 1].cpu().float().numpy()
            p_non = probs[:, 0].cpu().float().numpy()
            targets = y.numpy()

            for i in range(len(targets)):
                t = int(targets[i])
                p_f = float(p_frac[i])
                p_n = float(p_non[i])
                val_true_list.append(t)
                val_prob_list.append(p_f)

                pred_05 = int(p_f >= 0.5)
                val_preds_records.append({
                    "image_id": meta["image_id"][i],
                    "file_path": meta["file_path"][i],
                    "anatomic_region": meta["anatomic_region"][i],
                    "view_projection": meta["view_projection"][i],
                    "true_label": t,
                    "true_label_name": "acute_bone_fracture" if t == 1 else "candidate_control_radiograph",
                    "prob_non_fractured": round(p_n, 6),
                    "prob_fractured": round(p_f, 6),
                    "pred_label_default_05": pred_05,
                    "pred_label_name_default_05": "acute_bone_fracture" if pred_05 == 1 else "candidate_control_radiograph",
                })

    # Save validation predictions
    val_preds_csv_path = output_dir / "validation_predictions.csv"
    with open(val_preds_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_preds_records[0].keys()))
        writer.writeheader()
        writer.writerows(val_preds_records)
    print(f"[Threshold Calibration] Saved 612 validation predictions to {val_preds_csv_path}", flush=True)

    # Perform threshold sweep
    y_true_arr = np.array(val_true_list, dtype=int)
    y_prob_arr = np.array(val_prob_list, dtype=float)

    threshold_records, operating_points = perform_threshold_analysis(
        y_true_arr,
        y_prob_arr,
        output_dir / "threshold_analysis.csv",
    )
    print(f"[Threshold Calibration] Sweep complete. Saved threshold analysis to {output_dir / 'threshold_analysis.csv'}", flush=True)

    # Save validation metrics JSON
    final_val_metrics = {
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_ckpt_path),
        "total_validation_samples": len(val_records),
        "validation_fractured": val_frac,
        "validation_non_fractured": val_non_frac,
        "roc_auc": round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4),
        "pr_auc": round(float(average_precision_score(y_true_arr, y_prob_arr)), 4),
        "default_threshold_0_50": {
            "threshold": 0.50,
            "tp": int(np.sum((y_true_arr == 1) & (y_prob_arr >= 0.50))),
            "fp": int(np.sum((y_true_arr == 0) & (y_prob_arr >= 0.50))),
            "tn": int(np.sum((y_true_arr == 0) & (y_prob_arr < 0.50))),
            "fn": int(np.sum((y_true_arr == 1) & (y_prob_arr < 0.50))),
            "sensitivity": round(float(recall_score(y_true_arr, (y_prob_arr >= 0.50).astype(int), zero_division=0.0)), 4),
            "specificity": round(float(np.sum((y_true_arr == 0) & (y_prob_arr < 0.50)) / val_non_frac), 4),
            "precision": round(float(precision_score(y_true_arr, (y_prob_arr >= 0.50).astype(int), zero_division=0.0)), 4),
            "f1_score": round(float(f1_score(y_true_arr, (y_prob_arr >= 0.50).astype(int), zero_division=0.0)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true_arr, (y_prob_arr >= 0.50).astype(int))), 4),
        },
        "operating_points": operating_points,
    }
    with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_val_metrics, f, indent=2)

    # 12. Visualizations
    generate_visual_artifacts(history, y_true_arr, y_prob_arr, operating_points, output_dir)
    print(f"[Visuals] Saved loss_curves.png, roc_curve.png, precision_recall_curve.png, and confusion_matrix.png to {output_dir}", flush=True)

    # 13. Training Summary JSON
    summary_dict = {
        "project": "OsteoGuard AI",
        "stage": "STAGE 09 — FULL-SCALE FRACATLAS RESNET101 TRAINING",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(training_start_time)),
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_minutes": round(total_training_duration / 60, 2),
        "device": gpu_name,
        "total_vram_mb": round(total_vram_mb, 1),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_pr_auc": round(best_val_pr_auc, 4),
        "best_val_roc_auc": final_val_metrics["roc_auc"],
        "early_stopped": patience_counter >= patience,
        "quarantined_test_samples": test_count,
        "best_checkpoint_file": str(best_ckpt_path),
        "last_checkpoint_file": str(last_ckpt_path),
        "operating_points": operating_points,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"[Summary] Saved complete training summary to {output_dir / 'training_summary.json'}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print("STAGE 09 TRAINING AND CALIBRATION COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OsteoGuard AI Stage 09 FracAtlas Full Trainer")
    parser.add_argument("--manifest", type=str, default="ml_service/data/splits/fracatlas_classification_split.csv")
    parser.add_argument("--output_dir", type=str, default="ml_artifacts/fracatlas_full_training")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--fc_lr", type=float, default=1e-4)
    parser.add_argument("--layer4_lr", type=float, default=1e-5)
    parser.add_argument("--layer3_lr", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_full_training(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        fc_lr=args.fc_lr,
        layer4_lr=args.layer4_lr,
        layer3_lr=args.layer3_lr,
        weight_decay=args.weight_decay,
        random_seed=args.seed,
    )
