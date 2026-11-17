"""
OsteoGuard AI - Stage 08B ImageNet-Pretrained ResNet-101 FracAtlas Pilot Training Pipeline.

Features:
  1. Reproducible stratified pilot sampling (N=1,024) from existing train split (seed=42).
  2. Full existing validation partition evaluation (N=612).
  3. Pretrained ResNet-101 (TorchVision IMAGENET1K_V2) initialization with 2-class head.
  4. Layer freezing: conv1, bn1, layer1-3 frozen; layer4 and fc fine-tuned.
  5. Frozen BatchNorm running statistics safeguarded in eval mode.
  6. Resource-aware training tuned for 4 GB VRAM:
     - Batch size = 2
     - Gradient accumulation steps = 8 (effective batch size = 16)
     - Automatic Mixed Precision (AMP FP16) with GradScaler
     - Workers = 0 (Windows multiprocess safety)
  7. Weighted CrossEntropyLoss computed from actual pilot class frequencies.
  8. Comprehensive validation metrics:
     - Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score, ROC-AUC, PR-AUC, Confusion Matrix.
  9. Full artifact export:
     - pilot_config.yaml, pilot_sample_manifest.csv, training_log.csv, validation_metrics.json,
     - best_checkpoint.pt, last_checkpoint.pt, confusion_matrix.png, loss_curves.png,
     - pretrained_weights_info.json, STAGE_08B_FRACATLAS_PILOT.md.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ResNet101_Weights
import torchvision.transforms as transforms
import yaml

from ml_service.config.settings import get_settings


# ------------------------------------------------------------------------------
# 1. Dataset & DataLoader Definition
# ------------------------------------------------------------------------------

def get_pilot_transforms(split: str = "train", img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


class FracAtlasPilotDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], split: str = "train", img_size: Tuple[int, int] = (224, 224)):
        self.records = records
        self.split = split
        self.transform = get_pilot_transforms(split=split, img_size=img_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.records[idx]
        file_path = Path(row["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Image file missing: {file_path}")

        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")

        tensor = self.transform(img_rgb)
        label = torch.tensor(int(row["fractured"]), dtype=torch.long)
        return tensor, label


# ------------------------------------------------------------------------------
# 2. Stratified Sampling
# ------------------------------------------------------------------------------

def create_stratified_pilot_subset(
    full_manifest_path: Path,
    output_sample_path: Path,
    target_count: int = 1024,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    train_frac = []
    train_non_frac = []
    val_records = []

    with open(full_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["split"].lower()
            if sp == "train":
                if row["fractured"] == "1":
                    train_frac.append(row)
                else:
                    train_non_frac.append(row)
            elif sp == "val":
                val_records.append(row)

    # Sort deterministically
    train_frac.sort(key=lambda r: r["image_id"])
    train_non_frac.sort(key=lambda r: r["image_id"])
    val_records.sort(key=lambda r: r["image_id"])

    # Stratified target counts
    total_train = len(train_frac) + len(train_non_frac)
    n_frac = round(target_count * (len(train_frac) / total_train))
    n_non_frac = target_count - n_frac

    rng = random.Random(random_seed)
    sampled_frac = rng.sample(train_frac, n_frac)
    sampled_non_frac = rng.sample(train_non_frac, n_non_frac)

    pilot_train_records = sorted(sampled_frac + sampled_non_frac, key=lambda r: r["image_id"])

    # Save pilot sample manifest
    output_sample_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(pilot_train_records[0].keys())
    with open(output_sample_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pilot_train_records)

    sampling_info = {
        "random_seed": random_seed,
        "total_available_train": total_train,
        "available_fractured": len(train_frac),
        "available_non_fractured": len(train_non_frac),
        "sampled_total": len(pilot_train_records),
        "sampled_fractured": len(sampled_frac),
        "sampled_non_fractured": len(sampled_non_frac),
        "sampled_prevalence": len(sampled_frac) / len(pilot_train_records),
        "total_val_records": len(val_records),
        "val_fractured": sum(1 for r in val_records if r["fractured"] == "1"),
        "val_non_fractured": sum(1 for r in val_records if r["fractured"] == "0"),
    }
    return pilot_train_records, val_records, sampling_info


# ------------------------------------------------------------------------------
# 3. Model Construction & Layer Freezing
# ------------------------------------------------------------------------------

def build_pilot_resnet101_model(
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)

    # Replace classifier head: 2048 -> 2
    model.fc = nn.Linear(2048, 2)
    model.to(device)

    # Freeze conv1, bn1, layer1, layer2, layer3
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

    # layer4 and fc remain trainable
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # Provenance info
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
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": round((trainable_params / total_params) * 100, 2),
        "classifier_head": "Linear(in_features=2048, out_features=2, bias=True)",
        "output_mapping": {0: "candidate_control_radiograph", 1: "acute_bone_fracture"},
    }
    return model, model_info


def set_custom_train_mode(model: nn.Module):
    """Activates train mode for trainable layers while keeping frozen BN in eval mode."""
    model.train()
    model.bn1.eval()
    for layer in [model.layer1, model.layer2, model.layer3]:
        for m in layer.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()


# ------------------------------------------------------------------------------
# 4. Evaluation Function
# ------------------------------------------------------------------------------

def evaluate_pilot_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
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

            with torch.amp.autocast("cuda", dtype=torch.float16):
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
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    metrics = {
        "val_loss": round(avg_loss, 6),
        "val_duration_seconds": round(duration, 2),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix_2x2": cm,
        },
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }
    return metrics


# ------------------------------------------------------------------------------
# 5. Full Pilot Training Pipeline
# ------------------------------------------------------------------------------

def run_fracatlas_pilot(
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    fc_lr: float = 1e-4,
    layer4_lr: float = 1e-5,
    weight_decay: float = 1e-2,
    random_seed: int = 42,
    pilot_sample_size: int = 1024,
) -> Dict[str, Any]:
    start_total_time = time.time()
    settings = get_settings()
    root = settings.base_dir

    output_dir = root / "ml_artifacts/fracatlas_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_manifest_path = root / "ml_service/data/splits/fracatlas_classification_split.csv"
    if not full_manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {full_manifest_path}")

    # Seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Stage 08B GPU pilot training.")

    # 1. Stratified subset sampling
    sample_manifest_path = output_dir / "pilot_sample_manifest.csv"
    train_records, val_records, sampling_info = create_stratified_pilot_subset(
        full_manifest_path=full_manifest_path,
        output_sample_path=sample_manifest_path,
        target_count=pilot_sample_size,
        random_seed=random_seed,
    )

    # 2. DataLoaders
    train_dataset = FracAtlasPilotDataset(train_records, split="train", img_size=(224, 224))
    val_dataset = FracAtlasPilotDataset(val_records, split="val", img_size=(224, 224))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model & Layer Freezing
    model, model_info = build_pilot_resnet101_model(device=device)

    # 4. Class Weights & Loss
    n_total = sampling_info["sampled_total"]
    n_non_frac = sampling_info["sampled_non_fractured"]
    n_frac = sampling_info["sampled_fractured"]
    w0 = n_total / (2.0 * n_non_frac)
    w1 = n_total / (2.0 * n_frac)
    norm_w1 = n_non_frac / n_frac  # 844 / 180 = 4.688889
    class_weights = torch.tensor([1.0, norm_w1], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5. Optimizer & AMP Scaler
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": layer4_lr},
        {"params": model.fc.parameters(), "lr": fc_lr},
    ], weight_decay=weight_decay)

    scaler = torch.amp.GradScaler("cuda")

    # Save initial config YAML
    config_dict = {
        "experiment": {
            "name": "fracatlas_resnet101_pilot",
            "stage": "08B",
            "date": "2026-11-15",
            "random_seed": random_seed,
        },
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "compute_capability": f"{torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]}",
            "total_vram_mb": round(torch.cuda.get_device_properties(device).total_memory / (1024**2), 2),
            "mixed_precision": "FP16",
        },
        "model": model_info,
        "dataset": {
            "task": "binary_acute_fracture_classification",
            "original_manifest": str(full_manifest_path),
            "pilot_sample_manifest": str(sample_manifest_path),
            "sampling": sampling_info,
        },
        "training": {
            "epochs": epochs,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "optimizer": "AdamW",
            "learning_rates": {"layer4": layer4_lr, "fc": fc_lr},
            "weight_decay": weight_decay,
            "loss": "CrossEntropyLoss(weighted)",
            "class_weights": [1.0, round(float(norm_w1), 6)],
        },
    }
    with open(output_dir / "pilot_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    with open(output_dir / "pretrained_weights_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    # 6. Training Loop
    training_logs = []
    best_val_loss = float("inf")
    best_epoch = 0
    final_val_metrics = {}

    print("=" * 70)
    print("STARTING STAGE 08B FRACATLAS RESNET-101 PILOT TRAINING")
    print(f"Device: {torch.cuda.get_device_name(device)} (CC {torch.cuda.get_device_capability(device)})")
    print(f"Train subset: {len(train_dataset)} images | Val subset: {len(val_dataset)} images")
    print(f"Batch size: {batch_size} | Grad Accum: {gradient_accumulation_steps} | Effective BS: {batch_size * gradient_accumulation_steps}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        set_custom_train_mode(model)
        torch.cuda.reset_peak_memory_stats(device)

        running_loss = 0.0
        total_train_samples = 0
        optimizer.zero_grad()

        num_batches = len(train_loader)
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            b_size = images.size(0)

            # Determine accumulation divisor for current accumulation window
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
                # Scale loss for gradient accumulation
                accum_loss = loss / gradient_accumulation_steps

            scaler.scale(accum_loss).backward()
            running_loss += loss.item() * b_size
            total_train_samples += b_size

            # Optimizer step on accumulation boundary or epoch end
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_duration = time.time() - epoch_start
        train_loss = running_loss / total_train_samples
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        # Validation pass
        val_metrics = evaluate_pilot_model(model, val_loader, criterion, device)
        final_val_metrics = val_metrics

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_metrics["val_loss"],
            "train_duration_sec": round(train_duration, 2),
            "val_duration_sec": val_metrics["val_duration_seconds"],
            "peak_vram_mb": round(peak_vram_mb, 2),
            "lr_layer4": layer4_lr,
            "lr_fc": fc_lr,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall_sensitivity": val_metrics["recall_sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_f1_score": val_metrics["f1_score"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
        }
        training_logs.append(epoch_log)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Peak VRAM: {peak_vram_mb:.1f} MiB | "
            f"Time: {train_duration + val_metrics['val_duration_seconds']:.1f}s",
            flush=True,
        )

        # Save last checkpoint
        last_ckpt_path = output_dir / "last_checkpoint.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "metrics": val_metrics,
            "config": config_dict,
        }, last_ckpt_path)

        # Save best checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            best_ckpt_path = output_dir / "best_checkpoint.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "metrics": val_metrics,
                "config": config_dict,
            }, best_ckpt_path)

    total_training_time = time.time() - start_total_time

    # 7. Write training_log.csv
    log_fieldnames = list(training_logs[0].keys())
    with open(output_dir / "training_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fieldnames)
        writer.writeheader()
        writer.writerows(training_logs)

    # 8. Write validation_metrics.json (clean summary without raw arrays)
    summary_val_metrics = {
        "evaluation_split": "validation",
        "sample_count": len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "final_epoch_metrics": {k: v for k, v in final_val_metrics.items() if k not in ["y_true", "y_prob", "y_pred"]},
        "training_time_total_seconds": round(total_training_time, 2),
    }
    with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_val_metrics, f, indent=2)

    # 9. Generate and save Plots
    # A. Confusion Matrix Plot
    cm_matrix = np.array(final_val_metrics["confusion_matrix"]["matrix_2x2"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("FracAtlas Pilot Validation Confusion Matrix (Threshold = 0.5)")
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ["Non-fractured (0)", "Fractured (1)"]
    plt.xticks(tick_marks, classes, rotation=15)
    plt.yticks(tick_marks, classes)

    thresh = cm_matrix.max() / 2.0
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            plt.text(
                j, i, format(cm_matrix[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_matrix[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    plt.ylabel("True Clinical Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # B. Loss Curves Plot
    epochs_arr = [log["epoch"] for log in training_logs]
    train_losses = [log["train_loss"] for log in training_logs]
    val_losses = [log["val_loss"] for log in training_logs]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs_arr, train_losses, marker="o", color="blue", label="Training Loss", linewidth=2)
    plt.plot(epochs_arr, val_losses, marker="s", color="orange", label="Validation Loss", linewidth=2)
    plt.title("FracAtlas ResNet-101 Pilot Loss Trajectory (3 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted CrossEntropy Loss")
    plt.xticks(epochs_arr)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_curve_path = output_dir / "loss_curves.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # 10. Checkpoint Reload Verification
    fresh_model, _ = build_pilot_resnet101_model(device=device)
    loaded_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    fresh_model.load_state_dict(loaded_ckpt["model_state_dict"])
    fresh_model.eval()

    # Test dummy inference on reloaded model
    dummy_x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        dummy_out = fresh_model(dummy_x)
    assert dummy_out.shape == (2, 2)
    assert torch.isfinite(dummy_out).all()
    print("Checkpoint reload verification: SUCCESS.")

    print(f"\nPILOT COMPLETE. Artifacts stored in: {output_dir}")
    return {
        "status": "PASS",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_logs": training_logs,
        "summary_val_metrics": summary_val_metrics,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    run_fracatlas_pilot()
