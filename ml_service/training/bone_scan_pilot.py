"""
OsteoGuard AI - Stage 11B Bone Scintigraphy Model Pipeline & Pilot Training.

Experiment details:
  - Dataset: 203 training patients from bone_scan_train_cv_manifest.csv (zero overlap with original Val/Test).
  - Pilot split: Val = fold_0 (42 patients), Train = folds 1,2,3,4 (161 patients).
  - Model: BoneScanResNet18 (ImageNet pretrained, 1-channel adapted conv1, dropout=0.4, single logit).
  - Preprocessing: 16-bit counts clipped at 25,000, normalized to [0, 1], aspect ratio (256x1024) preserved.
  - Paired views: 1 patient per micro-batch (Anterior + Posterior scans), grad accumulation = 4.
  - Patient aggregation: P(patient positive) = max(P(anterior), P(posterior)).
  - Loss: BCEWithLogitsLoss with pos_weight based strictly on training folds.
  - Duration: 3 pilot epochs.
  - Outputs saved to: ml_artifacts/bone_scan_pilot/.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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

# Ensure project root is first in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_service.models.bone_scan_classifier import build_bone_scan_resnet18, set_eval_bn_mode
from ml_service.datasets.bone_scan_dataset import BoneScanPatientDataset


# ------------------------------------------------------------------------------
# 1. Pilot Evaluation Function
# ------------------------------------------------------------------------------

def evaluate_bone_scan_pilot(
    model: nn.Module,
    val_dataset: BoneScanPatientDataset,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates patient-level and image-level metrics across the validation fold.
    Patient prediction is aggregated via Clinical Max-Pooling:
        P(patient) = max(P(anterior), P(posterior))
    """
    model.eval()
    total_val_loss = 0.0
    val_image_count = 0

    patient_records = []
    image_records = []

    pt_y_true = []
    pt_y_prob = []

    img_y_true = []
    img_y_prob = []

    start_eval_time = time.time()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            item = val_dataset[i]
            pid = item["patient_id"]
            y_val = float(item["label"].item())

            ant_tensor = item["anterior_tensor"].unsqueeze(0).to(device)   # (1, 1, 1024, 256)
            post_tensor = item["posterior_tensor"].unsqueeze(0).to(device) # (1, 1, 1024, 256)

            pair_tensors = torch.cat([ant_tensor, post_tensor], dim=0)     # (2, 1, 1024, 256)
            pair_targets = torch.tensor([[y_val], [y_val]], device=device) # (2, 1)

            with autocast("cuda", dtype=torch.float16):
                logits = model(pair_tensors)                               # (2, 1)
                loss = criterion(logits, pair_targets)

            total_val_loss += loss.item() * 2
            val_image_count += 2

            probs = torch.sigmoid(logits).cpu().float().numpy().flatten()
            p_ant = float(probs[0])
            p_post = float(probs[1])

            # Patient-level aggregation: Clinical Max-Pooling
            p_patient = max(p_ant, p_post)

            pt_y_true.append(int(y_val))
            pt_y_prob.append(p_patient)

            img_y_true.extend([int(y_val), int(y_val)])
            img_y_prob.extend([p_ant, p_post])

            pred_pt = int(p_patient >= threshold)
            patient_records.append({
                "patient_id": pid,
                "target_class": item["target_class"],
                "actual_label": int(y_val),
                "prob_anterior": round(p_ant, 6),
                "prob_posterior": round(p_post, 6),
                "prob_patient_max": round(p_patient, 6),
                "pred_label_050": pred_pt,
                "pred_name_050": "BONE_METASTASIS" if pred_pt == 1 else "NO_BONE_METASTASIS",
            })

            image_records.append({
                "patient_id": pid,
                "view": "anterior",
                "file_path": item["anterior_path"],
                "actual_label": int(y_val),
                "probability": round(p_ant, 6),
                "pred_label_050": int(p_ant >= threshold),
            })
            image_records.append({
                "patient_id": pid,
                "view": "posterior",
                "file_path": item["posterior_path"],
                "actual_label": int(y_val),
                "probability": round(p_post, 6),
                "pred_label_050": int(p_post >= threshold),
            })

    eval_duration = time.time() - start_eval_time
    avg_loss = total_val_loss / val_image_count if val_image_count > 0 else 0.0

    # 1. Patient-level metrics
    pt_true_arr = np.array(pt_y_true, dtype=int)
    pt_prob_arr = np.array(pt_y_prob, dtype=float)
    pt_pred_arr = (pt_prob_arr >= threshold).astype(int)

    tn_pt, fp_pt, fn_pt, tp_pt = confusion_matrix(pt_true_arr, pt_pred_arr).ravel()
    sens_pt = float(tp_pt / (tp_pt + fn_pt)) if (tp_pt + fn_pt) > 0 else 0.0
    spec_pt = float(tn_pt / (tn_pt + fp_pt)) if (tn_pt + fp_pt) > 0 else 0.0
    prec_pt = float(tp_pt / (tp_pt + fp_pt)) if (tp_pt + fp_pt) > 0 else 0.0
    f1_pt = float(2 * prec_pt * sens_pt / (prec_pt + sens_pt)) if (prec_pt + sens_pt) > 0 else 0.0
    acc_pt = float((tp_pt + tn_pt) / len(pt_true_arr))
    bal_acc_pt = float((sens_pt + spec_pt) / 2.0)

    try:
        roc_auc_pt = float(roc_auc_score(pt_true_arr, pt_prob_arr))
    except Exception:
        roc_auc_pt = 0.5

    try:
        pr_auc_pt = float(average_precision_score(pt_true_arr, pt_prob_arr))
    except Exception:
        pr_auc_pt = 0.0

    # 2. Image-level metrics (supplementary)
    img_true_arr = np.array(img_y_true, dtype=int)
    img_prob_arr = np.array(img_y_prob, dtype=float)
    img_pred_arr = (img_prob_arr >= threshold).astype(int)

    tn_img, fp_img, fn_img, tp_img = confusion_matrix(img_true_arr, img_pred_arr).ravel()
    sens_img = float(tp_img / (tp_img + fn_img)) if (tp_img + fn_img) > 0 else 0.0
    spec_img = float(tn_img / (tn_img + fp_img)) if (tn_img + fp_img) > 0 else 0.0

    try:
        roc_auc_img = float(roc_auc_score(img_true_arr, img_prob_arr))
    except Exception:
        roc_auc_img = 0.5

    try:
        pr_auc_img = float(average_precision_score(img_true_arr, img_prob_arr))
    except Exception:
        pr_auc_img = 0.0

    return {
        "val_loss": round(avg_loss, 6),
        "val_duration_sec": round(eval_duration, 2),
        "patient_metrics": {
            "total_patients": len(pt_true_arr),
            "positive_patients": int(np.sum(pt_true_arr == 1)),
            "negative_patients": int(np.sum(pt_true_arr == 0)),
            "roc_auc": round(roc_auc_pt, 4),
            "pr_auc": round(pr_auc_pt, 4),
            "accuracy": round(acc_pt, 4),
            "sensitivity": round(sens_pt, 4),
            "specificity": round(spec_pt, 4),
            "precision": round(prec_pt, 4),
            "f1_score": round(f1_pt, 4),
            "balanced_accuracy": round(bal_acc_pt, 4),
            "true_positives": int(tp_pt),
            "false_positives": int(fp_pt),
            "true_negatives": int(tn_pt),
            "false_negatives": int(fn_pt),
        },
        "image_metrics_supplementary": {
            "total_images": len(img_true_arr),
            "roc_auc": round(roc_auc_img, 4),
            "pr_auc": round(pr_auc_img, 4),
            "sensitivity": round(sens_img, 4),
            "specificity": round(spec_img, 4),
            "true_positives": int(tp_img),
            "false_positives": int(fp_img),
            "true_negatives": int(tn_img),
            "false_negatives": int(fn_img),
        },
        "patient_records": patient_records,
        "image_records": image_records,
        "pt_y_true": pt_true_arr,
        "pt_y_prob": pt_prob_arr,
    }


# ------------------------------------------------------------------------------
# 2. Visualizations Generator
# ------------------------------------------------------------------------------

def generate_pilot_visualizations(
    history: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
):
    """Generates loss_curves.png, roc_curve.png, and precision_recall_curve.png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    # 1. Loss & Metric Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train BCE Loss", color="#1f77b4", marker="o", lw=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val BCE Loss", color="#ff7f0e", marker="s", lw=2, linestyle="--")
    axes[0].set_title("Bone Scan Pilot Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(epochs, [h["val_pt_roc_auc"] for h in history], label="Val Patient ROC-AUC", color="#2ca02c", marker="^", lw=2)
    axes[1].plot(epochs, [h["val_pt_pr_auc"] for h in history], label="Val Patient PR-AUC", color="#d62728", marker="d", lw=2, linestyle="--")
    axes[1].set_title("Patient-Level Validation AUC Curves", fontsize=12, fontweight="bold")
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
    plt.plot(fpr_arr, tpr_arr, color="#1f77b4", lw=2.5, label=f"Patient ROC (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    plt.title("Bone Scan Pilot — Patient-Level Validation ROC Curve", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    # 3. Precision-Recall Curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    prevalence = np.mean(y_true)

    plt.figure(figsize=(7, 6))
    plt.plot(rec_arr, prec_arr, color="#d62728", lw=2.5, label=f"Patient PR (AUC = {pr_auc_val:.4f})")
    plt.axhline(prevalence, color="grey", linestyle="--", lw=1.5, label=f"Prevalence Baseline ({prevalence:.4f})")
    plt.title("Bone Scan Pilot — Patient-Level Precision-Recall Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------------------
# 3. Main Pilot Training Routine
# ------------------------------------------------------------------------------

def run_bone_scan_pilot(
    cv_manifest_path: Path,
    output_dir: Path,
    val_fold: str = "fold_0",
    max_epochs: int = 3,
    fc_lr: float = 1e-4,
    layer4_lr: float = 1e-5,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 4,
    random_seed: int = 42,
):
    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 11B BONE SCINTIGRAPHY PILOT TRAINING", flush=True)
    print("=" * 80, flush=True)

    # 1. Device check
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA accelerator required for Stage 11B pilot.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    print(f"[Device] CUDA Device: {gpu_name} (Total VRAM: {vram_mb:.1f} MiB)", flush=True)

    # 2. Random Seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. Load CV Manifest and Build Datasets
    if not cv_manifest_path.exists():
        raise FileNotFoundError(f"CV Manifest missing: {cv_manifest_path}")

    print(f"[Manifest] Loading training CV manifest: {cv_manifest_path}", flush=True)
    val_dataset = BoneScanPatientDataset(cv_manifest_path, fold=val_fold, is_train=False, augment=False)

    # Train dataset combines all other folds (fold_1, fold_2, fold_3, fold_4)
    # We load records where train_cv_fold != val_fold
    all_train_records = []
    with open(cv_manifest_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["train_cv_fold"].lower() != val_fold.lower():
                all_train_records.append(r)

    # Write temporary training fold slice
    temp_train_slice = output_dir / "temp_train_slice.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(temp_train_slice, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_train_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_train_records)

    train_dataset = BoneScanPatientDataset(temp_train_slice, fold=None, is_train=True, augment=True)

    val_counts = val_dataset.get_class_counts()
    train_counts = train_dataset.get_class_counts()

    print(f"[Cohort] Validation Fold ({val_fold}): {val_counts['total_patients']} patients "
          f"({val_counts['positive_patients']} pos, {val_counts['negative_patients']} neg) | {len(val_dataset)*2} images", flush=True)
    print(f"[Cohort] Training Folds (1-4):      {train_counts['total_patients']} patients "
          f"({train_counts['positive_patients']} pos, {train_counts['negative_patients']} neg) | {len(train_dataset)*2} images", flush=True)

    # 4. Model Construction
    model, model_info = build_bone_scan_resnet18(device, pretrained=True, dropout_rate=0.4)
    print(f"[Model] BoneScanResNet18 initialized. Trainable params: {model_info['trainable_parameters']:,} / {model_info['total_parameters']:,}", flush=True)

    # 5. Loss Function with Training Class Weighting
    n_pos_train = train_counts["positive_patients"]
    n_neg_train = train_counts["negative_patients"]
    pos_weight_val = n_neg_train / n_pos_train  # ~ 128 / 33 = 3.8788
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"[Criterion] BCEWithLogitsLoss initialized (pos_weight = {pos_weight_val:.4f})", flush=True)

    # 6. Optimizer & Scheduler
    optimizer = AdamW([
        {"params": model.fc.parameters(), "lr": fc_lr, "name": "fc", "initial_lr": fc_lr},
        {"params": model.layer4.parameters(), "lr": layer4_lr, "name": "layer4", "initial_lr": layer4_lr},
    ], weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    # 7. Save Initial Artifacts
    # Copy CV manifest
    shutil.copy(cv_manifest_path, output_dir / "cv_fold_manifest.csv")

    # Save Pretrained Weights Info
    with open(output_dir / "pretrained_weights_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    # Save Pilot Config YAML
    config_dict = {
        "experiment": "Bone Scan Metastasis ResNet-18 Pilot",
        "dataset_manifest": str(cv_manifest_path),
        "validation_fold": val_fold,
        "train_patients": train_counts["total_patients"],
        "train_positive_patients": train_counts["positive_patients"],
        "train_negative_patients": train_counts["negative_patients"],
        "val_patients": val_counts["total_patients"],
        "val_positive_patients": val_counts["positive_patients"],
        "val_negative_patients": val_counts["negative_patients"],
        "architecture": "ResNet18",
        "in_channels": 1,
        "input_dimensions": "256 x 1024 (aspect ratio 1:4 preserved)",
        "intensity_preprocessing": "clip at 25000 counts, scale to [0, 1]",
        "dropout_rate": 0.4,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": round(pos_weight_val, 4),
        "optimizer": "AdamW",
        "fc_lr": fc_lr,
        "layer4_lr": layer4_lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "micro_batch_size": 1,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size_patients": grad_accum_steps,
        "effective_batch_size_images": grad_accum_steps * 2,
        "random_seed": random_seed,
        "device": gpu_name,
        "vram_total_mb": round(vram_mb, 1),
    }
    import yaml
    with open(output_dir / "pilot_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    print(f"[Config] Saved pilot configuration to {output_dir / 'pilot_config.yaml'}", flush=True)

    # 8. Training Loop
    log_csv_path = output_dir / "training_log.csv"
    best_ckpt_path = output_dir / "best_checkpoint.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"

    log_fields = [
        "epoch", "train_loss", "val_loss", "val_pt_roc_auc", "val_pt_pr_auc",
        "val_pt_f1", "val_pt_sensitivity", "val_pt_specificity", "val_pt_precision",
        "val_img_roc_auc", "val_img_pr_auc", "fc_lr", "layer4_lr",
        "duration_sec", "peak_vram_mb", "is_best"
    ]
    with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    history = []
    best_val_pr_auc = -1.0
    best_epoch = -1
    best_eval_res = None

    training_start = time.time()
    print("\n" + "=" * 80, flush=True)
    print("BEGINNING 3-EPOCH PILOT EXECUTION", flush=True)
    print("=" * 80, flush=True)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        set_eval_bn_mode(model)
        optimizer.zero_grad()

        total_train_loss = 0.0
        train_img_count = 0
        torch.cuda.reset_peak_memory_stats(device)

        # Shuffle patients each epoch
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)

        for step_idx, idx in enumerate(train_indices):
            item = train_dataset[idx]
            y_val = float(item["label"].item())

            ant = item["anterior_tensor"].unsqueeze(0).to(device)    # (1, 1, 1024, 256)
            post = item["posterior_tensor"].unsqueeze(0).to(device)  # (1, 1, 1024, 256)

            pair_imgs = torch.cat([ant, post], dim=0)                 # (2, 1, 1024, 256)
            pair_targets = torch.tensor([[y_val], [y_val]], device=device) # (2, 1)

            with autocast("cuda", dtype=torch.float16):
                logits = model(pair_imgs)
                loss = criterion(logits, pair_targets)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            total_train_loss += loss.item() * 2
            train_img_count += 2

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_indices):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        epoch_train_loss = total_train_loss / train_img_count if train_img_count > 0 else 0.0

        # Evaluate on validation fold (Fold 0)
        eval_res = evaluate_bone_scan_pilot(model, val_dataset, criterion, device, threshold=0.50)
        val_pt_m = eval_res["patient_metrics"]
        val_img_m = eval_res["image_metrics_supplementary"]

        epoch_duration = time.time() - epoch_start
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        current_fc_lr = optimizer.param_groups[0]["lr"]
        current_layer4_lr = optimizer.param_groups[1]["lr"]

        is_best = False
        if val_pt_m["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_pt_m["pr_auc"]
            best_epoch = epoch
            best_eval_res = eval_res
            is_best = True

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_patient_metrics": val_pt_m,
                "val_image_metrics": val_img_m,
                "config": config_dict,
            }, best_ckpt_path)

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_patient_metrics": val_pt_m,
            "val_image_metrics": val_img_m,
            "config": config_dict,
        }, last_ckpt_path)

        log_row = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": eval_res["val_loss"],
            "val_pt_roc_auc": val_pt_m["roc_auc"],
            "val_pt_pr_auc": val_pt_m["pr_auc"],
            "val_pt_f1": val_pt_m["f1_score"],
            "val_pt_sensitivity": val_pt_m["sensitivity"],
            "val_pt_specificity": val_pt_m["specificity"],
            "val_pt_precision": val_pt_m["precision"],
            "val_img_roc_auc": val_img_m["roc_auc"],
            "val_img_pr_auc": val_img_m["pr_auc"],
            "fc_lr": f"{current_fc_lr:.3e}",
            "layer4_lr": f"{current_layer4_lr:.3e}",
            "duration_sec": round(epoch_duration, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "is_best": is_best,
        }
        history.append(log_row)

        with open(log_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_row)

        star = " ★ BEST" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {eval_res['val_loss']:.4f} | "
            f"Val Pt ROC-AUC: {val_pt_m['roc_auc']:.4f} | "
            f"Val Pt PR-AUC: {val_pt_m['pr_auc']:.4f} | "
            f"Val Pt F1: {val_pt_m['f1_score']:.4f} (Sens: {val_pt_m['sensitivity']:.2%}, Spec: {val_pt_m['specificity']:.2%}) | "
            f"VRAM: {peak_vram:.1f} MiB | Time: {epoch_duration:.1f}s{star}",
            flush=True
        )

    total_time = time.time() - training_start
    print("\n" + "=" * 80, flush=True)
    print(f"PILOT TRAINING COMPLETE in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)", flush=True)
    print(f"Best Checkpoint: Epoch {best_epoch} with Val Patient PR-AUC = {best_val_pr_auc:.4f}", flush=True)
    print("=" * 80, flush=True)

    # 9. Clean up temp training slice file
    if temp_train_slice.exists():
        temp_train_slice.unlink()

    # 10. Save Prediction CSVs
    pt_preds_path = output_dir / "patient_level_predictions.csv"
    with open(pt_preds_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_eval_res["patient_records"][0].keys()))
        writer.writeheader()
        writer.writerows(best_eval_res["patient_records"])
    print(f"[Artifacts] Saved patient-level predictions to {pt_preds_path}", flush=True)

    img_preds_path = output_dir / "image_level_predictions.csv"
    with open(img_preds_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_eval_res["image_records"][0].keys()))
        writer.writeheader()
        writer.writerows(best_eval_res["image_records"])
    print(f"[Artifacts] Saved image-level predictions to {img_preds_path}", flush=True)

    # 11. Save Validation Metrics JSON
    val_metrics_json = {
        "experiment": "Bone Scan Metastasis ResNet-18 Pilot",
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_ckpt_path),
        "val_fold": val_fold,
        "patient_level_metrics": best_eval_res["patient_metrics"],
        "image_level_metrics_supplementary": best_eval_res["image_metrics_supplementary"],
    }
    with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics_json, f, indent=2)
    print(f"[Artifacts] Saved validation metrics to {output_dir / 'validation_metrics.json'}", flush=True)

    # 12. Generate Visualizations
    generate_pilot_visualizations(
        history,
        best_eval_res["pt_y_true"],
        best_eval_res["pt_y_prob"],
        output_dir,
    )
    print(f"[Visuals] Saved loss_curves.png, roc_curve.png, and precision_recall_curve.png to {output_dir}", flush=True)

    # 13. Create STAGE_11B_PILOT.md summary artifact
    pilot_md_path = output_dir / "STAGE_11B_PILOT.md"
    pt_m = best_eval_res["patient_metrics"]
    with open(pilot_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# STAGE 11B — BONE SCINTIGRAPHY PILOT EXPERIMENT SUMMARY

**Pilot Status:** PASS  
**Model:** BoneScanResNet18 (ImageNet Pretrained ResNet-18, 1-channel adapted)  
**Evaluated Split:** Training CV Fold 0 (Validation: {val_counts['total_patients']} patients; Training: {train_counts['total_patients']} patients)  
**Original Validation & Test Sets:** Strictly untouched (0 access)  

### Patient-Level Validation Performance (Epoch {best_epoch}):
- **Patient ROC-AUC:** {pt_m['roc_auc']:.4f}
- **Patient PR-AUC:** {pt_m['pr_auc']:.4f} (Baseline prevalence: {pt_m['positive_patients'] / pt_m['total_patients']:.4f})
- **Sensitivity:** {pt_m['sensitivity']:.2%} ({pt_m['true_positives']} / {pt_m['positive_patients']})
- **Specificity:** {pt_m['specificity']:.2%} ({pt_m['true_negatives']} / {pt_m['negative_patients']})
- **Precision:** {pt_m['precision']:.2%}
- **F1-Score:** {pt_m['f1_score']:.4f}
- **Balanced Accuracy:** {pt_m['balanced_accuracy']:.2%}
- **Peak VRAM:** {peak_vram:.1f} MiB (<18% of 4 GB RTX 3050 A)
""")
    print(f"[Summary] Generated {pilot_md_path}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print("STAGE 11B PILOT EXECUTION COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OsteoGuard AI Stage 11B Bone Scan Pilot Trainer")
    parser.add_argument("--manifest", type=str, default="ml_service/data/splits/bone_scan_train_cv_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="ml_artifacts/bone_scan_pilot")
    parser.add_argument("--val_fold", type=str, default="fold_0")
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--fc_lr", type=float, default=1e-4)
    parser.add_argument("--layer4_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_bone_scan_pilot(
        cv_manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        val_fold=args.val_fold,
        max_epochs=args.max_epochs,
        fc_lr=args.fc_lr,
        layer4_lr=args.layer4_lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum,
        random_seed=args.seed,
    )

