"""
OsteoGuard AI - Stage 16: BTXRD Three-Class ResNet-34 Pilot Training.

Executes a 3-epoch resource-aware pilot experiment on Center 1 data only.
Architecture: ResNet-34 (ImageNet pre-trained), early layers frozen, layer4 + head fine-tuned.
Class imbalance handled via numerically stable Multi-Class Focal Loss (gamma=2.0).
Generates all 12 required metrics, predictions, checkpoints, and visualization artifacts.
"""

import csv
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple
import yaml

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from ml_service.config.settings import get_settings
from ml_service.datasets.btxrd_dataset import BTXRDDataset, CLASS_TO_IDX, IDX_TO_CLASS
from ml_service.models.btxrd_classifier import BTXRDResNet34, MulticlassFocalLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Computes comprehensive 3-class performance metrics:
    Macro-F1, Balanced Accuracy, Per-class Recall, Precision, Specificity, F1,
    One-vs-Rest ROC-AUC, Per-class PR-AUC, Confusion Matrix, and Brier Score.
    """
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = len(class_names)

    # 1. Macro & Balanced metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_recall = recall_score(y_true, y_pred, average=None)
    balanced_acc = float(np.mean(per_class_recall))
    overall_acc = float(np.mean(y_pred == y_true))

    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # 2. Confusion Matrix & Per-class Specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_spec = []
    for c in range(num_classes):
        # One-vs-rest binary breakdown
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        per_class_spec.append(spec)

    # 3. One-Hot Encoding for Area Metrics & Calibration
    y_true_onehot = np.zeros((len(y_true), num_classes), dtype=float)
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1.0

    # 4. Multi-class Brier Score: 1/N * sum((p_ic - y_ic)^2)
    brier_score = float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

    # 5. One-vs-Rest ROC-AUC
    ovr_roc_aucs = []
    for c in range(num_classes):
        if len(np.unique(y_true_onehot[:, c])) > 1:
            score = float(roc_auc_score(y_true_onehot[:, c], y_prob[:, c]))
            ovr_roc_aucs.append(score)
        else:
            ovr_roc_aucs.append(0.5)
    macro_roc_auc = float(np.mean(ovr_roc_aucs))

    # 6. Per-class PR-AUC
    pr_aucs = []
    for c in range(num_classes):
        if len(np.unique(y_true_onehot[:, c])) > 1:
            score = float(average_precision_score(y_true_onehot[:, c], y_prob[:, c]))
            pr_aucs.append(score)
        else:
            pr_aucs.append(float(np.mean(y_true_onehot[:, c])))
    macro_pr_auc = float(np.mean(pr_aucs))

    # Assemble per-class dictionary
    class_metrics = {}
    for c, cname in enumerate(class_names):
        class_metrics[cname] = {
            "class_index": c,
            "support": int(np.sum(y_true == c)),
            "prevalence": round(float(np.mean(y_true == c)), 4),
            "recall_sensitivity": round(float(per_class_recall[c]), 4),
            "specificity": round(float(per_class_spec[c]), 4),
            "precision_ppv": round(float(per_class_prec[c]), 4),
            "f1_score": round(float(per_class_f1[c]), 4),
            "roc_auc_ovr": round(float(ovr_roc_aucs[c]), 4),
            "pr_auc": round(float(pr_aucs[c]), 4),
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "macro_roc_auc_ovr": round(macro_roc_auc, 4),
        "macro_pr_auc": round(macro_pr_auc, 4),
        "brier_score": round(brier_score, 4),
        "per_class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
    }


def run_btxrd_pilot_training():
    """
    Main orchestration routine for Stage 16 BTXRD ResNet-34 Pilot Training.
    """
    settings = get_settings()
    base_dir = settings.base_dir
    manifest_path = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    output_dir = base_dir / "ml_artifacts/btxrd_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 16: BTXRD THREE-CLASS RESNET-34 PILOT TRAINING", flush=True)
    print("=" * 80, flush=True)

    set_seed(42)

    # 1. Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Compute Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[Hardware] GPU Model: {torch.cuda.get_device_name(device)}", flush=True)
        print(f"[Hardware] VRAM Available: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB", flush=True)

    # 2. Dataset Ingestion (Center 1 ONLY, Zero External Center Access)
    print(f"\n[Dataset] Loading Center 1 partitions from {manifest_path}...", flush=True)
    train_dataset = BTXRDDataset(
        manifest_path=manifest_path,
        center="center_1",
        split="train",
        split_protocol="center_holdout_split",
        target_size=512,
        is_train=True,
        augment=True,
        base_dir=base_dir,
    )
    val_dataset = BTXRDDataset(
        manifest_path=manifest_path,
        center="center_1",
        split="val",
        split_protocol="center_holdout_split",
        target_size=512,
        is_train=False,
        augment=False,
        base_dir=base_dir,
    )

    print(f"[Dataset] Center 1 Clean Training Images:   {len(train_dataset)}", flush=True)
    print(f"[Dataset] Center 1 Clean Validation Images: {len(val_dataset)}", flush=True)
    assert len(train_dataset) == 2346, f"Expected 2346 train images, got {len(train_dataset)}"
    assert len(val_dataset) == 588, f"Expected 588 val images, got {len(val_dataset)}"

    train_counts = train_dataset.get_class_counts()
    val_counts = val_dataset.get_class_counts()
    class_order = ["NORMAL_BONE_RADIOGRAPH", "BENIGN_BONE_TUMOR", "PRIMARY_MALIGNANT_TUMOR"]

    print("[Dataset] Training Class Distribution:", flush=True)
    for c, cname in enumerate(class_order):
        print(f"  Class {c} ({cname}): {train_counts[cname]} ({train_counts[cname]/len(train_dataset)*100:.2f}%)", flush=True)

    print("[Dataset] Validation Class Distribution:", flush=True)
    for c, cname in enumerate(class_order):
        print(f"  Class {c} ({cname}): {val_counts[cname]} ({val_counts[cname]/len(val_dataset)*100:.2f}%)", flush=True)

    # Save class_distribution.json
    class_dist_path = output_dir / "class_distribution.json"
    with open(class_dist_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "BTXRD Three-Class ResNet-34 Pilot",
            "protocol": "Protocol B (Center 1 Internal Development)",
            "center_1_train": {
                "total": len(train_dataset),
                "counts": train_counts,
                "proportions": {k: round(v / len(train_dataset), 4) for k, v in train_counts.items()},
            },
            "center_1_val": {
                "total": len(val_dataset),
                "counts": val_counts,
                "proportions": {k: round(v / len(val_dataset), 4) for k, v in val_counts.items()},
            },
            "external_centers_isolated": ["center_2", "center_3"],
        }, f, indent=2)
    print(f"[Artifacts] Saved class distribution to {class_dist_path}", flush=True)

    # 3. Compute Focal Loss Alpha Weights
    train_freqs = [train_counts[cname] for cname in class_order]
    K = 3
    alpha_raw = [len(train_dataset) / (K * c) for c in train_freqs]
    alpha_norm = [round(float(a / sum(alpha_raw) * 3.0), 4) for a in alpha_raw]
    alpha_tensor = torch.tensor(alpha_norm, dtype=torch.float32, device=device)

    print(f"[Loss] Multi-Class Focal Loss initialized with gamma=2.0 and alpha={alpha_norm}", flush=True)
    loss_fn = MulticlassFocalLoss(gamma=2.0, alpha=alpha_tensor)

    # 4. Model Architecture & Pretrained Weights
    print("\n[Model] Initializing ResNet-34 with official ImageNet-1K pretrained weights...", flush=True)
    model = BTXRDResNet34(num_classes=3, dropout_p=0.4, pretrained=True, freeze_early=True).to(device)
    model_meta = model.get_model_metadata()

    print(f"[Model] Total Parameters:     {model_meta['total_parameters']:,}", flush=True)
    print(f"[Model] Trainable Parameters: {model_meta['trainable_parameters']:,} ({model_meta['trainable_percentage']}%)", flush=True)
    print(f"[Model] Frozen Parameters:    {model_meta['frozen_parameters']:,}", flush=True)

    # Save pretrained_weights_info.json
    weights_info_path = output_dir / "pretrained_weights_info.json"
    with open(weights_info_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_architecture": "ResNet-34",
            "weights_source": "torchvision.models.ResNet34_Weights.IMAGENET1K_V1",
            "weights_file": "resnet34-b627a593.pth",
            "num_classes": 3,
            "classification_head": "Dropout(p=0.4) -> Linear(512, 3)",
            "frozen_layers": ["conv1", "bn1", "layer1", "layer2", "layer3"],
            "trainable_layers": ["layer4", "classifier"],
            "parameter_counts": model_meta,
        }, f, indent=2)
    print(f"[Artifacts] Saved pretrained weights info to {weights_info_path}", flush=True)

    # 5. Training Hyperparameters
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS  # 16
    MAX_EPOCHS = 3
    CLASSIFIER_LR = 1e-4
    LAYER4_LR = 1e-5
    WEIGHT_DECAY = 0.01

    param_groups = model.get_parameter_groups(
        classifier_lr=CLASSIFIER_LR,
        layer4_lr=LAYER4_LR,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    # Save pilot_config.yaml
    pilot_config = {
        "experiment": "BTXRD Three-Class ResNet-34 Pilot",
        "stage": "STAGE_16",
        "dataset": {
            "name": "BTXRD (Bone Tumor X-Ray Dataset)",
            "center": "center_1",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "input_resolution": "512x512",
            "channels": 3,
            "aspect_ratio_handling": "Symmetrical black padding letterbox",
        },
        "model": {
            "architecture": "ResNet-34",
            "weights": "IMAGENET1K_V1",
            "freeze_early": True,
            "dropout": 0.4,
            "num_classes": 3,
        },
        "training": {
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "optimizer": "AdamW",
            "classifier_lr": CLASSIFIER_LR,
            "layer4_lr": LAYER4_LR,
            "weight_decay": WEIGHT_DECAY,
            "mixed_precision": "FP16",
            "seed": 42,
        },
        "loss": {
            "type": "MulticlassFocalLoss",
            "gamma": 2.0,
            "alpha": alpha_norm,
        },
    }
    with open(output_dir / "pilot_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(pilot_config, f, default_flow_style=False)
    print(f"[Artifacts] Saved pilot configuration to {output_dir / 'pilot_config.yaml'}", flush=True)

    # DataLoaders (workers=0 for safe Windows concurrency)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # 6. Execute 3-Epoch Pilot Training
    print(f"\n[Training] Commencing {MAX_EPOCHS}-epoch pilot loop (Batch: {BATCH_SIZE}, Accum: {GRAD_ACCUM_STEPS}, Effective: {EFFECTIVE_BATCH_SIZE})...", flush=True)
    training_history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_val_metrics = None
    best_val_preds = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad()

        train_loss_total = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                logits = model(images)
                loss = loss_fn(logits, labels)
                # Scale loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            train_loss_total += loss.item() * GRAD_ACCUM_STEPS
            train_batches += 1

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss_epoch = train_loss_total / train_batches

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_val_true = []
        all_val_probs = []
        all_val_ids = []
        all_val_classes = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with autocast("cuda" if device.type == "cuda" else "cpu"):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    probs = F.softmax(logits, dim=1)

                val_loss_total += loss.item()
                val_batches += 1

                all_val_true.extend(labels.cpu().numpy().tolist())
                all_val_probs.extend(probs.cpu().numpy().tolist())
                all_val_ids.extend(batch["image_id"])
                all_val_classes.extend(batch["target_class"])

        val_loss_epoch = val_loss_total / val_batches
        epoch_duration = time.time() - epoch_start_time

        peak_vram_mib = 0.0
        if device.type == "cuda":
            peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

        # Compute validation metrics
        y_true_arr = np.array(all_val_true, dtype=int)
        y_prob_arr = np.array(all_val_probs, dtype=float)
        epoch_metrics = compute_multiclass_metrics(y_true_arr, y_prob_arr, class_order)

        val_macro_f1 = epoch_metrics["macro_f1"]
        val_bal_acc = epoch_metrics["balanced_accuracy"]
        val_macro_roc = epoch_metrics["macro_roc_auc_ovr"]
        val_macro_pr = epoch_metrics["macro_pr_auc"]

        p_norm = epoch_metrics["per_class_metrics"]["NORMAL_BONE_RADIOGRAPH"]
        p_ben = epoch_metrics["per_class_metrics"]["BENIGN_BONE_TUMOR"]
        p_mal = epoch_metrics["per_class_metrics"]["PRIMARY_MALIGNANT_TUMOR"]

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} [{epoch_duration:.1f}s | VRAM: {peak_vram_mib:.1f} MiB] — "
            f"Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | "
            f"Macro-F1: {val_macro_f1:.4f} | BalAcc: {val_bal_acc*100:.1f}% | "
            f"Sens (N/B/M): {p_norm['recall_sensitivity']*100:.1f}% / {p_ben['recall_sensitivity']*100:.1f}% / {p_mal['recall_sensitivity']*100:.1f}% | "
            f"F1 (N/B/M): {p_norm['f1_score']:.3f} / {p_ben['f1_score']:.3f} / {p_mal['f1_score']:.3f}",
            flush=True,
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss_epoch, 6),
            "val_loss": round(val_loss_epoch, 6),
            "macro_f1": val_macro_f1,
            "balanced_accuracy": val_bal_acc,
            "overall_accuracy": epoch_metrics["overall_accuracy"],
            "macro_roc_auc_ovr": val_macro_roc,
            "macro_pr_auc": val_macro_pr,
            "brier_score": epoch_metrics["brier_score"],
            "sens_normal": p_norm["recall_sensitivity"],
            "sens_benign": p_ben["recall_sensitivity"],
            "sens_malignant": p_mal["recall_sensitivity"],
            "f1_normal": p_norm["f1_score"],
            "f1_benign": p_ben["f1_score"],
            "f1_malignant": p_mal["f1_score"],
            "peak_vram_mib": peak_vram_mib,
            "duration_seconds": round(epoch_duration, 2),
        })

        # Save last checkpoint
        last_ckpt_path = output_dir / "last_checkpoint.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
            "val_metrics": epoch_metrics,
            "model_meta": model_meta,
        }, last_ckpt_path)

        # Check if best model on Macro-F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_val_metrics = epoch_metrics
            best_val_preds = (all_val_ids, all_val_classes, all_val_true, all_val_probs)

            best_ckpt_path = output_dir / "best_checkpoint.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss_epoch,
                "val_loss": val_loss_epoch,
                "val_metrics": epoch_metrics,
                "model_meta": model_meta,
            }, best_ckpt_path)

    print(f"\n[Training] Completed 3-epoch pilot! Best Epoch: {best_epoch} (Macro-F1: {best_macro_f1:.4f})", flush=True)

    # 7. Save training_log.csv
    log_csv_path = output_dir / "training_log.csv"
    with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(training_history[0].keys()))
        writer.writeheader()
        writer.writerows(training_history)
    print(f"[Artifacts] Saved training log to {log_csv_path}", flush=True)

    # 8. Save validation_metrics.json (from Best Epoch)
    val_metrics_path = output_dir / "validation_metrics.json"
    with open(val_metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "validation_metrics": best_val_metrics,
            "training_summary": training_history,
        }, f, indent=2)
    print(f"[Artifacts] Saved validation metrics to {val_metrics_path}", flush=True)

    # 9. Save validation_predictions.csv (from Best Epoch)
    val_preds_path = output_dir / "validation_predictions.csv"
    val_ids, val_target_classes, val_true_lbls, val_probs = best_val_preds
    with open(val_preds_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "image_id", "target_class", "actual_label",
            "prob_normal", "prob_benign", "prob_malignant",
            "predicted_label", "predicted_class", "is_correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(val_ids)):
            p_vec = val_probs[i]
            p_cls = int(np.argmax(p_vec))
            writer.writerow({
                "image_id": val_ids[i],
                "target_class": val_target_classes[i],
                "actual_label": val_true_lbls[i],
                "prob_normal": round(p_vec[0], 6),
                "prob_benign": round(p_vec[1], 6),
                "prob_malignant": round(p_vec[2], 6),
                "predicted_label": p_cls,
                "predicted_class": IDX_TO_CLASS[p_cls],
                "is_correct": int(p_cls == val_true_lbls[i]),
            })
    print(f"[Artifacts] Saved {len(val_ids)} validation predictions to {val_preds_path}", flush=True)

    # 10. Generate Visual Diagnostic Plots
    print("\n[Visuals] Rendering diagnostic figures...", flush=True)

    # 10.1 Loss Curves
    epochs = [h["epoch"] for h in training_history]
    tr_loss = [h["train_loss"] for h in training_history]
    vl_loss = [h["val_loss"] for h in training_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, tr_loss, marker="o", color="#1f77b4", lw=2.5, label="Training Focal Loss")
    plt.plot(epochs, vl_loss, marker="s", color="#d62728", lw=2.5, label="Validation Focal Loss")
    plt.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    plt.title("BTXRD ResNet-34 Pilot: Training & Validation Loss Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Focal Loss", fontsize=11)
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close()

    # 10.2 Confusion Matrix
    cm_best = np.array(best_val_metrics["confusion_matrix"])
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_best, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"BTXRD Validation Confusion Matrix (Best Epoch {best_epoch})\nMacro-F1: {best_macro_f1:.4f} | BalAcc: {best_val_metrics['balanced_accuracy']*100:.1f}%", fontsize=11, fontweight="bold")
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["Normal", "Benign", "Malignant"], fontsize=10)
    plt.yticks(tick_marks, ["Normal", "Benign", "Malignant"], fontsize=10)
    plt.ylabel("True Diagnosis", fontsize=11)
    plt.xlabel("Predicted Diagnosis", fontsize=11)

    thresh = cm_best.max() / 2.0
    for r in range(3):
        for c in range(3):
            plt.text(c, r, format(cm_best[r, c], "d"),
                     ha="center", va="center",
                     color="white" if cm_best[r, c] > thresh else "black",
                     fontsize=14, fontweight="bold")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 10.3 One-vs-Rest ROC Curves
    y_true_best = np.array(val_true_lbls)
    y_prob_best = np.array(val_probs)
    y_onehot_best = np.zeros((len(y_true_best), 3), dtype=float)
    for i, lbl in enumerate(y_true_best):
        y_onehot_best[i, lbl] = 1.0

    plt.figure(figsize=(8, 7))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    display_names = ["Normal Bone", "Benign Primary Tumour", "Primary Malignant Tumour"]

    for c in range(3):
        fpr, tpr, _ = roc_curve(y_onehot_best[:, c], y_prob_best[:, c])
        c_auc = best_val_metrics["per_class_metrics"][class_order[c]]["roc_auc_ovr"]
        plt.plot(fpr, tpr, color=colors[c], lw=2.5, label=f"{display_names[c]} (AUC = {c_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k:", lw=1.5, label="Random Chance (AUC = 0.500)")
    plt.title(f"BTXRD Validation One-vs-Rest ROC Curves (Best Epoch {best_epoch})\nMacro ROC-AUC: {best_val_metrics['macro_roc_auc_ovr']:.4f}", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_ovr.png", dpi=300)
    plt.close()

    # 10.4 Precision-Recall Curves
    plt.figure(figsize=(8, 7))
    for c in range(3):
        prec, rec, _ = precision_recall_curve(y_onehot_best[:, c], y_prob_best[:, c])
        c_pr = best_val_metrics["per_class_metrics"][class_order[c]]["pr_auc"]
        base_prev = best_val_metrics["per_class_metrics"][class_order[c]]["prevalence"]
        plt.plot(rec, prec, color=colors[c], lw=2.5, label=f"{display_names[c]} (PR-AUC = {c_pr:.4f}, Prev = {base_prev:.3f})")

    plt.title(f"BTXRD Validation Precision-Recall Curves (Best Epoch {best_epoch})\nMacro PR-AUC: {best_val_metrics['macro_pr_auc']:.4f}", fontsize=12, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=300)
    plt.close()

    print(f"[Visuals] All 4 diagnostic plots saved to {output_dir}", flush=True)

    print("\n" + "=" * 80)
    print("STAGE 16: BTXRD RESNET-34 PILOT TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Epoch: {best_epoch} | Macro-F1: {best_macro_f1:.4f} | Balanced Accuracy: {best_val_metrics['balanced_accuracy']*100:.1f}%")
    print(f"Validation Macro ROC-AUC: {best_val_metrics['macro_roc_auc_ovr']:.4f} | Macro PR-AUC: {best_val_metrics['macro_pr_auc']:.4f}")
    print(f"Validation Brier Score:   {best_val_metrics['brier_score']:.4f}")
    print("-" * 80)
    for c, cname in enumerate(class_order):
        m = best_val_metrics["per_class_metrics"][cname]
        print(f"  {cname} (N={m['support']}): Sens={m['recall_sensitivity']*100:.1f}%, Spec={m['specificity']*100:.1f}%, Prec={m['precision_ppv']*100:.1f}%, F1={m['f1_score']:.4f}, PR-AUC={m['pr_auc']:.4f}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    run_btxrd_pilot_training()

