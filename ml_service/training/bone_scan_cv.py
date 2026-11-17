"""
OsteoGuard AI - Stage 12: Five-Fold Cross-Validation for Bone Scintigraphy.

Trains independent ResNet-18 classifiers across 5 stratified patient-disjoint folds.
Evaluates out-of-fold patient predictions using clinical max-pooling:
    P(patient positive) = max(P(anterior positive), P(posterior positive))

Calculates fold-wise metrics, pooled out-of-fold metrics, and saves all required artifacts.
"""

import csv
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ml_service.config.settings import get_settings
from ml_service.datasets.bone_scan_dataset import BoneScanPatientDataset
from ml_service.models.bone_scan_classifier import (
    BoneScanResNet18,
    build_bone_scan_resnet18,
    set_eval_bn_mode,
)


def seed_everything(seed: int = 42):
    """Sets random seeds for full reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_fold_validation(
    model: nn.Module,
    val_dataset: BoneScanPatientDataset,
    criterion: nn.Module,
    device: torch.device,
    val_fold_name: str,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates model on the validation fold patients.
    Returns patient-level and image-level metrics, and per-patient out-of-fold records.
    """
    model.eval()
    total_val_loss = 0.0
    val_img_count = 0

    patient_records = []
    pt_y_true = []
    pt_y_prob = []

    img_y_true = []
    img_y_prob = []

    start_time = time.time()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            item = val_dataset[i]
            pid = item["patient_id"]
            y_val = float(item["label"].item())
            tgt_cls = item["target_class"]

            ant = item["anterior_tensor"].unsqueeze(0).to(device)
            post = item["posterior_tensor"].unsqueeze(0).to(device)

            pair_imgs = torch.cat([ant, post], dim=0)
            pair_targets = torch.tensor([[y_val], [y_val]], device=device)

            with autocast("cuda", dtype=torch.float16):
                logits = model(pair_imgs)
                loss = criterion(logits, pair_targets)

            total_val_loss += loss.item() * 2
            val_img_count += 2

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            p_ant = float(probs[0])
            p_post = float(probs[1])
            p_patient = max(p_ant, p_post)

            pt_y_true.append(int(y_val))
            pt_y_prob.append(p_patient)

            img_y_true.extend([int(y_val), int(y_val)])
            img_y_prob.extend([p_ant, p_post])

            pred_label = int(p_patient >= threshold)
            pred_name = "BONE_METASTASIS" if pred_label == 1 else "NO_BONE_METASTASIS"

            patient_records.append({
                "patient_id": pid,
                "val_fold": val_fold_name,
                "target_class": tgt_cls,
                "actual_label": int(y_val),
                "prob_anterior": round(p_ant, 6),
                "prob_posterior": round(p_post, 6),
                "prob_patient_max": round(p_patient, 6),
                "pred_label_050": pred_label,
                "pred_name_050": pred_name,
            })

    eval_duration = time.time() - start_time
    avg_loss = total_val_loss / val_img_count if val_img_count > 0 else 0.0

    pt_true_arr = np.array(pt_y_true, dtype=int)
    pt_prob_arr = np.array(pt_y_prob, dtype=float)
    pt_pred_arr = (pt_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(pt_true_arr, pt_pred_arr).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    acc = float((tp + tn) / len(pt_true_arr))
    bal_acc = float((sens + spec) / 2.0)

    try:
        roc_auc = float(roc_auc_score(pt_true_arr, pt_prob_arr))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(pt_true_arr, pt_prob_arr))
    except Exception:
        pr_auc = 0.0

    metrics = {
        "val_loss": round(avg_loss, 6),
        "val_duration_sec": round(eval_duration, 2),
        "total_patients": len(pt_true_arr),
        "positive_patients": int(np.sum(pt_true_arr == 1)),
        "negative_patients": int(np.sum(pt_true_arr == 0)),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    return {
        "metrics": metrics,
        "patient_records": patient_records,
        "pt_y_true": pt_true_arr,
        "pt_y_prob": pt_prob_arr,
    }


def train_single_fold(
    fold_name: str,
    cv_manifest_path: Path,
    fold_output_dir: Path,
    device: torch.device,
    max_epochs: int = 20,
    patience: int = 5,
    grad_accum_steps: int = 4,
    fc_lr: float = 1e-4,
    layer4_lr: float = 1e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains one fold of 5-fold cross validation with early stopping.
    Returns best fold metrics, best patient predictions, and training history.
    """
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = fold_output_dir / "best_checkpoint.pt"
    last_ckpt_path = fold_output_dir / "last_checkpoint.pt"
    log_csv_path = fold_output_dir / "training_log.csv"
    metrics_json_path = fold_output_dir / "fold_metrics.json"

    print(f"\n================================================================================", flush=True)
    print(f"TRAINING FOLD: {fold_name.upper()} (Validation: {fold_name}, Train: other 4 folds)", flush=True)
    print(f"================================================================================", flush=True)

    # 1. Datasets
    val_dataset = BoneScanPatientDataset(cv_manifest_path, fold=fold_name, is_train=False, augment=False)
    train_dataset = BoneScanPatientDataset(cv_manifest_path, exclude_fold=fold_name, is_train=True, augment=True)

    val_counts = val_dataset.get_class_counts()
    train_counts = train_dataset.get_class_counts()

    print(f"[{fold_name}] Val Cohort:   {val_counts['total_patients']} patients ({val_counts['positive_patients']} pos, {val_counts['negative_patients']} neg)", flush=True)
    print(f"[{fold_name}] Train Cohort: {train_counts['total_patients']} patients ({train_counts['positive_patients']} pos, {train_counts['negative_patients']} neg)", flush=True)

    # 2. Fresh Model Instance
    model, model_info = build_bone_scan_resnet18(device, pretrained=True, dropout_rate=0.4)

    # 3. Loss with fold-specific pos_weight
    n_pos_train = train_counts["positive_patients"]
    n_neg_train = train_counts["negative_patients"]
    pos_weight_val = n_neg_train / n_pos_train
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"[{fold_name}] BCEWithLogitsLoss pos_weight = {pos_weight_val:.4f} ({n_neg_train}/{n_pos_train})", flush=True)

    # 4. Optimizer & Scheduler
    optimizer = AdamW([
        {"params": model.fc.parameters(), "lr": fc_lr, "name": "fc", "initial_lr": fc_lr},
        {"params": model.layer4.parameters(), "lr": layer4_lr, "name": "layer4", "initial_lr": layer4_lr},
    ], weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    # 5. Training Loop with Early Stopping on Val Patient PR-AUC
    best_val_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_eval_res = None
    history = []

    train_indices = list(range(len(train_dataset)))

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        set_eval_bn_mode(model)
        optimizer.zero_grad()

        total_train_loss = 0.0
        train_img_count = 0
        torch.cuda.reset_peak_memory_stats(device)

        # Shuffle indices reproducibly
        random.seed(seed + epoch * 100)
        random.shuffle(train_indices)

        for step_idx, idx in enumerate(train_indices):
            item = train_dataset[idx]
            y_val = float(item["label"].item())

            ant = item["anterior_tensor"].unsqueeze(0).to(device)
            post = item["posterior_tensor"].unsqueeze(0).to(device)

            pair_imgs = torch.cat([ant, post], dim=0)
            pair_targets = torch.tensor([[y_val], [y_val]], device=device)

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

        # Evaluate on validation fold
        eval_res = evaluate_fold_validation(model, val_dataset, criterion, device, val_fold_name=fold_name)
        m = eval_res["metrics"]
        val_pr_auc = m["pr_auc"]
        val_roc_auc = m["roc_auc"]
        val_loss = m["val_loss"]

        epoch_duration = time.time() - epoch_start
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        is_best = False
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_eval_res = eval_res
            patience_counter = 0
            is_best = True

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "fold": fold_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_patient_metrics": m,
                "val_patient_pr_auc": val_pr_auc,
                "val_patient_roc_auc": val_roc_auc,
                "config": {
                    "fc_lr": fc_lr,
                    "layer4_lr": layer4_lr,
                    "weight_decay": weight_decay,
                    "pos_weight": pos_weight_val,
                    "max_epochs": max_epochs,
                    "patience": patience,
                },
            }, best_ckpt_path)
        else:
            patience_counter += 1

        # Always save last checkpoint
        torch.save({
            "epoch": epoch,
            "fold": fold_name,
            "model_state_dict": model.state_dict(),
            "val_patient_metrics": m,
            "val_patient_pr_auc": val_pr_auc,
            "val_patient_roc_auc": val_roc_auc,
        }, last_ckpt_path)

        flag_str = "★ BEST" if is_best else f"(patience: {patience_counter}/{patience})"
        print(f"[{fold_name}] Epoch {epoch:02d}/{max_epochs:02d} | Train: {epoch_train_loss:.4f} | Val: {val_loss:.4f} | "
              f"ROC-AUC: {val_roc_auc:.4f} | PR-AUC: {val_pr_auc:.4f} | F1: {m['f1_score']:.4f} (Sens: {m['sensitivity']*100:.1f}%, Spec: {m['specificity']*100:.1f}%) | "
              f"VRAM: {peak_vram:.1f}MB | {epoch_duration:.1f}s | {flag_str}", flush=True)

        log_row = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": val_loss,
            "val_pt_roc_auc": val_roc_auc,
            "val_pt_pr_auc": val_pr_auc,
            "val_pt_f1": m["f1_score"],
            "val_pt_sensitivity": m["sensitivity"],
            "val_pt_specificity": m["specificity"],
            "val_pt_precision": m["precision"],
            "val_pt_accuracy": m["accuracy"],
            "val_pt_balanced_accuracy": m["balanced_accuracy"],
            "duration_sec": round(epoch_duration, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "is_best": is_best,
        }
        history.append(log_row)

        if patience_counter >= patience:
            print(f"[{fold_name}] Early stopping triggered at Epoch {epoch} (Patience: {patience} epochs reached). Best epoch was {best_epoch}.", flush=True)
            break

    # Save training_log.csv
    with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Save fold_metrics.json (from the best checkpoint)
    best_metrics = best_eval_res["metrics"]
    best_metrics["best_epoch"] = best_epoch
    best_metrics["fold"] = fold_name
    best_metrics["pos_weight"] = round(pos_weight_val, 4)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"[{fold_name}] Complete! Best Epoch: {best_epoch} | Best Val PR-AUC: {best_val_pr_auc:.4f} | ROC-AUC: {best_metrics['roc_auc']:.4f}", flush=True)

    return {
        "fold": fold_name,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "patient_records": best_eval_res["patient_records"],
        "history": history,
        "y_true": best_eval_res["pt_y_true"],
        "y_prob": best_eval_res["pt_y_prob"],
    }


def generate_cv_visualizations(
    fold_results: List[Dict[str, Any]],
    pooled_y_true: np.ndarray,
    pooled_y_prob: np.ndarray,
    output_dir: Path,
):
    """
    Generates:
      1. fold_comparison.png (metric bars per fold + pooled)
      2. pooled_roc_curve.png (pooled ROC curve + dashed fold curves)
      3. pooled_precision_recall_curve.png (pooled PR curve + dashed fold curves + baseline)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [r["fold"] for r in fold_results]

    # Compute pooled metrics
    pooled_pred = (pooled_y_prob >= 0.50).astype(int)
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(pooled_y_true, pooled_pred).ravel()
    pooled_sens = float(tp_p / (tp_p + fn_p))
    pooled_spec = float(tn_p / (tn_p + fp_p))
    pooled_roc_auc = float(roc_auc_score(pooled_y_true, pooled_y_prob))
    pooled_pr_auc = float(average_precision_score(pooled_y_true, pooled_y_prob))

    # 1. Fold Comparison Bar Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("OsteoGuard AI - Bone Scintigraphy 5-Fold Cross-Validation Comparison", fontsize=15, fontweight="bold")

    plot_categories = folds + ["Pooled"]
    roc_aucs = [r["best_metrics"]["roc_auc"] for r in fold_results] + [round(pooled_roc_auc, 4)]
    pr_aucs = [r["best_metrics"]["pr_auc"] for r in fold_results] + [round(pooled_pr_auc, 4)]
    sensitivities = [r["best_metrics"]["sensitivity"] for r in fold_results] + [round(pooled_sens, 4)]
    specificities = [r["best_metrics"]["specificity"] for r in fold_results] + [round(pooled_spec, 4)]

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#d62728"]

    # (a) ROC-AUC
    bars_a = axes[0, 0].bar(plot_categories, roc_aucs, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 0].set_title("Patient-Level ROC-AUC", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars_a:
        h = bar.get_height()
        axes[0, 0].annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (b) PR-AUC
    bars_b = axes[0, 1].bar(plot_categories, pr_aucs, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 1].set_title("Patient-Level PR-AUC", fontsize=12, fontweight="bold")
    axes[0, 1].axhline(42 / 203, color="gray", linestyle=":", label=f"Prevalence Baseline ({42/203:.3f})")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0, 1].legend(loc="upper right")
    for bar in bars_b:
        h = bar.get_height()
        axes[0, 1].annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (c) Sensitivity
    bars_c = axes[1, 0].bar(plot_categories, sensitivities, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("Patient Sensitivity (θ = 0.50)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars_c:
        h = bar.get_height()
        axes[1, 0].annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (d) Specificity
    bars_d = axes[1, 1].bar(plot_categories, specificities, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("Patient Specificity (θ = 0.50)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars_d:
        h = bar.get_height()
        axes[1, 1].annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "fold_comparison.png", dpi=300)
    plt.close()

    # 2. Pooled ROC Curve
    plt.figure(figsize=(8, 7))
    for r in fold_results:
        f_name = r["fold"]
        f_auc = r["best_metrics"]["roc_auc"]
        fpr_f, tpr_f, _ = roc_curve(r["y_true"], r["y_prob"])
        plt.plot(fpr_f, tpr_f, linestyle="--", alpha=0.45, label=f"{f_name} (AUC = {f_auc:.3f})")

    fpr_pool, tpr_pool, _ = roc_curve(pooled_y_true, pooled_y_prob)
    plt.plot(fpr_pool, tpr_pool, color="#d62728", lw=3, label=f"Pooled Out-of-Fold (AUC = {pooled_roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="black", linestyle=":", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.title("Bone Scintigraphy Five-Fold Cross-Validation ROC Curve", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_roc_curve.png", dpi=300)
    plt.close()

    # 3. Pooled Precision-Recall Curve
    plt.figure(figsize=(8, 7))
    for r in fold_results:
        f_name = r["fold"]
        f_pr_auc = r["best_metrics"]["pr_auc"]
        prec_f, rec_f, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        plt.plot(rec_f, prec_f, linestyle="--", alpha=0.45, label=f"{f_name} (PR-AUC = {f_pr_auc:.3f})")

    prec_pool, rec_pool, _ = precision_recall_curve(pooled_y_true, pooled_y_prob)
    plt.plot(rec_pool, prec_pool, color="#1f77b4", lw=3, label=f"Pooled Out-of-Fold (PR-AUC = {pooled_pr_auc:.4f})")
    baseline = 42 / 203
    plt.axhline(baseline, color="black", linestyle=":", lw=1.5, label=f"Prevalence Baseline ({baseline:.3f})")
    plt.title("Bone Scintigraphy Five-Fold Cross-Validation Precision-Recall Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_precision_recall_curve.png", dpi=300)
    plt.close()


def run_stage_12_five_fold_cv():
    """
    Main orchestration routine for Stage 12.
    """
    settings = get_settings()
    base_dir = settings.base_dir
    cv_manifest_path = base_dir / "ml_service/data/splits/bone_scan_train_cv_manifest.csv"
    artifacts_cv_dir = base_dir / "ml_artifacts/bone_scan_cv"
    artifacts_cv_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 12: FIVE-FOLD CROSS-VALIDATION TRAINING", flush=True)
    print("=" * 80, flush=True)

    # Device
    assert torch.cuda.is_available(), "CUDA device unavailable! Cannot proceed with GPU CV training."
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    print(f"[Device] {gpu_name} (Total VRAM: {vram_mb:.1f} MiB)", flush=True)

    # Copy CV manifest to artifacts
    cv_manifest_copy = artifacts_cv_dir / "cv_manifest.csv"
    shutil.copy2(cv_manifest_path, cv_manifest_copy)
    print(f"[Manifest] Copied CV manifest to {cv_manifest_copy}", flush=True)

    # Seed
    seed_everything(42)

    fold_names = [f"fold_{i}" for i in range(5)]
    fold_results = []
    all_oof_records = []

    overall_start_time = time.time()

    # Train each fold
    for fold_name in fold_names:
        fold_dir = artifacts_cv_dir / fold_name
        res = train_single_fold(
            fold_name=fold_name,
            cv_manifest_path=cv_manifest_path,
            fold_output_dir=fold_dir,
            device=device,
            max_epochs=20,
            patience=5,
            grad_accum_steps=4,
            fc_lr=1e-4,
            layer4_lr=1e-5,
            weight_decay=0.01,
            seed=42,
        )
        fold_results.append(res)
        all_oof_records.extend(res["patient_records"])

    total_cv_duration = time.time() - overall_start_time
    print(f"\n================================================================================", flush=True)
    print(f"ALL 5 FOLDS COMPLETED in {total_cv_duration:.2f} seconds ({total_cv_duration/60:.2f} minutes)", flush=True)
    print(f"================================================================================", flush=True)

    # Save out_of_fold_patient_predictions.csv
    oof_csv_path = artifacts_cv_dir / "out_of_fold_patient_predictions.csv"
    fieldnames = [
        "patient_id",
        "val_fold",
        "target_class",
        "actual_label",
        "prob_anterior",
        "prob_posterior",
        "prob_patient_max",
        "pred_label_050",
        "pred_name_050",
    ]
    with open(oof_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_oof_records)
    print(f"[Artifacts] Saved {len(all_oof_records)} OOF predictions to {oof_csv_path}", flush=True)

    # Verify 203 unique patients
    unique_pids = set(r["patient_id"] for r in all_oof_records)
    assert len(all_oof_records) == 203, f"Expected 203 records, got {len(all_oof_records)}"
    assert len(unique_pids) == 203, f"Expected 203 unique patient IDs, got {len(unique_pids)}"

    # Pooled arrays
    pooled_y_true = np.array([r["actual_label"] for r in all_oof_records], dtype=int)
    pooled_y_prob = np.array([r["prob_patient_max"] for r in all_oof_records], dtype=float)
    pooled_pred = (pooled_y_prob >= 0.50).astype(int)

    tn_p, fp_p, fn_p, tp_p = confusion_matrix(pooled_y_true, pooled_pred).ravel()
    pooled_sens = float(tp_p / (tp_p + fn_p))
    pooled_spec = float(tn_p / (tn_p + fp_p))
    pooled_prec = float(tp_p / (tp_p + fp_p)) if (tp_p + fp_p) > 0 else 0.0
    pooled_f1 = float(2 * pooled_prec * pooled_sens / (pooled_prec + pooled_sens)) if (pooled_prec + pooled_sens) > 0 else 0.0
    pooled_acc = float((tp_p + tn_p) / len(pooled_y_true))
    pooled_bal_acc = float((pooled_sens + pooled_spec) / 2.0)
    pooled_roc_auc = float(roc_auc_score(pooled_y_true, pooled_y_prob))
    pooled_pr_auc = float(average_precision_score(pooled_y_true, pooled_y_prob))

    # Mean and std across folds
    fold_roc_aucs = [r["best_metrics"]["roc_auc"] for r in fold_results]
    fold_pr_aucs = [r["best_metrics"]["pr_auc"] for r in fold_results]
    fold_sensitivities = [r["best_metrics"]["sensitivity"] for r in fold_results]
    fold_specificities = [r["best_metrics"]["specificity"] for r in fold_results]
    fold_f1s = [r["best_metrics"]["f1_score"] for r in fold_results]
    fold_bal_accs = [r["best_metrics"]["balanced_accuracy"] for r in fold_results]

    summary = {
        "experiment": "Bone Scan Scintigraphy Five-Fold Cross-Validation",
        "total_cv_patients": 203,
        "total_metastasis_positive": int(np.sum(pooled_y_true == 1)),
        "total_metastasis_negative": int(np.sum(pooled_y_true == 0)),
        "total_duration_sec": round(total_cv_duration, 2),
        "hardware": {
            "gpu": gpu_name,
            "total_vram_mb": round(vram_mb, 1),
        },
        "hyperparameters": {
            "batch_size_patients": 1,
            "images_per_patient": 2,
            "grad_accum_steps": 4,
            "fc_lr": 1e-4,
            "layer4_lr": 1e-5,
            "weight_decay": 0.01,
            "dropout": 0.4,
            "max_epochs": 20,
            "patience": 5,
            "mixed_precision": "fp16",
            "seed": 42,
        },
        "fold_metrics": {r["fold"]: r["best_metrics"] for r in fold_results},
        "cross_validation_statistics": {
            "mean_roc_auc": round(float(np.mean(fold_roc_aucs)), 4),
            "std_roc_auc": round(float(np.std(fold_roc_aucs)), 4),
            "mean_pr_auc": round(float(np.mean(fold_pr_aucs)), 4),
            "std_pr_auc": round(float(np.std(fold_pr_aucs)), 4),
            "mean_sensitivity": round(float(np.mean(fold_sensitivities)), 4),
            "std_sensitivity": round(float(np.std(fold_sensitivities)), 4),
            "mean_specificity": round(float(np.mean(fold_specificities)), 4),
            "std_specificity": round(float(np.std(fold_specificities)), 4),
            "mean_f1": round(float(np.mean(fold_f1s)), 4),
            "std_f1": round(float(np.std(fold_f1s)), 4),
            "mean_balanced_accuracy": round(float(np.mean(fold_bal_accs)), 4),
            "std_balanced_accuracy": round(float(np.std(fold_bal_accs)), 4),
        },
        "pooled_out_of_fold_metrics": {
            "total_patients": len(pooled_y_true),
            "roc_auc": round(pooled_roc_auc, 4),
            "pr_auc": round(pooled_pr_auc, 4),
            "accuracy": round(pooled_acc, 4),
            "sensitivity": round(pooled_sens, 4),
            "specificity": round(pooled_spec, 4),
            "precision": round(pooled_prec, 4),
            "f1_score": round(pooled_f1, 4),
            "balanced_accuracy": round(pooled_bal_acc, 4),
            "true_positives": int(tp_p),
            "false_positives": int(fp_p),
            "true_negatives": int(tn_p),
            "false_negatives": int(fn_p),
        },
    }

    summary_json_path = artifacts_cv_dir / "cv_metrics_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Artifacts] Saved CV summary metrics to {summary_json_path}", flush=True)

    # Visualizations
    generate_cv_visualizations(fold_results, pooled_y_true, pooled_y_prob, artifacts_cv_dir)
    print(f"[Visuals] Saved fold_comparison.png, pooled_roc_curve.png, and pooled_precision_recall_curve.png", flush=True)

    print("\n" + "=" * 80)
    print("STAGE 12: FIVE-FOLD CROSS-VALIDATION SUMMARY RESULTS")
    print("=" * 80)
    for r in fold_results:
        f_m = r["best_metrics"]
        print(f"  {r['fold']}: ROC-AUC = {f_m['roc_auc']:.4f} | PR-AUC = {f_m['pr_auc']:.4f} | Sens = {f_m['sensitivity']*100:.1f}% | Spec = {f_m['specificity']*100:.1f}% | F1 = {f_m['f1_score']:.4f} (Epoch {r['best_epoch']})")
    print("-" * 80)
    cv_stat = summary["cross_validation_statistics"]
    print(f"  CV Mean ROC-AUC: {cv_stat['mean_roc_auc']:.4f} ± {cv_stat['std_roc_auc']:.4f}")
    print(f"  CV Mean PR-AUC:  {cv_stat['mean_pr_auc']:.4f} ± {cv_stat['std_pr_auc']:.4f}")
    print(f"  CV Mean Sens:    {cv_stat['mean_sensitivity']*100:.1f}% ± {cv_stat['std_sensitivity']*100:.1f}%")
    print(f"  CV Mean Spec:    {cv_stat['mean_specificity']*100:.1f}% ± {cv_stat['std_specificity']*100:.1f}%")
    print("-" * 80)
    pool_m = summary["pooled_out_of_fold_metrics"]
    print(f"  Pooled OOF ROC-AUC: {pool_m['roc_auc']:.4f}")
    print(f"  Pooled OOF PR-AUC:  {pool_m['pr_auc']:.4f} (Prevalence baseline: {42/203:.4f})")
    print(f"  Pooled OOF Sens:    {pool_m['sensitivity']*100:.1f}% (TP={pool_m['true_positives']}, FN={pool_m['false_negatives']})")
    print(f"  Pooled OOF Spec:    {pool_m['specificity']*100:.1f}% (TN={pool_m['true_negatives']}, FP={pool_m['false_positives']})")
    print(f"  Pooled OOF F1:      {pool_m['f1_score']:.4f}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    run_stage_12_five_fold_cv()

