"""
OsteoGuard AI - Stage 10 Holdout Test Partition Independent Evaluation.

Features:
  - Strict zero-training evaluation on the isolated test partition (614 images).
  - Loads the frozen Stage 09 best checkpoint: ml_artifacts/fracatlas_full_training/best_checkpoint.pt.
  - Frozen primary operating threshold: THRESHOLD = 0.31 (calibrated on validation data for >=85% sensitivity).
  - Secondary comparator threshold: THRESHOLD = 0.50.
  - Generates full diagnostic metrics, confusion matrix, 95% bootstrap confidence intervals.
  - Direct comparison against validation performance to quantify generalization gap.
  - Visualizations: confusion_matrix_test.png, roc_curve_test.png,
    precision_recall_curve_test.png, validation_vs_test_comparison.png.
  - Artifacts saved to: ml_artifacts/fracatlas_test_evaluation/.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

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

from ml_service.training.fracatlas_full_trainer import build_full_resnet101_model


# ------------------------------------------------------------------------------
# 1. Dataset & Transform
# ------------------------------------------------------------------------------

def get_test_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Deterministic evaluation preprocessing pipeline."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


class FracAtlasTestDataset(Dataset):
    """Dataset for test partition evaluation with complete metadata."""

    def __init__(self, records: List[Dict[str, str]], img_size: Tuple[int, int] = (224, 224)):
        self.records = records
        self.transform = get_test_transforms(img_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        file_path = Path(row["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Test image file missing: {file_path}")

        with Image.open(file_path) as img:
            image_rgb = img.convert("RGB")

        tensor = self.transform(image_rgb)
        label = torch.tensor(int(row["fractured"]), dtype=torch.long)
        meta = {
            "image_id": row["image_id"],
            "file_path": str(file_path),
            "anatomic_region": row.get("anatomic_region", "unknown"),
            "view_projection": row.get("view_projection", "unknown"),
            "sha256_hash": row.get("sha256_hash", ""),
            "split": row.get("split", "test"),
        }
        return tensor, label, meta


# ------------------------------------------------------------------------------
# 2. Metric Computation & Bootstrap 95% Confidence Intervals
# ------------------------------------------------------------------------------

def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Computes full set of binary classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    acc = float((tp + tn) / len(y_true))
    bal_acc = float((sens + spec) / 2.0)
    youden = float(sens + spec - 1.0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "threshold": round(threshold, 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": round(acc, 4),
        "sensitivity_recall": round(sens, 4),
        "specificity": round(spec, 4),
        "precision_ppv": round(prec, 4),
        "npv": round(npv, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "youden_j": round(youden, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Calculates 95% empirical bootstrap confidence intervals (2.5% to 97.5%)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1_score": [],
        "balanced_accuracy": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t = y_true[idx]
        y_p = y_prob[idx]

        # Require at least one positive and one negative sample in the bootstrap draw
        if len(np.unique(y_t)) < 2:
            continue

        y_pr = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pr).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        bal_acc = (sens + spec) / 2.0

        try:
            roc = roc_auc_score(y_t, y_p)
        except Exception:
            roc = 0.5

        try:
            pr = average_precision_score(y_t, y_p)
        except Exception:
            pr = 0.0

        metrics["roc_auc"].append(roc)
        metrics["pr_auc"].append(pr)
        metrics["sensitivity"].append(sens)
        metrics["specificity"].append(spec)
        metrics["precision"].append(prec)
        metrics["f1_score"].append(f1)
        metrics["balanced_accuracy"].append(bal_acc)

    ci_results = {}
    for metric_name, values in metrics.items():
        arr = np.array(values)
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        ci_results[metric_name] = {
            "ci_lower_95": round(ci_lower, 4),
            "ci_upper_95": round(ci_upper, 4),
        }
    return ci_results


# ------------------------------------------------------------------------------
# 3. Visualizations
# ------------------------------------------------------------------------------

def generate_test_visualizations(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    primary_metrics: Dict[str, Any],
    secondary_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    output_dir: Path,
):
    """Generates all Stage 10 visual artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Heatmap (Primary Threshold 0.31)
    cm = np.array([
        [primary_metrics["tn"], primary_metrics["fp"]],
        [primary_metrics["fn"], primary_metrics["tp"]]
    ])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix (Frozen Threshold = {primary_metrics['threshold']:.2f})\n"
              f"Sens: {primary_metrics['sensitivity_recall']:.2%}, Spec: {primary_metrics['specificity']:.2%}",
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
    plt.savefig(output_dir / "confusion_matrix_test.png", dpi=200)
    plt.close()

    # 2. Test ROC Curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_arr, tpr_arr, color="#1f77b4", lw=2.5, label=f"Holdout Test ROC (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")

    plt.scatter([primary_metrics["false_positive_rate"]], [primary_metrics["sensitivity_recall"]],
                color="#d62728", s=90, zorder=5,
                label=f"Frozen Primary (th={primary_metrics['threshold']:.2f}, Sens={primary_metrics['sensitivity_recall']:.2f})")
    plt.scatter([secondary_metrics["false_positive_rate"]], [secondary_metrics["sensitivity_recall"]],
                color="#2ca02c", s=90, zorder=5, marker="^",
                label=f"Secondary (th={secondary_metrics['threshold']:.2f}, Sens={secondary_metrics['sensitivity_recall']:.2f})")

    plt.title("Holdout Test ROC Curve — FracAtlas ResNet-101", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_test.png", dpi=200)
    plt.close()

    # 3. Test Precision-Recall Curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    test_prevalence = np.mean(y_true)

    plt.figure(figsize=(7, 6))
    plt.plot(rec_arr, prec_arr, color="#d62728", lw=2.5, label=f"Holdout Test PR (AUC = {pr_auc_val:.4f})")
    plt.axhline(test_prevalence, color="grey", linestyle="--", lw=1.5, label=f"Test Prevalence Baseline ({test_prevalence:.4f})")

    plt.scatter([primary_metrics["sensitivity_recall"]], [primary_metrics["precision_ppv"]],
                color="#1f77b4", s=90, zorder=5,
                label=f"Frozen Primary (th={primary_metrics['threshold']:.2f}, Prec={primary_metrics['precision_ppv']:.2f})")
    plt.scatter([secondary_metrics["sensitivity_recall"]], [secondary_metrics["precision_ppv"]],
                color="#2ca02c", s=90, zorder=5, marker="^",
                label=f"Secondary (th={secondary_metrics['threshold']:.2f}, Prec={secondary_metrics['precision_ppv']:.2f})")

    plt.title("Holdout Test Precision-Recall Curve — FracAtlas ResNet-101", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve_test.png", dpi=200)
    plt.close()

    # 4. Validation vs Test Side-by-Side Comparison Bar Chart
    val_high_sens = val_metrics["operating_points"]["high_sensitivity_operating_point"]
    metrics_to_plot = ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Precision", "F1-Score", "Balanced Acc"]

    val_vals = [
        val_metrics["roc_auc"],
        val_metrics["pr_auc"],
        val_high_sens["sensitivity"],
        val_high_sens["specificity"],
        val_high_sens["precision"],
        val_high_sens["f1_score"],
        val_high_sens["balanced_accuracy"],
    ]

    test_vals = [
        roc_auc_val,
        pr_auc_val,
        primary_metrics["sensitivity_recall"],
        primary_metrics["specificity"],
        primary_metrics["precision_ppv"],
        primary_metrics["f1_score"],
        primary_metrics["balanced_accuracy"],
    ]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    rects1 = plt.bar(x - width/2, [v * 100 for v in val_vals], width, label="Validation (N=612)", color="#1f77b4")
    rects2 = plt.bar(x + width/2, [v * 100 for v in test_vals], width, label="Holdout Test (N=614)", color="#ff7f0e")

    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Validation vs. Holdout Test Generalization (Operating Threshold = 0.31)", fontsize=12, fontweight="bold")
    plt.xticks(x, metrics_to_plot, fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)

    for r in rects1:
        h = r.get_height()
        plt.text(r.get_x() + r.get_width()/2., h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    for r in rects2:
        h = r.get_height()
        plt.text(r.get_x() + r.get_width()/2., h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_dir / "validation_vs_test_comparison.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------------------
# 4. Main Evaluation Routine
# ------------------------------------------------------------------------------

def run_test_evaluation(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    primary_threshold: float = 0.31,
    secondary_threshold: float = 0.50,
):
    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 10 FRACATLAS HOLDOUT TEST EVALUATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Check GPU Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Target Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # 2. Verify Manifest and Extract Test Partition
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing at {manifest_path}")

    all_records = {"train": [], "val": [], "test": []}
    all_hashes = {"train": set(), "val": set(), "test": set()}

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["split"].lower()
            all_records[sp].append(row)
            all_hashes[sp].add(row["sha256_hash"])

    test_records = all_records["test"]
    n_test = len(test_records)
    n_frac = sum(1 for r in test_records if r["fractured"] == "1")
    n_non_frac = sum(1 for r in test_records if r["fractured"] == "0")

    print(f"[Manifest] Total test records: {n_test}", flush=True)
    print(f"[Manifest] Test fractured: {n_frac} ({n_frac / n_test:.2%})", flush=True)
    print(f"[Manifest] Test non-fractured: {n_non_frac} ({n_non_frac / n_test:.2%})", flush=True)

    # Strict partition isolation assertions
    assert n_test == 614, f"Expected 614 test records, got {n_test}"
    assert n_frac == 110, f"Expected 110 fractured, got {n_frac}"
    assert n_non_frac == 504, f"Expected 504 non-fractured, got {n_non_frac}"

    overlap_train = all_hashes["test"].intersection(all_hashes["train"])
    overlap_val = all_hashes["test"].intersection(all_hashes["val"])
    assert len(overlap_train) == 0, f"CRITICAL LEAKAGE: {len(overlap_train)} hashes overlap test & train!"
    assert len(overlap_val) == 0, f"CRITICAL LEAKAGE: {len(overlap_val)} hashes overlap test & val!"
    print("[Manifest] Partition hygiene verified: 0 hash overlap across all splits.", flush=True)

    # 3. Load Frozen Checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    print(f"[Model] Loading frozen checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_epoch = ckpt.get("epoch", "unknown")
    val_pr_auc_ckpt = ckpt.get("val_pr_auc", "unknown")
    val_roc_auc_ckpt = ckpt.get("val_roc_auc", "unknown")
    print(f"[Model] Checkpoint Epoch: {best_epoch} | Stored Val PR-AUC: {val_pr_auc_ckpt} | Stored Val ROC-AUC: {val_roc_auc_ckpt}", flush=True)

    model, _ = build_full_resnet101_model(device, pi_prior=502.0 / 2857.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Run Test Inference
    test_dataset = FracAtlasTestDataset(test_records)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"[Inference] Executing single-pass evaluation across all {n_test} test images...", flush=True)
    start_eval_time = time.time()

    all_targets = []
    all_prob_fractured = []
    all_prob_non_fractured = []
    prediction_records = []

    with torch.no_grad():
        for x, y, meta in test_loader:
            x = x.to(device)
            with autocast("cuda", dtype=torch.float16):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

            p_frac = probs[:, 1].cpu().float().numpy()
            p_non = probs[:, 0].cpu().float().numpy()
            targets = y.numpy()

            for i in range(len(targets)):
                t = int(targets[i])
                pf = float(p_frac[i])
                pn = float(p_non[i])

                all_targets.append(t)
                all_prob_fractured.append(pf)
                all_prob_non_fractured.append(pn)

                pred_031 = int(pf >= primary_threshold)
                pred_050 = int(pf >= secondary_threshold)

                prediction_records.append({
                    "image_id": meta["image_id"][i],
                    "file_path": meta["file_path"][i],
                    "anatomic_region": meta["anatomic_region"][i],
                    "view_projection": meta["view_projection"][i],
                    "actual_label": t,
                    "actual_label_name": "acute_bone_fracture" if t == 1 else "candidate_control_radiograph",
                    "non_fracture_probability": round(pn, 6),
                    "fracture_probability": round(pf, 6),
                    "pred_threshold_031": pred_031,
                    "pred_name_threshold_031": "acute_bone_fracture" if pred_031 == 1 else "candidate_control_radiograph",
                    "pred_threshold_050": pred_050,
                    "pred_name_threshold_050": "acute_bone_fracture" if pred_050 == 1 else "candidate_control_radiograph",
                })

    eval_duration = time.time() - start_eval_time
    print(f"[Inference] Evaluation complete in {eval_duration:.2f} seconds ({eval_duration / n_test * 1000:.1f} ms/image).", flush=True)

    y_true_arr = np.array(all_targets, dtype=int)
    y_prob_arr = np.array(all_prob_fractured, dtype=float)

    # 5. Compute Metrics at Frozen Thresholds
    primary_metrics = compute_metrics_at_threshold(y_true_arr, y_prob_arr, primary_threshold)
    secondary_metrics = compute_metrics_at_threshold(y_true_arr, y_prob_arr, secondary_threshold)

    roc_auc_val = float(roc_auc_score(y_true_arr, y_prob_arr))
    pr_auc_val = float(average_precision_score(y_true_arr, y_prob_arr))

    print(f"\n[Test Metrics] Frozen Threshold ({primary_threshold}):", flush=True)
    print(f"  - Sensitivity: {primary_metrics['sensitivity_recall']:.2%} ({primary_metrics['tp']}/{n_frac})", flush=True)
    print(f"  - Specificity: {primary_metrics['specificity']:.2%} ({primary_metrics['tn']}/{n_non_frac})", flush=True)
    print(f"  - Precision:   {primary_metrics['precision_ppv']:.2%} ({primary_metrics['tp']}/{primary_metrics['tp'] + primary_metrics['fp']})", flush=True)
    print(f"  - F1-Score:    {primary_metrics['f1_score']:.4f}", flush=True)
    print(f"  - Balanced Acc:{primary_metrics['balanced_accuracy']:.2%}", flush=True)
    print(f"  - ROC-AUC:     {roc_auc_val:.4f}", flush=True)
    print(f"  - PR-AUC:      {pr_auc_val:.4f}", flush=True)

    print(f"\n[Test Metrics] Secondary Threshold ({secondary_threshold}):", flush=True)
    print(f"  - Sensitivity: {secondary_metrics['sensitivity_recall']:.2%} ({secondary_metrics['tp']}/{n_frac})", flush=True)
    print(f"  - Specificity: {secondary_metrics['specificity']:.2%} ({secondary_metrics['tn']}/{n_non_frac})", flush=True)
    print(f"  - Precision:   {secondary_metrics['precision_ppv']:.2%} ({secondary_metrics['tp']}/{secondary_metrics['tp'] + secondary_metrics['fp']})", flush=True)
    print(f"  - F1-Score:    {secondary_metrics['f1_score']:.4f}", flush=True)

    # 6. Bootstrap Confidence Intervals
    print("\n[Bootstrap] Computing 95% Confidence Intervals (1,000 resamples)...", flush=True)
    ci_results = compute_bootstrap_confidence_intervals(y_true_arr, y_prob_arr, primary_threshold, n_bootstraps=1000)
    for m, ci in ci_results.items():
        print(f"  - {m}: 95% CI [{ci['ci_lower_95']}, {ci['ci_upper_95']}]", flush=True)

    # 7. Save Test Predictions CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_csv_path = output_dir / "test_predictions.csv"
    with open(preds_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_records[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_records)
    print(f"[Artifacts] Saved test predictions to {preds_csv_path}", flush=True)

    # 8. Load Validation Metrics for Comparison
    val_metrics_path = checkpoint_path.parent / "validation_metrics.json"
    with open(val_metrics_path, "r", encoding="utf-8") as f:
        val_metrics = json.load(f)

    # 9. Save Test Metrics JSON
    test_metrics_json = {
        "dataset": "FracAtlas Holdout Test Split",
        "checkpoint_used": str(checkpoint_path),
        "checkpoint_epoch": best_epoch,
        "total_test_samples": n_test,
        "test_fractured": n_frac,
        "test_non_fractured": n_non_frac,
        "test_prevalence": round(n_frac / n_test, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
        "frozen_primary_threshold_031": primary_metrics,
        "secondary_threshold_050": secondary_metrics,
        "confidence_intervals_95": ci_results,
        "validation_comparison_primary_031": {
            "roc_auc": {"val": val_metrics["roc_auc"], "test": round(roc_auc_val, 4), "delta": round(roc_auc_val - val_metrics["roc_auc"], 4)},
            "pr_auc": {"val": val_metrics["pr_auc"], "test": round(pr_auc_val, 4), "delta": round(pr_auc_val - val_metrics["pr_auc"], 4)},
            "sensitivity": {"val": val_metrics["operating_points"]["high_sensitivity_operating_point"]["sensitivity"], "test": primary_metrics["sensitivity_recall"], "delta": round(primary_metrics["sensitivity_recall"] - val_metrics["operating_points"]["high_sensitivity_operating_point"]["sensitivity"], 4)},
            "specificity": {"val": val_metrics["operating_points"]["high_sensitivity_operating_point"]["specificity"], "test": primary_metrics["specificity"], "delta": round(primary_metrics["specificity"] - val_metrics["operating_points"]["high_sensitivity_operating_point"]["specificity"], 4)},
            "precision": {"val": val_metrics["operating_points"]["high_sensitivity_operating_point"]["precision"], "test": primary_metrics["precision_ppv"], "delta": round(primary_metrics["precision_ppv"] - val_metrics["operating_points"]["high_sensitivity_operating_point"]["precision"], 4)},
            "f1_score": {"val": val_metrics["operating_points"]["high_sensitivity_operating_point"]["f1_score"], "test": primary_metrics["f1_score"], "delta": round(primary_metrics["f1_score"] - val_metrics["operating_points"]["high_sensitivity_operating_point"]["f1_score"], 4)},
            "balanced_accuracy": {"val": val_metrics["operating_points"]["high_sensitivity_operating_point"]["balanced_accuracy"], "test": primary_metrics["balanced_accuracy"], "delta": round(primary_metrics["balanced_accuracy"] - val_metrics["operating_points"]["high_sensitivity_operating_point"]["balanced_accuracy"], 4)},
        }
    }
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics_json, f, indent=2)
    print(f"[Artifacts] Saved test metrics to {output_dir / 'test_metrics.json'}", flush=True)

    # 10. Generate Visualizations
    generate_test_visualizations(y_true_arr, y_prob_arr, primary_metrics, secondary_metrics, val_metrics, output_dir)
    print(f"[Visuals] Saved 4 test evaluation plots to {output_dir}", flush=True)

    # 11. Save Test Evaluation Summary JSON
    summary_dict = {
        "project": "OsteoGuard AI",
        "stage": "STAGE 10 — FINAL FRACATLAS HOLDOUT TEST EVALUATION",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint_path),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "evaluation_samples": n_test,
        "frozen_primary_threshold": primary_threshold,
        "test_roc_auc": round(roc_auc_val, 4),
        "test_pr_auc": round(pr_auc_val, 4),
        "test_sensitivity_031": primary_metrics["sensitivity_recall"],
        "test_specificity_031": primary_metrics["specificity"],
        "test_precision_031": primary_metrics["precision_ppv"],
        "test_f1_031": primary_metrics["f1_score"],
        "test_balanced_acc_031": primary_metrics["balanced_accuracy"],
        "generalization_delta_roc_auc": round(roc_auc_val - val_metrics["roc_auc"], 4),
        "generalization_delta_pr_auc": round(pr_auc_val - val_metrics["pr_auc"], 4),
    }
    with open(output_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[Summary] Saved test evaluation summary to {output_dir / 'test_evaluation_summary.json'}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("STAGE 10 HOLDOUT EVALUATION COMPLETE", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OsteoGuard AI Stage 10 FracAtlas Test Evaluator")
    parser.add_argument("--manifest", type=str, default="ml_service/data/splits/fracatlas_classification_split.csv")
    parser.add_argument("--checkpoint", type=str, default="ml_artifacts/fracatlas_full_training/best_checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="ml_artifacts/fracatlas_test_evaluation")
    parser.add_argument("--primary_threshold", type=float, default=0.31)
    parser.add_argument("--secondary_threshold", type=float, default=0.50)

    args = parser.parse_args()

    run_test_evaluation(
        manifest_path=Path(args.manifest),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        primary_threshold=args.primary_threshold,
        secondary_threshold=args.secondary_threshold,
    )

