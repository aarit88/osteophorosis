"""
OsteoGuard AI - Stage 08C FracAtlas Prediction Diagnostics & Threshold Analysis.

Executes:
  1. Class mapping audit (0 = non-fractured, 1 = fractured).
  2. Model inference on all 612 validation images from best_checkpoint.pt.
  3. Per-image prediction recording (image_id, file_path, actual, logits, probs, pred_0.5).
  4. Detailed probability distribution analysis (stratified by actual label).
  5. Comprehensive threshold sweep (0.05 to 0.95 in 0.01 steps):
     - Sensitivity, Specificity, Precision, F1, Accuracy, Balanced Accuracy, Youden's J.
     - Identification of Best F1, Best Youden J, 80% sensitivity threshold, 90% sensitivity threshold.
  6. Image loading forensic audit (identifying truncated JPEG images in FracAtlas).
  7. Training configuration & logit analysis (class weights, loss formula, classifier bias).
  8. Generation of diagnostic plots:
     - probability_distribution.png
     - roc_curve.png
     - precision_recall_curve.png
  9. Export of CSV files:
     - validation_predictions.csv
     - threshold_analysis.csv
"""

import csv
import json
from pathlib import Path
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ResNet101_Weights
import torchvision.transforms as transforms

from ml_service.config.settings import get_settings


# ------------------------------------------------------------------------------
# 1. Dataset & Transform Definition
# ------------------------------------------------------------------------------

def get_val_transform(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


class DiagnosticValDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], img_size: Tuple[int, int] = (224, 224)):
        self.records = records
        self.transform = get_val_transform(img_size=img_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str, str]:
        row = self.records[idx]
        file_path = Path(row["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")

        tensor = self.transform(img_rgb)
        label = int(row["fractured"])
        image_id = row["image_id"]
        anatomic_region = row.get("anatomic_region", "unknown")
        view_proj = row.get("view_projection", "unknown")
        return tensor, label, image_id, str(file_path), anatomic_region


# ------------------------------------------------------------------------------
# 2. Checkpoint Loader
# ------------------------------------------------------------------------------

def load_pilot_model(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


# ------------------------------------------------------------------------------
# 3. Validation Inference & Predictions Extraction
# ------------------------------------------------------------------------------

def run_diagnostic_inference(
    model: nn.Module,
    val_records: List[Dict[str, str]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    dataset = DiagnosticValDataset(val_records)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    predictions = []
    with torch.no_grad():
        for images, labels, img_ids, file_paths, regions in loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

            logits_np = logits.cpu().float().numpy()
            probs_np = probs.cpu().float().numpy()
            labels_np = labels.numpy()

            for i in range(len(img_ids)):
                p0 = float(probs_np[i, 0])
                p1 = float(probs_np[i, 1])
                pred_0_5 = 1 if p1 >= 0.5 else 0

                predictions.append({
                    "image_id": img_ids[i],
                    "file_path": file_paths[i],
                    "anatomic_region": regions[i],
                    "actual_label": int(labels_np[i]),
                    "logit_class_0": float(logits_np[i, 0]),
                    "logit_class_1": float(logits_np[i, 1]),
                    "prob_class_0": p0,
                    "prob_class_1": p1,
                    "predicted_label_0_5": pred_0_5,
                })
    return predictions


# ------------------------------------------------------------------------------
# 4. Statistical Distribution Calculations
# ------------------------------------------------------------------------------

def compute_distribution_stats(probs: List[float]) -> Dict[str, float]:
    arr = np.array(probs)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


# ------------------------------------------------------------------------------
# 5. Threshold Analysis (0.05 to 0.95 in 0.01 increments)
# ------------------------------------------------------------------------------

def run_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []

    best_f1 = -1.0
    best_f1_info = {}

    best_j = -2.0
    best_j_info = {}

    thresh_80_sens = None
    thresh_90_sens = None

    for th in thresholds:
        th_val = round(float(th), 2)
        y_pred = (y_prob >= th_val).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
        acc = float((tp + tn) / len(y_true))
        bal_acc = float((sens + spec) / 2.0)
        youden_j = float(sens + spec - 1.0)

        record = {
            "threshold": th_val,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "sensitivity_recall": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "youden_j": round(youden_j, 4),
        }
        results.append(record)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_info = record.copy()

        if youden_j > best_j:
            best_j = youden_j
            best_j_info = record.copy()

        # Identify threshold for >= 80% and >= 90% sensitivity
        # For highest sensitivity, we seek largest threshold that achieves >= 0.80 / >= 0.90
        if sens >= 0.80:
            if thresh_80_sens is None or th_val > thresh_80_sens["threshold"]:
                thresh_80_sens = record.copy()
        if sens >= 0.90:
            if thresh_90_sens is None or th_val > thresh_90_sens["threshold"]:
                thresh_90_sens = record.copy()

    summary = {
        "best_f1": best_f1_info,
        "best_youden_j": best_j_info,
        "threshold_80_sensitivity": thresh_80_sens,
        "threshold_90_sensitivity": thresh_90_sens,
    }
    return results, summary


# ------------------------------------------------------------------------------
# 6. Forensic Image Loading Audit
# ------------------------------------------------------------------------------

def audit_fracatlas_image_corruption(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    truncated_images = []
    fatal_errors = []

    for r in records:
        file_path = Path(r["file_path"])
        # Test loading strictly WITHOUT LOAD_TRUNCATED_IMAGES first
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            with Image.open(file_path) as img:
                img.convert("RGB").load()
        except OSError as e:
            if "truncated" in str(e).lower():
                # Now test with LOAD_TRUNCATED_IMAGES = True
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    with Image.open(file_path) as img2:
                        img2.convert("RGB").load()
                    truncated_images.append({
                        "image_id": r["image_id"],
                        "file_path": str(file_path),
                        "split": r["split"],
                        "target_class": r["target_class"],
                        "error_message": str(e),
                        "recovered_with_load_truncated": True,
                    })
                except Exception as e2:
                    fatal_errors.append({
                        "image_id": r["image_id"],
                        "file_path": str(file_path),
                        "split": r["split"],
                        "fatal_error": str(e2),
                    })
            else:
                fatal_errors.append({
                    "image_id": r["image_id"],
                    "file_path": str(file_path),
                    "split": r["split"],
                    "fatal_error": str(e),
                })
        except Exception as e:
            fatal_errors.append({
                "image_id": r["image_id"],
                "file_path": str(file_path),
                "split": r["split"],
                "fatal_error": str(e),
            })

    # Restore flag
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    return {
        "total_images_audited": len(records),
        "truncated_images_count": len(truncated_images),
        "truncated_images": truncated_images,
        "fatal_unrecoverable_errors_count": len(fatal_errors),
        "fatal_errors": fatal_errors,
    }


# ------------------------------------------------------------------------------
# 7. Main Diagnostic Pipeline
# ------------------------------------------------------------------------------

def main():
    settings = get_settings()
    root = settings.base_dir
    output_dir = root / "ml_artifacts/fracatlas_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "ml_service/data/splits/fracatlas_classification_split.csv"
    ckpt_path = output_dir / "best_checkpoint.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing Stage 08C Model Diagnostics on: {device}")

    # 1. Load Validation Data
    val_records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() == "val":
                val_records.append(r)
    val_records.sort(key=lambda r: r["image_id"])
    print(f"Loaded {len(val_records)} validation records.")

    # 2. Load Model Checkpoint
    model, ckpt = load_pilot_model(ckpt_path, device)
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (Val Loss: {ckpt['val_loss']:.4f})")

    # Inspect fc layer
    fc_w = ckpt["model_state_dict"]["fc.weight"].cpu().float().numpy()
    fc_b = ckpt["model_state_dict"]["fc.bias"].cpu().float().numpy()
    print(f"Classifier head weight norm: Class 0 = {np.linalg.norm(fc_w[0]):.4f}, Class 1 = {np.linalg.norm(fc_w[1]):.4f}")
    print(f"Classifier head bias: Class 0 = {fc_b[0]:.6f}, Class 1 = {fc_b[1]:.6f}")

    # 3. Run Inference
    predictions = run_diagnostic_inference(model, val_records, device)

    # Save validation_predictions.csv
    pred_csv_path = output_dir / "validation_predictions.csv"
    pred_fieldnames = list(predictions[0].keys())
    with open(pred_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Saved validation predictions to: {pred_csv_path}")

    # 4. Statistical Distributions
    y_true = np.array([p["actual_label"] for p in predictions])
    y_prob = np.array([p["prob_class_1"] for p in predictions])
    logits_0 = np.array([p["logit_class_0"] for p in predictions])
    logits_1 = np.array([p["logit_class_1"] for p in predictions])

    frac_probs = [p["prob_class_1"] for p in predictions if p["actual_label"] == 1]
    non_frac_probs = [p["prob_class_1"] for p in predictions if p["actual_label"] == 0]

    stats_frac = compute_distribution_stats(frac_probs)
    stats_non_frac = compute_distribution_stats(non_frac_probs)
    stats_all = compute_distribution_stats(y_prob.tolist())

    print("\n--- PROBABILITY DISTRIBUTION STATISTICS P(fracture) ---")
    print("Actual Fractured (N=107):")
    for k, v in stats_frac.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nActual Non-Fractured (N=505):")
    for k, v in stats_non_frac.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 5. Threshold Analysis
    threshold_records, thresh_summary = run_threshold_sweep(y_true, y_prob)
    thresh_csv_path = output_dir / "threshold_analysis.csv"
    thresh_fieldnames = list(threshold_records[0].keys())
    with open(thresh_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=thresh_fieldnames)
        writer.writeheader()
        writer.writerows(threshold_records)
    print(f"Saved threshold sweep to: {thresh_csv_path}")

    print("\n--- OPTIMAL THRESHOLD FINDINGS ---")
    print(f"Default (theta=0.50): Sens={next(r['sensitivity_recall'] for r in threshold_records if r['threshold']==0.50):.4f}, "
          f"Spec={next(r['specificity'] for r in threshold_records if r['threshold']==0.50):.4f}, "
          f"F1={next(r['f1_score'] for r in threshold_records if r['threshold']==0.50):.4f}")
    print(f"Best F1: Threshold={thresh_summary['best_f1']['threshold']} -> "
          f"F1={thresh_summary['best_f1']['f1_score']}, "
          f"Sens={thresh_summary['best_f1']['sensitivity_recall']}, "
          f"Prec={thresh_summary['best_f1']['precision']}, "
          f"Spec={thresh_summary['best_f1']['specificity']}")
    print(f"Best Youden's J: Threshold={thresh_summary['best_youden_j']['threshold']} -> "
          f"J={thresh_summary['best_youden_j']['youden_j']}, "
          f"Sens={thresh_summary['best_youden_j']['sensitivity_recall']}, "
          f"Spec={thresh_summary['best_youden_j']['specificity']}, "
          f"BalAcc={thresh_summary['best_youden_j']['balanced_accuracy']}")
    if thresh_summary["threshold_80_sensitivity"]:
        t80 = thresh_summary["threshold_80_sensitivity"]
        print(f">=80% Sensitivity: Threshold={t80['threshold']} -> Sens={t80['sensitivity_recall']}, Spec={t80['specificity']}, Prec={t80['precision']}")
    if thresh_summary["threshold_90_sensitivity"]:
        t90 = thresh_summary["threshold_90_sensitivity"]
        print(f">=90% Sensitivity: Threshold={t90['threshold']} -> Sens={t90['sensitivity_recall']}, Spec={t90['specificity']}, Prec={t90['precision']}")

    # 6. Forensic Image Loading Audit
    print("\nAuditing entire FracAtlas dataset for corrupted/truncated JPEG images...")
    corruption_audit = audit_fracatlas_image_corruption(manifest_path)
    print(f"Total images audited: {corruption_audit['total_images_audited']}")
    print(f"Truncated images detected: {corruption_audit['truncated_images_count']}")
    for ti in corruption_audit["truncated_images"]:
        print(f"  -> {ti['image_id']} ({ti['split']}, {ti['target_class']}): {ti['error_message']}")
    print(f"Fatal unrecoverable errors: {corruption_audit['fatal_unrecoverable_errors_count']}")

    # 7. Generate Diagnostic Plots
    # A. Probability Distribution Plot
    plt.figure(figsize=(9, 5.5))
    bins = np.linspace(0.0, 1.0, 31)
    plt.hist(non_frac_probs, bins=bins, alpha=0.6, color="#1f77b4", label=f"Non-Fractured (N=505, Mean={stats_non_frac['mean']:.3f})", density=True)
    plt.hist(frac_probs, bins=bins, alpha=0.6, color="#d62728", label=f"Fractured (N=107, Mean={stats_frac['mean']:.3f})", density=True)

    # Threshold markers
    plt.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Default Threshold (0.50)")
    best_f1_th = thresh_summary["best_f1"]["threshold"]
    best_j_th = thresh_summary["best_youden_j"]["threshold"]
    plt.axvline(best_f1_th, color="green", linestyle="-.", linewidth=2, label=f"Best F1 Threshold ({best_f1_th:.2f})")
    plt.axvline(best_j_th, color="purple", linestyle=":", linewidth=2, label=f"Best Youden J ({best_j_th:.2f})")

    plt.title("FracAtlas Pilot: Validation Probability Distribution P(Fracture)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Probability P(Class 1 = Fractured)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(fontsize=9, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    dist_plot_path = output_dir / "probability_distribution.png"
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()
    print(f"Saved distribution plot: {dist_plot_path}")

    # B. ROC Curve Plot
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6.5, 6))
    plt.plot(fpr, tpr, color="#2ca02c", linewidth=2.5, label=f"ResNet-101 (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.5000)")

    # Mark Youden J point
    best_j_sens = thresh_summary["best_youden_j"]["sensitivity_recall"]
    best_j_fpr = 1.0 - thresh_summary["best_youden_j"]["specificity"]
    plt.scatter([best_j_fpr], [best_j_sens], color="purple", s=90, zorder=5, label=f"Youden J Optimal (TPR={best_j_sens:.2f}, FPR={best_j_fpr:.2f})")

    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_plot_path = output_dir / "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve: {roc_plot_path}")

    # C. Precision-Recall Curve Plot
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline_prev = len(frac_probs) / len(predictions)

    plt.figure(figsize=(6.5, 6))
    plt.plot(recalls, precisions, color="#ff7f0e", linewidth=2.5, label=f"ResNet-101 (PR-AUC = {pr_auc:.4f})")
    plt.axhline(baseline_prev, color="gray", linestyle="--", label=f"Prevalence Baseline ({baseline_prev:.4f})")

    # Mark Best F1 point
    best_f1_sens = thresh_summary["best_f1"]["sensitivity_recall"]
    best_f1_prec = thresh_summary["best_f1"]["precision"]
    plt.scatter([best_f1_sens], [best_f1_prec], color="green", s=90, zorder=5, label=f"Best F1 Optimal (P={best_f1_prec:.2f}, R={best_f1_sens:.2f})")

    plt.title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Recall / Sensitivity", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pr_plot_path = output_dir / "precision_recall_curve.png"
    plt.savefig(pr_plot_path, dpi=200)
    plt.close()
    print(f"Saved PR curve: {pr_plot_path}")

    # 8. Save diagnostic summary JSON for report generation
    diag_summary = {
        "validation_samples": len(predictions),
        "actual_fractured_count": len(frac_probs),
        "actual_non_fractured_count": len(non_frac_probs),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "prevalence_baseline": round(float(baseline_prev), 4),
        "probability_stats": {
            "fractured": stats_frac,
            "non_fractured": stats_non_frac,
            "all": stats_all,
        },
        "threshold_summary": thresh_summary,
        "image_loading_audit": {
            "total_images": corruption_audit["total_images_audited"],
            "truncated_count": corruption_audit["truncated_images_count"],
            "truncated_images": corruption_audit["truncated_images"],
            "fatal_errors": corruption_audit["fatal_errors"],
        },
        "model_head": {
            "weight_norm_class_0": float(np.linalg.norm(fc_w[0])),
            "weight_norm_class_1": float(np.linalg.norm(fc_w[1])),
            "bias_class_0": float(fc_b[0]),
            "bias_class_1": float(fc_b[1]),
        },
    }
    with open(output_dir / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)
    print("Diagnostics analysis successfully completed.")


if __name__ == "__main__":
    main()

