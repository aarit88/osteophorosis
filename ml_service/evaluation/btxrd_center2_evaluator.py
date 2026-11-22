"""
OsteoGuard AI - Stage 19: BTXRD Center 2 External Holdout Evaluation.

Executes an independent, single-pass evaluation of the cryptographically frozen
Stage 18 ResNet-34 classifier on the Center 2 external multi-center benchmark (N=549).

Preserves:
- Frozen ResNet-34 weights (Epoch 18, SHA-256: 21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD)
- 512x512 aspect-ratio-preserving letterbox with symmetrical black border (0, 0, 0)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Standard argmax classification rule over raw, uncalibrated softmax probabilities
- Complete isolation of Center 3 (strictly sealed)
"""

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
from torch.utils.data import DataLoader, Dataset

from ml_service.config.settings import get_settings
from ml_service.datasets.btxrd_dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    preprocess_btxrd_image,
)
from ml_service.models.btxrd_classifier import BTXRDResNet34

# Authoritative Constants
FROZEN_MODEL_VERSION = "1.0.0"
EXPECTED_CHECKPOINT_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
EXPECTED_CENTER2_COUNT = 549
EXPECTED_CENTER2_DISTRIBUTION = {
    "NORMAL_BONE_RADIOGRAPH": 259,
    "BENIGN_BONE_TUMOR": 214,
    "PRIMARY_MALIGNANT_TUMOR": 76,
}


def compute_file_sha256(filepath: Path) -> str:
    """Computes uppercase SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


class BTXRDCenter2Dataset(Dataset):
    """
    Dedicated dataset strictly isolating Center 2 external holdout radiographs.
    """

    def __init__(self, manifest_path: Path, base_dir: Path):
        self.manifest_path = manifest_path
        self.base_dir = base_dir
        self.records: List[Dict[str, Any]] = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                center = row.get("center", "").strip().lower()
                split = row.get("center_holdout_split", "").strip().lower()

                # Isolation check: ONLY center_2 external holdout
                if center != "center_2":
                    continue
                if split != "external_test_center2":
                    continue

                # Conflict quarantine check
                if row.get("conflict_flag", "").strip().lower() == "true":
                    continue

                tgt_cls = row.get("target_class", "").strip()
                if tgt_cls not in CLASS_TO_IDX:
                    continue

                fpath = Path(row["file_path"])
                if not fpath.exists():
                    alt_path = base_dir / "ml_service/data/raw/btxrd/images" / row["image_id"]
                    if alt_path.exists():
                        fpath = alt_path
                    else:
                        raise FileNotFoundError(f"Image file not found: {fpath}")

                self.records.append({
                    "image_id": row["image_id"],
                    "file_path": str(fpath),
                    "target_class": tgt_cls,
                    "label": CLASS_TO_IDX[tgt_cls],
                    "center": "center_2",
                })

        if len(self.records) != EXPECTED_CENTER2_COUNT:
            raise ValueError(
                f"Center 2 cohort count mismatch! Expected {EXPECTED_CENTER2_COUNT}, found {len(self.records)}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        tensor = preprocess_btxrd_image(rec["file_path"], target_size=512, is_train=False)
        return {
            "image": tensor,
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "image_id": rec["image_id"],
            "target_class": rec["target_class"],
        }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    class_idx: Optional[int] = None,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes point estimate and 95% bootstrap confidence interval over image observations.
    Note: Does not claim patient independence due to missing BTXRD patient identifiers.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    def calc_score(indices: np.ndarray) -> Optional[float]:
        yt = y_true[indices]
        yp = y_pred[indices]
        ypr = y_prob[indices]

        if metric_name == "balanced_accuracy":
            recalls = recall_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
            return float(np.mean(recalls))
        elif metric_name == "overall_accuracy":
            return float(np.mean(yt == yp))
        elif metric_name == "macro_f1":
            return float(f1_score(yt, yp, average="macro", zero_division=0))
        elif metric_name == "weighted_f1":
            return float(f1_score(yt, yp, average="weighted", zero_division=0))
        elif metric_name == "macro_roc_auc":
            aucs = []
            for c in range(3):
                yb = (yt == c).astype(int)
                if len(np.unique(yb)) > 1:
                    aucs.append(roc_auc_score(yb, ypr[:, c]))
            return float(np.mean(aucs)) if len(aucs) == 3 else None
        elif metric_name == "macro_pr_auc":
            prs = []
            for c in range(3):
                yb = (yt == c).astype(int)
                if len(np.unique(yb)) > 1:
                    prs.append(average_precision_score(yb, ypr[:, c]))
            return float(np.mean(prs)) if len(prs) == 3 else None
        elif metric_name == "sensitivity" and class_idx is not None:
            tp = np.sum((yt == class_idx) & (yp == class_idx))
            fn = np.sum((yt == class_idx) & (yp != class_idx))
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        elif metric_name == "specificity" and class_idx is not None:
            tn = np.sum((yt != class_idx) & (yp != class_idx))
            fp = np.sum((yt != class_idx) & (yp == class_idx))
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        elif metric_name == "precision" and class_idx is not None:
            tp = np.sum((yt == class_idx) & (yp == class_idx))
            fp = np.sum((yt != class_idx) & (yp == class_idx))
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        elif metric_name == "npv" and class_idx is not None:
            tn = np.sum((yt != class_idx) & (yp != class_idx))
            fn = np.sum((yt == class_idx) & (yp != class_idx))
            return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        elif metric_name == "f1" and class_idx is not None:
            p = calc_score_prec(indices, class_idx)
            r = calc_score_rec(indices, class_idx)
            return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        elif metric_name == "roc_auc" and class_idx is not None:
            yb = (yt == class_idx).astype(int)
            if len(np.unique(yb)) > 1:
                return float(roc_auc_score(yb, ypr[:, class_idx]))
            return None
        elif metric_name == "pr_auc" and class_idx is not None:
            yb = (yt == class_idx).astype(int)
            if len(np.unique(yb)) > 1:
                return float(average_precision_score(yb, ypr[:, class_idx]))
            return None
        return None

    def calc_score_prec(indices, c):
        yt = y_true[indices]
        yp = y_pred[indices]
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def calc_score_rec(indices, c):
        yt = y_true[indices]
        yp = y_pred[indices]
        tp = np.sum((yt == c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    point_est = calc_score(np.arange(n))
    if point_est is None:
        point_est = 0.0

    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        s = calc_score(idx)
        if s is not None:
            scores.append(s)

    if len(scores) > 100:
        ci_lower = float(np.percentile(scores, 2.5))
        ci_upper = float(np.percentile(scores, 97.5))
    else:
        ci_lower, ci_upper = point_est, point_est

    return round(float(point_est), 4), round(float(ci_lower), 4), round(float(ci_upper), 4)


def run_stage_19_center2_evaluation() -> Dict[str, Any]:
    """
    Main evaluation pipeline for Stage 19.
    """
    settings = get_settings()
    base_dir = settings.base_dir
    manifest_path = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    frozen_ckpt_path = base_dir / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
    output_dir = base_dir / "ml_artifacts/btxrd_center2_external_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 19: BTXRD CENTER 2 EXTERNAL HOLDOUT EVALUATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Pre-flight verification
    print("[Pre-flight] Verifying frozen checkpoint existence and SHA-256 hash...", flush=True)
    assert frozen_ckpt_path.exists(), f"Frozen checkpoint missing at: {frozen_ckpt_path}"
    pre_eval_hash = compute_file_sha256(frozen_ckpt_path)
    print(f"[Pre-flight] Checkpoint SHA-256: {pre_eval_hash}", flush=True)
    assert pre_eval_hash == EXPECTED_CHECKPOINT_SHA256, (
        f"Cryptographic hash mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {pre_eval_hash}"
    )

    # 2. Dataset loading and verification
    print(f"\n[Dataset] Loading Center 2 external holdout cohort from {manifest_path}...", flush=True)
    dataset = BTXRDCenter2Dataset(manifest_path, base_dir)
    print(f"[Dataset] Verified Center 2 total image count: {len(dataset)}", flush=True)

    class_counts = {
        "NORMAL_BONE_RADIOGRAPH": sum(1 for r in dataset.records if r["target_class"] == "NORMAL_BONE_RADIOGRAPH"),
        "BENIGN_BONE_TUMOR": sum(1 for r in dataset.records if r["target_class"] == "BENIGN_BONE_TUMOR"),
        "PRIMARY_MALIGNANT_TUMOR": sum(1 for r in dataset.records if r["target_class"] == "PRIMARY_MALIGNANT_TUMOR"),
    }
    print(f"[Dataset] Center 2 Class Breakdown: {class_counts}", flush=True)
    assert class_counts == EXPECTED_CENTER2_DISTRIBUTION, (
        f"Distribution mismatch! Expected {EXPECTED_CENTER2_DISTRIBUTION}, got {class_counts}"
    )

    # 3. Model setup on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Model] Loading ResNet-34 on device: {device}...", flush=True)
    model = BTXRDResNet34(num_classes=3, pretrained=False, dropout_p=0.4, freeze_early=True)
    model.unfreeze_layer3()
    ckpt = torch.load(frozen_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Inference pass
    print("\n[Inference] Executing single-pass evaluation across all 549 Center 2 radiographs...", flush=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_ids: List[str] = []
    all_classes: List[str] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            targets = batch["label"].numpy()
            image_ids = batch["image_id"]
            target_classes = batch["target_class"]

            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.append(probs)
            all_ids.extend(image_ids)
            all_classes.extend(target_classes)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)

    assert len(all_targets) == EXPECTED_CENTER2_COUNT, f"Expected 549 predictions, got {len(all_targets)}"

    # 5. Core Metric Calculations
    print("\n[Metrics] Computing primary performance metrics and bootstrap 95% CIs (N=1000)...", flush=True)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Multiclass Brier score: (1/N) * sum_{i,k} (p_{ik} - y_{ik})^2
    y_one_hot = np.zeros_like(all_probs)
    for i, t in enumerate(all_targets):
        y_one_hot[i, t] = 1.0
    brier_score = float(np.mean(np.sum((all_probs - y_one_hot) ** 2, axis=1)))

    overall_acc, acc_ci_l, acc_ci_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "overall_accuracy")
    bal_acc, bal_ci_l, bal_ci_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "balanced_accuracy")
    macro_f1, mf1_ci_l, mf1_ci_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "macro_f1")
    weighted_f1, wf1_ci_l, wf1_ci_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "weighted_f1")
    macro_roc, mroc_ci_l, mroc_ci_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "macro_roc_auc")
    macro_pr, mpr_ci_l, mpr_ci_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "macro_pr_auc")

    # Per-class metrics
    per_class_summary = {}
    for c_idx in range(3):
        c_name = IDX_TO_CLASS[c_idx]
        supp = int(np.sum(all_targets == c_idx))

        sens, sens_l, sens_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "sensitivity", class_idx=c_idx)
        spec, spec_l, spec_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "specificity", class_idx=c_idx)
        prec, prec_l, prec_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "precision", class_idx=c_idx)
        npv, npv_l, npv_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "npv", class_idx=c_idx)
        f1, f1_l, f1_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "f1", class_idx=c_idx)
        roc, roc_l, roc_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "roc_auc", class_idx=c_idx)
        pr, pr_l, pr_u = bootstrap_metric_ci(all_targets, all_preds, all_probs, "pr_auc", class_idx=c_idx)

        per_class_summary[c_name] = {
            "class_index": c_idx,
            "support": supp,
            "prevalence": round(float(supp / len(all_targets)), 4),
            "sensitivity": {"estimate": sens, "ci_95": [sens_l, sens_u]},
            "specificity": {"estimate": spec, "ci_95": [spec_l, spec_u]},
            "precision_ppv": {"estimate": prec, "ci_95": [prec_l, prec_u]},
            "npv": {"estimate": npv, "ci_95": [npv_l, npv_u]},
            "f1_score": {"estimate": f1, "ci_95": [f1_l, f1_u]},
            "roc_auc_ovr": {"estimate": roc, "ci_95": [roc_l, roc_u]},
            "pr_auc": {"estimate": pr, "ci_95": [pr_l, pr_u]},
        }

    # Malignant specific error analysis
    mal_total = int(np.sum(all_targets == 2))
    mal_correct = int(np.sum((all_targets == 2) & (all_preds == 2)))
    mal_as_benign = int(np.sum((all_targets == 2) & (all_preds == 1)))
    mal_as_normal = int(np.sum((all_targets == 2) & (all_preds == 0)))
    non_mal_as_mal = int(np.sum((all_targets != 2) & (all_preds == 2)))

    malignant_errors = []
    for i in range(len(all_targets)):
        if all_targets[i] == 2 and all_preds[i] != 2:
            malignant_errors.append({
                "image_id": all_ids[i],
                "actual_class": "PRIMARY_MALIGNANT_TUMOR",
                "predicted_class": IDX_TO_CLASS[all_preds[i]],
                "prob_normal": round(float(all_probs[i, 0]), 4),
                "prob_benign": round(float(all_probs[i, 1]), 4),
                "prob_malignant": round(float(all_probs[i, 2]), 4),
            })
        elif all_targets[i] != 2 and all_preds[i] == 2:
            malignant_errors.append({
                "image_id": all_ids[i],
                "actual_class": IDX_TO_CLASS[all_targets[i]],
                "predicted_class": "PRIMARY_MALIGNANT_TUMOR",
                "prob_normal": round(float(all_probs[i, 0]), 4),
                "prob_benign": round(float(all_probs[i, 1]), 4),
                "prob_malignant": round(float(all_probs[i, 2]), 4),
            })

    # 6. Save Predictions CSV
    csv_path = output_dir / "center2_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "target_class",
            "actual_label",
            "prob_normal",
            "prob_benign",
            "prob_malignant",
            "predicted_label",
            "predicted_class",
            "is_correct",
        ])
        for i in range(len(all_targets)):
            writer.writerow([
                all_ids[i],
                all_classes[i],
                all_targets[i],
                f"{all_probs[i, 0]:.6f}",
                f"{all_probs[i, 1]:.6f}",
                f"{all_probs[i, 2]:.6f}",
                all_preds[i],
                IDX_TO_CLASS[all_preds[i]],
                1 if all_targets[i] == all_preds[i] else 0,
            ])
    print(f"[Artifacts] Saved predictions to {csv_path}", flush=True)

    # 7. Save Classification Metrics JSON
    classif_metrics = {
        "evaluation_cohort": "Center 2 External Holdout (N=549)",
        "overall_accuracy": {"estimate": overall_acc, "ci_95": [acc_ci_l, acc_ci_u]},
        "balanced_accuracy": {"estimate": bal_acc, "ci_95": [bal_ci_l, bal_ci_u]},
        "macro_f1": {"estimate": macro_f1, "ci_95": [mf1_ci_l, mf1_ci_u]},
        "weighted_f1": {"estimate": weighted_f1, "ci_95": [wf1_ci_l, wf1_ci_u]},
        "macro_roc_auc_ovr": {"estimate": macro_roc, "ci_95": [mroc_ci_l, mroc_ci_u]},
        "macro_pr_auc_ovr": {"estimate": macro_pr, "ci_95": [mpr_ci_l, mpr_ci_u]},
        "multiclass_brier_score": round(brier_score, 4),
        "calibration_status": "uncalibrated_raw_softmax",
        "decision_rule": "argmax(softmax(logits))",
    }
    with open(output_dir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(classif_metrics, f, indent=2)

    # 8. Save Per-Class Metrics JSON
    with open(output_dir / "per_class_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_class_summary, f, indent=2)

    # 9. Save Confusion Matrix JSON
    cm_dict = {
        "labels": ["NORMAL_BONE_RADIOGRAPH", "BENIGN_BONE_TUMOR", "PRIMARY_MALIGNANT_TUMOR"],
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_percentages": [[round(float(v) * 100, 2) for v in row] for row in cm_norm],
        "malignant_oncological_audit": {
            "total_malignant": mal_total,
            "correct_malignant_detected": mal_correct,
            "sensitivity": round(float(mal_correct / mal_total), 4),
            "false_benign_count": mal_as_benign,
            "false_normal_count": mal_as_normal,
            "false_positives_non_malignant_called_malignant": non_mal_as_mal,
        },
    }
    with open(output_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)

    # 10. Render Publication Diagnostic Visuals
    print("\n[Visuals] Rendering diagnostic figures for Center 2...", flush=True)

    # A. Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    classes_short = ["Normal", "Benign", "Malignant"]

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Center 2 Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_xticklabels(classes_short)
    axes[0].set_yticklabels(classes_short)
    axes[0].set_xlabel("Predicted Class", fontweight="bold")
    axes[0].set_ylabel("True Class", fontweight="bold")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{cm[i, j]}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)

    im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Center 2 Confusion Matrix (Row Recall %)", fontsize=13, fontweight="bold")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(classes_short)
    axes[1].set_yticklabels(classes_short)
    axes[1].set_xlabel("Predicted Class", fontweight="bold")
    axes[1].set_ylabel("True Class", fontweight="bold")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=12)

    plt.tight_layout()
    cm_plot_path = output_dir / "center2_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # B. ROC Curves OvR
    plt.figure(figsize=(8, 7))
    colors = ["#2b5c8f", "#d95f02", "#e41a1c"]
    for c_idx in range(3):
        yb = (all_targets == c_idx).astype(int)
        ypr = all_probs[:, c_idx]
        fpr, tpr, _ = roc_curve(yb, ypr)
        auc = roc_auc_score(yb, ypr)
        c_name = IDX_TO_CLASS[c_idx].replace("_BONE_RADIOGRAPH", "").replace("_BONE_TUMOR", "").replace("_TUMOR", "")
        plt.plot(fpr, tpr, color=colors[c_idx], lw=2.5,
                 label=f"{c_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Chance Level")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    plt.title(f"Center 2 External ROC Curves (Macro AUC = {macro_roc:.4f})", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = output_dir / "center2_roc_curves.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()

    # C. Precision-Recall Curves
    plt.figure(figsize=(8, 7))
    for c_idx in range(3):
        yb = (all_targets == c_idx).astype(int)
        ypr = all_probs[:, c_idx]
        prec_c, rec_c, _ = precision_recall_curve(yb, ypr)
        pr_score = average_precision_score(yb, ypr)
        c_name = IDX_TO_CLASS[c_idx].replace("_BONE_RADIOGRAPH", "").replace("_BONE_TUMOR", "").replace("_TUMOR", "")
        base_prev = np.mean(yb)
        plt.plot(rec_c, prec_c, color=colors[c_idx], lw=2.5,
                 label=f"{c_name} (PR-AUC = {pr_score:.4f}, Prev = {base_prev*100:.1f}%)")

    plt.xlabel("Recall (Sensitivity)", fontweight="bold")
    plt.ylabel("Precision (PPV)", fontweight="bold")
    plt.title(f"Center 2 External Precision-Recall Curves (Macro PR = {macro_pr:.4f})", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = output_dir / "center2_precision_recall_curves.png"
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()

    # 11. Post-evaluation SHA-256 Checksum Verification
    print("\n[Post-flight] Recomputing frozen checkpoint SHA-256 after evaluation...", flush=True)
    post_eval_hash = compute_file_sha256(frozen_ckpt_path)
    assert post_eval_hash == pre_eval_hash, "CRITICAL ERROR: Checkpoint was modified during evaluation!"
    print(f"[Post-flight] Checkpoint SHA-256 verified unchanged: {post_eval_hash}", flush=True)

    hash_verif_path = output_dir / "checkpoint_hash_verification.txt"
    with open(hash_verif_path, "w", encoding="utf-8") as f:
        f.write(f"Pre-Evaluation SHA-256:  {pre_eval_hash}\n")
        f.write(f"Post-Evaluation SHA-256: {post_eval_hash}\n")
        f.write("Status: CONFIRMED_IDENTICAL_BIT_LEVEL_INTEGRITY\n")

    # 12. Save Evaluation Manifest JSON
    eval_manifest = {
        "stage": "STAGE_19",
        "evaluation_name": "BTXRD Center 2 External Holdout Evaluation",
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_sha256": post_eval_hash,
        "model_version": FROZEN_MODEL_VERSION,
        "dataset_manifest": str(manifest_path),
        "cohort_center": "center_2",
        "cohort_partition": "external_test_center2",
        "total_images_evaluated": len(all_targets),
        "excluded_or_unreadable_images": 0,
        "class_mapping": CLASS_TO_IDX,
        "preprocessing_configuration": {
            "target_resolution": [512, 512],
            "aspect_ratio_mode": "letterbox_black_padding",
            "padding_color": [0, 0, 0],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "decision_rule": "argmax(softmax(logits))",
        "calibration_status": "uncalibrated_raw_softmax",
        "summary_metrics": {
            "overall_accuracy": overall_acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_roc_auc_ovr": macro_roc,
            "macro_pr_auc_ovr": macro_pr,
            "multiclass_brier_score": round(brier_score, 4),
        },
        "center_3_status": "SEALED_UNTOUCHED",
    }
    with open(output_dir / "evaluation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 19: CENTER 2 EXTERNAL HOLDOUT EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Overall Accuracy:  {overall_acc*100:.2f}% (95% CI: [{acc_ci_l*100:.2f}%, {acc_ci_u*100:.2f}%])")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}% (95% CI: [{bal_ci_l*100:.2f}%, {bal_ci_u*100:.2f}%])")
    print(f"Macro-F1 Score:    {macro_f1:.4f} (95% CI: [{mf1_ci_l:.4f}, {mf1_ci_u:.4f}])")
    print(f"Weighted-F1:       {weighted_f1:.4f} (95% CI: [{wf1_ci_l:.4f}, {wf1_ci_u:.4f}])")
    print(f"Macro ROC-AUC:     {macro_roc:.4f} (95% CI: [{mroc_ci_l:.4f}, {mroc_ci_u:.4f}])")
    print(f"Macro PR-AUC:      {macro_pr:.4f} (95% CI: [{mpr_ci_l:.4f}, {mpr_ci_u:.4f}])")
    print(f"Brier Score:       {brier_score:.4f} (uncalibrated)")
    print("-" * 80)
    print("Per-Class Performance:")
    for c_name, m in per_class_summary.items():
        print(f"  {c_name} (N={m['support']}):")
        print(f"    Sensitivity (Recall): {m['sensitivity']['estimate']*100:.2f}% (95% CI: [{m['sensitivity']['ci_95'][0]*100:.1f}%, {m['sensitivity']['ci_95'][1]*100:.1f}%])")
        print(f"    Specificity:          {m['specificity']['estimate']*100:.2f}% (95% CI: [{m['specificity']['ci_95'][0]*100:.1f}%, {m['specificity']['ci_95'][1]*100:.1f}%])")
        print(f"    Precision (PPV):      {m['precision_ppv']['estimate']*100:.2f}% (95% CI: [{m['precision_ppv']['ci_95'][0]*100:.1f}%, {m['precision_ppv']['ci_95'][1]*100:.1f}%])")
        print(f"    F1-Score:             {m['f1_score']['estimate']:.4f}")
        print(f"    ROC-AUC (OvR):        {m['roc_auc_ovr']['estimate']:.4f}")
        print(f"    PR-AUC:               {m['pr_auc']['estimate']:.4f}")
    print("-" * 80)
    print(f"Confusion Matrix (Counts):\n{cm}")
    print(f"Primary Malignancy Audit: {mal_correct}/{mal_total} ({mal_correct/mal_total*100:.1f}%) detected.")
    print(f"  False Benign: {mal_as_benign} | False Normal: {mal_as_normal} | False Positives: {non_mal_as_mal}")
    print("=" * 80)

    return {
        "overall_metrics": classif_metrics,
        "per_class_metrics": per_class_summary,
        "confusion_matrix": cm.tolist(),
        "malignant_oncological_audit": cm_dict["malignant_oncological_audit"],
        "post_eval_hash": post_eval_hash,
    }


if __name__ == "__main__":
    run_stage_19_center2_evaluation()

