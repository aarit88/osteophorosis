"""
OsteoGuard AI - Stage 20: BTXRD Center 3 External Stress Test.

Executes an independent, single-pass evaluation of the cryptographically frozen
Stage 18 ResNet-34 classifier on the Center 3 external stress test cohort (N=259).

Center 3 Epidemiological Profile:
- Total: 259 radiographs
- Specialized oncology referral cohort with severe imbalance:
  - NORMAL_BONE_RADIOGRAPH: 27 (10.42%)
  - BENIGN_BONE_TUMOR: 201 (77.61%)
  - PRIMARY_MALIGNANT_TUMOR: 31 (11.97%)

Preserves:
- Frozen ResNet-34 weights (Epoch 18, SHA-256: 21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD)
- 512x512 aspect-ratio-preserving letterbox with symmetrical black border (0, 0, 0)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Standard argmax classification rule over raw, uncalibrated softmax probabilities
- Zero parameter tuning, calibration fitting, or threshold adjustment
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
EXPECTED_CENTER3_COUNT = 259
EXPECTED_CENTER3_DISTRIBUTION = {
    "NORMAL_BONE_RADIOGRAPH": 27,
    "BENIGN_BONE_TUMOR": 201,
    "PRIMARY_MALIGNANT_TUMOR": 31,
}


def compute_file_sha256(filepath: Path) -> str:
    """Computes uppercase SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


class BTXRDCenter3Dataset(Dataset):
    """
    Dedicated dataset strictly isolating Center 3 external stress test radiographs.
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

                # Isolation check: ONLY center_3 external holdout
                if center != "center_3":
                    continue
                if split != "external_test_center3":
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
                    "center": "center_3",
                })

        if len(self.records) != EXPECTED_CENTER3_COUNT:
            raise ValueError(
                f"Center 3 cohort count mismatch! Expected {EXPECTED_CENTER3_COUNT}, found {len(self.records)}"
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


def compute_fast_bootstrap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes point estimates and 95% bootstrap confidence intervals for all metrics
    using vectorized iteration over image observations.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_indices = rng.choice(n, size=(n_bootstraps, n), replace=True)

    # Point estimates
    overall_acc = float(np.mean(y_true == y_pred))
    recalls_pt = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    bal_acc = float(np.mean(recalls_pt))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    roc_aucs_pt = []
    pr_aucs_pt = []
    for c in range(3):
        yb = (y_true == c).astype(int)
        roc_aucs_pt.append(float(roc_auc_score(yb, y_prob[:, c])))
        pr_aucs_pt.append(float(average_precision_score(yb, y_prob[:, c])))
    macro_roc = float(np.mean(roc_aucs_pt))
    macro_pr = float(np.mean(pr_aucs_pt))

    # Bootstrap distributions
    boot_accs = []
    boot_bal_accs = []
    boot_macro_f1s = []
    boot_weighted_f1s = []
    boot_macro_rocs = []
    boot_macro_prs = []

    per_class_boots = {c: {
        "sens": [], "spec": [], "prec": [], "npv": [], "f1": [], "roc": [], "pr": []
    } for c in range(3)}

    for b in range(n_bootstraps):
        idx = boot_indices[b]
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]

        # Overall
        boot_accs.append(np.mean(yt == yp))
        recs = recall_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
        boot_bal_accs.append(np.mean(recs))
        boot_macro_f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        boot_weighted_f1s.append(f1_score(yt, yp, average="weighted", zero_division=0))

        # Per class & ROC/PR
        rocs_b = []
        prs_b = []
        for c in range(3):
            yb = (yt == c).astype(int)
            if len(np.unique(yb)) > 1:
                r_c = roc_auc_score(yb, ypr[:, c])
                p_c = average_precision_score(yb, ypr[:, c])
                rocs_b.append(r_c)
                prs_b.append(p_c)
                per_class_boots[c]["roc"].append(r_c)
                per_class_boots[c]["pr"].append(p_c)

            tp = np.sum((yt == c) & (yp == c))
            fp = np.sum((yt != c) & (yp == c))
            tn = np.sum((yt != c) & (yp != c))
            fn = np.sum((yt == c) & (yp != c))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

            per_class_boots[c]["sens"].append(sens)
            per_class_boots[c]["spec"].append(spec)
            per_class_boots[c]["prec"].append(prec)
            per_class_boots[c]["npv"].append(npv)
            per_class_boots[c]["f1"].append(f1)

        if len(rocs_b) == 3:
            boot_macro_rocs.append(np.mean(rocs_b))
        if len(prs_b) == 3:
            boot_macro_prs.append(np.mean(prs_b))

    def get_ci(arr, pt):
        if len(arr) > 50:
            return round(float(pt), 4), round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)
        return round(float(pt), 4), round(float(pt), 4), round(float(pt), 4)

    overall_results = {
        "overall_accuracy": get_ci(boot_accs, overall_acc),
        "balanced_accuracy": get_ci(boot_bal_accs, bal_acc),
        "macro_f1": get_ci(boot_macro_f1s, macro_f1),
        "weighted_f1": get_ci(boot_weighted_f1s, weighted_f1),
        "macro_roc_auc_ovr": get_ci(boot_macro_rocs, macro_roc),
        "macro_pr_auc_ovr": get_ci(boot_macro_prs, macro_pr),
    }

    per_class_results = {}
    for c in range(3):
        c_name = IDX_TO_CLASS[c]
        supp = int(np.sum(y_true == c))

        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        fn = np.sum((y_true == c) & (y_pred != c))

        sens_pt = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec_pt = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec_pt = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv_pt = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1_pt = float(2 * prec_pt * sens_pt / (prec_pt + sens_pt)) if (prec_pt + sens_pt) > 0 else 0.0

        per_class_results[c_name] = {
            "class_index": c,
            "support": supp,
            "prevalence": round(float(supp / len(y_true)), 4),
            "sensitivity": {"estimate": get_ci(per_class_boots[c]["sens"], sens_pt)[0], "ci_95": [get_ci(per_class_boots[c]["sens"], sens_pt)[1], get_ci(per_class_boots[c]["sens"], sens_pt)[2]]},
            "specificity": {"estimate": get_ci(per_class_boots[c]["spec"], spec_pt)[0], "ci_95": [get_ci(per_class_boots[c]["spec"], spec_pt)[1], get_ci(per_class_boots[c]["spec"], spec_pt)[2]]},
            "precision_ppv": {"estimate": get_ci(per_class_boots[c]["prec"], prec_pt)[0], "ci_95": [get_ci(per_class_boots[c]["prec"], prec_pt)[1], get_ci(per_class_boots[c]["prec"], prec_pt)[2]]},
            "npv": {"estimate": get_ci(per_class_boots[c]["npv"], npv_pt)[0], "ci_95": [get_ci(per_class_boots[c]["npv"], npv_pt)[1], get_ci(per_class_boots[c]["npv"], npv_pt)[2]]},
            "f1_score": {"estimate": get_ci(per_class_boots[c]["f1"], f1_pt)[0], "ci_95": [get_ci(per_class_boots[c]["f1"], f1_pt)[1], get_ci(per_class_boots[c]["f1"], f1_pt)[2]]},
            "roc_auc_ovr": {"estimate": get_ci(per_class_boots[c]["roc"], roc_aucs_pt[c])[0], "ci_95": [get_ci(per_class_boots[c]["roc"], roc_aucs_pt[c])[1], get_ci(per_class_boots[c]["roc"], roc_aucs_pt[c])[2]]},
            "pr_auc": {"estimate": get_ci(per_class_boots[c]["pr"], pr_aucs_pt[c])[0], "ci_95": [get_ci(per_class_boots[c]["pr"], pr_aucs_pt[c])[1], get_ci(per_class_boots[c]["pr"], pr_aucs_pt[c])[2]]},
        }

    return overall_results, per_class_results


def run_stage_20_center3_evaluation() -> Dict[str, Any]:
    """
    Main orchestration routine for Stage 20 Center 3 External Stress Test.
    """
    settings = get_settings()
    base_dir = settings.base_dir
    manifest_path = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    frozen_ckpt_path = base_dir / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
    output_dir = base_dir / "ml_artifacts/btxrd_center3_external_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 20: BTXRD CENTER 3 EXTERNAL STRESS TEST EVALUATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Pre-flight Checkpoint Verification
    print("[Pre-flight] Verifying frozen checkpoint existence and SHA-256 hash...", flush=True)
    assert frozen_ckpt_path.exists(), f"Frozen checkpoint missing at: {frozen_ckpt_path}"
    pre_eval_hash = compute_file_sha256(frozen_ckpt_path)
    print(f"[Pre-flight] Checkpoint SHA-256: {pre_eval_hash}", flush=True)
    assert pre_eval_hash == EXPECTED_CHECKPOINT_SHA256, (
        f"Cryptographic hash mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {pre_eval_hash}"
    )

    # 2. Dataset Loading & Cohort Verification
    print(f"\n[Dataset] Loading Center 3 external stress test cohort from {manifest_path}...", flush=True)
    dataset = BTXRDCenter3Dataset(manifest_path, base_dir)
    print(f"[Dataset] Verified Center 3 total image count: {len(dataset)}", flush=True)

    class_counts = {
        "NORMAL_BONE_RADIOGRAPH": sum(1 for r in dataset.records if r["target_class"] == "NORMAL_BONE_RADIOGRAPH"),
        "BENIGN_BONE_TUMOR": sum(1 for r in dataset.records if r["target_class"] == "BENIGN_BONE_TUMOR"),
        "PRIMARY_MALIGNANT_TUMOR": sum(1 for r in dataset.records if r["target_class"] == "PRIMARY_MALIGNANT_TUMOR"),
    }
    print(f"[Dataset] Center 3 Class Breakdown: {class_counts}", flush=True)
    assert class_counts == EXPECTED_CENTER3_DISTRIBUTION, (
        f"Distribution mismatch! Expected {EXPECTED_CENTER3_DISTRIBUTION}, got {class_counts}"
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
    print("\n[Inference] Executing single-pass evaluation across all 259 Center 3 radiographs...", flush=True)
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

    assert len(all_targets) == EXPECTED_CENTER3_COUNT, f"Expected 259 predictions, got {len(all_targets)}"

    # 5. Core Metric Calculations
    print("\n[Metrics] Computing primary performance metrics and bootstrap 95% CIs (N=1000)...", flush=True)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Multiclass Brier score: (1/N) * sum_{i,k} (p_{ik} - y_{ik})^2
    y_one_hot = np.zeros_like(all_probs)
    for i, t in enumerate(all_targets):
        y_one_hot[i, t] = 1.0
    brier_score = float(np.mean(np.sum((all_probs - y_one_hot) ** 2, axis=1)))

    overall_results, per_class_summary = compute_fast_bootstrap_cis(all_targets, all_preds, all_probs, n_bootstraps=1000, seed=42)

    # Malignant specific error analysis
    mal_total = int(np.sum(all_targets == 2))
    mal_correct = int(np.sum((all_targets == 2) & (all_preds == 2)))
    mal_as_benign = int(np.sum((all_targets == 2) & (all_preds == 1)))
    mal_as_normal = int(np.sum((all_targets == 2) & (all_preds == 0)))
    non_mal_as_mal = int(np.sum((all_targets != 2) & (all_preds == 2)))

    # Normal specific error analysis
    norm_total = int(np.sum(all_targets == 0))
    norm_correct = int(np.sum((all_targets == 0) & (all_preds == 0)))
    norm_as_benign = int(np.sum((all_targets == 0) & (all_preds == 1)))
    norm_as_mal = int(np.sum((all_targets == 0) & (all_preds == 2)))

    # Benign specific error analysis
    ben_total = int(np.sum(all_targets == 1))
    ben_correct = int(np.sum((all_targets == 1) & (all_preds == 1)))
    ben_as_norm = int(np.sum((all_targets == 1) & (all_preds == 0)))
    ben_as_mal = int(np.sum((all_targets == 1) & (all_preds == 2)))

    # 6. Save Predictions CSV
    csv_path = output_dir / "center3_predictions.csv"
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
        "evaluation_cohort": "Center 3 External Stress Test (N=259)",
        "overall_accuracy": {"estimate": overall_results["overall_accuracy"][0], "ci_95": [overall_results["overall_accuracy"][1], overall_results["overall_accuracy"][2]]},
        "balanced_accuracy": {"estimate": overall_results["balanced_accuracy"][0], "ci_95": [overall_results["balanced_accuracy"][1], overall_results["balanced_accuracy"][2]]},
        "macro_f1": {"estimate": overall_results["macro_f1"][0], "ci_95": [overall_results["macro_f1"][1], overall_results["macro_f1"][2]]},
        "weighted_f1": {"estimate": overall_results["weighted_f1"][0], "ci_95": [overall_results["weighted_f1"][1], overall_results["weighted_f1"][2]]},
        "macro_roc_auc_ovr": {"estimate": overall_results["macro_roc_auc_ovr"][0], "ci_95": [overall_results["macro_roc_auc_ovr"][1], overall_results["macro_roc_auc_ovr"][2]]},
        "macro_pr_auc_ovr": {"estimate": overall_results["macro_pr_auc_ovr"][0], "ci_95": [overall_results["macro_pr_auc_ovr"][1], overall_results["macro_pr_auc_ovr"][2]]},
        "multiclass_brier_score": round(brier_score, 4),
        "calibration_status": "uncalibrated_raw_softmax",
        "decision_rule": "argmax(softmax(logits))",
        "epidemiological_notice": (
            "Specialized oncology referral cohort with severe imbalance (only 27 normal controls, 10.42%). "
            "Normal specificity and sensitivity estimates have wider confidence intervals."
        ),
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
        "error_analysis": {
            "malignant": {
                "support": mal_total,
                "correct_detected": mal_correct,
                "sensitivity": round(float(mal_correct / mal_total), 4),
                "misclassified_as_benign": mal_as_benign,
                "misclassified_as_normal": mal_as_normal,
                "false_positives_from_other_classes": non_mal_as_mal,
            },
            "normal": {
                "support": norm_total,
                "correct_detected": norm_correct,
                "sensitivity": round(float(norm_correct / norm_total), 4),
                "misclassified_as_benign": norm_as_benign,
                "misclassified_as_malignant": norm_as_mal,
            },
            "benign": {
                "support": ben_total,
                "correct_detected": ben_correct,
                "sensitivity": round(float(ben_correct / ben_total), 4),
                "misclassified_as_normal": ben_as_norm,
                "misclassified_as_malignant": ben_as_mal,
            },
        },
    }
    with open(output_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)

    # 10. Render Publication Diagnostic Visuals
    print("\n[Visuals] Rendering diagnostic figures for Center 3...", flush=True)

    # A. Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    classes_short = ["Normal", "Benign", "Malignant"]

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Center 3 Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
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
    axes[1].set_title("Center 3 Confusion Matrix (Row Recall %)", fontsize=13, fontweight="bold")
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
    cm_plot_path = output_dir / "center3_confusion_matrix.png"
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
    plt.title(f"Center 3 External ROC Curves (Macro AUC = {overall_results['macro_roc_auc_ovr'][0]:.4f})", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = output_dir / "center3_roc_curves.png"
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
    plt.title(f"Center 3 External Precision-Recall Curves (Macro PR = {overall_results['macro_pr_auc_ovr'][0]:.4f})", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = output_dir / "center3_precision_recall_curves.png"
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
        "stage": "STAGE_20",
        "evaluation_name": "BTXRD Center 3 External Stress Test",
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_sha256": post_eval_hash,
        "model_version": FROZEN_MODEL_VERSION,
        "dataset_manifest": str(manifest_path),
        "cohort_center": "center_3",
        "cohort_partition": "external_test_center3",
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
            "overall_accuracy": overall_results["overall_accuracy"][0],
            "balanced_accuracy": overall_results["balanced_accuracy"][0],
            "macro_f1": overall_results["macro_f1"][0],
            "weighted_f1": overall_results["weighted_f1"][0],
            "macro_roc_auc_ovr": overall_results["macro_roc_auc_ovr"][0],
            "macro_pr_auc_ovr": overall_results["macro_pr_auc_ovr"][0],
            "multiclass_brier_score": round(brier_score, 4),
        },
        "center_1_status": "UNTOUCHED_ISOLATED",
        "center_2_status": "UNTOUCHED_ISOLATED",
    }
    with open(output_dir / "evaluation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 20: CENTER 3 EXTERNAL STRESS TEST COMPLETE")
    print("=" * 80)
    print(f"Overall Accuracy:  {overall_results['overall_accuracy'][0]*100:.2f}% (95% CI: [{overall_results['overall_accuracy'][1]*100:.2f}%, {overall_results['overall_accuracy'][2]*100:.2f}%])")
    print(f"Balanced Accuracy: {overall_results['balanced_accuracy'][0]*100:.2f}% (95% CI: [{overall_results['balanced_accuracy'][1]*100:.2f}%, {overall_results['balanced_accuracy'][2]*100:.2f}%])")
    print(f"Macro-F1 Score:    {overall_results['macro_f1'][0]:.4f} (95% CI: [{overall_results['macro_f1'][1]:.4f}, {overall_results['macro_f1'][2]:.4f}])")
    print(f"Weighted-F1:       {overall_results['weighted_f1'][0]:.4f} (95% CI: [{overall_results['weighted_f1'][1]:.4f}, {overall_results['weighted_f1'][2]:.4f}])")
    print(f"Macro ROC-AUC:     {overall_results['macro_roc_auc_ovr'][0]:.4f} (95% CI: [{overall_results['macro_roc_auc_ovr'][1]:.4f}, {overall_results['macro_roc_auc_ovr'][2]:.4f}])")
    print(f"Macro PR-AUC:      {overall_results['macro_pr_auc_ovr'][0]:.4f} (95% CI: [{overall_results['macro_pr_auc_ovr'][1]:.4f}, {overall_results['macro_pr_auc_ovr'][2]:.4f}])")
    print(f"Brier Score:       {brier_score:.4f} (uncalibrated)")
    print("-" * 80)
    print("Per-Class Performance:")
    for c_name, m in per_class_summary.items():
        print(f"  {c_name} (N={m['support']}):")
        print(f"    Sensitivity (Recall): {m['sensitivity']['estimate']*100:.2f}% (95% CI: [{m['sensitivity']['ci_95'][0]*100:.1f}%, {m['sensitivity']['ci_95'][1]*100:.1f}%])")
        print(f"    Specificity:          {m['specificity']['estimate']*100:.2f}% (95% CI: [{m['specificity']['ci_95'][0]*100:.1f}%, {m['specificity']['ci_95'][1]*100:.1f}%])")
        print(f"    Precision (PPV):      {m['precision_ppv']['estimate']*100:.2f}% (95% CI: [{m['precision_ppv']['ci_95'][0]*100:.1f}%, {m['precision_ppv']['ci_95'][1]*100:.1f}%])")
        print(f"    NPV:                  {m['npv']['estimate']*100:.2f}% (95% CI: [{m['npv']['ci_95'][0]*100:.1f}%, {m['npv']['ci_95'][1]*100:.1f}%])")
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
        "error_analysis": cm_dict["error_analysis"],
        "post_eval_hash": post_eval_hash,
    }


if __name__ == "__main__":
    run_stage_20_center3_evaluation()

