"""
OsteoGuard AI - Stage 13: Final Bone Metastasis Ensemble Construction and Validation Evaluation.

Evaluates the five-model ResNet-18 ensemble on the separate original 44-patient validation cohort.
Computes comprehensive metrics, bootstrap confidence intervals, threshold analysis,
calibration diagnostics, model disagreement analysis, and publication-ready visual artifacts.
"""

import csv
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from scipy.optimize import minimize

from ml_service.config.settings import get_settings
from ml_service.datasets.bone_scan_dataset import BoneScanPatientDataset
from ml_service.models.bone_scan_ensemble import BoneScanEnsemble


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes point estimate and 95% percentile bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    point_est = float(metric_func(y_true, y_prob))
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        # Ensure both classes are present in sample
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    if len(bootstrapped_scores) > 0:
        ci_lower = float(np.percentile(bootstrapped_scores, 100 * (alpha / 2)))
        ci_upper = float(np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2)))
    else:
        ci_lower, ci_upper = point_est, point_est

    return round(point_est, 4), round(ci_lower, 4), round(ci_upper, 4)


def optimize_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits temperature scaling parameter T on logits to minimize negative log-likelihood.
    Returns (optimal_T, original_nll, calibrated_nll).
    """
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss()

    orig_nll = float(bce(logits_tensor, labels_tensor).item())

    def loss_func(T):
        scaled_logits = logits_tensor / T[0]
        return float(bce(scaled_logits, labels_tensor).item())

    res = minimize(loss_func, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    optimal_T = float(res.x[0])
    calibrated_nll = float(res.fun)

    return round(optimal_T, 4), round(orig_nll, 4), round(calibrated_nll, 4)


def run_stage_13_ensemble_validation():
    """
    Main orchestration routine for Stage 13.
    """
    settings = get_settings()
    base_dir = settings.base_dir
    master_manifest = base_dir / "ml_service/data/splits/bone_scan_split_manifest.csv"
    cv_dir = base_dir / "ml_artifacts/bone_scan_cv"
    output_dir = base_dir / "ml_artifacts/bone_scan_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("OSTEO GUARD AI - STAGE 13: BONE SCINTIGRAPHY ENSEMBLE VALIDATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})", flush=True)

    # 2. Build Ensemble
    print("[Ensemble] Loading 5 cross-validation best checkpoints...", flush=True)
    ensemble = BoneScanEnsemble(checkpoint_dir=cv_dir, device=device)
    ensemble_meta = ensemble.get_ensemble_metadata()
    print(f"[Ensemble] Initialized with {ensemble_meta['num_models']} models ({ensemble_meta['total_ensemble_parameters']:,} parameters total)", flush=True)

    # 3. Load Validation Dataset
    print(f"[Dataset] Loading original validation cohort from {master_manifest}...", flush=True)
    val_dataset = BoneScanPatientDataset(master_manifest, split="val", is_train=False, augment=False)
    counts = val_dataset.get_class_counts()
    print(f"[Dataset] Validation cohort: {counts['total_patients']} patients "
          f"({counts['positive_patients']} positive, {counts['negative_patients']} negative)", flush=True)
    assert counts["total_patients"] == 44, f"Expected 44 val patients, got {counts['total_patients']}"
    assert counts["positive_patients"] == 9, f"Expected 9 val pos patients, got {counts['positive_patients']}"
    assert counts["negative_patients"] == 35, f"Expected 35 val neg patients, got {counts['negative_patients']}"

    # 4. Ensemble Inference
    print("\n[Inference] Running ensemble inference on 44 validation patients...", flush=True)
    patient_predictions = []
    y_true_list = []
    y_prob_list = []
    y_disagree_list = []
    individual_fold_probs = {f"fold_{i}": [] for i in range(5)}

    start_infer_time = time.time()
    for i in range(len(val_dataset)):
        item = val_dataset[i]
        pid = item["patient_id"]
        y_val = float(item["label"].item())
        tgt_cls = item["target_class"]

        ant_tensor = item["anterior_tensor"]
        post_tensor = item["posterior_tensor"]

        pred_res = ensemble(ant_tensor, post_tensor)

        p_ens = pred_res["prob_patient_ensemble"]
        std_dis = pred_res["std_disagreement"]
        min_p = pred_res["min_patient_prob"]
        max_p = pred_res["max_patient_prob"]
        fold_pts = pred_res["fold_patient_probs"]

        y_true_list.append(int(y_val))
        y_prob_list.append(p_ens)
        y_disagree_list.append(std_dis)

        for f_idx in range(5):
            individual_fold_probs[f"fold_{f_idx}"].append(fold_pts[f_idx])

        pred_label_050 = int(p_ens >= 0.50)
        pred_name_050 = "BONE_METASTASIS" if pred_label_050 == 1 else "NO_BONE_METASTASIS"

        patient_predictions.append({
            "patient_id": pid,
            "target_class": tgt_cls,
            "actual_label": int(y_val),
            "prob_anterior_mean": round(pred_res["prob_anterior_mean"], 6),
            "prob_posterior_mean": round(pred_res["prob_posterior_mean"], 6),
            "prob_patient_ensemble": round(p_ens, 6),
            "std_disagreement": round(std_dis, 6),
            "min_patient_prob": round(min_p, 6),
            "max_patient_prob": round(max_p, 6),
            "fold_0_prob": round(fold_pts[0], 6),
            "fold_1_prob": round(fold_pts[1], 6),
            "fold_2_prob": round(fold_pts[2], 6),
            "fold_3_prob": round(fold_pts[3], 6),
            "fold_4_prob": round(fold_pts[4], 6),
            "pred_label_050": pred_label_050,
            "pred_name_050": pred_name_050,
        })

    infer_duration = time.time() - start_infer_time
    print(f"[Inference] Completed in {infer_duration:.2f} seconds ({infer_duration/44*1000:.1f} ms/patient)", flush=True)

    # Save ensemble_validation_predictions.csv
    preds_csv_path = output_dir / "ensemble_validation_predictions.csv"
    with open(preds_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(patient_predictions[0].keys()))
        writer.writeheader()
        writer.writerows(patient_predictions)
    print(f"[Artifacts] Saved predictions to {preds_csv_path}", flush=True)

    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_prob_list, dtype=float)
    y_disagree = np.array(y_disagree_list, dtype=float)

    # 5. Core Performance Metrics with Bootstrap CIs
    roc_pt, roc_lower, roc_upper = compute_bootstrap_ci(y_true, y_prob, roc_auc_score)
    pr_pt, pr_lower, pr_upper = compute_bootstrap_ci(y_true, y_prob, average_precision_score)
    brier = round(float(brier_score_loss(y_true, y_prob)), 4)
    baseline_pr = round(9 / 44, 4)

    # Default threshold 0.50
    pred_050 = (y_prob >= 0.50).astype(int)
    tn_050, fp_050, fn_050, tp_050 = confusion_matrix(y_true, pred_050).ravel()
    sens_050 = float(tp_050 / (tp_050 + fn_050))
    spec_050 = float(tn_050 / (tn_050 + fp_050))
    prec_050 = float(tp_050 / (tp_050 + fp_050)) if (tp_050 + fp_050) > 0 else 0.0
    f1_050 = float(2 * prec_050 * sens_050 / (prec_050 + sens_050)) if (prec_050 + sens_050) > 0 else 0.0
    acc_050 = float((tp_050 + tn_050) / 44)
    bal_acc_050 = float((sens_050 + spec_050) / 2.0)
    npv_050 = float(tn_050 / (tn_050 + fn_050)) if (tn_050 + fn_050) > 0 else 0.0

    # 6. Threshold Sweep (0.05 to 0.95)
    thresholds = np.arange(0.05, 0.96, 0.05)
    threshold_records = []
    candidate_thresholds_data = {}

    for th in thresholds:
        th_val = round(float(th), 2)
        pred_th = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_th).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
        acc = float((tp + tn) / 44)
        bal_acc = float((sens + spec) / 2.0)
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        rec = {
            "threshold": th_val,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "negative_predictive_value": round(npv, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }
        threshold_records.append(rec)

        if th_val in (0.35, 0.50, 0.60):
            candidate_thresholds_data[str(th_val)] = rec

    th_csv_path = output_dir / "ensemble_threshold_analysis.csv"
    with open(th_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_records[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_records)
    print(f"[Artifacts] Saved threshold analysis to {th_csv_path}", flush=True)

    # 7. Calibration Assessment & Temperature Scaling
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
    # ECE on 5 bins (due to N=44)
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_uncalibrated = 0.0
    cal_bins = []
    for b in range(n_bins):
        mask = (y_prob > bin_edges[b]) & (y_prob <= bin_edges[b + 1])
        cnt = int(np.sum(mask))
        if cnt > 0:
            b_acc = float(np.mean(y_true[mask]))
            b_conf = float(np.mean(y_prob[mask]))
            b_err = abs(b_acc - b_conf)
            ece_uncalibrated += (cnt / len(y_true)) * b_err
            cal_bins.append({
                "bin": b + 1,
                "range": [round(bin_edges[b], 2), round(bin_edges[b + 1], 2)],
                "count": cnt,
                "mean_pred": round(b_conf, 4),
                "actual_fraction": round(b_acc, 4),
                "error": round(b_err, 4),
            })

    # Logit calculation for temperature scaling: logit = log(p / (1 - p))
    eps = 1e-7
    clipped_probs = np.clip(y_prob, eps, 1.0 - eps)
    logits = np.log(clipped_probs / (1.0 - clipped_probs))

    opt_T, orig_nll, cal_nll = optimize_temperature_scaling(logits, y_true)
    scaled_probs = 1.0 / (1.0 + np.exp(-logits / opt_T))
    brier_scaled = round(float(brier_score_loss(y_true, scaled_probs)), 4)

    # Scaled ECE
    ece_scaled = 0.0
    for b in range(n_bins):
        mask = (scaled_probs > bin_edges[b]) & (scaled_probs <= bin_edges[b + 1])
        cnt = int(np.sum(mask))
        if cnt > 0:
            b_acc = float(np.mean(y_true[mask]))
            b_conf = float(np.mean(scaled_probs[mask]))
            ece_scaled += (cnt / len(y_true)) * abs(b_acc - b_conf)

    calibration_analysis = {
        "validation_cohort_size": 44,
        "positive_cases": 9,
        "negative_cases": 35,
        "prevalence_baseline": baseline_pr,
        "uncalibrated_metrics": {
            "brier_score": brier,
            "expected_calibration_error_ece": round(float(ece_uncalibrated), 4),
            "reliability_bins_5": cal_bins,
        },
        "temperature_scaling_experimental": {
            "optimal_temperature_T": opt_T,
            "original_nll": orig_nll,
            "scaled_nll": cal_nll,
            "scaled_brier_score": brier_scaled,
            "scaled_ece": round(float(ece_scaled), 4),
            "clinical_caveat": (
                "Temperature scaling fitted on N=44 patients with only 9 positive cases is prone to "
                "sampling noise. It demonstrates that temperature scaling can compress over-confident logits, "
                "but raw probabilities should be presented as ranking scores or dual-threshold bands rather "
                "than definitive Bayesian risk probabilities until tested on larger multi-center cohorts."
            ),
        },
    }

    cal_json_path = output_dir / "ensemble_calibration_analysis.json"
    with open(cal_json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_analysis, f, indent=2)
    print(f"[Artifacts] Saved calibration analysis to {cal_json_path}", flush=True)

    # 8. Save Metrics Summary JSON
    metrics_summary = {
        "experiment": "Bone Scan 5-Model Ensemble Validation Cohort Evaluation",
        "validation_cohort": {
            "total_patients": 44,
            "positive_patients": 9,
            "negative_patients": 35,
            "prevalence": baseline_pr,
            "cohort_note": "Original sealed validation cohort (44 patients). Zero holdout test patients accessed.",
        },
        "ensemble_provenance": ensemble_meta,
        "discriminative_performance": {
            "roc_auc": roc_pt,
            "roc_auc_95_ci": [roc_lower, roc_upper],
            "pr_auc": pr_pt,
            "pr_auc_95_ci": [pr_lower, pr_upper],
            "pr_auc_baseline": baseline_pr,
            "brier_score": brier,
        },
        "operating_points": {
            "default_threshold_050": {
                "threshold": 0.50,
                "sensitivity": round(sens_050, 4),
                "specificity": round(spec_050, 4),
                "precision": round(prec_050, 4),
                "f1_score": round(f1_050, 4),
                "accuracy": round(acc_050, 4),
                "balanced_accuracy": round(bal_acc_050, 4),
                "negative_predictive_value": round(npv_050, 4),
                "confusion_matrix": {
                    "true_positives": int(tp_050),
                    "false_positives": int(fp_050),
                    "true_negatives": int(tn_050),
                    "false_negatives": int(fn_050),
                },
            },
            "candidate_screening_threshold_035": candidate_thresholds_data.get("0.35"),
            "candidate_balanced_threshold_060": candidate_thresholds_data.get("0.6"),
        },
        "model_disagreement": {
            "overall_mean_std": round(float(np.mean(y_disagree)), 4),
            "positive_mean_std": round(float(np.mean(y_disagree[y_true == 1])), 4),
            "negative_mean_std": round(float(np.mean(y_disagree[y_true == 0])), 4),
            "interpretation": (
                "Standard deviation across the 5 models serves as an experimental disagreement index. "
                "Higher disagreement occurs at intermediate ambiguous lesions and lower at consensus extremes."
            ),
        },
    }

    metrics_json_path = output_dir / "ensemble_validation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[Artifacts] Saved validation metrics to {metrics_json_path}", flush=True)

    # 9. Save ensemble_config.json
    config_data = {
        "stage": "STAGE_13",
        "ensemble_type": "5-Fold ResNet-18 Averaging Ensemble",
        "input_resolution": [1024, 256],
        "aspect_ratio": "1:4 (W=256, H=1024)",
        "preprocessing": "16-bit counts clipped at 25000, normalized to [0, 1]",
        "patient_aggregation": "P_patient_k = max(P_ant_k, P_post_k)",
        "ensemble_aggregation": "P_ensemble = mean(P_patient_0..4)",
        "disagreement_metric": "std(P_patient_0..4)",
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "checkpoints": ensemble_meta["checkpoints"],
    }
    with open(output_dir / "ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"[Artifacts] Saved ensemble config to {output_dir / 'ensemble_config.json'}", flush=True)

    # 10. Visualizations
    print("[Visuals] Generating publication-quality diagnostic plots...", flush=True)

    # 10.1 Confusion Matrices Side-by-Side (θ=0.50 and θ=0.35)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_050 = np.array([[tn_050, fp_050], [fn_050, tp_050]])
    rec_035 = candidate_thresholds_data.get("0.35", {})
    cm_035 = np.array([[rec_035.get("true_negatives", 0), rec_035.get("false_positives", 0)],
                       [rec_035.get("false_negatives", 0), rec_035.get("true_positives", 0)]])

    for ax, cm, th_title, sens_val, spec_val in zip(
        axes,
        [cm_050, cm_035],
        ["Default Threshold (θ = 0.50)", "Screening Candidate (θ = 0.35)"],
        [sens_050, rec_035.get("sensitivity", 0.0)],
        [spec_050, rec_035.get("specificity", 0.0)],
    ):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{th_title}\nSens: {sens_val*100:.1f}% | Spec: {spec_val*100:.1f}%", fontsize=11, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Non-Metastatic", "Metastatic"])
        ax.set_yticklabels(["Non-Metastatic", "Metastatic"])
        ax.set_ylabel("True Diagnosis", fontsize=10)
        ax.set_xlabel("Predicted Diagnosis", fontsize=10)

        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                ax.text(c, r, format(cm[r, c], "d"),
                        ha="center", va="center",
                        color="white" if cm[r, c] > thresh else "black",
                        fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_confusion_matrix.png", dpi=300)
    plt.close()

    # 10.2 Ensemble ROC Curve
    plt.figure(figsize=(8, 7))
    for f_idx in range(5):
        f_p = individual_fold_probs[f"fold_{f_idx}"]
        f_auc = roc_auc_score(y_true, f_p)
        fpr_f, tpr_f, _ = roc_curve(y_true, f_p)
        plt.plot(fpr_f, tpr_f, linestyle="--", alpha=0.4, label=f"Fold {f_idx} Model (AUC = {f_auc:.3f})")

    fpr_ens, tpr_ens, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr_ens, tpr_ens, color="#d62728", lw=3, label=f"5-Model Ensemble (AUC = {roc_pt:.4f} [95% CI: {roc_lower:.3f}-{roc_upper:.3f}])")
    plt.plot([0, 1], [0, 1], "k:", lw=1.5, label="Random Chance (AUC = 0.500)")
    plt.title("Ensemble ROC Curve — Original Validation Cohort (N=44)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_roc_curve.png", dpi=300)
    plt.close()

    # 10.3 Ensemble PR Curve
    plt.figure(figsize=(8, 7))
    for f_idx in range(5):
        f_p = individual_fold_probs[f"fold_{f_idx}"]
        f_pr = average_precision_score(y_true, f_p)
        prec_f, rec_f, _ = precision_recall_curve(y_true, f_p)
        plt.plot(rec_f, prec_f, linestyle="--", alpha=0.4, label=f"Fold {f_idx} Model (PR-AUC = {f_pr:.3f})")

    prec_ens, rec_ens, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(rec_ens, prec_ens, color="#1f77b4", lw=3, label=f"5-Model Ensemble (PR-AUC = {pr_pt:.4f} [95% CI: {pr_lower:.3f}-{pr_upper:.3f}])")
    plt.axhline(baseline_pr, color="black", linestyle=":", lw=1.5, label=f"Prevalence Baseline ({baseline_pr:.3f})")
    plt.title("Ensemble Precision-Recall Curve — Validation Cohort (N=44)", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_precision_recall_curve.png", dpi=300)
    plt.close()

    # 10.4 Probability Distributions Strip & Box Plot
    plt.figure(figsize=(8, 6))
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    plt.boxplot([neg_probs, pos_probs], tick_labels=["Non-Metastatic (N=35)", "Metastatic (N=9)"], patch_artist=True,
                boxprops=dict(facecolor="#e0e0e0", alpha=0.7), medianprops=dict(color="black", lw=2))

    # Jitter scatter
    x_neg = np.random.normal(1, 0.04, size=len(neg_probs))
    x_pos = np.random.normal(2, 0.04, size=len(pos_probs))
    plt.scatter(x_neg, neg_probs, color="#1f77b4", alpha=0.7, s=40, label="Non-Metastatic")
    plt.scatter(x_pos, pos_probs, color="#d62728", alpha=0.85, s=60, edgecolors="black", label="Metastatic")

    plt.axhline(0.50, color="black", linestyle="--", lw=1.5, label="Default θ = 0.50")
    plt.axhline(0.35, color="#e377c2", linestyle=":", lw=1.5, label="Screening θ = 0.35")
    plt.title("Ensemble Predicted Probability Distribution by Diagnostic Class", fontsize=12, fontweight="bold")
    plt.ylabel("Ensemble Predicted Probability P(Metastasis)", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_probability_distribution.png", dpi=300)
    plt.close()

    # 10.5 Model Disagreement Plot
    plt.figure(figsize=(9, 6))
    plt.scatter(neg_probs, y_disagree[y_true == 0], color="#1f77b4", alpha=0.75, s=50, label="Non-Metastatic (N=35)")
    plt.scatter(pos_probs, y_disagree[y_true == 1], color="#d62728", alpha=0.85, s=70, edgecolors="black", label="Metastatic (N=9)")
    plt.axvline(0.50, color="gray", linestyle=":", alpha=0.6)
    plt.title("Model Disagreement Index (Std Dev) vs. Ensemble Predicted Probability", fontsize=12, fontweight="bold")
    plt.xlabel("Ensemble Mean Probability P_ensemble", fontsize=11)
    plt.ylabel("Cross-Model Standard Deviation (Disagreement Index)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_model_disagreement.png", dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print("STAGE 13: ENSEMBLE VALIDATION EVALUATION RESULTS")
    print("=" * 80)
    print(f"Validation Cohort: {len(y_true)} patients (9 positive, 35 negative)")
    print(f"Ensemble ROC-AUC:  {roc_pt:.4f} (95% CI: [{roc_lower:.4f}, {roc_upper:.4f}])")
    print(f"Ensemble PR-AUC:   {pr_pt:.4f} (95% CI: [{pr_lower:.4f}, {pr_upper:.4f}], Baseline: {baseline_pr:.4f})")
    print(f"Brier Score:       {brier:.4f}")
    print("-" * 80)
    print(f"Default θ = 0.50:")
    print(f"  Sensitivity:  {sens_050*100:.1f}% ({tp_050}/9 detected)")
    print(f"  Specificity:  {spec_050*100:.1f}% ({tn_050}/35 correctly rejected)")
    print(f"  Precision:    {prec_050*100:.1f}% ({tp_050}/{tp_050+fp_050})")
    print(f"  NPV:          {npv_050*100:.1f}% ({tn_050}/{tn_050+fn_050})")
    print(f"  F1-Score:     {f1_050:.4f}")
    print(f"  Balanced Acc: {bal_acc_050*100:.1f}%")
    print(f"  Confusion Matrix: TP={tp_050}, FP={fp_050}, TN={tn_050}, FN={fn_050}")
    print("-" * 80)
    if "0.35" in candidate_thresholds_data:
        c035 = candidate_thresholds_data["0.35"]
        print(f"Screening Candidate θ = 0.35:")
        print(f"  Sensitivity:  {c035['sensitivity']*100:.1f}% ({c035['true_positives']}/9 detected)")
        print(f"  Specificity:  {c035['specificity']*100:.1f}% ({c035['true_negatives']}/35 correctly rejected)")
        print(f"  NPV:          {c035['negative_predictive_value']*100:.1f}%")
        print(f"  Balanced Acc: {c035['balanced_accuracy']*100:.1f}%")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    run_stage_13_ensemble_validation()

