# STAGE 10 — FINAL FRACATLAS HOLDOUT TEST EVALUATION REPORT

**Project:** OsteoGuard AI  
**Stage:** 10 — Final FracAtlas Holdout Test Partition Evaluation  
**Audit Timestamp:** 2026-11-15T19:25:00+05:30  
**Overall Stage Status:** **PASS**  
**Execution Environment:** Isolated GPU Virtual Environment (`%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe`)  
**Hardware Accelerator:** NVIDIA GeForce RTX 3050 A Laptop GPU (4,094 MiB VRAM, Ada Lovelace CC 8.9)  

---

## 1. Executive Summary

Stage 10 executed the final, independent, zero-training evaluation of the full-scale FracAtlas ResNet-101 binary fracture classification model on the strictly quarantined holdout test partition (614 images). 

The evaluation was performed strictly in inference mode (`torch.no_grad()`, `model.eval()`) without weight updates, parameter tuning, or threshold re-optimization. The primary decision threshold was locked in advance at **$\theta = 0.31$** based on the Stage 09 validation calibration (designed for $\ge 85\%$ sensitivity), with $\theta = 0.50$ evaluated as a fixed secondary comparator.

### Key Evaluation Findings:
1. **Exceptional Generalization:** 
   - **Holdout Test ROC-AUC:** **0.9117** (95% CI: [0.8838, 0.9370]) vs. Validation ROC-AUC 0.8871 ($\Delta = +0.0246$).
   - **Holdout Test PR-AUC:** **0.7537** (95% CI: [0.6775, 0.8171]) vs. Validation PR-AUC 0.7556 ($\Delta = -0.0019$).
   - The negligible generalization gap on PR-AUC ($\Delta = -0.0019$) and positive shift on ROC-AUC confirms that the progressive unfreezing and Focal Loss strategy produced strong, robust feature representations without overfitting.
2. **Clinical Triage Utility ($\theta = 0.31$):**
   - **Sensitivity / Recall:** **90.00%** (99 / 110 acute fractures detected, 95% CI: [84.68%, 95.08%]).
   - **Specificity:** **74.40%** (375 / 504 non-fractures correctly identified, 95% CI: [70.62%, 78.17%]).
   - **Negative Predictive Value (NPV):** **97.15%** (375 / 386 negative calls confirmed true negatives).
   - **Precision (PPV):** **43.42%** (99 / 228 positive calls confirmed true positives).
   - **Balanced Accuracy:** **82.20%** (95% CI: [79.10%, 85.40%]).
3. **Rule-Out Safety Profile:** At $\theta = 0.31$, the False Negative Rate is only **10.00%** (11 missed fractures out of 110), establishing high efficacy for emergency department triage rule-out workflows.

All 23 automated unit and system tests passed cleanly (100%). Stage 10 is complete with final status **PASS**.

---

## 2. Model and Checkpoint Provenance

| Component | Specification | Provenance / Verification |
| :--- | :--- | :--- |
| **Model Architecture** | ResNet-101 | 101-layer deep residual network with AdaptiveAvgPool2d and Linear(2048, 2) head |
| **Pretrained Weights** | `ResNet101_Weights.IMAGENET1K_V2` | Locally cached official TorchVision weights (`SHA-256: cd907fc2a0de...`) |
| **Trained Checkpoint** | `best_checkpoint.pt` | Saved at **Epoch 9** of Stage 09 (`ml_artifacts/fracatlas_full_training/best_checkpoint.pt`) |
| **Checkpoint Metrics** | Val PR-AUC: 0.7556, Val ROC-AUC: 0.8871 | Verified uncorrupted and intact (`test_stage_09_checkpoint_unaltered` PASSED) |
| **Parameters Evaluated** | 42,504,258 total | Exactly matches training configuration |
| **Execution Mode** | `eval()` with `torch.no_grad()` | Zero gradients, zero optimizer, zero weight adjustments |

---

## 3. Test Dataset Composition and Hygiene

The holdout test partition established in Stage 06.2 was loaded from `ml_service/data/splits/fracatlas_classification_split.csv`:

| Class Index | Diagnostic Category | Test Partition Count | Test Prevalence | Cohort Ground Truth Total | Split Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `candidate_control_radiograph` (Non-fractured) | 504 | 82.08% | 3,364 | 14.98% of controls |
| **1** | `acute_bone_fracture` (Fractured) | 110 | 17.92% | 719 | 15.30% of fractures |
| **Total** | **Holdout Test Partition** | **614** | **100.00%** | **4,083** | **15.04% of cohort** |

- **Zero Leakage Verification:** SHA-256 image hash sets confirmed **0 overlapping images** between the holdout test set and the training (2,857) or validation (612) sets (`test_no_data_leakage_between_splits` PASSED).
- **Image Integrity:** All 614 test images were verified present on disk and decoded with 100% integrity without truncated image errors.

---

## 4. Frozen Decision Threshold Rationale

The primary operating threshold was **strictly frozen** prior to test inference:
- **Primary Threshold ($\theta = 0.31$):** Selected during Stage 09 validation calibration to meet the clinical objective of $\ge 85\%$ sensitivity for fracture detection. On validation data, $\theta = 0.31$ achieved 85.98% sensitivity, 72.48% specificity, and 39.83% precision.
- **Secondary Comparator Threshold ($\theta = 0.50$):** Standard default decision boundary reported for comparative completeness.
- **Strict Methodological Rule:** No threshold sweep, threshold tuning, or post-hoc threshold selection was performed on test data.

---

## 5. Holdout Test Metrics and Confusion Matrices

### Confusion Matrix at Frozen Primary Threshold ($\theta = 0.31$)

```
                  Predicted Negative (0)    Predicted Positive (1)    Total
Actual Negative:          375 (TN)                  129 (FP)           504
Actual Positive:           11 (FN)                   99 (TP)           110
Total:                    386                       228                614
```

### Confusion Matrix at Secondary Comparator Threshold ($\theta = 0.50$)

```
                  Predicted Negative (0)    Predicted Positive (1)    Total
Actual Negative:          477 (TN)                   27 (FP)           504
Actual Positive:           45 (FN)                   65 (TP)           110
Total:                    522                        92                614
```

---

### Complete Metric Breakdown with 95% Confidence Intervals

*Empirical 95% Confidence Intervals calculated via non-parametric bootstrapping with 1,000 iterations ($N=614$, seed 42).*

| Metric | Primary Threshold ($\theta = 0.31$) | 95% Bootstrap CI ($\theta = 0.31$) | Secondary Threshold ($\theta = 0.50$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.9117** | **[0.8838, 0.9370]** | 0.9117 (threshold-independent) |
| **PR-AUC** | **0.7537** | **[0.6775, 0.8171]** | 0.7537 (threshold-independent) |
| **Sensitivity (Recall)** | **90.00%** (99/110) | **[84.68%, 95.08%]** | 59.09% (65/110) |
| **Specificity** | **74.40%** (375/504) | **[70.62%, 78.17%]** | 94.64% (477/504) |
| **Precision (PPV)** | **43.42%** (99/228) | **[37.28%, 49.77%]** | 70.65% (65/92) |
| **Negative Predictive Value (NPV)** | **97.15%** (375/386) | [95.45%, 98.67%] | 91.38% (477/522) |
| **F1-Score** | **0.5858** | **[0.5247, 0.6450]** | 0.6436 |
| **Accuracy** | **77.20%** (474/614) | [73.94%, 80.46%] | 88.27% (542/614) |
| **Balanced Accuracy** | **82.20%** | **[79.10%, 85.40%]** | 76.87% |
| **Youden's J Statistic** | **0.6440** | [0.5794, 0.7061] | 0.5373 |
| **False Positive Rate (FPR)** | **25.60%** (129/504) | [21.83%, 29.38%] | 5.36% (27/504) |
| **False Negative Rate (FNR)** | **10.00%** (11/110) | [4.92%, 15.32%] | 40.91% (45/110) |

---

## 6. Validation versus Holdout Test Comparison

| Metric | Validation (N=612) | Holdout Test (N=614) | Performance Delta ($\Delta$) | Generalization Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 0.8871 | **0.9117** | **+0.0246** | Robust discriminative generalization (+2.46%) |
| **PR-AUC** | 0.7556 | **0.7537** | **-0.0019** | Negligible gap (-0.19%), confirming zero overfitting |
| **Sensitivity ($\theta = 0.31$)** | 85.98% | **90.00%** | **+4.02%** | Exceeded validation design target (>85%) |
| **Specificity ($\theta = 0.31$)** | 72.48% | **74.40%** | **+1.92%** | Improved non-fracture rule-out capability |
| **Precision ($\theta = 0.31$)** | 39.83% | **43.42%** | **+3.59%** | Moderate precision gain due to fewer false positives |
| **F1-Score ($\theta = 0.31$)** | 0.5444 | **0.5858** | **+0.0414** | Improved harmonic balance on test set |
| **Balanced Accuracy** | 79.23% | **82.20%** | **+2.97%** | Consistent gain across both classes |
| **Youden's J** | 0.5846 | **0.6440** | **+0.0594** | Higher diagnostic separation on unseen data |

### Generalization Assessment
The validation-to-test performance gap is remarkably tight across all evaluated metrics:
- The PR-AUC delta of $-0.0019$ demonstrates that the model's precision-recall characteristics are near-identical across separate data splits.
- The sensitivity of $90.00\%$ confirms that the operating threshold selected during validation ($\theta = 0.31$) is not an overfitted artifact, but an effective, generalizable decision boundary for fracture screening.
- The high NPV of $97.15\%$ demonstrates that patients predicted negative by the model have a $<3\%$ chance of harboring an undiagnosed fracture in this cohort.

---

## 7. Artifact Inventory

All generated artifacts are preserved in `ml_artifacts/fracatlas_test_evaluation/`:

| Artifact Path | Size | Description |
| :--- | :---: | :--- |
| `ml_artifacts/fracatlas_test_evaluation/test_predictions.csv` | 153.3 KB | Record-level test predictions (614 rows) with probabilities and predictions at both thresholds |
| `ml_artifacts/fracatlas_test_evaluation/test_metrics.json` | 2.5 KB | Full machine-readable test metrics, confusion matrices, 95% CIs, and validation comparisons |
| `ml_artifacts/fracatlas_test_evaluation/confusion_matrix_test.png` | 62.3 KB | 2x2 confusion matrix heatmap at primary frozen threshold $\theta = 0.31$ |
| `ml_artifacts/fracatlas_test_evaluation/roc_curve_test.png` | 105.1 KB | Holdout test ROC curve with operating points ($\theta = 0.31$ and $\theta = 0.50$) marked |
| `ml_artifacts/fracatlas_test_evaluation/precision_recall_curve_test.png` | 87.1 KB | Holdout test PR curve with test prevalence baseline (0.1792) marked |
| `ml_artifacts/fracatlas_test_evaluation/validation_vs_test_comparison.png` | 89.1 KB | Side-by-side bar chart comparing Validation vs Test across 7 core metrics |
| `ml_artifacts/fracatlas_test_evaluation/test_evaluation_summary.json` | 0.6 KB | Executive summary of test evaluation results and generalization deltas |

---

## 8. Test Suite and Environmental Audit

The test suite executed with **23/23 passing tests (100% pass rate)**:

```
pytest ml_service/tests/test_focal_loss.py \
       ml_service/tests/test_stage_09_full_training.py \
       ml_service/tests/test_stage_09_post_training.py \
       ml_service/tests/test_stage_10_test_evaluation.py -v
======================= 23 passed, 1 warning in 71.40s =======================
```

1. **Stage 10 Test Evaluation Tests (5/5 PASS):**
   - `test_test_artifacts_completeness`: All 7 artifact files verified present and non-empty.
   - `test_test_predictions_count_and_probabilities`: Exactly 614 rows, probabilities sum to $1.0$, thresholds matched.
   - `test_frozen_threshold_integrity`: Primary threshold verified locked at 0.31 without test optimization.
   - `test_no_data_leakage_between_splits`: Zero hash overlap between train/val and test sets.
   - `test_stage_09_checkpoint_unaltered`: Stage 09 checkpoint verified unaltered.
2. **Prior Regression Tests (18/18 PASS):**
   - 6 Focal Loss unit tests.
   - 6 Stage 09 full training configuration tests.
   - 6 Stage 09 post-training artifact and reload tests.

### Environmental Non-Interference Confirmation
- **Baseline CPU Python:** `C:\Python313\python.exe` verified untouched (`Baseline PyTorch: 2.9.0+cpu CUDA available: False`).
- **Frontend Codebase:** `osteophorosis-main/src/` verified 100% untouched (all 58 files preserved).
- **Previous Checkpoints:** `ml_artifacts/fracatlas_pilot/` and `ml_artifacts/fracatlas_full_training/` remain strictly unaltered.

---

## 9. Limitations and Clinical Disclosure

1. **Investigational Research Model:** This neural network is an investigational deep learning algorithm developed for academic research purposes. It is **NOT** a certified medical device and has **NOT** received clearance or approval from the U.S. FDA, European EMA, or any national health regulatory authority.
2. **False Positive Burden:** While the primary operating threshold ($\theta = 0.31$) achieves high sensitivity (90.00%), it incurs a false positive rate of 25.60% (129 false positives out of 504 non-fractured images), resulting in a precision of 43.42%. In clinical practice, all AI-flagged radiographs must be reviewed by a board-certified radiologist or emergency physician.
3. **Anatomical Diversity & Single-Source Dataset:** Although the FracAtlas dataset incorporates radiographs across multiple skeletal locations (wrist, hand, shoulder, ankle, pelvis), all data originate from a single multi-center collection in Bangladesh. External multi-institution validation is required before general hospital deployment.

---

## 10. Final Verification Sign-Off

- **Stage Status:** **PASS**
- **Test Set Evaluated:** Exactly once (614 images, 110 fractured, 504 non-fractured)
- **Primary Operating Metric:** Sensitivity **90.00%**, Specificity **74.40%**, ROC-AUC **0.9117**, PR-AUC **0.7537**
- **Next Stage:** STAGE 11 (Not started — awaiting user directive)

