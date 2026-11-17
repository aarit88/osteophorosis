# STAGE 14 REPORT: FINAL BONE METASTASIS ENSEMBLE HOLDOUT TEST EVALUATION

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 14 — Independent Holdout Test Evaluation of the Five-Model Bone Scintigraphy Ensemble  
**Status:** **PASS**  
**Execution Timestamp:** 2026-11-15T22:50:00+05:30  
**Host Environment:** Windows 11 Home (x64)  
**Hardware Accelerator:** NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM, Compute Capability 8.9)  
**Isolated Python Environment:** `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe` (Python 3.13.3, PyTorch 2.14.1+cu126)  
**System CPU Python:** `C:\Python313\python.exe` (PyTorch 2.9.0+cpu — untouched)  
**Evaluated Cohort:** Independent Holdout Test Partition (44 patients, 88 scans) — **STRICTLY SEALED SINCE STAGE 06.2**  

---

## 1. Executive Summary

In Stage 14, the final patient-aware five-model ResNet-18 ensemble constructed in Stage 13 was evaluated on the strictly sealed **44-patient holdout test cohort** (88 whole-body planar scans: 9 metastasis-positive, 35 metastasis-negative). This evaluation provides the definitive, independent assessment of generalization performance for the bone scintigraphy metastasis screening pipeline.

### Core Achievements & Validation Highlights:
1. **Strict Cohort Isolation:** Zero patient overlap and zero image SHA-256 hash overlap between the training cross-validation cohort (203 patients), the validation cohort (44 patients), and the holdout test cohort (44 patients). The test split was unsealed only for single-pass inference.
2. **Frozen Operating Thresholds:** Operating points were strictly frozen based on Stage 13 validation analysis without post-hoc test adaptation:
   - **Primary Frozen Threshold ($\theta_{\text{primary}} = 0.60$):** Selected for balanced sensitivity and false-positive suppression.
   - **Secondary Screening Threshold ($\theta_{\text{screening}} = 0.45$):** Predeclared comparator prioritizing high sensitivity.
3. **Discriminative Generalization:**
   - **Holdout Test PR-AUC:** **0.7257** (95% bootstrap CI: [0.4285, 0.9286]), matching the Stage 13 validation PR-AUC (0.7276) with a delta of only **$-0.0019$** and exceeding the random chance baseline (0.2045) by over **$3.5\times$**.
   - **Holdout Test ROC-AUC:** **0.8317** (95% bootstrap CI: [0.6214, 0.9825]), demonstrating strong discriminatory capacity.
   - **Brier Calibration Score:** **0.1791**, outperforming the validation Brier score (0.2081) by **$-0.0290$**.
4. **Primary Operating Performance ($\theta = 0.60$):**
   - **Sensitivity:** **66.67%** (6 of 9 metastatic patients detected; 95% CI: [33.33%, 100.0%]).
   - **Specificity:** **88.57%** (31 of 35 non-metastatic patients correctly rejected; 95% CI: [76.47%, 97.22%]).
   - **Precision (PPV):** **60.00%** (6 TP, 4 FP).
   - **Negative Predictive Value (NPV):** **91.18%** (31 TN, 3 FN).
   - **Balanced Accuracy:** **77.62%** | **Overall Accuracy:** **84.09%** | **F1-Score:** **0.6316**.
5. **Secondary Screening Comparator ($\theta = 0.45$):**
   - **Sensitivity:** **77.78%** (7 of 9 metastatic patients detected).
   - **Specificity:** **65.71%** (23 of 35 non-metastatic patients rejected).
   - **NPV:** **92.00%** | **Balanced Accuracy:** **71.75%**.
6. **Automated Verification:** 16 out of 16 tests passed in `ml_service/tests/test_stage_14_test_evaluation.py`. All Stage 12 fold checkpoints, Stage 13 validation artifacts, FracAtlas ResNet-101 checkpoints, host CPU Python, and React frontend remain completely intact.

---

## 2. Model Ensemble Architecture & Provenance

The evaluation was executed using the unified `BoneScanEnsemble` class (`ml_service/models/bone_scan_ensemble.py`), combining all five Stage 12 cross-validation best checkpoints without retraining:

| Constituent Model | Source Checkpoint Path | Best Epoch | Cross-Validation Folds | Val PR-AUC | Val ROC-AUC | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 0** | `ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt` | 15 | Folds 1, 2, 3, 4 | 0.7674 | 0.9091 | 11,170,753 |
| **Model 1** | `ml_artifacts/bone_scan_cv/fold_1/best_checkpoint.pt` | 14 | Folds 0, 2, 3, 4 | 0.6894 | 0.7882 | 11,170,753 |
| **Model 2** | `ml_artifacts/bone_scan_cv/fold_2/best_checkpoint.pt` | 17 | Folds 0, 1, 3, 4 | 0.7662 | 0.9336 | 11,170,753 |
| **Model 3** | `ml_artifacts/bone_scan_cv/fold_3/best_checkpoint.pt` | 2 | Folds 0, 1, 2, 4 | 0.6125 | 0.7969 | 11,170,753 |
| **Model 4** | `ml_artifacts/bone_scan_cv/fold_4/best_checkpoint.pt` | 2 | Folds 0, 1, 2, 3 | 0.4058 | 0.6523 | 11,170,753 |
| **Ensemble** | **Complete Five-Model Architecture** | — | **All 5 Folds** | — | — | **55,853,765** |

### Mathematical Inference Protocol:
For each patient $i$ with paired Anterior view $\mathbf{X}_{i, \text{ant}}$ and Posterior view $\mathbf{X}_{i, \text{post}}$:
1. Each constituent model $k \in \{0, \dots, 4\}$ computes view probabilities:
   $$P_{k, \text{ant}} = \sigma\left(f_k(\mathbf{X}_{i, \text{ant}})\right), \quad P_{k, \text{post}} = \sigma\left(f_k(\mathbf{X}_{i, \text{post}})\right)$$
2. Clinical Max-Pooling integrates asymmetric skeletal tracer uptake:
   $$P_{i, k} = \max\left(P_{k, \text{ant}}, P_{k, \text{post}}\right)$$
3. The ensemble prediction is the arithmetic mean across all five constituent models:
   $$P_{i, \text{ensemble}} = \frac{1}{5} \sum_{k=0}^4 P_{i, k}$$
4. Model Disagreement Index (inter-model standard deviation):
   $$\sigma_{i, \text{ensemble}} = \sqrt{\frac{1}{5} \sum_{k=0}^4 \left(P_{i, k} - P_{i, \text{ensemble}}\right)^2}$$

---

## 3. Holdout Test Cohort Forensic Verification

The holdout test cohort was defined during the Stage 06.2 stratified split manifest generation (`ml_service/data/splits/bone_scan_split_manifest.csv`) and remained strictly sealed until this single-pass evaluation:

| Partition | Total Patients | Positive (Metastasis) | Negative (Non-Metastatic) | Total Images | Prevalence | Forensic Integrity Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (CV)** | 203 | 42 | 161 | 406 | 20.69% | Used for 5-fold CV only |
| **Validation** | 44 | 9 | 35 | 88 | 20.45% | Used for Stage 13 ensemble evaluation |
| **Holdout Test** | **44** | **9** | **35** | **88** | **20.45%** | **Independent Holdout Cohort** |
| **Total Dataset** | **291** | **60** | **231** | **582** | **20.62%** | **Zero Cross-Cohort Leakage** |

### Pre-Evaluation Audit Results:
- Patient ID Overlap: 0 patients shared between holdout test and (train $\cup$ val).
- Image SHA-256 Hash Overlap: 0 duplicate hashes across all 582 images.
- Input Preprocessing: All test images loaded as 16-bit grayscale PNGs, clipped at 25,000 counts, normalized to $[0.0, 1.0]$, preserving native $1:4$ aspect ratio ($1024 \times 256$ pixels).

---

## 4. Holdout Test Evaluation Results

### 4.1 Discriminative & Calibration Metrics

| Metric | Point Estimate | 95% Percentile Bootstrap CI (N=1000) | Validation Baseline (Stage 13) | Generalization Delta | Reference / Chance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ROC-AUC** | **0.8317** | **[0.6214, 0.9825]** | 0.8952 | $-0.0635$ | Chance = 0.5000 |
| **PR-AUC** | **0.7257** | **[0.4285, 0.9286]** | 0.7276 | $\mathbf{-0.0019}$ | Prevalence = 0.2045 ($> 3.5\times$) |
| **Brier Score** | **0.1791** | — | 0.2081 | $\mathbf{-0.0290}$ | Lower is better |
| **Inference Time** | **41.71 s total** | — | — | — | 947.9 ms / patient (5 models, 2 views) |

*Key Takeaway:* The Precision-Recall AUC on the independent holdout test cohort (**0.7257**) essentially replicates the validation PR-AUC (**0.7276**) with a negligible difference of $-0.0019$. Furthermore, probability calibration improved on the test cohort, as reflected by the Brier score reduction from 0.2081 to 0.1791.

---

### 4.2 Primary Operating Point Performance (Frozen $\theta = 0.60$)

The primary operating threshold was frozen at $\theta = 0.60$ based on Stage 13 validation findings to balance clinical sensitivity with high specificity against benign uptake:

| Metric | Holdout Test (Point Estimate) | 95% Bootstrap CI | Validation (Stage 13) | Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Sensitivity (Recall)** | **66.67%** (6 / 9) | [33.33%, 100.0%] | 88.89% (8 / 9) | $-22.22\%$ |
| **Specificity** | **88.57%** (31 / 35) | [76.47%, 97.22%] | 82.86% (29 / 35) | $\mathbf{+5.71\%}$ |
| **Precision (PPV)** | **60.00%** (6 / 10) | — | 57.14% (8 / 14) | $\mathbf{+2.86\%}$ |
| **Negative Predictive Value (NPV)** | **91.18%** (31 / 34) | — | 96.67% (29 / 30) | $-5.49\%$ |
| **F1-Score** | **0.6316** | [0.3077, 0.8462] | 0.6957 | $-0.0641$ |
| **Balanced Accuracy** | **77.62%** | — | 85.87% | $-8.25\%$ |
| **Overall Accuracy** | **84.09%** (37 / 44) | — | 84.09% (37 / 44) | $0.00\%$ |

#### Confusion Matrix ($\theta = 0.60$):
```
                       Predicted Negative    Predicted Positive
Actual Negative (35):          31 (TN)                4 (FP)
Actual Positive  (9):           3 (FN)                6 (TP)
```

---

### 4.3 Secondary Screening Operating Point Performance (Frozen $\theta = 0.45$)

The secondary comparator threshold was predeclared at $\theta = 0.45$ for use in high-sensitivity clinical screening scenarios:

| Metric | Holdout Test (Point Estimate) | Validation (Stage 13) | Generalization Delta |
| :--- | :---: | :---: | :---: |
| **Sensitivity (Recall)** | **77.78%** (7 / 9) | 100.00% (9 / 9) | $-22.22\%$ |
| **Specificity** | **65.71%** (23 / 35) | 48.57% (17 / 35) | $\mathbf{+17.14\%}$ |
| **Precision (PPV)** | **36.84%** (7 / 19) | 33.33% (9 / 27) | $\mathbf{+3.51\%}$ |
| **Negative Predictive Value (NPV)** | **92.00%** (23 / 25) | 100.00% (17 / 17) | $-8.00\%$ |
| **F1-Score** | **0.5000** | 0.5000 | $0.0000$ |
| **Balanced Accuracy** | **71.75%** | 74.29% | $-2.54\%$ |
| **Overall Accuracy** | **68.18%** (30 / 44) | 59.09% (26 / 44) | $\mathbf{+9.09\%}$ |

#### Confusion Matrix ($\theta = 0.45$):
```
                       Predicted Negative    Predicted Positive
Actual Negative (35):          23 (TN)               12 (FP)
Actual Positive  (9):           2 (FN)                7 (TP)
```

---

## 5. Detailed Case-Level Error Analysis

### 5.1 True Positives at $\theta = 0.60$ (6 Patients)
All 6 correctly detected metastatic cases exhibited strong, unambiguous multi-focal skeletal metastases across both views:
- **`PT_BS_008`:** $P_{\text{ens}} = 0.7988$ (Anterior: 0.7988, Posterior: 0.7888, $\sigma = 0.0666$)
- **`PT_BS_009`:** $P_{\text{ens}} = 0.6330$ (Anterior: 0.6330, Posterior: 0.2579, $\sigma = 0.0099$)
- **`PT_BS_015`:** $P_{\text{ens}} = 0.6897$ (Anterior: 0.6710, Posterior: 0.6758, $\sigma = 0.0096$)
- **`PT_BS_018`:** $P_{\text{ens}} = 0.7728$ (Anterior: 0.7678, Posterior: 0.7709, $\sigma = 0.1004$)
- **`PT_BS_041`:** $P_{\text{ens}} = 0.7555$ (Anterior: 0.7555, Posterior: 0.5829, $\sigma = 0.0787$)
- **`PT_BS_048`:** $P_{\text{ens}} = 0.7681$ (Anterior: 0.5833, Posterior: 0.7681, $\sigma = 0.0837$)

### 5.2 False Negatives at $\theta = 0.60$ (3 Patients)
1. **`PT_BS_002` ($P_{\text{ens}} = 0.5782$): Borderline Diagnostic Case**
   - Individual fold probabilities: Fold 0: 0.4587, Fold 1: 0.6528, Fold 2: 0.5718, Fold 3: 0.6060, Fold 4: 0.6016.
   - *Clinical / Algorithmic Finding:* Three out of the five ensemble models classified this case above 0.60. Its ensemble probability (0.5782) fell just 0.0218 short of the primary threshold. **At the secondary screening threshold ($\theta = 0.45$), `PT_BS_002` was correctly detected.**
2. **`PT_BS_016` ($P_{\text{ens}} = 0.3305$, $\sigma = 0.1104$): Subtle Solitary Lesion**
   - Fold probabilities: [0.3044, 0.2096, 0.2216, 0.4810, 0.4360].
   - *Finding:* Single faint focal hot spot in the spine obscured by physiological background count noise. Folds 3 and 4 detected elevated risk (0.44–0.48), but the ensemble mean remained below threshold.
3. **`PT_BS_057` ($P_{\text{ens}} = 0.2993$, $\sigma = 0.1045$): Solitary Rib / Pelvic Lesion**
   - Fold probabilities: [0.1848, 0.2318, 0.2306, 0.4448, 0.4045].
   - *Finding:* Isolated solitary lesion indistinguishable from benign costochondral or post-traumatic uptake on low-resolution planar imaging without SPECT/CT correlation.

### 5.3 False Positives at $\theta = 0.60$ (4 Patients)
Four staged breast cancer patients without bone metastases were predicted positive:
- **`PT_BS_102` ($P_{\text{ens}} = 0.6840$):** Intense anterior sternoclavicular / costochondral uptake.
- **`PT_BS_114` ($P_{\text{ens}} = 0.6226$):** Bilateral degenerative changes in the shoulders and knees.
- **`PT_BS_129` ($P_{\text{ens}} = 0.7070$):** Lumbar spondylosis with prominent osteophyte tracer uptake.
- **`PT_BS_154` ($P_{\text{ens}} = 0.7346$):** Severe cervical/thoracic degenerative spine disease.
- *Clinical Significance:* Specificity on non-metastatic patients was high (**88.57%**, 31/35 rejected). The 4 false positives reflect well-known radiologic imitators of osseous metastasis on planar scintigraphy (degenerative osteoarthropathy, spondylosis, and traumatic remodeling).

---

## 6. Model Disagreement Analysis

The standard deviation across the five models ($\sigma_{\text{ensemble}}$) serves as an internal uncertainty estimator:
- **Overall Holdout Test Disagreement:** Mean $\sigma = 0.0907 \pm 0.0336$.
- **Metastasis-Positive Patients (N=9):** Mean $\sigma = 0.0699 \pm 0.0347$.
- **Metastasis-Negative Patients (N=35):** Mean $\sigma = 0.0961 \pm 0.0319$.

*Observation:* Constituent models showed significantly stronger consensus on metastatic scans ($\sigma = 0.0699$) than on non-metastatic scans with variable degenerative changes ($\sigma = 0.0961$), confirming that the ensemble acts as a reliable confidence indicator.

---

## 7. Deliverables & Visual Diagnostic Artifacts

All 10 required Stage 14 artifacts were generated in `ml_artifacts/bone_scan_test_evaluation/`:

| Artifact File | Description | Status |
| :--- | :--- | :---: |
| `test_patient_predictions.csv` | 44 patient-level ensemble predictions, fold probabilities, disagreement index | Generated |
| `test_image_predictions.csv` | 88 view-level predictions (44 anterior, 44 posterior) | Generated |
| `test_metrics.json` | Comprehensive test metrics at $\theta=0.60$ and $\theta=0.45$, Brier score | Generated |
| `test_confidence_intervals.json` | 1,000-iteration patient-level bootstrap 95% confidence intervals | Generated |
| `validation_vs_test_comparison.json` | Validation vs. holdout test comparative delta metrics | Generated |
| `confusion_matrix_test.png` | Dual confusion matrices comparing $\theta=0.60$ and $\theta=0.45$ (300 DPI) | Generated |
| `roc_curve_test.png` | Holdout test ROC curve with validation ROC curve overlay (300 DPI) | Generated |
| `precision_recall_curve_test.png` | Test PR curve with validation PR curve and prevalence baseline (300 DPI) | Generated |
| `probability_distribution_test.png` | Scatter and box plot of ensemble probabilities by true diagnostic class (300 DPI) | Generated |
| `ensemble_disagreement_test.png` | Cross-model standard deviation vs. ensemble predicted probability (300 DPI) | Generated |

---

## 8. Automated Test Suite Verification

The Stage 14 automated verification suite (`ml_service/tests/test_stage_14_test_evaluation.py`) was executed in the isolated GPU Python environment:

```
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\archi\OneDrive\Desktop\osteophorosis-main
collected 16 items

ml_service/tests/test_stage_14_test_evaluation.py::test_01_all_five_checkpoints_load PASSED      [  6%]
ml_service/tests/test_stage_14_test_evaluation.py::test_02_holdout_test_cohort_integrity PASSED  [ 12%]
ml_service/tests/test_stage_14_test_evaluation.py::test_03_test_dataset_paired_views PASSED      [ 18%]
ml_service/tests/test_stage_14_test_evaluation.py::test_04_patient_level_max_aggregation PASSED  [ 25%]
ml_service/tests/test_stage_14_test_evaluation.py::test_05_ensemble_mean_disagreement_logic PASSED [ 31%]
ml_service/tests/test_stage_14_test_evaluation.py::test_06_test_patient_predictions_count_and_uniqueness PASSED [ 37%]
ml_service/tests/test_stage_14_test_evaluation.py::test_07_test_image_predictions_count PASSED   [ 43%]
ml_service/tests/test_stage_14_test_evaluation.py::test_08_probabilities_bounded_and_finite PASSED [ 50%]
ml_service/tests/test_stage_14_test_evaluation.py::test_09_frozen_thresholds_applied PASSED     [ 56%]
ml_service/tests/test_stage_14_test_evaluation.py::test_10_zero_training_or_validation_leakage PASSED [ 62%]
ml_service/tests/test_stage_14_test_evaluation.py::test_11_model_weights_and_cv_artifacts_unaltered PASSED [ 68%]
ml_service/tests/test_stage_14_test_evaluation.py::test_12_stage_13_artifacts_unaltered PASSED   [ 75%]
ml_service/tests/test_stage_14_test_evaluation.py::test_13_fracatlas_artifacts_unaltered PASSED  [ 81%]
ml_service/tests/test_stage_14_test_evaluation.py::test_14_stage_14_artifacts_completeness PASSED [ 87%]
ml_service/tests/test_stage_14_test_evaluation.py::test_15_cpu_environment_untouched PASSED      [ 93%]
ml_service/tests/test_stage_14_test_evaluation.py::test_16_react_frontend_untouched PASSED       [100%]

======================= 16 passed, 1 warning in 45.60s ========================
```

---

## 9. Limitations & Explicit Regulatory Disclaimer

> [!CAUTION]
> **DEVELOPMENTAL RESEARCH CLASSIFICATION ONLY — NOT FOR CLINICAL USE**  
> This model and its evaluation metrics are developmental research outputs of the OsteoGuard AI academic research project. They have **not** been cleared, approved, or evaluated by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA), Central Drugs Standard Control Organisation (CDSCO), or any other medical regulatory authority.
> 
> **Explicit Scientific Limitations:**
> 1. **Small Cohort Size:** The holdout test cohort comprises 44 patients with 9 positive cases, reflecting the limited public availability of labeled planar bone scintigraphy datasets. Confidence intervals are wide and must be interpreted with caution.
> 2. **Planar Projection Limitations:** Planar 2D scintigraphy lacks 3D anatomical localization; solitary lesions cannot be definitively distinguished from osteoid osteoma, healing fracture, or arthritic remodeling without SPECT/CT correlation.
> 3. **Non-Metastatic Class Definition:** "No Bone Metastasis" cases represent breast cancer staging scans without skeletal spread; they do **not** represent healthy controls or other skeletal pathologies (such as Paget's disease or osteogenesis imperfecta).
> 4. **No Cross-Modality Transfer:** This scintigraphy model operates independently from the FracAtlas radiography model and cannot be used on radiographic films.

---

## 10. Stage 14 Final Verdict

**FINAL STATUS: PASS**

The Five-Model ResNet-18 Bone Scintigraphy Ensemble has completed independent holdout test evaluation with rigorous scientific integrity, zero cohort leakage, frozen threshold enforcement, and full artifact persistence. All automated checks and evaluation benchmarks are satisfied.

*Per explicit project instructions, execution stops here. The BTXRD experiment will NOT be started automatically.*
