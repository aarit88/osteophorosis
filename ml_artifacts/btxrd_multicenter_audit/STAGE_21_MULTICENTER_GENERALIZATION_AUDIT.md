# STAGE 21 REPORT: BTXRD MULTI-CENTER GENERALIZATION AUDIT AND MODEL DISPOSITION

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 21 — BTXRD Multi-Center Generalization Audit and Model Disposition  
**Status:** **PASS**  
**Execution Timestamp:** 2026-11-16T08:40:00+05:30  
**Host Environment:** Windows 11 Home (x64)  
**Execution Environment:** `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe` (Python 3.13.3, PyTorch 2.14.1+cu126)  
**System CPU Python:** `C:\Python313\python.exe` (Untouched)  
**React Frontend:** Untouched  
**Audit Scope:** Stages 15 through 20 BTXRD Three-Class Bone Tumour Classification Benchmark  

---

## 1. Executive Summary & Audit Mandate

Stage 21 executes the comprehensive multi-center generalization audit, statistical evaluation, error profiling, and formal disposition decision for the frozen BTXRD ResNet-34 classifier across all three clinical cohorts:
- **Center 1:** Internal Development Validation Cohort ($N=588$)
- **Center 2:** Primary External Multi-Center Holdout Benchmark ($N=549$)
- **Center 3:** Secondary External Tertiary Referral Stress Test ($N=259$)

This stage is strictly **AUDITING AND REPORTING ONLY**. No models were trained, fine-tuned, recalibrated, or modified.

### Key Audit Findings:
1. **Cryptographic Immutability Confirmed:** Checkpoint SHA-256 (`21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`) remained invariant throughout Stages 18, 19, 20, and 21.
2. **Pronounced External Generalization Gap:** Macro-F1 degraded from `0.8103` on Center 1 to `0.5069` on Center 2 ($-37.4\%$) and `0.3592` on Center 3 ($-55.7\%$).
3. **Malignant Detection Variability:** Primary bone malignancy recall degraded from `90.24%` on Center 1 to `59.21%` on Center 2 and `67.74%` on Center 3. Critical false-negative predictions (malignancies labeled as normal) occurred in both external cohorts (10 in Center 2, 6 in Center 3).
4. **Referral Prior Distortion:** In Center 3, where 89.58% of incoming cases have tumours, normal-class sensitivity collapsed to `18.52%` (5 / 27), illustrating severe overcalling driven by training loss class weighting under extreme prevalence skew.
5. **Disposition Verdicts Issued:**
   - *Research Prototype:* **CONDITIONAL PASS**
   - *Cross-Center Generalization:* **FAIL**
   - *Clinical Deployment:* **STRICTLY PROHIBITED (FAIL)**
   - *Integration into Final OsteoGuard Classifier:* **UNSUITABLE IN CURRENT FORM**

---

## 2. Source Artifact Verification & Integrity Audit

A comprehensive forensic audit of all source files across Stages 17, 18, 19, and 20 was conducted:

| Evaluated Dimension | Target Milestone | Source Artifact Path | Observed Value | Expected Value | Audit Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Model Checkpoint** | Stage 18 Freeze | `ml_artifacts/btxrd_frozen_model/best_checkpoint.pt` | `21064670FA62...` | `21064670FA62...` | **MATCH (BIT-LEVEL)** |
| **Stage 19 Checkpoint** | Stage 19 Pre/Post | `ml_artifacts/btxrd_center2_external_evaluation/checkpoint_hash_verification.txt` | `21064670FA62...` | `21064670FA62...` | **MATCH** |
| **Stage 20 Checkpoint** | Stage 20 Pre/Post | `ml_artifacts/btxrd_center3_external_evaluation/checkpoint_hash_verification.txt` | `21064670FA62...` | `21064670FA62...` | **MATCH** |
| **Center 1 Val Rows** | Stage 17 Training | `ml_artifacts/btxrd_full_training/validation_predictions.csv` | 588 rows | 588 rows | **MATCH** |
| **Center 2 Test Rows** | Stage 19 Holdout | `ml_artifacts/btxrd_center2_external_evaluation/center2_predictions.csv` | 549 rows | 549 rows | **MATCH** |
| **Center 3 Test Rows** | Stage 20 Stress | `ml_artifacts/btxrd_center3_external_evaluation/center3_predictions.csv` | 259 rows | 259 rows | **MATCH** |
| **Center 1 Supports** | Stage 17 Manifest | `ml_service/data/splits/btxrd_split_manifest.csv` | [319, 228, 41] | [319, 228, 41] | **MATCH** |
| **Center 2 Supports** | Stage 19 Manifest | `ml_service/data/splits/btxrd_split_manifest.csv` | [259, 214, 76] | [259, 214, 76] | **MATCH** |
| **Center 3 Supports** | Stage 20 Manifest | `ml_service/data/splits/btxrd_split_manifest.csv` | [27, 201, 31] | [27, 201, 31] | **MATCH** |
| **Center 1 CM Sum** | Stage 17 Metrics | `ml_artifacts/btxrd_full_training/validation_metrics.json` | 588 | 588 | **MATCH** |
| **Center 2 CM Sum** | Stage 19 Metrics | `ml_artifacts/btxrd_center2_external_evaluation/confusion_matrix.json` | 549 | 549 | **MATCH** |
| **Center 3 CM Sum** | Stage 20 Metrics | `ml_artifacts/btxrd_center3_external_evaluation/confusion_matrix.json` | 259 | 259 | **MATCH** |

### Forensic Investigation of Initial Stage 20 Test-Suite Failure:
- **Incident Description:** In task `task-5271`, the initial execution of `test_stage_20_center3_evaluation.py` returned an exit code of 1 with 1 failure: `test_09_multicenter_isolation_and_center2_preservation`.
- **Root-Cause Analysis:** The test function attempted to read `m2["external_cohort"]` from `ml_artifacts/btxrd_center2_external_evaluation/evaluation_manifest.json`, but the actual manifest keys created in Stage 19 were `cohort_center`, `cohort_partition`, and `total_images_evaluated`. This represented a test assertion key mismatch, not a defect in model execution, predictions, or evaluations.
- **Remediation & Final Rerun:** The test assertion was corrected to verify `m2["cohort_center"] == "center_2"`, `m2["cohort_partition"] == "external_test_center2"`, and `m2["total_images_evaluated"] == 549`. The test suite was re-executed in task `task-5281`, passing **15 of 15 tests cleanly in 12.76s** with zero unresolved defects.
- **Historical Stage 19 Test Note:** Re-running historical `test_stage_19_center2_evaluation.py` after Stage 20 fails on its historical isolation test (`assert not c3_dir.exists()`) because Stage 20 subsequently created the Center 3 directory. In accordance with strict audit rules, historical test files remain unedited and this chronological progression is documented.

---

## 3. Three-Center Performance Synthesis

The synthesis below presents side-by-side performance across all three independent hospital centers. Center 1 represents development validation, while Centers 2 and 3 represent separate, unpooled external evaluations:

| Performance Metric | Center 1 Internal Val ($N=588$) | Center 2 External Holdout ($N=549$) | Center 3 External Stress Test ($N=259$) | Cross-Center Generalization Dynamic |
| :--- | :---: | :---: | :---: | :--- |
| **Cohort Function** | Internal Development Validation | Primary External Holdout | Secondary External Stress Test | Validates progressive distribution shift |
| **Prevalence (N / B / M)** | 54.2% / 38.8% / 7.0% | 47.2% / 39.0% / 13.8% | **10.4% / 77.6% / 12.0%** | Center 3 has extreme tumor referral skew |
| **Overall Accuracy** | **81.80%** | **51.73%** `[47.72%, 55.74%]` | **47.10%** `[40.93%, 53.28%]` | Progressive degradation under external shift |
| **Balanced Accuracy** | **82.90%** | **53.74%** `[49.16%, 58.44%]` | **44.67%** `[36.94%, 52.60%]` | Equal-weight recall compressed by normal class |
| **Macro-F1 Score** | **0.8103** | **0.5069** `[0.4648, 0.5494]` | **0.3592** `[0.3007, 0.4183]` | $-37.4\%$ in Center 2, $-55.7\%$ in Center 3 |
| **Weighted-F1 Score** | **0.8141** | **0.5210** `[0.4802, 0.5598]` | **0.5285** `[0.4687, 0.5866]` | Stabilized in Center 3 by benign dominance |
| **Macro ROC-AUC (OvR)** | **0.9114** | **0.7180** `[0.6841, 0.7512]` | **0.6468** `[0.5853, 0.7043]` | Discriminative ranking maintained above chance |
| **Macro PR-AUC (OvR)** | **0.8700** | **0.5683** `[0.5240, 0.6212]` | **0.4558** `[0.4060, 0.5327]` | Unweighted precision-recall performance |
| **Multiclass Brier Score** | **0.3168** | **0.5731** | **0.6486** | Progressive probability entropy inflation |

### Per-Class Diagnostic Performance Breakdown:

#### Class 0: Normal Bone Radiograph
- **Center 1 (Support 319, Prev 54.25%):** Sensitivity **90.91%**, Specificity **75.46%**, PPV **81.46%**, NPV **87.50%**, F1 **0.8593**, ROC-AUC **0.8935**, PR-AUC **0.8771**
- **Center 2 (Support 259, Prev 47.18%):** Sensitivity **45.95%** `[39.4%, 51.8%]`, Specificity **78.28%** `[73.9%, 82.6%]`, PPV **65.38%** `[58.3%, 72.1%]`, NPV **61.85%** `[57.1%, 66.9%]`, F1 **0.5397**, ROC-AUC **0.7234**, PR-AUC **0.6703**
- **Center 3 (Support 27, Prev 10.42%):** Sensitivity **18.52%** `[5.0%, 34.5%]`, Specificity **78.45%** `[72.8%, 83.4%]`, PPV **9.09%** `[2.2%, 17.2%]`, NPV **89.22%** `[84.7%, 93.3%]`, F1 **0.1220**, ROC-AUC **0.5121**, PR-AUC **0.1141**

#### Class 1: Benign Primary Bone Tumour
- **Center 1 (Support 228, Prev 38.78%):** Sensitivity **67.54%**, Specificity **91.94%**, PPV **84.15%**, NPV **81.73%**, F1 **0.7494**, ROC-AUC **0.8538**, PR-AUC **0.8382**
- **Center 2 (Support 214, Prev 38.98%):** Sensitivity **56.07%** `[49.5%, 62.7%]`, Specificity **61.79%** `[56.1%, 66.8%]`, PPV **48.39%** `[42.7%, 54.4%]`, NPV **68.77%** `[63.5%, 73.9%]`, F1 **0.5195**, ROC-AUC **0.6109**, PR-AUC **0.5322**
- **Center 3 (Support 201, Prev 77.61%):** Sensitivity **47.76%** `[40.9%, 55.1%]`, Specificity **70.69%** `[58.7%, 81.2%]`, PPV **84.96%** `[78.6%, 91.3%]`, NPV **28.08%** `[20.6%, 35.6%]`, F1 **0.6115**, ROC-AUC **0.6793**, PR-AUC **0.8718**

#### Class 2: Primary Malignant Bone Tumour
- **Center 1 (Support 41, Prev 6.97%):** Sensitivity **90.24%**, Specificity **97.81%**, PPV **75.51%**, NPV **99.26%**, F1 **0.8222**, ROC-AUC **0.9868**, PR-AUC **0.8949**
- **Center 2 (Support 76, Prev 13.84%):** Sensitivity **59.21%** `[47.9%, 70.8%]`, Specificity **84.36%** `[81.1%, 87.3%]`, PPV **37.82%** `[29.7%, 46.8%]`, NPV **92.79%** `[90.2%, 95.1%]`, F1 **0.4615**, ROC-AUC **0.8198**, PR-AUC **0.5023**
- **Center 3 (Support 31, Prev 11.97%):** Sensitivity **67.74%** `[51.3%, 83.3%]`, Specificity **69.30%** `[63.4%, 75.2%]`, PPV **23.08%** `[14.8%, 32.3%]`, NPV **94.05%** `[89.9%, 97.1%]`, F1 **0.3443**, ROC-AUC **0.7490**, PR-AUC **0.3815**

---

## 4. Generalization Gap and In-Depth Error Analysis

### 4.1 Comparative Confusion Matrices

$$\mathbf{C_1} = \begin{pmatrix} 290 & 26 & 3 \\ 65 & 154 & 9 \\ 1 & 3 & 37 \end{pmatrix}, \quad \mathbf{C_2} = \begin{pmatrix} 119 & 107 & 33 \\ 53 & 120 & 41 \\ 10 & 21 & 45 \end{pmatrix}, \quad \mathbf{C_3} = \begin{pmatrix} 5 & 13 & 9 \\ 44 & 96 & 61 \\ 6 & 4 & 21 \end{pmatrix}$$

### 4.2 Malignant Class Oncological Safety Audit
Across both external cohorts, primary malignant bone tumours suffered substantial false-negative misclassifications:
- **Center 2 Malignant Misclassifications (31 / 76 = 40.79%):**
  - 10 labeled as **Normal Bone Radiographs** (critical delay risk).
  - 21 labeled as **Benign Bone Tumours** (sub-critical risk; flagged as neoplasm).
  - 74 non-malignant cases erroneously predicted as malignant (33 normal, 41 benign).
- **Center 3 Malignant Misclassifications (10 / 31 = 32.26%):**
  - 6 labeled as **Normal Bone Radiographs** (critical delay risk).
  - 4 labeled as **Benign Bone Tumours** (flagged as neoplasm).
  - 70 non-malignant cases erroneously predicted as malignant (9 normal, 61 benign).
- **Clinical Implication:** In high-stakes orthopedic oncology, a 32–41% miss rate for primary osteosarcoma or chondrosarcoma is clinically unacceptable.

### 4.3 Root-Cause Analysis of Center 3 Normal-Class Collapse ($N=27$)
In Center 3, only 5 of 27 normal control radiographs were correctly identified (Sensitivity: 18.52%):
1. **Prior Shift:** Center 1 training data consisted of 54.2% normal radiographs, whereas Center 3 had only 10.42% normal radiographs.
2. **Focal Loss Weighting Artifact:** Stage 17 training applied $\alpha = [0.3333, 0.4667, 2.1850]$ to prevent missing rare malignant lesions. When exposed to an external distribution where 89.58% of cases are neoplastic, this penalty caused the network to over-predict pathology on normal scans.
3. **Small Sample Uncertainty:** With only 27 normal images, random variation and subtle non-neoplastic degenerative changes heavily amplify the observed error rate, resulting in a wide 95% CI of `[5.0%, 34.5%]`.

---

## 5. Statistical and Clinical Interpretation Limitations

1. **Image-Level Resampling Limitation:** BTXRD provides image-level data without patient accession identifiers. 1,000 bootstrap iterations were conducted over image observations; patient-level independence is unverified.
2. **Prevalence-Induced High NPV:** Center 3's 94.05% Malignant NPV is driven by the 88.03% non-malignant prevalence and must **not** be presented as evidence that the model reliably excludes malignancy.
3. **Uncalibrated Softmax Probabilities:** Brier score escalated to 0.6486 in Center 3, demonstrating unmitigated overconfidence under distribution shift. Post-hoc calibration on holdouts was strictly avoided to prevent leakage.
4. **Non-Clinical Software Status:** The system is an investigational research prototype and has not been cleared by any regulatory agency for clinical diagnostic use.

---

## 6. Model Readiness Decisions

| Domain | Verdict | Decision Summary |
| :--- | :---: | :--- |
| **Research Prototype** | **CONDITIONAL PASS** | Validated as an experimental baseline for deep radiomic research on skeletal radiographs. |
| **Cross-Center Generalization** | **FAIL** | Unacceptable performance drop on external hospitals (Macro-F1 drops $>37\%$). |
| **Clinical Deployment** | **STRICTLY PROHIBITED** | 32–41% malignant miss rate creates unacceptable patient safety hazards. |
| **Final OsteoGuard Integration** | **UNSUITABLE IN CURRENT FORM** | Predicts 3 classes (Normal/Benign/Malignant primary tumour); does **not** classify secondary bone metastasis, Paget's disease, or osteogenesis imperfecta. |

---

## 7. Evidence-Based Remediation Priorities

1. **Patient-Level Clustering:** Enforce cryptographic patient IDs and Group K-Fold splitting.
2. **Acquisition Standardization:** Ingest raw DICOM metadata and auto-calibrate anatomical window/levels.
3. **Biopsy-Proven Consensus:** Ground truth curated via 3-reader MSK radiologist panels correlated with histological pathology.
4. **Logit Adjustment & Domain Adaptation:** Incorporate Class-Dependent Logit Adjustment and Domain-Adversarial Neural Networks (DANN).
5. **Calibrated Prediction Sets:** Fit temperature scaling on an isolated development calibration split; implement Split Conformal Prediction.
6. **Blinded Center 4 & Center 5 Validation:** Evaluate next-generation models on untouched prospective external hospitals.

---

## 8. Automated Test Suite Results

Dedicated audit tests were authored in `ml_service/tests/test_stage_21_multicenter_audit.py` verifying hash preservation, artifact completeness, metrics consistency, and non-regression across all previous stages.

---

## 9. Stage Gate Enforcement & Stop Boundary

- **STAGE 21 IS COMPLETE AND PASSED.**
- **HARD GATE ENFORCED:**
  - Audit and model disposition are officially finalized.
  - No further training or fine-tuning will be launched.
  - **STOP AFTER STAGE 21.** Awaiting user direction before initiating any future stages.

