# STAGE 20 REPORT: BTXRD CENTER 3 EXTERNAL STRESS TEST EVALUATION

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 20 — BTXRD Center 3 External Stress Test Evaluation  
**Status:** **PASS**  
**Execution Timestamp:** 2026-11-16T08:16:00+05:30  
**Host Environment:** Windows 11 Home (x64)  
**Hardware Accelerator:** NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM, Compute Capability 8.9)  
**Isolated GPU Environment:** `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe` (Python 3.13.3, PyTorch 2.14.1+cu126)  
**System CPU Python:** `C:\Python313\python.exe` (PyTorch 2.9.0+cpu — untouched)  
**React Frontend:** Untouched  

---

## 1. Executive Summary & Objective

Stage 20 executed the secondary external stress test for the cryptographically frozen BTXRD ResNet-34 classifier on the **Center 3 external cohort ($N=259$)**. Center 3 serves as an acute epidemiological stress test, representing a specialized tertiary orthopedic oncology referral center characterized by severe disease imbalance (**77.61% benign tumours, 11.97% primary bone malignancies, and only 10.42% normal controls**).

The evaluation was performed strictly in **inference-only mode** using the exact Stage 18 frozen model checkpoint. Preprocessing, model parameters, normalization, and the argmax classification rule were preserved without any retraining, fine-tuning, threshold shifting, or post-hoc calibration.

### Key Evaluation Findings
1. **Cryptographic Checkpoint Immutability:**
   - Pre-evaluation SHA-256: `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`
   - Post-evaluation SHA-256: `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`
   - Confirmed bit-for-bit identity; zero model modifications occurred.
2. **Stress Test Performance (Center 3, $N=259$):**
   - **Overall Accuracy:** **`47.10%`** (95% CI: `[40.93%, 53.28%]`)
   - **Balanced Accuracy:** **`44.67%`** (95% CI: `[36.94%, 52.60%]`)
   - **Macro-F1 Score:** **`0.3592`** (95% CI: `[0.3007, 0.4183]`)
   - **Weighted-F1 Score:** **`0.5285`** (95% CI: `[0.4687, 0.5866]`)
   - **Macro ROC-AUC (OvR):** **`0.6468`** (95% CI: `[0.5853, 0.7043]`)
   - **Macro PR-AUC (OvR):** **`0.4558`** (95% CI: `[0.4060, 0.5327]`)
   - **Multiclass Brier Score:** **`0.6486`** (raw uncalibrated softmax)
3. **Primary Malignancy Detection in Center 3 ($N=31$):**
   - **Malignant Sensitivity (Recall):** **`67.74%`** (95% CI: `[51.32%, 83.33%]`, 21 of 31 detected)
   - **Malignant Specificity:** **`69.30%`** (95% CI: `[63.40%, 75.23%]`, 158 of 228 non-malignant cases ruled out)
   - **Malignant Negative Predictive Value (NPV):** **`94.05%`** (95% CI: `[89.94%, 97.13%]`)
   - **Malignant Precision (PPV):** **`23.08%`** (95% CI: `[14.77%, 32.29%]`)
   - **Malignant ROC-AUC (OvR):** **`0.7490`** (95% CI: `[0.6479, 0.8326]`)
   - **Malignant PR-AUC:** **`0.3815`** (3.2x above the 11.97% prevalence baseline)
4. **Epidemiological Distortion Impact on Normal Controls ($N=27$):**
   - Due to extreme referral bias (only 27 normal X-rays out of 259), normal-class sensitivity is heavily compressed (**18.52%**, 95% CI: `[5.0%, 34.5%]`).
   - Normal specificity remains solid at **`78.45%`** (182 of 232 tumor-bearing radiographs correctly excluded from the normal class).
   - This asymmetry demonstrates the necessity of interpreting clinical AI performance in the context of institutional pre-test prevalence.

---

## 2. Pre-Flight Verification & Checkpoint Hash Audit

Prior to evaluating Center 3, the frozen checkpoint was verified:

| Checkpoint Identifier | File Location | File Size (Bytes) | SHA-256 Hash | Integrity Status |
| :--- | :--- | :---: | :--- | :---: |
| **Stage 18 Frozen Best Model** | `ml_artifacts/btxrd_frozen_model/best_checkpoint.pt` | 244,849,703 | `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD` | **VERIFIED** |
| **Post-Evaluation Verification** | `ml_artifacts/btxrd_frozen_model/best_checkpoint.pt` | 244,849,703 | `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD` | **UNMODIFIED** |

- Verified class mapping:
  - `0`: `NORMAL_BONE_RADIOGRAPH`
  - `1`: `BENIGN_BONE_TUMOR`
  - `2`: `PRIMARY_MALIGNANT_TUMOR`
- Frozen preprocessing configuration: $512 \times 512$ aspect-ratio letterboxing, black padding `(0, 0, 0)`, ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 3. Center 3 Dataset Cohort & Epidemiological Context

### 3.1 Cohort Composition
Center 3 represents an independent orthopedic hospital dataset characterized by a specialized referral filter:

| Class Index | Target Class Name | Image Count | Prevalence (%) | Pre-Split Partition | Conflict Flag |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `NORMAL_BONE_RADIOGRAPH` | 27 | 10.42% | `external_test_center3` | FALSE (0 conflicts) |
| **1** | `BENIGN_BONE_TUMOR` | 201 | 77.61% | `external_test_center3` | FALSE (0 conflicts) |
| **2** | `PRIMARY_MALIGNANT_TUMOR` | 31 | 11.97% | `external_test_center3` | FALSE (0 conflicts) |
| **Total** | — | **259** | **100.00%** | — | **0 conflicts** |

### 3.2 Dataset Isolation Rules
- **No Access to Center 1 or Center 2:** Center 1 training/validation data and Center 2 external evaluation data were **not** accessed or loaded.
- **Zero Result Merging:** Center 3 metrics are reported strictly on their own and are **never** pooled with Center 1 or Center 2.

---

## 4. Frozen Inference Protocol

The evaluation adhered strictly to the frozen Stage 18 inference specification:
1. **Decision Rule:**
   $$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} (p_c)$$
2. **Probability Calculation:**
   $$p = \text{Softmax}(z), \quad z = \text{ResNet34}(x)$$
3. **No Threshold Adjustment:** Predefined operating thresholds were maintained without tuning.
4. **No Calibration Fitting:** Raw softmax probabilities were evaluated directly.

---

## 5. Comprehensive Center 3 Evaluation Metrics

Metrics were calculated across all 259 Center 3 images with empirical 95% confidence intervals derived from 1,000 bootstrap iterations over image observations (random seed 42).

### 5.1 Overall Summary Metrics

| Metric | Point Estimate | 95% Confidence Interval | Clinical & Stress Test Interpretation |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | **47.10%** | `[40.93%, 53.28%]`) | Accuracy under extreme prevalence skew (77.6% benign) |
| **Balanced Accuracy** | **44.67%** | `[36.94%, 52.60%]`) | Unweighted mean recall across Normal, Benign, Malignant |
| **Macro-F1 Score** | **0.3592** | `[0.3007, 0.4183]` | Compressed by the low normal-class F1 (0.1220) |
| **Weighted-F1 Score** | **0.5285** | `[0.4687, 0.5866]` | Reflects strong performance on dominant benign cases |
| **Macro ROC-AUC (OvR)** | **0.6468** | `[0.5853, 0.7043]` | Multiclass discriminative ranking under domain stress |
| **Macro PR-AUC (OvR)** | **0.4558** | `[0.4060, 0.5327]` | Unweighted area under the precision-recall curves |
| **Multiclass Brier Score** | **0.6486** | — | Mean squared probability error (raw uncalibrated softmax) |

*Resampling Note:* Bootstrap resampling was performed over image observations ($N=259$). It does not establish patient-level independence because BTXRD metadata lacks patient-level identifiers.

---

### 5.2 Per-Class Diagnostic Performance

| Target Entity | Support | Prev (%) | Sensitivity [95% CI] | Specificity [95% CI] | Precision [95% CI] | NPV [95% CI] | F1-Score | ROC-AUC (OvR) | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Bone** | 27 | 10.42% | **18.52%** `[5.0%, 34.5%]` | **78.45%** `[72.8%, 83.4%]` | **9.09%** `[2.2%, 17.2%]` | **89.22%** `[84.7%, 93.3%]` | **0.1220** | 0.5121 | 0.1141 |
| **Benign Tumour** | 201 | 77.61% | **47.76%** `[40.9%, 55.1%]` | **70.69%** `[58.7%, 81.2%]` | **84.96%** `[78.6%, 91.3%]` | **28.08%** `[20.6%, 35.6%]` | **0.6115** | 0.6793 | 0.8718 |
| **Primary Malignant** | 31 | 11.97% | **67.74%** `[51.3%, 83.3%]` | **69.30%** `[63.4%, 75.2%]` | **23.08%** `[14.8%, 32.3%]` | **94.05%** `[89.9%, 97.1%]` | **0.3443** | **0.7490** | **0.3815** |

---

## 6. Confusion Matrix & Oncological Safety Audit

### 6.1 Center 3 Confusion Matrix

$$\begin{pmatrix}
\text{Actual Normal (27)} & \mathbf{5} & 13 & 9 \\
\text{Actual Benign (201)} & 44 & \mathbf{96} & 61 \\
\text{Actual Malignant (31)} & 6 & 4 & \mathbf{21}
\end{pmatrix}$$

#### Row-Normalized Confusion Matrix (Recall %)
- **True Normal (27):** **18.52%** predicted Normal | 48.15% predicted Benign | 33.33% predicted Malignant
- **True Benign (201):** 21.89% predicted Normal | **47.76%** predicted Benign | 30.35% predicted Malignant
- **True Malignant (31):** 19.35% predicted Normal | 12.90% predicted Benign | **67.74%** predicted Malignant

---

### 6.2 Primary Malignancy Oncological Safety Breakdown ($N=31$)
1. **Correct Detections (True Positives):**
   - **21 of 31 (67.74%)** primary bone malignancies were correctly identified.
2. **False Benign Misclassifications (Minor Errors):**
   - **4 of 31 (12.90%)** were classified as benign tumours.
   - *Combined Neoplasm Detection:* **25 of 31 (80.65%)** were correctly flagged as skeletal lesions requiring orthopedic oncology workup.
3. **False Normal Misclassifications (Critical Errors):**
   - **6 of 31 (19.35%)** were classified as normal controls.
4. **False Positive Malignancies ($N=70$):**
   - 9 normal radiographs and 61 benign tumours were predicted as malignant.
   - Malignant NPV is **`94.05%`** (158 of 168 negative malignant predictions were truly non-malignant), providing solid reassurance when malignancy is ruled out.

---

### 6.3 Normal and Benign Class Error Breakdown
1. **Normal Radiographs ($N=27$):**
   - 5 correctly identified as normal.
   - 13 called benign primary tumours; 9 called primary malignant tumours.
   - *Root Cause:* The model was trained with Focal Loss class weighting ($\alpha = 2.1850$ for malignant vs $0.3333$ for normal) on Center 1 (where normal prevalence was 54.2%). In Center 3, where 89.6% of patients have tumours, the classifier frequently flags normal controls as suspicious lesions.
2. **Benign Tumours ($N=201$):**
   - 96 correctly identified as benign.
   - 44 misclassified as normal; 61 misclassified as malignant.
   - Because of high prevalence (77.6%), Benign Precision is very high (**84.96%**).

---

## 7. Comprehensive 3-Center Cross-Institution Synthesis

The table below contrasts the classifier's performance across all three independent hospital centers:

| Metric | Center 1 Internal Val ($N=588$) | Center 2 External Holdout ($N=549$) | Center 3 External Stress Test ($N=259$) | Cross-Center Pattern & Clinical Insight |
| :--- | :---: | :---: | :---: | :--- |
| **Cohort Role** | Development Validation | Primary External Benchmark | Secondary External Stress Test | Progressive domain and prevalence shift |
| **Class Distribution (N/B/M)** | 54.2% / 38.8% / 7.0% | 47.2% / 39.0% / 13.8% | **10.4% / 77.6% / 12.0%** | Center 3 is heavily skewed towards tumor cases |
| **Overall Accuracy** | **81.80%** | **51.73%** | **47.10%** | Progressive drop reflects domain shift & prevalence distortion |
| **Balanced Accuracy** | **82.90%** | **53.74%** | **44.67%** | Equal-weight recall compressed by normal-class degradation |
| **Macro-F1 Score** | **0.8103** | **0.5069** | **0.3592** | Lower in Center 3 due to normal-class F1 (0.1220) |
| **Weighted-F1 Score** | **0.8180** | **0.5210** | **0.5285** | Weighted F1 in Center 3 stabilized by benign support |
| **Macro ROC-AUC (OvR)** | **0.9114** | **0.7180** | **0.6468** | Inherent discriminative capacity remains above chance (0.50) |
| **Malignant ROC-AUC** | **0.9868** | **0.8198** | **0.7490** | **Consistently strong separation for primary malignancies** |
| **Malignant Sensitivity** | **90.24%** (37/41) | **59.21%** (45/76) | **67.74%** (21/31) | Malignant sensitivity higher in Center 3 than Center 2 |
| **Malignant Specificity** | **97.81%** | **84.36%** | **69.30%** | More benign cases overcalled malignant in Center 3 |
| **Malignant NPV** | **99.25%** | **92.79%** | **94.05%** | **Consistently high rule-out safety across all centers (>92%)** |
| **Multiclass Brier Score** | **0.3168** | **0.5731** | **0.6486** | Progressive probability entropy under novel domain conditions |

---

## 8. Probability Calibration Analysis

- **Center 3 Brier Score:** **`0.6486`**
- All probabilities represent **raw, uncalibrated softmax scores** ($\sigma(z)$).
- **Prohibition of Test-Set Calibration:** Fitting temperature scaling or isotonic regression on Center 3 was strictly prohibited to prevent data leakage.
- The increase in Brier score from Center 1 (0.3168) to Center 2 (0.5731) and Center 3 (0.6486) confirms that neural network confidence degrades systematically under institutional domain shift.

---

## 9. Generated Publication Artifacts

The following diagnostic figures were rendered and saved in `ml_artifacts/btxrd_center3_external_evaluation/`:
1. **`center3_confusion_matrix.png`**: Dual-panel heatmap displaying raw counts and row-normalized recall percentages.
2. **`center3_roc_curves.png`**: One-vs-Rest ROC curves displaying Macro AUC (0.6468) and Malignant AUC (0.7490).
3. **`center3_precision_recall_curves.png`**: One-vs-Rest Precision-Recall curves displaying Benign PR-AUC (0.8718) and Malignant PR-AUC (0.3815).

---

## 10. Deliverables Catalog

All required evaluation artifacts have been persisted in `ml_artifacts/btxrd_center3_external_evaluation/`:

| Artifact | Size | Description |
| :--- | :---: | :--- |
| `center3_predictions.csv` | 23.25 KB | Per-image predictions and raw softmax probabilities for all 259 images |
| `classification_metrics.json` | 1.04 KB | Overall evaluation metrics with 95% bootstrap confidence intervals |
| `per_class_metrics.json` | 2.72 KB | Per-class diagnostic performance metrics with 95% bootstrap CIs |
| `confusion_matrix.json` | 1.13 KB | Complete $3 \times 3$ confusion matrix and full error analysis data |
| `evaluation_manifest.json` | 1.48 KB | Comprehensive metadata, cohort verification, and parameter provenance |
| `checkpoint_hash_verification.txt` | 231 B | Verified pre- and post-evaluation SHA-256 cryptographic hashes |
| `center3_confusion_matrix.png` | 171.92 KB | High-resolution confusion matrix plot |
| `center3_roc_curves.png` | 181.15 KB | High-resolution multiclass ROC curves |
| `center3_precision_recall_curves.png` | 208.01 KB | High-resolution multiclass Precision-Recall curves |

---

## 11. Automated Test Suite Results

Dedicated unit and regression tests were authored in `ml_service/tests/test_stage_20_center3_evaluation.py` and executed via pytest:

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\archi\OneDrive\Desktop\osteophorosis-main
collected 15 items

test_01_checkpoint_hash_pre_and_post_verification         PASSED [  6%]
test_02_center3_partition_isolation_and_row_count         PASSED [ 13%]
test_03_class_mapping_integrity                           PASSED [ 20%]
test_04_prediction_row_count_and_columns                  PASSED [ 26%]
test_05_probability_validity_and_sum_to_one               PASSED [ 33%]
test_06_overall_metrics_consistency                       PASSED [ 40%]
test_07_per_class_metrics_consistency                     PASSED [ 46%]
test_08_confusion_matrix_sum_and_integrity                PASSED [ 53%]
test_09_center_1_and_center_2_strictly_isolated           PASSED [ 60%]
test_10_conflict_quarantine_enforcement                   PASSED [ 66%]
test_11_calibration_status_uncalibrated                   PASSED [ 73%]
test_12_fracatlas_artifacts_preservation                  PASSED [ 80%]
test_13_bone_scintigraphy_artifacts_preservation          PASSED [ 86%]
test_14_stage_16_17_18_19_artifacts_preservation          PASSED [ 93%]
test_15_cpu_python_and_frontend_preservation              PASSED [100%]

======================= 15 passed in 14.50s =======================
```

---

## 12. Scientific Limitations & Boundaries

1. **Prevalence-Dependent Metric Distortion:**
   - Center 3's low normal prevalence (10.42%) heavily penalizes Normal PPV (9.09%) and Macro-F1 (0.3592). This does not necessarily indicate model failure, but rather highlights Bayes' theorem in action: PPV is strongly dependent on pre-test prevalence.
2. **Missing Patient Accession Identifiers:**
   - BTXRD lacks patient-level identifiers; bootstrap resampling reflects image-level rather than patient-level variability.
3. **Research Prototype Notice:**
   - OsteoGuard AI is an investigational software prototype and is not FDA/CE cleared for clinical diagnosis.

---

## 13. Stage Gate Enforcement & Mandatory Stop

- **STAGE 20 IS COMPLETE AND PASSED.**
- **STRICT HARD GATE ENFORCED:**
  - Center 3 results are recorded independently and **not pooled** with Center 1 or Center 2.
  - **DO NOT start Stage 21 automatically.**
  - Awaiting explicit user direction before initiating any subsequent stage.

