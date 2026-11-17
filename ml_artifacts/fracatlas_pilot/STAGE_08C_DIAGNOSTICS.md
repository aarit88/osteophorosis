# STAGE 08C — FRACATLAS MODEL PREDICTION DIAGNOSTICS & THRESHOLD ANALYSIS REPORT

**Project:** OsteoGuard AI — A Unified Deep Learning and Radiomic Framework for Differential Diagnosis and Biomechanical Fracture Risk Stratification  
**Stage:** 08C (Post-Pilot Forensic Prediction Diagnostics)  
**Date:** November 15, 2026  
**Status:** **PASS**  
**Author:** Senior Machine Learning Architect & Lead Medical Vision Engineer  

---

## 1. EXECUTIVE SUMMARY & INVESTIGATION DIRECTIVE

In Stage 08B, the ImageNet-pretrained ResNet-101 pilot model trained for 3 epochs on FracAtlas achieved strong discriminative representation learning:
- **Validation ROC-AUC:** **0.7675** (Area Under the Receiver Operating Characteristic)
- **Validation PR-AUC:** **0.4068** (Baseline prevalence = 0.1748; >2.3x higher than random chance)
- **Validation Loss:** Steady monotonic descent from **0.5756 to 0.5368**

However, evaluation at the default binary decision threshold of $\theta = 0.50$ produced an apparent anomaly:
- **True Positives (TP): 0**
- **False Positives (FP): 0**
- **True Negatives (TN): 505**
- **False Negatives (FN): 107**
- **Sensitivity / Recall:** **0.00%**
- **Specificity:** **100.0%**

### Core Forensic Finding:
The occurrence of zero positive predictions at threshold 0.50 was **not** caused by inverted class mapping, broken weights, or corrupted data.
1. The **maximum predicted fracture probability across the entire 612-image validation cohort was $P_{\max} = 0.4915$**.
2. Because every single validation image satisfied $P(\text{fracture} \mid X) < 0.5000$, setting the decision boundary at $\theta = 0.50$ mathematically guaranteed **100% assignment to class 0**.
3. The model's predicted probabilities for fractured images ($\mu = 0.3767, \text{median} = 0.3804$) are clearly separated from non-fractured controls ($\mu = 0.3102, \text{median} = 0.3096$).
4. When calibrated on the validation set at **$\theta = 0.35$ (Best F1 and Best Youden's $J$)**, the model achieves **71.96% sensitivity and 69.11% specificity** (F1 = 0.4529, Balanced Accuracy = 70.54%).
5. At **$\theta = 0.32$**, the model achieves **83.18% sensitivity** (89 out of 107 fractures detected).

---

## 2. STEP 1 — CLASS MAPPING & PROBABILITY EXTRACTION AUDIT

A rigorous audit was conducted across `fracatlas_dataset.py`, `fracatlas_pilot.py`, and the model checkpoint:

| Pipeline Component | Inspected Code / Configuration | Verified Status |
| :--- | :--- | :---: |
| **Dataset Label Mapping** | `row["fractured"] == "0"` $\to$ Control (0)<br>`row["fractured"] == "1"` $\to$ Fractured (1) | **CONFIRMED** |
| **Model Classification Head** | `nn.Linear(2048, 2)`: Logit Index 0 = Control, Logit Index 1 = Fractured | **CONFIRMED** |
| **Softmax Function** | `probs = torch.softmax(logits, dim=1)` | **CONFIRMED** |
| **Positive Class Extraction** | `probs[:, 1]` extracted as $P(\text{fracture} \mid X)$ | **CONFIRMED** |
| **ROC-AUC Calculation** | `roc_auc_score(y_true, y_prob)` with `y_prob = probs[:, 1]` | **CONFIRMED** |
| **Confusion Matrix Labels** | `confusion_matrix(y_true, y_pred, labels=[0, 1])` | **CONFIRMED** |

**Conclusion:** Zero class mapping inversion exists. The labels, logits, probabilities, and ROC-AUC calculation are 100% mathematically and semantically concordant.

---

## 3. STEP 2 — VALIDATION PROBABILITY DISTRIBUTION ANALYSIS

All 612 validation radiographs were evaluated using `best_checkpoint.pt` on `cuda:0`. Predictions were exported to `ml_artifacts/fracatlas_pilot/validation_predictions.csv`.

### Probability Distributions for $P(\text{Class 1 = Fractured})$:

| Statistical Metric | Actual Fractured ($N = 107$) | Actual Non-Fractured ($N = 505$) | Full Cohort ($N = 612$) |
| :--- | :---: | :---: | :---: |
| **Count** | 107 | 505 | 612 |
| **Mean ($\mu$)** | **0.3767** | **0.3102** | **0.3218** |
| **Standard Deviation ($\sigma$)** | 0.0588 | 0.0682 | 0.0713 |
| **Minimum** | 0.2242 | 0.1444 | 0.1444 |
| **5th Percentile (p5)** | 0.2554 | 0.1992 | 0.2043 |
| **10th Percentile (p10)** | 0.3001 | 0.2194 | 0.2264 |
| **25th Percentile (p25)** | 0.3476 | 0.2572 | 0.2650 |
| **Median (p50)** | **0.3804** | **0.3096** | **0.3263** |
| **75th Percentile (p75)** | 0.4194 | 0.3636 | 0.3752 |
| **90th Percentile (p90)** | 0.4473 | 0.3946 | 0.4118 |
| **95th Percentile (p95)** | 0.4622 | 0.4133 | 0.4392 |
| **Maximum** | **0.4782** | **0.4915** | **0.4915** |

### Key Diagnostic Observations:
1. **$P_{\max} = 0.4915 < 0.5000$:** The entire output space for the positive class is strictly bounded below 0.50. Under $\theta = 0.50$, positive predictions are impossible.
2. **Distribution Separation:** The fractured cohort's mean ($0.3767$) is $+0.0665$ higher than controls ($0.3102$). The 25th percentile of fractured cases ($0.3476$) exceeds the median of controls ($0.3096$).
3. **Probability Compression:** Softmax probabilities are compressed within the interval $[0.14, 0.49]$, reflecting the uncalibrated posterior distribution of an imbalanced training set (~17.5% prevalence) after 3 transfer-learning epochs.

---

## 4. STEP 3 — THRESHOLD SWEEP ANALYSIS (0.05 TO 0.95)

We evaluated 91 threshold increments from $\theta = 0.05$ to $\theta = 0.95$ with step size $0.01$. Results were exported to `ml_artifacts/fracatlas_pilot/threshold_analysis.csv`.

### 4.1 Performance at Critical Operating Points

| Operating Point | Threshold ($\theta$) | TP | FP | TN | FN | Sensitivity | Specificity | Precision | F1-Score | Accuracy | Balanced Acc | Youden's $J$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Threshold** | **0.50** | **0** | **0** | **505** | **107** | **0.00%** | **100.0%** | 0.00% | 0.0000 | 82.52% | 50.00% | 0.0000 |
| **Conservative Cutoff** | 0.40 | 40 | 45 | 460 | 67 | 37.38% | 91.09% | 47.06% | 0.4167 | 81.70% | 64.24% | 0.2847 |
| **Best F1-Score** | **0.35** | **77** | **156** | **349** | **30** | **71.96%** | **69.11%** | **33.05%** | **0.4529** | **69.61%** | **70.54%** | **0.4107** |
| **Best Youden's $J$** | **0.35** | **77** | **156** | **349** | **30** | **71.96%** | **69.11%** | **33.05%** | **0.4529** | **69.61%** | **70.54%** | **0.4107** |
| **$\ge 80\%$ Sensitivity** | **0.32** | **89** | **238** | **267** | **18** | **83.18%** | **52.87%** | **27.22%** | **0.4101** | **58.17%** | **68.02%** | **0.3605** |
| **$\ge 90\%$ Sensitivity** | **0.29** | **97** | **301** | **204** | **10** | **90.65%** | **40.40%** | **24.37%** | **0.3842** | **49.18%** | **65.53%** | **0.3105** |

### 4.2 Sensitivity-Specificity Trade-off in Clinical Radiology
- In acute musculoskeletal injury triage, a **false negative (missed acute fracture)** is clinically hazardous, risking displaced bone union, avascular necrosis, and chronic deformity.
- A **false positive (false alarm)** merely prompts secondary radiologist review or dedicated localized projection views.
- **Clinical Conclusion:** Operating at $\theta = 0.50$ is unacceptable for clinical screening. Operating at $\theta = 0.35$ captures 72% of fractures with 69% specificity; operating at $\theta = 0.32$ captures 83.2% of fractures while eliminating more than half of normal radiographs from the manual review queue.

---

## 5. STEP 4 — FORENSIC IMAGE LOADING & CORRUPTION AUDIT

A complete forensic scan of all **4,083 images** in the FracAtlas dataset was performed using Pillow:

### 5.1 Truncated Images Discovery
- **Total Images Audited:** 4,083
- **Truncated Images Identified:** Exactly **59 images** (1.45% of dataset).
- **Pathology & Range:** All 59 truncated images belong to `candidate_control_radiograph` (non-fractured) and reside in the high-ID range `IMG0004028.jpg` to `IMG0004347.jpg`.
- **Partition Distribution:**
  - Training set: 40 images
  - Validation set: 7 images (`IMG0004028`, `IMG0004092`, `IMG0004120`, `IMG0004256`, `IMG0004273`, `IMG0004278`, `IMG0004297`)
  - Test set: 12 images
- **Nature of Truncation:** Unprocessed trailing bytes at end-of-file (range: 0 to 67 bytes). This is a well-known artifact caused by digital radiography PACS export tools truncating trailing End-Of-Image (EOI) zero-padding.
- **Recovery & Integrity:** With `ImageFile.LOAD_TRUNCATED_IMAGES = True`, 100% of the 59 images decode their full image matrices without pixel distortion, clipping, or missing channels.
- **Fatal / Unrecoverable Corruptions:** **0**. Zero images were discarded or skipped.

---

## 6. STEP 5 — TRAINING CONFIGURATION & LOGIT ANALYSIS

### 6.1 Logits and Classifier Head Weights
From `best_checkpoint.pt`:
- Classifier Head: `fc = nn.Linear(2048, 2)`
- Weight Euclidean Norms: $\|\mathbf{w}_0\|_2 = 0.6098, \quad \|\mathbf{w}_1\|_2 = 0.6031$ (balanced feature projection)
- Classifier Bias: $b_0 = -0.009690, \quad b_1 = +0.004890$ (initialized near 0 and minimally moved in 3 epochs)
- Mean Validation Logits:
  - Class 0 Logit: $\mu = 0.7932$
  - Class 1 Logit: $\mu = 0.0468$
  - Logit Difference: $\Delta z = z_1 - z_0 \approx -0.7464$
- Softmax Mapping:
  $$P(\text{class } 1) = \frac{1}{1 + e^{-\Delta z}} = \frac{1}{1 + e^{0.7464}} = \frac{1}{1 + 2.109} \approx 0.3216$$
  The mean predicted probability of $0.3218$ is directly governed by this logit deficit.

### 6.2 Confirmed Bugs vs. Model Limitations

| Issue | Classification | Impact | Stage 09 Resolution |
| :--- | :--- | :--- | :--- |
| **Threshold Fixed at 0.50** | **Operational Flaw** | Causes $TP = 0$ when posterior probabilities are uncalibrated | Implement validation Youden's $J$ threshold calibration |
| **Short Pilot Training (3 Epochs)** | **Design Scope** | 512 updates insufficient to adjust logit dynamic range | Extend to 25–30 epochs with Cosine Annealing |
| **Weighted CrossEntropyLoss** | **Training Limitation** | Up-weights positive loss scalar but does not re-center softmax | Adopt Focal Loss ($\gamma=2.0$) or class-balanced loss |
| **Zero Bias Initialization** | **Training Limitation** | Standard Linear head starts at bias 0, ignoring 17.5% prior | Initialize bias to prior: $b_1 = \log(\frac{\pi}{1-\pi}) \approx -1.545$ |

---

## 7. STEP 6 — GENERATED ARTIFACTS & VISUALIZATIONS

The following diagnostic outputs were generated and validated:

1. `validation_predictions.csv` (127.5 KB) — Complete inference records for all 612 validation images.
2. `threshold_analysis.csv` (5.2 KB) — Detailed performance metrics across 91 threshold steps.
3. `probability_distribution.png` (98.3 KB) — Histogram comparing fractured vs. control probabilities with threshold markers.
4. `roc_curve.png` (113.0 KB) — ROC curve illustrating discriminative ability (ROC-AUC = 0.7675) with optimal Youden's $J$ marker.
5. `precision_recall_curve.png` (99.2 KB) — PR curve (PR-AUC = 0.4068) with optimal F1 marker.
6. `diagnostic_summary.json` (28.1 KB) — Machine-readable summary of statistical distributions and corruption audits.

---

## 8. TEST SUITE PASS CONFIRMATION

The diagnostic test suite was executed in `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe`:
- `test_stage_08c_diagnostics.py`: **5 passed** (100%)
- `test_stage_08b_pilot.py`: **6 passed** (100%)
- **Total: 11/11 tests passing.**

---

## 9. JUSTIFIED RECOMMENDATIONS FOR STAGE 09 FULL-SCALE TRAINING

1. **Prior-Matching Classifier Bias Initialization:**
   Initialize the linear head bias using the empirical training prevalence ($\pi = 0.1758$):
   $$b_1 = \log\left(\frac{0.1758}{1 - 0.1758}\right) \approx -1.545, \quad b_0 = 0.0$$
   This prevents high initial loss on majority examples and accelerates convergence.
2. **Focal Loss Formulation:**
   Replace weighted CrossEntropy with Focal Loss ($\alpha = 0.75, \gamma = 2.0$):
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Down-weights easy control radiographs and focuses backpropagation on ambiguous bone contours.
3. **Automated Validation Threshold Calibration:**
   Before freezing the production model, run validation threshold optimization to lock the operating point for target clinical sensitivity (e.g., 85% sensitivity for triage).
4. **Learning Rate Scheduling & Progressive Unfreezing:**
   Train for 25 epochs using Cosine Annealing learning rate schedule (`eta_min = 1e-6`), unfreezing `layer3` at epoch 10 for enhanced radiographic feature extraction.

---

**STAGE 08C STATUS: PASS.**  
**Model diagnostics complete. Full-scale training has NOT been started. Awaiting your directive.**

