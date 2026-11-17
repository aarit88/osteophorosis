# STAGE 18 REPORT: BTXRD FINAL MODEL FREEZE AND EXTERNAL EVALUATION READINESS AUDIT

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 18 — BTXRD Final Model Freeze & External Evaluation Readiness Audit  
**Status:** **PASS**  
**Execution Timestamp:** 2026-11-16T07:25:00+05:30  
**Host Environment:** Windows 11 Home (x64)  
**Hardware Accelerator:** NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM, Compute Capability 8.9)  
**Isolated GPU Environment:** `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe` (Python 3.13.3, PyTorch 2.14.1+cu126)  
**System CPU Python:** `C:\Python313\python.exe` (PyTorch 2.9.0+cpu — untouched)  
**React Frontend:** Untouched  

---

## 1. Executive Summary

Stage 18 establishes the formal freeze, cryptographic verification, and external evaluation protocol for the final BTXRD three-class primary bone tumour classifier trained during Stage 17. The model weights, preprocessing pipelines, class mapping schemas, and decision boundaries are now sealed in an immutable state.

### Key Milestones & Audit Outcomes
1. **Cryptographic Checkpoint Immutability:**
   - The Stage 17 authoritative best checkpoint (Epoch 18) was verified and copied to `ml_artifacts/btxrd_frozen_model/best_checkpoint.pt`.
   - The SHA-256 hash was calculated as:
     `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`
   - Bit-level identity between the training artifact and the frozen deployment model was strictly confirmed.
2. **Deterministic Validation Reproduction:**
   - Validation performance was independently audited on the Center 1 clean validation partition ($N=588$) using the frozen inference engine.
   - The reproduced performance matched Stage 17 metrics:
     - **Validation Macro-F1:** **`0.8103`**
     - **Validation Macro PR-AUC:** **`0.8700`**
     - **Balanced Accuracy:** **`82.90%`** | **Overall Accuracy:** **`81.80%`**
     - **Macro ROC-AUC (OvR):** **`0.9114`**
     - **Multiclass Brier Score:** **`0.3168`**
     - **Confusion Matrix:** $\begin{pmatrix} 290 & 26 & 3 \\ 65 & 154 & 9 \\ 1 & 3 & 37 \end{pmatrix}$
3. **Sealed External Multi-Center Holdouts:**
   - External validation cohorts from **Center 2** ($N=549$) and **Center 3** ($N=259$) were audited solely via tabular metadata.
   - Zero image pixels were accessed or loaded.
   - Separate, non-overlapping external evaluation specifications were formally registered.
   - Center 3's severe epidemiological skew towards benign lesions (only 27 normal radiographs, 10.42% prevalence) was formally documented to safeguard future metric interpretation.
4. **Probability Calibration Status Clarified:**
   - The model's probability outputs are confirmed as **raw, uncalibrated softmax probabilities**.
   - No post-hoc temperature scaling or isotonic regression was applied.
   - External data must **never** be used for calibration fitting.
5. **Quality Assurance:**
   - All 15 automated audit tests in `ml_service/tests/test_stage_18_btxrd_model_freeze.py` **PASSED** cleanly in 17.49s.
   - Regression testing over both Stage 17 and Stage 18 (30 tests total) completed with 100% pass rate in 27.94s.
   - FracAtlas, Bone Scintigraphy, host CPU Python, and React frontend remain 100% bit-level intact.

---

## 2. Model Provenance & Cryptographic Audit

### 2.1 Checkpoint Provenance
- **Origin Stage:** Stage 17 — Full-Scale BTXRD Three-Class Training
- **Training Epoch:** 18 of 25 (Predeclared Best Checkpoint)
- **Predeclared Model Selection Policy:** Primary: Validation Macro-F1; Secondary (tie-breaker): Validation Macro PR-AUC.
- **Source Path:** `ml_artifacts/btxrd_full_training/best_checkpoint.pt`
- **Frozen Path:** `ml_artifacts/btxrd_frozen_model/best_checkpoint.pt`
- **File Size:** `244,849,703 bytes` (244.85 MB)

### 2.2 SHA-256 Cryptographic Checksum
$$\text{SHA-256} = \mathbf{\text{21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD}}$$

Both source and frozen checkpoints share identical SHA-256 hashes. The hash is persisted in `ml_artifacts/btxrd_frozen_model/checkpoint_sha256.txt` and embedded in `model_freeze_manifest.json`.

---

## 3. Architecture & Parameter State

| Parameter Group | Modules | Parameter Count | Trainable at Epoch 18 | Frozen at Epoch 18 | Locked Eval Mode |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stem & Early Backbone** | `conv1`, `bn1`, `layer1`, `layer2` | 1,347,904 | 0 | 1,347,904 | YES (`eval()`) |
| **Mid-Level Backbone** | `layer3` (unfrozen at Epoch 6) | 7,084,544 | 7,084,544 | 0 | YES (`eval()` for BN) |
| **High-Level Backbone** | `layer4` | 12,851,200 | 12,851,200 | 0 | YES (`eval()` for BN) |
| **Classification Head** | `Dropout(0.4) -> Linear(512, 3)` | 2,563 | 2,563 | 0 | N/A |
| **Total** | — | **21,286,211** | **19,938,307 (93.67%)** | **1,347,904 (6.33%)** | — |

- **Weight Provenance:** ResNet-34 initialized with official ImageNet-1K pretrained weights (`ResNet34_Weights.IMAGENET1K_V1`, TorchVision SHA-256: `b627a593...`).

---

## 4. Frozen Preprocessing Pipeline

To guarantee that external evaluations reproduce the exact geometric conditions of training:
1. **Aspect-Ratio Preserving Letterboxing:**
   - Calculation: $s = \frac{512}{\max(W, H)}$, $W' = \text{round}(W \cdot s)$, $H' = \text{round}(H \cdot s)$.
   - Padding: $\text{pad}_x = (512 - W') // 2$, $\text{pad}_y = (512 - H') // 2$.
   - Symmetrical padding with pure black border: `RGB(0, 0, 0)`.
2. **Channel Format:** Single grayscale channel broadcast across 3 identical channels (`RGB`).
3. **Channel Normalization:**
   - Mean: $\mu = [0.485, 0.456, 0.406]$
   - Standard Deviation: $\sigma = [0.229, 0.224, 0.225]$
4. **Data Augmentations:** Strictly disabled (`eval_deterministic = True`).

---

## 5. Frozen Inference Protocol & API Contract

### 5.1 Primary Decision Rule
$$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} \left( \sigma(z)_c \right) = \arg\max_{c \in \{0, 1, 2\}} (z_c)$$
where $z \in \mathbb{R}^3$ represents raw logits and $\sigma(z)$ is the standard softmax vector.
- **Classification Thresholds:** Standard argmax (no post-hoc threshold tuning).
- **Threshold Freezing:** Thresholds must **not** be adjusted using external evaluation outcomes.

### 5.2 Authoritative Class Mapping

| Index | Authoritative Target Name | Clinical Scope |
| :---: | :--- | :--- |
| **0** | `NORMAL_BONE_RADIOGRAPH` | Normal skeletal control with intact cortex and physiological trabecular architecture |
| **1** | `BENIGN_BONE_TUMOR` | Benign primary bone neoplasm (e.g. osteochondroma, enchondroma, non-ossifying fibroma) |
| **2** | `PRIMARY_MALIGNANT_TUMOR` | Primary malignant bone sarcoma (e.g. osteosarcoma, chondrosarcoma, Ewing sarcoma) |

### 5.3 Output Schema Specification
For each evaluated radiograph, the inference engine returns:
```json
{
  "normal_probability": 0.1172,
  "benign_probability": 0.0910,
  "malignant_probability": 0.7920,
  "predicted_class_idx": 2,
  "predicted_class_name": "PRIMARY_MALIGNANT_TUMOR",
  "model_version": "1.0.0",
  "checkpoint_sha256": "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD",
  "calibration_status": "uncalibrated_raw_softmax"
}
```

---

## 6. Validation Performance Reproduction Audit

The frozen model was evaluated against the Center 1 clean validation partition ($N=588$) to verify deterministic reproducibility:

| Class Index | Target Diagnostic Entity | Support | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `NORMAL_BONE_RADIOGRAPH` | 319 | **90.91%** | 75.46% | 81.46% | **0.8593** | 0.8771 |
| **1** | `BENIGN_BONE_TUMOR` | 228 | **67.54%** | 91.94% | 84.15% | **0.7494** | 0.8382 |
| **2** | `PRIMARY_MALIGNANT_TUMOR` | 41 | **90.24%** | **97.81%** | **75.51%** | **0.8222** | **0.8949** |
| **Macro** | **Unweighted Average** | **588** | **82.90%** | **88.40%** | **80.37%** | **0.8103** | **0.8700** |

- **Audited Confusion Matrix:**
  $$\begin{pmatrix} 290 & 26 & 3 \\ 65 & 154 & 9 \\ 1 & 3 & 37 \end{pmatrix}$$
- **Result:** Exact reproduction confirmed across all metrics and confusion matrix elements.

---

## 7. Probability Calibration Policy & Clarifications

1. **Uncalibrated Softmax Confirmation:**
   - The Stage 17 multiclass Brier score is **`0.3168`**.
   - Model probabilities represent **raw softmax outputs** normalized via $\frac{e^{z_i}}{\sum_j e^{z_j}}$.
   - **No probability calibration function** (such as Platt scaling, vector scaling, temperature scaling, or isotonic regression) was fitted.
2. **Prohibition of External Calibration:**
   - Fitting a calibration function on external holdout datasets (Center 2 or Center 3) is **strictly forbidden**, as it would violate independent evaluation principles.
3. **Clinical Interpretation Note:**
   - While the raw Brier score reflects solid probability assignment, raw deep neural network probabilities are susceptible to miscalibration on out-of-distribution domain shifts. Downstream clinical risk models must treat these outputs as ordinal risk ranks rather than true posterior probabilities.

---

## 8. External Multi-Center Evaluation Protocol Design

Two independent external validation benchmarks have been configured. These cohorts remain strictly sealed.

### 8.1 Center 2: Primary External Three-Class Benchmark
- **Cohort Size:** 549 radiographs
- **Split Identifier:** `external_test_center2`
- **Class Breakdown:**
  - `NORMAL_BONE_RADIOGRAPH`: 259 (47.18%)
  - `BENIGN_BONE_TUMOR`: 214 (38.98%)
  - `PRIMARY_MALIGNANT_TUMOR`: 76 (13.84%)
- **Scientific Role:** Assesses generalizability across independent scanner hardware, imaging protocols, and patient demographics under balanced diagnostic conditions.

### 8.2 Center 3: Secondary External Three-Class Stress Test
- **Cohort Size:** 259 radiographs
- **Split Identifier:** `external_test_center3`
- **Class Breakdown:**
  - `NORMAL_BONE_RADIOGRAPH`: 27 (10.42%)
  - `BENIGN_BONE_TUMOR`: 201 (77.61%)
  - `PRIMARY_MALIGNANT_TUMOR`: 31 (11.97%)
- **Scientific Role & Clinical Caution:**
  - Represents an acute epidemiological stress test characterized by a specialized oncology referral pattern (89.58% abnormal / tumor-positive).
  - With **only 27 normal control radiographs**, specificity and negative predictive value (NPV) estimates will have wider confidence intervals and must be interpreted in clinical context.

### 8.3 Isolation Rules
- **No Result Merging:** Center 2 and Center 3 metrics must **never** be merged into a single pooled score.
- **Independent Reporting:** Separate confusion matrices, per-class sensitivity/specificity tables, and bootstrap confidence intervals must be computed independently for each center.

---

## 9. Immutable Model Directory Structure

The frozen model package in `ml_artifacts/btxrd_frozen_model/` contains:

```
ml_artifacts/btxrd_frozen_model/
├── best_checkpoint.pt          (244.85 MB — Verified Epoch 18 weights)
├── checkpoint_sha256.txt       (Cryptographic hash)
├── class_mapping.json          (Authoritative 3-class index and clinical dictionary)
├── model_config.json           (ResNet-34 architecture and layer freeze specification)
├── model_freeze_manifest.json  (Comprehensive model freeze provenance metadata)
└── preprocessing_config.json   (Deterministic letterbox and normalization parameters)
```

---

## 10. Automated Audit Test Suite Results

The dedicated test suite `ml_service/tests/test_stage_18_btxrd_model_freeze.py` was executed via pytest in the GPU virtual environment:

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\archi\OneDrive\Desktop\osteophorosis-main
collected 15 items

test_01_original_checkpoint_exists_and_hash               PASSED [  6%]
test_02_frozen_copy_hash_match                            PASSED [ 13%]
test_03_frozen_model_directory_manifest_completeness      PASSED [ 20%]
test_04_class_mapping_consistency                         PASSED [ 26%]
test_05_model_architecture_and_reload                     PASSED [ 33%]
test_06_frozen_classifier_predict_single_image            PASSED [ 40%]
test_07_validation_performance_reproduction_audit         PASSED [ 46%]
test_08_center_2_and_center_3_metadata_and_boundaries     PASSED [ 53%]
test_09_center_3_low_normal_prevalence_documentation     PASSED [ 60%]
test_10_conflict_quarantine_enforcement                   PASSED [ 66%]
test_11_calibration_policy_audit                          PASSED [ 73%]
test_12_fracatlas_artifacts_preservation                  PASSED [ 80%]
test_13_bone_scintigraphy_artifacts_preservation          PASSED [ 86%]
test_14_stage_16_and_17_artifacts_preservation            PASSED [ 93%]
test_15_cpu_python_and_frontend_preservation              PASSED [100%]

======================= 15 passed, 1 warning in 17.49s ========================
```

---

## 11. Scientific Limitations & Boundaries

1. **Patient-Level Independence within Center 1:**
   - BTXRD lacks patient-level identifiers. Internal validation performance ($N=588$) may reflect radiograph-level rather than patient-level cross-validation. True patient-level generalizability will be determined exclusively on external multi-center cohorts (Center 2 and Center 3).
2. **Primary Malignant Bone Tumour Scope:**
   - Class 2 models primary bone malignancies (e.g. osteosarcoma, chondrosarcoma) visible on plain radiographs. It does not model secondary skeletal metastases, which are evaluated separately via bone scintigraphy.
3. **Research Software Prototype:**
   - OsteoGuard AI is an investigational software prototype and is not cleared for medical diagnostic use.

---

## 12. Stage Gate Enforcement & Mandatory Stop

- **STAGE 18 IS COMPLETE AND PASSED.**
- **HARD STOP APPLIED:**
  - Zero external images have been accessed or processed.
  - No external evaluation runs have been launched.
  - Awaiting explicit user direction before commencing Stage 19.
