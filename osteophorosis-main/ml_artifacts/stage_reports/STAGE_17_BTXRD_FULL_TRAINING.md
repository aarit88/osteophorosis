# STAGE 17 REPORT: FULL-SCALE BTXRD THREE-CLASS RESNET-34 TRAINING

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 17 — Full-Scale BTXRD Three-Class Bone Tumour Training  
**Status:** **PASS**  
**Execution Timestamp:** 2026-11-16T06:40:00+05:30  
**Host Environment:** Windows 11 Home (x64)  
**Hardware Accelerator:** NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM, Compute Capability 8.9)  
**Isolated GPU Environment:** `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe` (Python 3.13.3, PyTorch 2.14.1+cu126)  
**System CPU Python:** `C:\Python313\python.exe` (PyTorch 2.9.0+cpu — untouched)  
**React Frontend:** Untouched  

---

## 1. Executive Summary

Stage 17 executed the full-scale production training run for the automated three-class differential diagnosis of bone tumours from plain radiographs using the BTXRD dataset under **Protocol B (Center 1 Development Cohort)**. The model was trained across 25 epochs with progressive backbone unfreezing, multi-class focal loss optimization, and automated early stopping patience monitoring. External multi-center validation cohorts from Center 2 ($N=549$) and Center 3 ($N=259$) were strictly preserved as sealed benchmarks and were **not accessed or evaluated** during this stage.

### Key Milestones & Empirical Results
1. **Convergence & Predeclared Best Model Selection:**
   - Model selection was governed strictly by the predeclared protocol: **Primary Criterion:** Validation Macro-F1; **Secondary Criterion (Tie-Breaker):** Validation Macro PR-AUC.
   - The authoritative best checkpoint occurred at **Epoch 18**, achieving:
     - **Validation Macro-F1:** **`0.8103`** (exceeding pilot performance of 0.6335 by +17.68 percentage points).
     - **Validation Macro PR-AUC:** **`0.8700`** (pilot: 0.7526, +11.74 percentage points).
     - **Validation Macro ROC-AUC (OvR):** **`0.9114`** (pilot: 0.8302).
     - **Balanced Accuracy:** **`82.90%`** | **Overall Accuracy:** **`81.80%`**.
     - **Validation Brier Score:** **`0.3168`** (indicating strong probability calibration).
2. **High-Stakes Primary Malignancy Detection:**
   - **Malignant Detection Sensitivity:** **`90.24%`** (37 of 41 primary malignant tumours detected; only a single case misclassified as normal).
   - **Malignant Specificity:** **`97.81%`** (535 of 547 non-malignant cases correctly ruled out; only 12 false positives across the entire 588-image validation set).
   - **Malignant Precision (PPV):** **`75.51%`** | **Malignant F1-Score:** **`0.8222`**.
   - **Malignant OvR ROC-AUC:** **`0.9868`** | **Malignant PR-AUC:** **`0.8949`**.
3. **Progressive Unfreezing Stability:**
   - During Epochs 1–5, the model trained with `layer4` and the classification head active ($13,115,907$ trainable parameters).
   - At **Epoch 6**, `layer3` was unfrozen with a fine-tuning learning rate of $3.0 \times 10^{-6}$, expanding trainable parameters to **$19,938,307$** (93.67%).
   - Batch normalization layers across frozen blocks remained strictly locked in `eval()` mode, preventing batch-statistic degradation from small mini-batches.
   - Training remained exceptionally resource-efficient, consuming a peak VRAM of only **`426.87 MiB`** (10.42% of 4 GB).
4. **Early Stopping Enforcement:**
   - Training completed all 25 planned epochs, with early stopping triggering at Epoch 25 following 7 consecutive epochs without exceeding Epoch 18's record Macro-F1 of 0.8103.
5. **Quality Assurance & Preservation:**
   - All 15 automated unit and regression tests in `ml_service/tests/test_stage_17_btxrd_full_training.py` **PASSED** (13.02s execution time).
   - FracAtlas Stage 09/10 checkpoints, Bone Scintigraphy 5-fold CV checkpoints, host CPU Python, and the React frontend remain 100% bit-level intact.

---

## 2. Dataset Cohorts & Boundary Controls

### 2.1 Center 1 Development Cohort
In strict adherence to Protocol B, training and validation occurred exclusively on Center 1 radiographs. All 4 duplicate images exhibiting conflicting diagnostic classifications (`IMG000591.jpeg`, `IMG003456.jpeg`, `IMG000707.jpeg`, `IMG003346.jpeg`) remained quarantined and excluded.

| Class Index | Target Class | Center 1 Train ($N=2,346$) | Train Prev (%) | Center 1 Val ($N=588$) | Val Prev (%) | Normalized Focal $\alpha$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | `NORMAL_BONE_RADIOGRAPH` | 1,272 | 54.22% | 319 | 54.25% | **0.3333** |
| **1** | `BENIGN_BONE_TUMOR` | 880 | 37.51% | 228 | 38.78% | **0.4817** |
| **2** | `PRIMARY_MALIGNANT_TUMOR` | 194 | 8.27% | 41 | 6.97% | **2.1850** |
| **Total** | — | **2,346** | 100.00% | **588** | 100.00% | **3.0000** |

### 2.2 Hospital Center Isolation
- **Center 2 External Holdout:** 549 radiographs (259 normal, 214 benign, 76 malignant) — **COMPLETELY SEALED**.
- **Center 3 External Holdout:** 259 radiographs (27 normal, 201 benign, 31 malignant) — **COMPLETELY SEALED**.
- *Verification:* Automated test `test_11_zero_external_center_access` verified that 100% of the 588 validation predictions originate from Center 1, confirming zero multi-center contamination.

---

## 3. Preprocessing & Architecture

### 3.1 Aspect-Ratio Letterboxing Preprocessing
To preserve anatomical proportions and fine cortical margin patterns without distortion:
1. Scaling factor: $s = \frac{512}{\max(W, H)}$.
2. Scaled dimensions: $W' = \text{round}(W \cdot s), \; H' = \text{round}(H \cdot s)$.
3. Symmetrical padding: $\text{pad}_x = (512 - W') // 2, \; \text{pad}_y = (512 - H') // 2$ with pure black canvas `(0, 0, 0)`.
4. Grayscale broadcast: Broadcast single luminance channel to 3 channels.
5. Standardized Normalization: ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 3.2 Progressive Unfreezing Architecture
- **Backbone:** TorchVision ResNet-34 (`IMAGENET1K_V1`).
- **Phase 1 (Epochs 1–5):**
  - Frozen: `conv1`, `bn1`, `layer1`, `layer2`, `layer3` (8,170,304 parameters).
  - Trainable: `layer4` + `classifier` (13,115,907 parameters, 61.62%).
- **Phase 2 (Epochs 6–25):**
  - `unfreeze_layer3()` called at Epoch 6.
  - Frozen: `conv1`, `bn1`, `layer1`, `layer2` (1,347,904 parameters).
  - Trainable: `layer3` ($3.0 \times 10^{-6}$ LR), `layer4` ($1.0 \times 10^{-5}$ LR), `classifier` ($1.0 \times 10^{-4}$ LR).
  - Trainable parameters: **19,938,307** (93.67%).
- **Locked BatchNorm:** All frozen BatchNorm modules kept in `eval()` mode across both phases.

### 3.3 Loss Function
Multi-Class Focal Loss with $\gamma = 2.0$ and inverse-frequency class weights:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $\alpha = [0.3333, 0.4817, 2.1850]$, evaluated via `F.log_softmax(logits, dim=1)` for numerical stability.

---

## 4. Full 25-Epoch Training History

| Epoch | Duration | Peak VRAM | Train Loss | Val Loss | Macro-F1 | Balanced Acc | Overall Acc | Macro ROC | Macro PR | Brier Score | Checkpoint Event |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 1331.6s | 305.5 MiB | 0.300391 | 0.177803 | 0.5645 | 67.25% | 60.88% | 0.7811 | 0.6803 | 0.5516 | ★ New Best |
| **02** | 1601.3s | 305.5 MiB | 0.238221 | 0.151849 | 0.6335 | 73.35% | 67.35% | 0.8300 | 0.7517 | 0.4861 | ★ New Best |
| **03** | 186.6s | 305.5 MiB | 0.217377 | 0.138977 | 0.6254 | 73.73% | 67.35% | 0.8525 | 0.7890 | 0.4726 | Patience 1/7 |
| **04** | 645.4s | 305.5 MiB | 0.207002 | 0.116175 | 0.7034 | 77.57% | 73.47% | 0.8738 | 0.8214 | 0.4188 | ★ New Best |
| **05** | 188.6s | 305.5 MiB | 0.177321 | 0.109243 | 0.7110 | 78.80% | 74.32% | 0.8812 | 0.8386 | 0.4013 | ★ New Best |
| **06** | 3955.5s | 426.9 MiB | 0.193717 | 0.118123 | 0.7095 | 78.85% | 76.36% | 0.8839 | 0.8360 | 0.3925 | Unfreeze layer3 |
| **07** | 169.7s | 426.9 MiB | 0.172464 | 0.114048 | 0.7179 | 80.22% | 76.36% | 0.8869 | 0.8349 | 0.3911 | ★ New Best |
| **08** | 173.8s | 426.9 MiB | 0.162014 | 0.101105 | 0.7430 | 79.91% | 77.89% | 0.8972 | 0.8504 | 0.3578 | ★ New Best |
| **09** | 1102.0s | 426.9 MiB | 0.155029 | 0.099479 | 0.7337 | 80.74% | 77.55% | 0.8996 | 0.8533 | 0.3570 | Patience 1/7 |
| **10** | 4538.8s | 426.9 MiB | 0.144355 | 0.086044 | 0.7771 | 80.16% | 77.89% | 0.9038 | 0.8584 | 0.3454 | ★ New Best |
| **11** | 352.2s | 426.9 MiB | 0.140823 | 0.109822 | 0.7269 | 79.89% | 76.70% | 0.8971 | 0.8551 | 0.3623 | Patience 1/7 |
| **12** | 433.1s | 426.9 MiB | 0.136609 | 0.098764 | 0.7446 | 81.91% | 76.87% | 0.8954 | 0.8588 | 0.3779 | Patience 2/7 |
| **13** | 519.7s | 426.9 MiB | 0.119580 | 0.107174 | 0.7394 | 81.43% | 78.40% | 0.8955 | 0.8596 | 0.3600 | Patience 3/7 |
| **14** | 423.0s | 426.9 MiB | 0.126832 | 0.095598 | 0.7799 | 81.88% | 80.61% | 0.9045 | 0.8631 | 0.3312 | ★ New Best |
| **15** | 352.3s | 426.9 MiB | 0.107240 | 0.094947 | 0.7586 | 81.85% | 78.74% | 0.9030 | 0.8681 | 0.3530 | Patience 1/7 |
| **16** | 664.6s | 426.9 MiB | 0.110770 | 0.089538 | 0.7936 | 82.23% | 80.78% | 0.9111 | 0.8703 | 0.3238 | ★ New Best |
| **17** | 747.0s | 426.9 MiB | 0.106204 | 0.110547 | 0.7188 | 81.39% | 76.70% | 0.8966 | 0.8573 | 0.3824 | Patience 1/7 |
| **18** | **839.4s** | **426.9 MiB** | **0.108757** | **0.090384** | **`0.8103`** | **`82.90%`** | **`81.80%`** | **`0.9114`** | **`0.8700`** | **`0.3168`** | ★ **OVERALL BEST** |
| **19** | 841.1s | 426.9 MiB | 0.113175 | 0.092698 | 0.7905 | 81.58% | 80.61% | 0.9134 | 0.8733 | 0.3148 | Patience 1/7 |
| **20** | 610.3s | 426.9 MiB | 0.104347 | 0.095259 | 0.7994 | 81.69% | 81.12% | 0.9137 | 0.8743 | 0.3127 | Patience 2/7 |
| **21** | 456.1s | 426.9 MiB | 0.115651 | 0.092019 | 0.7705 | 82.65% | 79.76% | 0.9083 | 0.8705 | 0.3477 | Patience 3/7 |
| **22** | 381.4s | 426.9 MiB | 0.106108 | 0.088225 | 0.7958 | 81.96% | 80.61% | 0.9131 | 0.8734 | 0.3204 | Patience 4/7 |
| **23** | 220.3s | 426.9 MiB | 0.093769 | 0.089826 | 0.7850 | 82.63% | 80.27% | 0.9111 | 0.8733 | 0.3318 | Patience 5/7 |
| **24** | 216.4s | 426.9 MiB | 0.107928 | 0.088048 | 0.7932 | 82.69% | 80.78% | 0.9147 | 0.8763 | 0.3164 | Patience 6/7 |
| **25** | 278.3s | 426.9 MiB | 0.098529 | 0.104224 | 0.7524 | 81.81% | 78.74% | 0.9051 | 0.8679 | 0.3494 | Early Stopping (7/7) |

---

## 5. Detailed Evaluation of the Final Model (Epoch 18)

### 5.1 Per-Class Performance Metrics

| Class Index | Target Diagnostic Entity | Support | Prevalence | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score | ROC-AUC (OvR) | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `NORMAL_BONE_RADIOGRAPH` | 319 | 54.25% | **90.91%** | 75.46% | 81.46% | **0.8593** | 0.8935 | 0.8771 |
| **1** | `BENIGN_BONE_TUMOR` | 228 | 38.78% | **67.54%** | 91.94% | 84.15% | **0.7494** | 0.8538 | 0.8382 |
| **2** | `PRIMARY_MALIGNANT_TUMOR` | 41 | 6.97% | **90.24%** | **97.81%** | **75.51%** | **0.8222** | **0.9868** | **0.8949** |
| **Macro** | **Unweighted Diagnostic Average** | **588** | — | **82.90%** | **88.40%** | **80.37%** | **0.8103** | **0.9114** | **0.8700** |

### 5.2 Confusion Matrix Analysis

$$\begin{pmatrix}
\text{Actual Normal (319)} & \mathbf{290} & 26 & 3 \\
\text{Actual Benign (228)} & 65 & \mathbf{154} & 9 \\
\text{Actual Malignant (41)} & 1 & 3 & \mathbf{37}
\end{pmatrix}$$

#### Critical Oncological Breakdown
1. **Malignant Detection (Class 2):**
   - Correctly identified: **37 / 41 (90.24%)**.
   - Misclassified as benign: **3 cases** (7.32%).
   - Misclassified as normal: **Only 1 case** (2.44%).
   - The combined miss rate for primary bone malignancy is under 10%, a critical safety attribute for differential AI systems.
2. **Normal Control Identification (Class 0):**
   - Correctly identified: **290 / 319 (90.91%)**.
   - Misclassified as benign: 26 cases (8.15%).
   - Misclassified as malignant: 3 cases (0.94%).
3. **Benign Differentiation (Class 1):**
   - Correctly identified: **154 / 228 (67.54%)**.
   - Misclassified as normal: 65 cases (28.51%).
   - Misclassified as malignant: 9 cases (3.95%).
   - The benign class represents the most challenging boundary due to subtle radiological overlap between indolent benign lesions (e.g. enchondromas, non-ossifying fibromas) and normal cortical variations.

---

## 6. Generated Visual Artifacts

All 4 high-resolution diagnostic figures were rendered and persisted in `ml_artifacts/btxrd_full_training/`:

1. **`confusion_matrix.png`**:
   - Dual-panel heatmap displaying both raw image count confusion and row-normalized recall percentages. Highlights the 90.2% detection rate on primary bone malignancies.
2. **`roc_curves_ovr.png`**:
   - One-vs-Rest ROC curves with macro-average. Malignant curve demonstrates near-perfect discriminative capacity ($\text{AUC} = 0.9868$), with Normal ($\text{AUC} = 0.8935$) and Benign ($\text{AUC} = 0.8538$).
3. **`precision_recall_curves.png`**:
   - Precision-Recall curves plotted against class prevalence baselines. Malignant PR-AUC reaches $0.8949$ against a $6.97\%$ prevalence baseline.
4. **`loss_curves.png`**:
   - Training and validation Focal loss progression, Macro-F1 trajectory, and per-class sensitivity evolution over all 25 epochs. Highlights the unfreezing event at Epoch 6.

---

## 7. Deliverables & Checkpoint Verification

All 12 required full training deliverables are confirmed present and populated in `ml_artifacts/btxrd_full_training/`:

| Artifact Name | Size | Purpose |
| :--- | :---: | :--- |
| `best_checkpoint.pt` | 244.85 MB | Predeclared Best Model weights (Epoch 18, Macro-F1: 0.8103) |
| `last_checkpoint.pt` | 244.85 MB | Terminal checkpoint weights (Epoch 25, Early Stopping) |
| `training_config.yaml` | 1.04 KB | Full training hyperparameters and optimizer parameters |
| `training_log.csv` | 3.40 KB | Complete 25-epoch per-epoch training and validation log |
| `validation_metrics.json` | 15.10 KB | Comprehensive metrics, classification report, and confusion matrices |
| `validation_predictions.csv` | 53.82 KB | Per-image probabilities and predictions for all 588 validation images |
| `class_distribution.json` | 1.18 KB | Center 1 training and validation cohort class counts |
| `training_summary.json` | 2.30 KB | Concise executive summary of training dynamics |
| `confusion_matrix.png` | 111.22 KB | Dual-panel raw and normalized confusion matrix visualization |
| `roc_curves_ovr.png` | 215.55 KB | One-vs-Rest multiclass ROC curves |
| `precision_recall_curves.png` | 201.57 KB | One-vs-Rest Precision-Recall curves |
| `loss_curves.png` | 255.88 KB | 4-panel training dynamics and convergence curves |

---

## 8. Automated Test Suite Results

The dedicated test suite `ml_service/tests/test_stage_17_btxrd_full_training.py` was executed via pytest in the GPU environment (`%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe`):

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\archi\OneDrive\Desktop\osteophorosis-main
collected 15 items

test_01_three_class_label_mapping                         PASSED [  6%]
test_02_dataloader_integrity_and_sample_counts            PASSED [ 13%]
test_03_aspect_ratio_letterboxing_and_padding             PASSED [ 20%]
test_04_label_consistency                                 PASSED [ 26%]
test_05_focal_loss_numerical_stability                    PASSED [ 33%]
test_06_center_1_class_weights_calculation                PASSED [ 40%]
test_07_model_output_shape_and_progressive_unfreezing     PASSED [ 46%]
test_08_cuda_forward_backward_pass                        PASSED [ 53%]
test_09_checkpoint_reload_and_prediction_reproducibility  PASSED [ 60%]
test_10_artifact_manifest_completeness                    PASSED [ 66%]
test_11_zero_external_center_access                       PASSED [ 73%]
test_12_conflict_quarantine_integrity                     PASSED [ 80%]
test_13_fracatlas_artifacts_preservation                  PASSED [ 86%]
test_14_bone_scintigraphy_artifacts_preservation          PASSED [ 93%]
test_15_stage_16_pilot_and_frontend_preservation          PASSED [100%]

======================= 15 passed, 1 warning in 13.02s ========================
```

---

## 9. Scientific Limitations & Regulatory Disclaimers

1. **Research Prototype Notice:**
   - The model and diagnostic scores produced by OsteoGuard AI are investigational algorithms developed strictly for research benchmarking. They are not approved by the FDA, CE, or CDSCO as Software as a Medical Device (SaMD) and must not be used for standalone patient diagnosis or treatment selection.
2. **Patient Independence Limitation within Center 1:**
   - As documented during Stage 06.1 and Stage 15 forensic audits, BTXRD does not provide ground-truth patient-level or study-level accession identifiers. While Center 1 was split randomly under stratified sampling, individual patients with multiple projections could theoretically appear across both training and validation splits. True patient-level independence is evaluated only on the external multi-center cohorts (Center 2 and Center 3).
3. **Primary Sarcoma vs. Secondary Skeletal Metastasis Distinction:**
   - Class 2 represents **primary bone tumours** (predominantly osteosarcoma and chondrosarcoma imaged via plain radiography). This is fundamentally distinct from the **secondary bone metastases** classified via whole-body Technetium-99m bone scintigraphy in Stages 11–14. These modalities and biological conditions are strictly segregated within OsteoGuard AI.

---

## 10. Stage Boundary & Hard Gate Enforcement

In compliance with project governance and safety protocols:
- **STAGE 17 IS COMPLETE AND PASSED.**
- **STRICT HARD GATE:** External validation cohorts (Center 2 and Center 3) remain **strictly sealed**.
- **DO NOT** begin Stage 18.
- **DO NOT** evaluate Center 2 or Center 3 holdout test sets without explicit user authorization.
