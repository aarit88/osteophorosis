# STAGE 16 REPORT: BTXRD THREE-CLASS RESNET-34 PILOT TRAINING

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 16 — BTXRD Three-Class Primary Bone Tumour Pilot Training  
**Status:** **PASS**  
**Execution Timestamp:** 2026-11-15T23:38:00+05:30  
**Host Environment:** Windows 11 Home (x64)  
**Hardware Accelerator:** NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM, Compute Capability 8.9)  
**Isolated Python Environment:** `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe` (Python 3.13.3, PyTorch 2.14.1+cu126)  
**System CPU Python:** `C:\Python313\python.exe` (PyTorch 2.9.0+cpu — untouched)  
**React Frontend:** Untouched  

---

## 1. Executive Summary

Stage 16 established the training and validation pipeline for automated three-class differential diagnosis of plain radiograph bone tumours using the BTXRD dataset. The experiment strictly implements Protocol B (Center 1 Development Cohort), holding Center 2 and Center 3 completely sealed as independent multi-center external validation benchmarks.

Key achievements in Stage 16:
1. **Aspect-Ratio Letterboxing & Radiographic Normalization:** High-resolution X-rays of varying aspect ratios (from $200 \times 300$ up to $2000 \times 2000$) were symmetrically padded with black borders (`(0, 0, 0)`) to $512 \times 512$, preserving trabecular and cortical geometry without anamorphic stretching. Grayscale single-channel radiographs were broadcast to 3 channels and standardized via ImageNet statistics.
2. **Transfer Learning with Controlled Freezing:** ResNet-34 initialized with official ImageNet-1K pretrained weights (`resnet34-b627a593.pth`) was deployed. Early feature extraction layers (`conv1`, `bn1`, `layer1`, `layer2`, `layer3` — 8,170,304 parameters) were frozen, while `layer4` and a custom head (`Dropout(0.4) -> Linear(512, 3)` — 13,115,907 parameters, 61.62%) remained trainable. Frozen BatchNorm layers were locked in `eval()` mode.
3. **Class Imbalance Mitigation via Normalized Focal Loss:** Multi-Class Focal Loss ($\gamma = 2.0$) with inverse-frequency $\alpha$ weights computed strictly from Center 1 training frequencies ($\alpha = [0.3333, 0.4817, 2.1850]$) counterbalanced the severe primary sarcoma minority (8.27% prevalence).
4. **Controlled 3-Epoch Pilot Execution:** Training completed across 3 epochs using batch size 2, gradient accumulation 8 (effective batch size 16), AdamW with differential learning rates, and FP16 mixed precision. Peak VRAM utilization was exceptionally light at **305.55 MiB** (7.46% of 4 GB).
5. **Discriminative Validation Performance:**
   - **Macro-F1:** **0.6335** (Best Checkpoint: Epoch 2)
   - **Balanced Accuracy:** **73.35%** (Overall Accuracy: 67.35%)
   - **Macro ROC-AUC (OvR):** **0.8302** | **Macro PR-AUC:** **0.7526**
   - **Malignant Detection Sensitivity:** **87.80%** (36/41 malignant tumors detected with only 1 false-normal prediction)
   - **Malignant ROC-AUC (OvR):** **0.9100** | **Malignant PR-AUC:** **0.6926**
6. **Verification & Audit Suite:** All 12 automated unit and integration tests passed in `ml_service/tests/test_stage_16_btxrd_pilot.py` (execution time: 16.72s).

---

## 2. Dataset Cohorts & Boundary Controls

### 2.1 Center 1 Development Cohort Distribution
Under Protocol B, the model is developed and validated solely on Center 1 radiographs. All 4 duplicate images with conflicting diagnostic labels (`IMG000591.jpeg`, `IMG003456.jpeg`, `IMG000707.jpeg`, `IMG003346.jpeg`) were quarantined and excluded.

| Class Index | Target Class | Center 1 Train ($N=2,346$) | Train Prev (%) | Center 1 Val ($N=588$) | Val Prev (%) | Normalized Focal $\alpha$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | `NORMAL_BONE_RADIOGRAPH` | 1,272 | 54.22% | 319 | 54.25% | **0.3333** |
| **1** | `BENIGN_BONE_TUMOR` | 880 | 37.51% | 228 | 38.78% | **0.4817** |
| **2** | `PRIMARY_MALIGNANT_TUMOR` | 194 | 8.27% | 41 | 6.97% | **2.1850** |
| **Total** | — | **2,346** | 100.00% | **588** | 100.00% | **3.0000** |

### 2.2 External Multi-Center Holdouts (Strictly Sealed)
- **Center 2 External Holdout:** 549 images (259 normal, 214 benign, 76 malignant) — **UNTOUCHED**.
- **Center 3 External Holdout:** 259 images (27 normal, 201 benign, 31 malignant) — **UNTOUCHED**.
- *Verification:* Automated test `test_09_zero_external_center_access` verified that zero images from Center 2 or Center 3 appeared in training or validation predictions.

---

## 3. Preprocessing & Architectural Design

### 3.1 Aspect-Ratio Preserving Letterbox Preprocessing
Unlike standard square resizing which warps bone trabeculae and osteolytic margins, the pipeline implements letterboxing:
$$\text{scale} = \frac{512}{\max(W, H)}, \quad W' = \text{round}(W \cdot \text{scale}), \quad H' = \text{round}(H \cdot \text{scale})$$
The image is bilinearly resized to $(W', H')$ and symmetrically centered on a $512 \times 512$ black canvas (`pad_x = (512 - W') // 2`, `pad_y = (512 - H') // 2`). For grayscale plain radiographs, the single luminance channel is broadcast across 3 RGB channels prior to ImageNet channel normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 3.2 Pretrained Backbone & Layer Freezing
- **Backbone:** TorchVision ResNet-34 (`IMAGENET1K_V1`, SHA-256: `b627a593e62057d197626359550302b1f0c2a868f000b0d367fa12bf13c8ffad`).
- **Freezing Scheme:**
  - Frozen Layers: `conv1`, `bn1`, `layer1`, `layer2`, `layer3` (8,170,304 frozen parameters, 38.38%).
  - Trainable Layers: `layer4` + `classifier` (13,115,907 trainable parameters, 61.62%).
  - Total Parameters: 21,286,211.
- **BatchNorm Locking:** Overridden `train()` method guarantees that frozen batch normalization layers remain locked in `eval()` mode during training, preventing disruption of running mean and variance from mini-batches of size 2.

### 3.3 Multi-Class Focal Loss Formulation
To penalize hard minority examples (specifically osteosarcomas and chondrosarcomas in Class 2):
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $\gamma = 2.0$, and $\alpha = [0.3333, 0.4817, 2.1850]$ acts as an inverse-prevalence multiplier. For numerical stability, $p_t$ is evaluated directly via `F.log_softmax(logits, dim=1)`.

---

## 4. Pilot Training Execution & Learning Dynamics

### 4.1 Training Configuration
- **Hardware:** NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM)
- **Batch Size:** 2 per forward pass
- **Gradient Accumulation:** 8 steps (effective batch size: 16)
- **Optimizer:** AdamW (`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`)
  - Head LR: $1.0 \times 10^{-4}$
  - Layer4 LR: $1.0 \times 10^{-5}$
- **Precision:** FP16 Automatic Mixed Precision (`torch.cuda.amp.GradScaler`)
- **Peak VRAM:** 305.55 MiB

### 4.2 Epoch Progression Log

| Epoch | Duration | Peak VRAM | Train Loss | Val Loss | Macro-F1 | Balanced Acc | Overall Acc | Macro ROC | Macro PR | Brier Score | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 268.48s | 305.55 MiB | 0.300391 | 0.177803 | 0.5645 | 67.25% | 60.88% | 0.7811 | 0.6809 | 0.5516 | — |
| **02** | 397.98s | 305.55 MiB | 0.238187 | 0.151777 | **0.6335** | **73.35%** | **67.35%** | **0.8302** | **0.7526** | **0.4858** | ★ **BEST** |
| **03** | 322.45s | 305.55 MiB | 0.217271 | 0.138819 | 0.6254 | 73.73% | 67.35% | 0.8527 | 0.7893 | 0.4724 | — |

---

## 5. Detailed Validation Evaluation (Best Model: Epoch 2)

### 5.1 Per-Class Performance Breakdown

| Class Index | Diagnostic Category | Support ($N$) | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score | ROC-AUC (OvR) | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `NORMAL_BONE_RADIOGRAPH` | 319 | 64.26% | 84.39% | 83.00% | 0.7244 | 0.8096 | 0.8219 |
| **1** | `BENIGN_BONE_TUMOR` | 228 | 67.98% | 75.56% | 63.79% | 0.6582 | 0.7711 | 0.7433 |
| **2** | `PRIMARY_MALIGNANT_TUMOR` | 41 | **87.80%** | **88.67%** | 36.73% | **0.5180** | **0.9100** | **0.6926** |
| **Macro** | **Macro Average / Total** | **588** | **73.35%** | **82.87%** | **61.17%** | **0.6335** | **0.8302** | **0.7526** |

### 5.2 Confusion Matrix Analysis ($N=588$)

```
                        Predicted Normal   Predicted Benign   Predicted Malignant    Total
Actual Normal (0)             205                 84                  30              319
Actual Benign (1)              41                155                  32              228
Actual Malignant (2)            1                  4                  36               41
-----------------------------------------------------------------------------------------
Total Predicted               247                243                  98              588
```

#### Clinical & Diagnostic Interpretation:
1. **Exceptional Sarcoma Sensitivity (87.80%):** Out of 41 primary malignant bone tumours, 36 were correctly identified as malignant. Crucially, only 1 malignant lesion was misclassified as normal (a false-normal rate of only 2.44%), and 4 were flagged as benign (where biopsy/referral is still standard clinical practice).
2. **Conservative Malignant Margin (PPV 36.73%):** Because Focal Loss $\alpha_2 = 2.1850$ aggressively penalizes missed malignant tumours, the model errs on the side of caution: 30 normal and 32 benign cases were flagged as suspect malignant. In oncological triage, a low false-negative rate is prioritized over specificity.
3. **Strong Benign vs Normal Discrimination:** Benign tumours achieved 67.98% sensitivity and 75.56% specificity with an F1 score of 0.6582.

---

## 6. Generated Deliverables in `ml_artifacts/btxrd_pilot/`

All 12 required pilot artifacts were successfully generated and validated:

| File Name | File Size | Description |
| :--- | :---: | :--- |
| `pilot_config.yaml` | 719 B | Hyperparameters, optimizer settings, and layer freezing definitions |
| `class_distribution.json` | 848 B | Center 1 class counts, percentages, and normalized Focal Loss weights |
| `pretrained_weights_info.json` | 762 B | TorchVision ResNet-34 ImageNet-1K pretrained weights SHA-256 hash |
| `training_log.csv` | 571 B | Epoch-by-epoch loss, macro metrics, and per-class sensitivities |
| `validation_metrics.json` | 3,087 B | Full 3-class confusion matrix, OvR ROC-AUC, PR-AUC, and Brier score |
| `validation_predictions.csv` | 53,582 B | All 588 validation image predictions with raw class probabilities |
| `best_checkpoint.pt` | 190.2 MB | Best model state dict, optimizer state, and validation metrics (Epoch 2) |
| `last_checkpoint.pt` | 190.2 MB | Final epoch model checkpoint (Epoch 3) |
| `confusion_matrix.png` | 113.5 KB | Seaborn 3-class normalized and count confusion matrix heatmap |
| `loss_curves.png` | 144.4 KB | Training vs. validation Focal Loss and Macro-F1 curves |
| `roc_curves_ovr.png` | 216.8 KB | One-vs-Rest ROC curves with per-class and macro AUC |
| `precision_recall_curves.png` | 222.6 KB | Per-class Precision-Recall curves with prevalence baselines |

---

## 7. Verification Test Suite Results

The automated test suite in `ml_service/tests/test_stage_16_btxrd_pilot.py` was executed with pytest using the isolated GPU Python environment:

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\archi\OneDrive\Desktop\osteophorosis-main
collected 12 items

ml_service/tests/test_stage_16_btxrd_pilot.py::test_01_three_class_label_mapping PASSED [  8%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_02_aspect_ratio_letterboxing_and_padding PASSED [ 16%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_03_grayscale_to_rgb_broadcast PASSED [ 25%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_04_center_1_class_weights_and_focal_loss PASSED [ 33%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_05_resnet34_architecture_and_parameter_counts PASSED [ 41%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_06_cuda_forward_backward_pass PASSED [ 50%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_07_pilot_artifact_manifest_completeness PASSED [ 58%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_08_checkpoint_reload_and_prediction_reproducibility PASSED [ 66%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_09_zero_external_center_access PASSED [ 75%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_10_conflict_quarantine_integrity PASSED [ 83%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_11_fracatlas_and_bone_scintigraphy_preservation PASSED [ 91%]
ml_service/tests/test_stage_16_btxrd_pilot.py::test_12_cpu_python_and_frontend_preservation PASSED [100%]

======================= 12 passed, 1 warning in 16.72s ========================
```

---

## 8. Clinical & Scientific Disclaimers

1. **Research Prototype Only:** This pilot model is strictly an experimental research prototype for radiographic differential diagnostic exploration. It is **NOT** FDA-cleared, CE-marked, or clinically validated for patient diagnosis, staging, or surgical treatment planning.
2. **Independence Scope:** As audited in Stage 15, the raw BTXRD dataset does not provide patient, visit, or study identifiers. Consequently, internal splits cannot guarantee patient-level independence. Generalizability will be established via multi-center external validation on Center 2 and Center 3.
3. **Oncology Scope Distinction:** The primary malignant class (`PRIMARY_MALIGNANT_TUMOR`, e.g. Osteosarcoma, Chondrosarcoma) is an independent biological and clinical entity from secondary bone metastasis. It is modeled strictly as a 3-class primary bone lesion classifier and is never conflated with bone scintigraphy metastasis detection.

---

## 9. Stage Gate Decision

- **Stage 16 Status:** **PASS**
- **Readiness for Stage 17:** All pipeline primitives, GPU training routines, Focal Loss stabilizers, artifact pipelines, and test suites are verified and ready.
- **Protocol Boundary Notice:** In accordance with the Stage 16 mandate, training has ceased. No full training (Stage 17) or multi-center holdout evaluation (Center 2 / Center 3) has been initiated.

