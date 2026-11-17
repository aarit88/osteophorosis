# STAGE 11A — BONE SCAN METASTASIS DATASET READINESS AUDIT

**Project:** OsteoGuard AI  
**Stage:** 11A — Bone Scan Scintigraphy Metastasis Dataset Readiness & Experimental Design Audit  
**Audit Timestamp:** 2026-11-15T19:35:00+05:30  
**Readiness Status:** **PASS (READY FOR STAGE 11 EXPERIMENTATION)**  
**Target Modality:** Planar Whole-Body Radionuclide Bone Scintigraphy ($^{99\text{m}}\text{Tc}$-MDP Gamma Camera)  
**Target Disease Category:** Secondary Osseous Metastasis in Adult Breast Cancer Patients  

---

## 1. Executive Summary

Stage 11A establishes a separate, scientifically rigorous dataset readiness audit for the secondary bone metastasis classification task within the OsteoGuard AI framework. 

In strict adherence to clinical AI governance rules:
- **Experiment Independence:** This bone scintigraphy dataset is evaluated as an isolated diagnostic pipeline. It is **never** merged with plain radiographic fracture data (FracAtlas) or tumor data (BTXRD).
- **Accurate Label Semantics:** Metastasis-negative scans represent adult breast cancer patients undergoing metastatic staging who showed no detectable osseous metastases. They are **not** healthy skeletal controls.
- **Disease Boundary:** This cohort represents metastatic breast carcinoma; it does **not** represent Paget's disease of bone or Osteogenesis Imperfecta (OI).
- **Patient-Level Isolation:** All partitioning, cross-validation schemes, and evaluation metrics are enforced strictly at the patient level (`patient_id`), ensuring that paired Anterior and Posterior scans are never split across folds.
- **FracAtlas Immutability:** Previous FracAtlas checkpoints (`ml_artifacts/fracatlas_full_training/best_checkpoint.pt`) and evaluation artifacts remain untouched and verified.

The dataset, manifests, paired views, and 16-bit image properties were forensically verified with **6/6 automated pytest tests passing (100%)**. The dataset is certified **READY (PASS)** for Stage 11 modeling.

---

## 2. Dataset Provenance and Clinical Context

| Dimension | Specification |
| :--- | :--- |
| **Originating Institution** | Instituto de Investigaciones en Ciencias de la Salud, Universidad Nacional de Asunción (IICS-UNA), San Lorenzo, Paraguay |
| **Clinical Target Population** | Adult breast cancer patients undergoing skeletal metastatic staging |
| **Cohort Demographics** | 291 adult patients: 289 females (99.31%), 2 males (0.69%) |
| **Age Distribution** | Range: 25 to 93 years; Mean: 54.7 years; Median: 55.0 years |
| **Imaging Modality** | Planar Whole-Body Radionuclide Bone Scintigraphy |
| **Radiopharmaceutical** | Technetium-99m labeled methylene diphosphonate ($^{99\text{m}}\text{Tc}$-MDP) |
| **Physical Views per Patient** | Exactly two full-body planar acquisitions: Anterior (`A*.png`) and Posterior (`P*.png`) |
| **Physical Files on Disk** | 582 PNG images located in `ml_service/data/raw/bone_scan_metastasis/` |
| **Primary Clinical Source** | `Bone Scan Images Dataset.xlsx` (11 metadata columns, 291 patient rows) |

---

## 3. Class Distribution and Clinical Label Semantics

| Diagnostic Label | Clinical Definition | Patients | Images | Prevalence |
| :--- | :--- | :---: | :---: | :---: |
| **`BONE_METASTASIS` (Class 1)** | Confirmed osseous metastatic involvement (focal osteoblastic/osteolytic radiotracer uptake outside normal physiological distribution) | **60** | **120** | **20.62%** |
| **`NO_BONE_METASTASIS` (Class 0)** | Staged breast cancer patients with physiological radiotracer distribution and no detectable skeletal metastasis | **231** | **462** | **79.38%** |
| **Cohort Total** | **Full Scintigraphy Staging Cohort** | **291** | **582** | **100.00%** |

### Critical Clinical Distinction:
- **Negative Label ($\ne$ Healthy):** Negative bone scans reflect patients with confirmed primary breast malignancies whose bone scintigram demonstrated physiological tracer clearance (calvarium, renal excretion, bladder) without metastatic lesions. These patients may have systemic chemotherapy toxicity, osteopenia, or prior surgical interventions; they must **never** be labeled as healthy skeletal controls.
- **Class Imbalance:** The positive class prevalence is $\pi = 20.62\%$. The class imbalance ratio is $231 / 60 \approx 3.85:1$.

---

## 4. Patient-Level Split Verification & Leakage Audit

The patient-level split manifest established in Stage 06.2 (`ml_service/data/splits/bone_scan_split_manifest.csv`) was audited:

### 4.1 Partition Distribution (70% / 15% / 15%)

| Partition | Patients (Mets / No-Mets) | Images (Ant + Post) | Class Balance (Mets / No-Mets) | Prevalence ($\pi$) |
| :--- | :---: | :---: | :---: | :---: |
| **Train (70%)** | **203** (42 / 161) | **406** | 84 Mets / 322 No-Mets | 20.69% |
| **Validation (15%)** | **44** (9 / 35) | **88** | 18 Mets / 70 No-Mets | 20.45% |
| **Holdout Test (15%)** | **44** (9 / 35) | **88** | 18 Mets / 70 No-Mets | 20.45% |
| **Cohort Total** | **291** (60 / 231) | **582** | **120 Mets / 462 No-Mets** | **20.62%** |

### 4.2 Leakage & Co-Location Audit Findings:
- **Patient ID Overlap:**
  - Train $\cap$ Validation: **0 patients**
  - Train $\cap$ Test: **0 patients**
  - Validation $\cap$ Test: **0 patients**
- **Cryptographic SHA-256 Hash Overlap:**
  - Train $\cap$ Validation: **0 hashes**
  - Train $\cap$ Test: **0 hashes**
  - Validation $\cap$ Test: **0 hashes**
- **Paired View Integrity:** 100% of 291 patients have both `anterior` and `posterior` views. Zero patients are unpaired.
- **Strict Co-Location:** For every patient, both Anterior and Posterior scans reside within the exact same split partition and the exact same cross-validation fold (`test_bone_scan_zero_leakage_and_colocation` PASSED).

---

## 5. Supplementary 5-Fold Stratified Cross-Validation Scheme

Because the 15% validation and test sets each contain only **9 metastasis-positive patients** ($N_{\text{pos}} = 9$), a single false negative causes an **11.11% drop in sensitivity**. To protect against small-sample variance during hyperparameter tuning and model selection, a supplementary stratified 5-fold cross-validation scheme is encoded into the manifest:

| Cross-Validation Fold | Total Patients | Positive Patients | Negative Patients | Total Images | Positive Images | Negative Images |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fold_0`** | 59 | **12** | 47 | 118 | 24 | 94 |
| **`fold_1`** | 58 | **12** | 46 | 116 | 24 | 92 |
| **`fold_2`** | 58 | **12** | 46 | 116 | 24 | 92 |
| **`fold_3`** | 58 | **12** | 46 | 116 | 24 | 92 |
| **`fold_4`** | 58 | **12** | 46 | 116 | 24 | 92 |
| **Total** | **291** | **60** | **231** | **582** | **120** | **462** |

*Verification:* Exactly 12 metastasis-positive patients (24 images) are assigned to each fold (`test_bone_scan_5fold_stratification` PASSED).

---

## 6. Image Quality and Technical Properties Audit

Forensic inspection of all 582 image files revealed technical properties unique to planar nuclear medicine scintigraphy:

```
Unique Image Dimensions:  (Width: 256, Height: 1024) [Aspect Ratio 1:4]
Image Mode / Bit Depth:   'I;16' (16-bit single-channel unsigned integer)
Global Pixel Min:         0.0 (Pure black background across 100% of cohort)
Global Pixel Max:         65535.0 (Saturated integer max)
Average Cohort Mean:      2135.19 counts
Average Cohort Std:       4734.47 counts
Background Intensity:     Corner samples = 0.0 (100% zero background)
Corrupted / Missing Files: 0 (100% decode integrity verified)
```

### Percentile Intensity Profile:
Analysis across representative images demonstrated the physiological count distribution of $^{99\text{m}}\text{Tc}$-MDP:
- **50th Percentile ($p_{50}$):** 0 counts (more than 50% of the $256 \times 1024$ image area is zero-count background surrounding the body).
- **90th Percentile ($p_{90}$):** 2,000 to 7,000 counts (physiological skeletal uptake: spine, ribs, pelvis).
- **95th Percentile ($p_{95}$):** 3,700 to 10,000 counts (high bone turnover: sacroiliac joints, sternum, joints).
- **99th Percentile ($p_{99}$):** 7,700 to 20,000 counts (hot focal spots, including active osteoblastic metastases).
- **99.9th Percentile & Max:** 15,000 to 65,535 counts (extreme tracer accumulation in the urinary bladder, renal pelvis, or injection site artifact).

---

## 7. Nuclear Medicine Preprocessing Recommendations

> [!WARNING]
> **Do NOT blindly reuse the FracAtlas plain radiograph preprocessing pipeline.**
> 1. FracAtlas resizes square X-rays to $224 \times 224$. Forcing a $256 \times 1024$ scintigram into $224 \times 224$ introduces severe **4:1 anamorphic aspect ratio distortion**, squashing the human skeleton vertically.
> 2. FracAtlas normalizes 8-bit $[0, 255]$ images with ImageNet mean/std. Dividing 16-bit counts ($0–65,535$) naively by 65,535 compresses diagnostic skeletal counts ($1,000–15,000$) into an ultra-narrow $[0.015, 0.23]$ dynamic range, destroying lesion contrast against noise.

### Recommended Scintigraphy Preprocessing Strategy:
1. **Aspect-Preserving Spatial Standardization:**
   - Standardize images to **$(W=256, H=1024)$** or $(W=128, H=512)$, preserving the natural $1:4$ full-body vertical aspect ratio.
2. **Robust Percentile Intensity Normalization:**
   - Apply non-linear windowing or percentile clipping at $[0, p_{99.5}]$ (or fixed ceiling of $25,000$ counts) to discard saturated bladder/injection artifacts while mapping skeletal counts to $[0.0, 1.0]$.
   - Alternatively, apply logarithmic scaling: $I_{\text{norm}} = \frac{\ln(1 + I)}{\ln(1 + I_{\text{clip}})}$.
3. **Channel Mapping for Transfer Learning:**
   - Replicate the single 16-bit normalized channel into 3 identical channels ($1 \to 3$) to allow standard pretrained convolutional backbones (ResNet, EfficientNet) to extract multi-scale spatial features without architectural modifications.
4. **Conservative Scintigraphy Augmentation:**
   - Allowed: Vertical translation ($\pm 5\%$), subtle scaling ($0.95–1.05$), small rotation ($\pm 5^\circ$).
   - **Caution on Horizontal Flipping:** In planar scintigraphy, flipping an anterior scan maps Left to Right, which mimics a posterior scan. If horizontal flipping is used, anatomical orientation metadata must be preserved.

---

## 8. Patient-Level Evaluation Protocol Design

In nuclear oncology, patient care decisions are made per-patient, not per-individual image. A patient with metastasis on *either* the anterior or posterior view is diagnosed with skeletal metastasis.

### 8.1 Dual-View Prediction Aggregation Strategy:
During inference, the model processes both paired views for patient $i$:
$$p_{\text{ant}, i} = P(\text{metastasis} \mid \text{Anterior}_i)$$
$$p_{\text{post}, i} = P(\text{metastasis} \mid \text{Posterior}_i)$$

Two primary aggregation methods must be compared:
1. **Clinical Max-Pooling (Recommended Primary):**
   $$P_{\text{patient}, i} = \max\left(p_{\text{ant}, i}, p_{\text{post}, i}\right)$$
   *Clinical Rationale:* Osseous metastases often present asymmetrically (e.g., dorsal vertebral body metastasis visible primarily on posterior view; sternal metastasis visible on anterior view). If *either* view is positive, the patient is metastasis-positive.
2. **Softmax Mean-Pooling (Secondary Comparator):**
   $$P_{\text{patient}, i} = \frac{p_{\text{ant}, i} + p_{\text{post}, i}}{2}$$

### 8.2 Primary Evaluation Metrics:
All primary metrics must be reported at the **patient level** ($N=44$ for holdout test, $N=291$ across 5-fold CV):
- Patient-level ROC-AUC
- Patient-level PR-AUC
- Patient-level Sensitivity (Recall)
- Patient-level Specificity
- Patient-level Precision (PPV)
- Patient-level Negative Predictive Value (NPV)
- Patient-level Balanced Accuracy
- Patient-level Confusion Matrix

---

## 9. Training Feasibility & Risk Analysis

| Risk Dimension | Risk Level | Mitigation Strategy |
| :--- | :---: | :--- |
| **Small Cohort Size** ($N=203$ train patients) | **HIGH** | Use compact architectures (ResNet-18, ResNet-34) with heavy backbone freezing. Avoid unconstrained training of 100M+ parameter backbones. |
| **Class Imbalance** ($20.6\%$ positive) | **MEDIUM** | Employ Focal Loss ($\gamma=2.0, \alpha=[0.25, 0.75]$) or weighted cross-entropy ($w_{\text{pos}} = 3.85$) with prior-matching logit bias $b_1 = \ln(42/161) \approx -1.3436$. |
| **Evaluation Variance** (9 test positive patients) | **HIGH** | Use 5-fold cross-validation ($N=12$ positive patients per fold) for model selection, architecture search, and hyperparameter tuning; reserve holdout test strictly for locked final verification. |
| **Spatial Aspect Ratio** ($1:4$) | **MEDIUM** | Standardize to $(256, 1024)$ or $(128, 512)$ using stride/pooling adjustments rather than square distortion. |
| **GPU Memory Footprint** | **LOW** | $256 \times 1024$ single-channel tensors with micro-batch size 2 require $<700\text{ MiB}$ VRAM, well within the 4,094 MiB RTX 3050 A VRAM budget. |

---

## 10. Automated Test Results

The test suite executed with **6/6 tests passing (100% pass rate)** in 1.38s:

```
pytest ml_service/tests/test_stage_11a_bone_scan_readiness.py -v
======================== 6 passed, 1 warning in 1.38s =========================
```

- `test_bone_scan_dataset_identity_and_metadata`: **PASSED** (291 patients in Excel: 60 pos, 231 neg; 289 female, 2 male; 582 PNGs on disk).
- `test_bone_scan_manifest_exact_counts`: **PASSED** (Train: 203 pts / 406 imgs; Val: 44 pts / 88 imgs; Test: 44 pts / 88 imgs).
- `test_bone_scan_zero_leakage_and_colocation`: **PASSED** (0 patient leakage, 0 hash leakage, paired views co-located).
- `test_bone_scan_16bit_image_properties`: **PASSED** (All images are 16-bit `I;16` with $(256, 1024)$ resolution).
- `test_bone_scan_5fold_stratification`: **PASSED** (Exactly 12 positive patients in each of the 5 folds).
- `test_fracatlas_artifacts_strictly_unaltered`: **PASSED** (Stage 09 ResNet-101 checkpoint verified intact).

---

## 11. Final Readiness Sign-Off

- **Stage Status:** **PASS**
- **Metastasis Dataset Certified Ready:** **YES**
- **Unified 4-Class Pipeline:** Remains locked (`dataset_ready: false`) pending acquisition of Paget's disease and Osteogenesis Imperfecta cohorts.
- **Next Stage:** STAGE 11B (Awaiting user directive — training has **NOT** been started).
