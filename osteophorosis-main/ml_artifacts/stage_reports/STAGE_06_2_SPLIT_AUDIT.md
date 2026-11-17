# OsteoGuard AI — Stage 06.2 Split Audit and Data Leakage Prevention Report

**Date:** 2026-11-15  
**Stage:** STAGE 06.2 — DATASET PARTITIONING AND DATA LEAKAGE PREVENTION  
**Status:** **PASS**  
**Author:** Senior Machine Learning Architect & Lead Python Engineer  
**Architectural Baseline:** ResNet-101 / Swin Transformer, U-Net Segmentation, IBSI Radiomics, Late Feature Fusion, Multi-Class Focal Loss  

---

## 1. Executive Summary & Objective

In **Stage 06.2**, we developed and executed the rigorous dataset partitioning and data leakage prevention protocol for the three physical medical imaging cohorts acquired in Stage 06.1:
1. **Bone Scan Metastasis Dataset** (IICS-UNA Paraguay, 291 adult breast cancer patients, 582 planar scintigrams).
2. **FracAtlas Dataset** (4,083 musculoskeletal radiographs, 719 fractures with COCO segmentation masks, 3,364 candidate controls).
3. **BTXRD Bone Tumor Dataset** (3,746 plain radiographs across 3 hospital centers, 1,867 tumors, 1,879 non-tumor controls).

### Strict Medical Imaging Principle:
These three datasets address distinct clinical and methodological tasks. They **MUST NEVER** be concatenated into a naive four-class disease classifier:
- Planar nuclear scintigraphy ($^{99\text{m}}\text{Tc}$-MDP) cannot be mixed with plain radiography.
- Negative staging bone scans (`NO_BONE_METASTASIS`) from oncology patients are **not** healthy skeletal controls.
- Primary bone sarcomas (Osteosarcoma) are distinct pathological entities and must **never** map to secondary carcinoma metastasis.

Stage 06.2 establishes isolated, reproducible, leakage-free partitions with deterministic pseudorandom seeds (`seed=42`).

---

## 2. Task 1: Bone Scan Metastasis Dataset Partitioning

### 2.1 Cohort Description & Grouping
- **Source:** IICS-UNA Paraguay (`Bone Scan Images Dataset.xlsx`).
- **Total Patients:** 291 adult patients (289 female, 2 male; mean age 52.8 years).
- **Total Images:** 582 PNG images (exactly 2 paired whole-body scans per patient: Anterior `A*.png` and Posterior `P*.png`).
- **Disease Prevalence:** 60 metastasis-positive patients (20.62%), 231 metastasis-negative patients (79.38%).
- **Negative Class Semantics:** Staged breast cancer patients without detectable osseous metastases. Target class: `NO_BONE_METASTASIS` (strictly excluded from `HEALTHY_CONTROL`).

### 2.2 Patient-Level Stratified Partitioning (70% / 15% / 15%)
Grouping was enforced strictly at the patient level (`patient_id`). Both Anterior and Posterior scans for every patient were co-located into identical splits and folds. Zero patient leakage and zero image hash overlap exist across partitions.

| Partition | Patients (Mets + NoMets) | Images (Ant + Post) | Class Balance (Mets / NoMets) | Prevalence |
| :--- | :--- | :--- | :--- | :--- |
| **Train (70%)** | 203 patients (42 + 161) | 406 images | 84 Mets / 322 NoMets | 20.69% |
| **Val (15%)** | 44 patients (9 + 35) | 88 images | 18 Mets / 70 NoMets | 20.45% |
| **Test (15%)** | 44 patients (9 + 35) | 88 images | 18 Mets / 70 NoMets | 20.45% |
| **Total** | **291 patients** | **582 images** | **120 Mets / 462 NoMets** | **20.62%** |

### 2.3 Supplementary 5-Fold Cross-Validation Protocol
Because the holdout test set contains only 9 metastasis-positive patients ($N_{\text{mets}} = 9$), a single false negative causes an **11.11% drop** in empirical sensitivity. This high sample variance necessitates a variance-reduced experimental protocol:
- Supplementary stratified 5-fold patient cross-validation (`fold_0` through `fold_4`).
- Exactly **12 metastasis patients** (24 images) assigned to each fold.
- 46 or 47 non-metastasis patients (92-94 images) assigned to each fold.
- **Methodological Recommendation:** Use the 5-fold cross-validation scheme for backbone comparison, loss weighting tuning, and threshold selection; reserve the holdout test partition strictly for locked final confirmation.

**Manifest File:** `ml_service/data/splits/bone_scan_split_manifest.csv` (582 rows, 9 columns).

---

## 3. Task 2: FracAtlas Partitioning & Duplicate Content Resolution

### 3.1 Cohort Description & Authoritative Resolution
- **Source:** FracAtlas (`dataset.csv`).
- **Archive Contents:** 4,085 image files on disk; 4,083 valid records in `dataset.csv`.
- **Duplicate Filename Resolution:** `IMG0003375.jpg` and `IMG0003376.jpg` were duplicated in both `Fractured/` and `Non_fractured/`. `dataset.csv` authoritatively confirms both images are fractured (`fractured=1, fracture_count=1`). The pipeline mapped both images to `Fractured/` and omitted the duplicate non-fractured copies.

### 3.2 Duplicate Content Discovery & Group-Aware Partitioning
Forensic SHA-256 analysis identified **12 duplicate content groups** (26 images total) with identical visual pixel content across different image IDs:
- 3 fractured groups (7 images: `IMG0000261`/`2299`/`2300`, `IMG0001763`/`2615`, `IMG0002298`/`2379`).
- 9 candidate control groups (19 images).
- All 12 duplicate groups exhibited 100% internal label concordance (0 conflicts).

To prevent test-set memorization and optimistic performance estimates, partitioning was grouped by SHA-256 hash:
- **Total Unique Hashes:** 4,068 unique content digests across 4,083 images.

| Partition | Unique Hashes | Total Images | Fractured Images | Control Images |
| :--- | :--- | :--- | :--- | :--- |
| **Train (70%)** | 2,842 | 2,857 | 502 (17.57%) | 2,355 (82.43%) |
| **Val (15%)** | 613 | 612 | 107 (17.48%) | 505 (82.52%) |
| **Test (15%)** | 613 | 614 | 110 (17.92%) | 504 (82.08%) |
| **Total** | **4,068** | **4,083** | **719 (17.61%)** | **3,364 (82.39%)** |

### 3.3 Limitations on Patient Independence
FracAtlas does not publish patient, study, or accession identifiers. While image-content deduplication guarantees zero duplicate image leakage, multiple views (AP, Lateral, Oblique) of a single patient could conceivably fall into different splits. This represents an inherent limitation of the public benchmark that is now transparently documented.

### 3.4 Official Segmentation Split (U-Net Baseline)
FracAtlas provides an official 574 / 82 / 63 split for the 719 fractured images:
- **Train Split:** 574 images (738 fracture polygons).
- **Val Split:** 82 images (108 fracture polygons).
- **Test Split:** 63 images (78 fracture polygons).
- **Deduplication Audit:** All 3 duplicate fractured groups are contained strictly within the training set. Zero cross-split leakage exists in the segmentation partition.

**Manifest Files:**
- `ml_service/data/splits/fracatlas_classification_split.csv` (4,083 rows, 8 columns).
- `ml_service/data/splits/fracatlas_segmentation_split.csv` (719 rows, 6 columns).

---

## 4. Task 3: BTXRD (Bone Tumor X-Ray Dataset) Partitioning

### 4.1 Cohort Description & Pathology Distribution
- **Source:** BTXRD (`dataset.xlsx`).
- **Total Images:** 3,746 plain radiographs.
- **Pathologies:** 1,867 tumors (1,525 benign, 342 primary malignant) + 1,879 non-tumor normal radiographs.
- **Center Distribution:** Center 1: 2,938 (78.43%), Center 2: 549 (14.66%), Center 3: 259 (6.91%).

### 4.2 Forensic Conflict Discovery & Quarantine
SHA-256 deduplication identified 21 duplicate hash groups. Crucially, **2 duplicate groups exhibited conflicting ground-truth labels** in the original dataset metadata:
1. `IMG000591.jpeg` (Benign Osteochondroma) vs `IMG003456.jpeg` (Normal Bone Radiograph) [SHA-256: `9922a655a6...`].
2. `IMG000707.jpeg` (Benign Osteochondroma) vs `IMG003346.jpeg` (Normal Bone Radiograph) [SHA-256: `2a821efbeb...`].

Both pairs (4 images total) have been explicitly flagged with `conflict_flag: TRUE` in the split manifest to ensure they are excluded or quarantined during clinical evaluation.

### 4.3 Dual Partitioning Protocols

#### Protocol A: Duplicate-Grouped Stratified Split (70% / 15% / 15%)
All duplicate hashes were grouped to guarantee identical partition assignment.
- **Train:** 2,622 images (1,068 benign, 239 malignant, 1,315 normal).
- **Val:** 562 images (229 benign, 51 malignant, 282 normal).
- **Test:** 562 images (228 benign, 52 malignant, 282 normal).

#### Protocol B: Multi-Center Holdout Split (External Generalization Benchmark)
Evaluating hospital-level distribution shift:
- **Center 1 (Internal Cohort, 2,938 images):** Random 80/20 train/val split (2,350 train / 588 val).
- **Center 2 (External Holdout Test, 549 images):** Multi-class evaluation benchmark containing 259 normal controls, 214 benign tumors, and 76 primary malignant tumors (13.84% malignant prevalence).
- **Center 3 (External Holdout Test, 259 images):** Benign-only generalization test cohort (259 benign osteochondromas).

### 4.4 Strict Oncology Governance
Primary bone malignancies (Osteosarcoma, Other MT) are strictly labeled `PRIMARY_MALIGNANT_TUMOR` and **never** mapped to secondary bone metastasis. Non-tumor normal radiographs (1,879 images) provide authentic negative controls.

**Manifest File:** `ml_service/data/splits/btxrd_split_manifest.csv` (3,746 rows, 11 columns).

---

## 5. Verification & Test Execution Results

Automated unit tests covering all data integrity, stratification, co-location, and leakage constraints were executed:

### ML Pytest Suite:
```powershell
python -m pytest ml_service/tests tests_ml -v
```
**Results:** **85 passed in 29.54s (100% PASS)**
- `test_stage_06_2_splits.py::test_bone_scan_split_integrity`: **PASSED** (0 patient leakage, paired views co-located, 5-fold balance).
- `test_stage_06_2_splits.py::test_fracatlas_classification_split`: **PASSED** (4,083 rows, 0 duplicate hash leakage, duplicate filename resolution).
- `test_stage_06_2_splits.py::test_fracatlas_segmentation_split`: **PASSED** (719 rows, official split verified, COCO polygon mapping verified).
- `test_stage_06_2_splits.py::test_btxrd_split_integrity`: **PASSED** (3,746 rows, 0 hash leakage across splits, 2 conflict groups flagged, center holdouts isolated).

### Frontend Vitest Suite:
```powershell
npm test --prefix osteophorosis-main -- --run --pool=threads
```
**Results:** **3 test files passed, 6/6 tests passed in 21.39s (100% PASS)**

---

## 6. Manifest File Registry Summary

| Manifest File | Rows | Target Pathology / Modality | Partition Strategy |
| :--- | :--- | :--- | :--- |
| `ml_service/data/splits/bone_scan_split_manifest.csv` | 582 | Secondary Bone Metastasis (Scintigraphy) | Patient-level 70/15/15 + 5-Fold Cross-Validation |
| `ml_service/data/splits/fracatlas_classification_split.csv` | 4,083 | Acute Bone Fracture (X-ray) | Duplicate-hash grouped 70/15/15 |
| `ml_service/data/splits/fracatlas_segmentation_split.csv` | 719 | Fracture Polygon Segmentation (X-ray) | Official 574 train / 82 val / 63 test |
| `ml_service/data/splits/btxrd_split_manifest.csv` | 3,746 | Primary Bone Tumors & Normal Controls (X-ray) | Duplicate-grouped 70/15/15 + Center 1/2/3 Holdout |

`ml_service/data/dataset_registry.yaml` has been updated with individual verified readiness statuses (`split_readiness`). The master four-class pipeline remains correctly locked (`dataset_ready: false`) pending acquisition of Paget's disease and Osteogenesis Imperfecta cohorts.

---

## 7. Next Stage Readiness

With verified, leak-free dataset partitions firmly established for all acquired cohorts, the project is fully primed for **Stage 07: Feature Extraction & Backbone Development** (ResNet-101 semantic backbone, U-Net skeletal segmentation, and IBSI-aligned radiomic feature extraction).

