# STAGE 15 REPORT: BTXRD BONE TUMOUR DATASET READINESS AND FORENSIC AUDIT

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 15 — BTXRD Bone Tumour Dataset Forensic Audit, Split Integrity, and Training Readiness  
**Status:** **PASS**  
**Execution Timestamp:** 2026-11-15T23:05:00+05:30  
**Lead AI Architect / Medical ML Engineer:** Antigravity AI (Google DeepMind Team)  
**Host Environment:** Windows 11 Home (x64)  
**Compute Accelerator:** NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM, Compute Capability 8.9)  
**Isolated GPU Environment:** `%LOCALAPPDATA%\venvs\osteoguard-gpu\Scripts\python.exe` (Python 3.13.3, PyTorch 2.14.1+cu126)  
**System CPU Python:** `C:\Python313\python.exe` (PyTorch 2.9.0+cpu — untouched)  
**Audited Cohort:** BTXRD (Bone Tumor X-Ray Dataset, 3,746 plain radiographs across 3 hospital centers)  

---

## 1. Executive Summary

In Stage 15, a comprehensive forensic audit, metadata cross-validation, annotation schema verification, and split integrity audit was conducted for the **BTXRD (Bone Tumor X-Ray Dataset)** archive located in `ml_service/data/raw/btxrd/`. This experiment establishes a completely independent, scientifically valid three-class primary bone tumour classification pipeline:
- **Class 0:** Normal Bone Radiograph (Non-Tumour Control)
- **Class 1:** Benign Primary Bone Tumour
- **Class 2:** Primary Malignant Bone Tumour (Primary Bone Sarcoma)

### Key Audit Findings & Core Governance Principles:
1. **Physical File & Image Audit:** Exactly **3,746 image files** exist on disk (3,719 `.jpeg` and 27 `.jpg` files). All 3,746 images open cleanly with zero corruption. Two format anomalies were identified (`IMG001373.jpeg` encoded as BMP and `IMG001560.jpeg` encoded as MPO), both handled seamlessly by standard PIL decoders.
2. **Strict Oncology Governance (Rule 2.1 Enforcement):**
   - **Primary malignant bone tumours (e.g., Osteosarcoma) must NEVER be mapped to secondary bone metastasis.** Osteosarcomas are primary mesenchymal bone neoplasms occurring predominantly in children and young adults; secondary bone metastases are hematogenous carcinoma disseminations occurring predominantly in older oncology patients.
   - **Negative Class Authenticity:** Unlike the oncology staging cohort in the bone scan dataset (where negative scans represent breast cancer patients undergoing staging), BTXRD Class 0 images are authentic non-tumour skeletal radiographs.
3. **Absence of Patient & Study Identifiers:** Comprehensive inspection of `dataset.xlsx`, JPEG EXIF tags, and LabelMe JSON files confirms that **no patient IDs, subject IDs, or accession numbers exist**. Consequently, **patient-independent generalization cannot be formally established on the internal split**. True distribution-shift generalization must be evaluated using external hospital centers (Center 2 and Center 3).
4. **Duplicate Content & Quarantine of Conflicting Records:**
   - 21 duplicate cryptographic SHA-256 hash groups (42 images total) were audited.
   - 19 duplicate groups (38 images) exhibit 100% internal label concordance.
   - **2 duplicate groups (4 images total) exhibit ground-truth label conflicts** between Benign Osteochondroma and Normal Bone:
     - Group 1: `IMG000591.jpeg` (Benign) vs `IMG003456.jpeg` (Normal) [Hash: `9922a655a63d9916...`]
     - Group 2: `IMG000707.jpeg` (Benign) vs `IMG003346.jpeg` (Normal) [Hash: `2a821efbeb008dad...`]
   - **Action Taken:** Both conflicting groups are explicitly tracked with `conflict_flag = 'TRUE'` in `btxrd_split_manifest.csv` and are **strictly quarantined (excluded)** from model training and evaluation. Zero silent relabeling was performed.
5. **Split Integrity (Zero Leakage):**
   - **Protocol A (Duplicate-Grouped Stratified Split, 70/15/15):** Exactly 0 duplicate hashes cross split boundaries between Train (2,618 clean), Val (561 clean), and Test (563 clean).
   - **Protocol B (Multi-Center Holdout Split):** Center 1 (2,934 clean images) is strictly restricted to train/val. Center 2 (549 clean images, multi-class) and Center 3 (259 clean images, predominantly benign) are completely isolated as independent external test cohorts. Zero hash leakage crosses center boundaries.
6. **Annotation Schema Audit:** Exactly **1,867 LabelMe JSON files** exist in `Annotations/`, mapping with 100% coverage to the 1,867 tumour images. Annotations contain 2,324 bounding box rectangles and 2,318 lesion boundary polygons (99.95% concordance with Excel pathology metadata). Normal images have zero annotations.
7. **Readiness Determination:** **PASS**. The dataset, manifests, and annotation schemas are fully verified, enabling the commencement of Stage 16 pipeline development without retraining or leakage.

---

## 2. Dataset Provenance and Physical File Inventory

The BTXRD archive was extracted into `ml_service/data/raw/btxrd/` in Stage 06.1 with path-traversal protection. The physical on-disk inventory was verified against `dataset.xlsx`:

| Component | Path on Disk | File Count / Size | Format / Schema | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Radiograph Images** | `ml_service/data/raw/btxrd/images/` | 3,746 files | 3,719 `.jpeg`, 27 `.jpg` (8-bit) | **VERIFIED (0 corrupt)** |
| **Tumour Annotations** | `ml_service/data/raw/btxrd/Annotations/` | 1,867 files | LabelMe JSON (Rectangles & Polygons) | **VERIFIED (0 corrupt)** |
| **Curator Metadata** | `ml_service/data/raw/btxrd/dataset.xlsx` | 553,439 bytes | Excel Workbook (Sheet1: 3,746 rows) | **VERIFIED** |
| **Split Manifest** | `ml_service/data/splits/btxrd_split_manifest.csv` | 3,746 rows | Master Split Manifest (11 columns) | **VERIFIED** |
| **Master Manifest** | `ml_service/data/dataset_manifest.csv` | 8,411 rows total | Global Registry (3,746 BTXRD rows) | **VERIFIED** |

### Extension & Compression Findings:
- Images `IMG000001.jpeg` through `IMG003719.jpeg` are suffixed `.jpeg`.
- Images `IMG003720.jpg` through `IMG003746.jpg` (27 images) are suffixed `.jpg`. Both `dataset.xlsx` and disk filenames reflect this naming.
- `IMG001373.jpeg` is internally encoded as Windows BMP format with a `.jpeg` extension.
- `IMG001560.jpeg` is internally encoded as Multi-Picture Object (MPO) format.
- PIL decodes 100% of the 3,746 images cleanly without decompression errors.

---

## 3. Verified Class Counts and Pathological Distribution

### 3.1 Three-Class Category Breakdown

The dataset partitions into three mutually exclusive diagnostic targets:

| Diagnostic Target Class | Target Class Enum | Raw Total (N=3,746) | Quarantined Conflicts | Clean Effective Dataset (N=3,742) | Clean Prevalence | Unique Content Hashes (N=3,723) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Bone Radiograph** | `NORMAL_BONE_RADIOGRAPH` | 1,879 | 2 | **1,877** | **50.16%** | 1,866 (50.12%) |
| **Benign Primary Tumour** | `BENIGN_BONE_TUMOR` | 1,525 | 2 | **1,523** | **40.70%** | 1,516 (40.72%) |
| **Primary Malignant Tumour** | `PRIMARY_MALIGNANT_TUMOR` | 342 | 0 | **342** | **9.14%** | 341 (9.16%) |
| **Total Cohort** | — | **3,746** | **4** | **3,742** | **100.0%** | **3,723** |

### 3.2 Specific Histopathological Sub-Pathology Breakdown (Clean N = 3,742)

The 1,865 clean bone tumour cases encompass diverse primary mesenchymal and chondrogenic neoplasms:

#### Benign Bone Tumour Subtypes (1,523 Clean Images | 40.70% of Cohort):
- **Osteochondroma (Solitary Exostosis):** 752 images (20.10%) — Most common benign skeletal lesion.
- **Multiple Osteochondromas (Hereditary Multiple Exostoses):** 263 images (7.03%) — Multi-focal cartilage-capped bony projections.
- **Simple Bone Cyst (Unicameral Bone Cyst):** 206 images (5.51%) — Fluid-filled expansile pseudotumour.
- **Giant Cell Tumor of Bone (GCTB):** 93 images (2.49%) — Epiphyseal/metaphyseal osteolytic neoplasm with locally aggressive potential.
- **Synovial Osteochondroma:** 51 images (1.36%) — Intra-articular chondroid proliferation.
- **Osteofibroma (Ossifying Fibroma / Fibrous Dysplasia):** 44 images (1.18%) — Benign fibro-osseous lesion.
- **Other Benign Tumours (Chondroblastoma, Enchondroma, Osteoma):** 115 images (3.07%).

#### Primary Malignant Bone Tumour Subtypes (342 Clean Images | 9.14% of Cohort):
- **Osteosarcoma (High-Grade Osteogenic Sarcoma):** 297 images (7.94%) — Predominant primary malignant bone sarcoma characterized by osteoid matrix production, periosteal Codman's triangles, and "sunburst" calcification.
- **Other Primary Malignant Tumours (Chondrosarcoma, Ewing Sarcoma):** 45 images (1.20%).

---

## 4. Duplicate Image Analysis and Quarantined Conflict Resolution

Cryptographic SHA-256 analysis of all 3,746 raw images revealed **21 duplicate hash groups** comprising **42 images** (exactly 2 images per group):

### 4.1 Concordant Duplicate Groups (19 Groups / 38 Images)
- 19 duplicate groups contain images with identical ground-truth clinical labels in `dataset.xlsx`.
- In Protocol A, all duplicate hashes were assigned to the same partition (e.g., both copies in train, or both in test), ensuring **zero memorization leakage**.

### 4.2 Conflicting Duplicate Groups (2 Groups / 4 Images) — QUARANTINED
Two duplicate hash groups exhibited contradictory ground-truth clinical diagnoses in the original metadata:

| Conflict Group | Duplicate Image A | Duplicate Image B | Center | Label A (Metadata) | Label B (Metadata) | SHA-256 Hash Digest | Quarantine Action |
| :---: | :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| **Group 1** | `IMG000591.jpeg` | `IMG003456.jpeg` | Center 1 | `BENIGN_BONE_TUMOR` (Osteochondroma) | `NORMAL_BONE_RADIOGRAPH` (Non-Tumor) | `9922a655a63d9916...` | **QUARANTINED** (`conflict_flag: TRUE`) |
| **Group 2** | `IMG000707.jpeg` | `IMG003346.jpeg` | Center 1 | `BENIGN_BONE_TUMOR` (Osteochondroma) | `NORMAL_BONE_RADIOGRAPH` (Non-Tumor) | `2a821efbeb008dad...` | **QUARANTINED** (`conflict_flag: TRUE`) |

#### Resolution Policy:
1. In accordance with clinical AI auditing standards, **no arbitrary or silent relabeling was performed**.
2. Both conflicting pairs are explicitly flagged with `conflict_flag: TRUE` in `btxrd_split_manifest.csv`.
3. In both Protocol A and Protocol B, the paired conflicting images were co-located in identical partitions (Group 1: `train` in Stratified Split, `val` in Center Split; Group 2: `train` in Stratified Split, `train` in Center Split).
4. **Mandatory Exclusion:** All 4 images must be excluded during model training, hyperparameter validation, and final test evaluation.

---

## 5. Dataset Partitioning Integrity Audit

Two distinct experimental evaluation protocols were established in Stage 06.2 (`ml_service/data/splits/btxrd_split_manifest.csv`):

### 5.1 Protocol A: Duplicate-Grouped Stratified Split (Internal Benchmark)
All duplicate hashes are strictly co-located within the same split partition. Class proportions remain rigorously balanced across splits:

| Partition Split | Total Images (Raw) | Quarantined Conflicts | Clean Sample Size | Normal Bone (Class 0) | Benign Tumour (Class 1) | Malignant Tumour (Class 2) | Malignant Prevalence | Duplicate Hash Leakage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (70%)** | 2,622 | 4 | **2,618** | 1,314 (50.19%) | 1,065 (40.68%) | 239 (9.13%) | 9.13% | **0 hashes** |
| **Validation (15%)** | 561 | 0 | **561** | 280 (49.91%) | 230 (41.00%) | 51 (9.09%) | 9.09% | **0 hashes** |
| **Holdout Test (15%)** | 563 | 0 | **563** | 283 (50.27%) | 228 (40.50%) | 52 (9.24%) | 9.24% | **0 hashes** |
| **Total Clean** | **3,746** | **4** | **3,742** | **1,877 (50.16%)** | **1,523 (40.70%)** | **342 (9.14%)** | **9.14%** | **0 hashes** |

### 5.2 Protocol B: Multi-Center Generalization Benchmark (External Generalization)
Evaluates real-world clinical hospital distribution shift by training on Center 1 and evaluating on independent external medical centers:

| Center Partition | Center Role | Clean Sample Size | Normal Bone (Class 0) | Benign Tumour (Class 1) | Malignant Tumour (Class 2) | Malignant Prevalence | Generalization Capability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Center 1 (Train)** | Internal Training Split | **2,346** | 1,272 (54.22%) | 880 (37.51%) | 194 (8.27%) | 8.27% | Baseline model training |
| **Center 1 (Val)** | Internal Validation Split | **588** | 319 (54.25%) | 228 (38.78%) | 41 (6.97%) | 6.97% | Model selection & early stopping |
| **Center 2 (External Holdout)** | Multi-Class External Test | **549** | 259 (47.18%) | 214 (38.98%) | 76 (13.84%) | 13.84% | **Balanced three-class external evaluation** |
| **Center 3 (External Holdout)** | Predominantly Benign Test | **259** | 27 (10.42%) | 201 (77.61%) | 31 (11.97%) | 11.97% | **High-benign specificity / stress test** |

#### Crucial Multi-Center Audit Finding:
- **Center 2** exhibits a representative, high-quality balance across all three classes (259 Normal, 214 Benign, 76 Malignant) with higher image resolution ($1274 \times 1600$). It represents the **gold standard external holdout test benchmark**.
- **Center 3** was previously mischaracterized as "benign-only" in early documentation. Our forensic audit confirms that **Center 3 actually contains all three classes**, but is heavily skewed toward benign osteochondromas (77.61% benign, 10.42% normal, 11.97% malignant). Due to having only 27 normal controls, Center 3 cannot serve as a standalone representative three-class performance benchmark, but serves as an effective specificity stress-test cohort.

---

## 6. Patient / Study Identification Availability Audit

A comprehensive forensic audit of all available metadata was conducted to assess whether patient-level grouping exists:

1. **`dataset.xlsx` Column Audit:**
   - Evaluated all 37 columns: `image_id, center, age, gender, hand, ulna, radius, humerus, foot, tibia, fibula, femur, hip bone, ankle-joint, knee-joint, hip-joint, wrist-joint, elbow-joint, shoulder-joint, tumor, benign, malignant, osteochondroma, multiple osteochondromas, simple bone cyst, giant cell tumor, osteofibroma, synovial osteochondroma, other bt, osteosarcoma, other mt, upper limb, lower limb, pelvis, frontal, lateral, oblique`.
   - **Result:** No `patient_id`, `subject_id`, `study_id`, `accession_number`, or medical record number (MRN) columns exist.
2. **DICOM / EXIF Headers:**
   - 100% of EXIF tags are stripped across all JPEG/JPG files.
3. **Filename Structure:**
   - Files are sequentially enumerated: `IMG000001.jpeg` through `IMG003746.jpeg`.
4. **LabelMe JSON Files:**
   - Contain only `imagePath`, dimensions, and shapes.

### Scientific Governance Finding:
> [!WARNING]
> **PATIENT INDEPENDENCE CANNOT BE ESTABLISHED ON PROTOCOL A**  
> Because the BTXRD public release does not include patient or accession identifiers, it is impossible to guarantee that multiple views (e.g., AP and Lateral projections) or longitudinal exams of a single patient are not distributed across the internal train, validation, and test splits.  
> 
> **Methodological Mandate:**
> - The project **must never claim patient-independent generalization** based solely on SHA-256 image-hash deduplication.
> - Rigorous, uncontaminated generalizability must be demonstrated through **Protocol B (Multi-Center Generalization)**, where entire hospital institutions (Center 2 and Center 3) are held out.

---

## 7. LabelMe Annotation Schema Audit

The `Annotations/` directory contains exactly **1,867 LabelMe JSON files** (one per tumour radiograph). A complete schema inspection revealed:
- **Corrupted JSON Files:** 0.
- **Empty Shape Files:** 0 (100% contain valid annotations).
- **Total Geometric Shapes:** **4,642 shapes**.
  - `rectangle`: 2,324 shapes (Axis-aligned bounding boxes).
  - `polygon`: 2,318 shapes (Closed boundary lesion contour polygons).
- **Lesion Density:**
  - 1,627 images (87.1%) contain exactly 2 shapes (1 bounding box + 1 polygon), indicating single focal tumours.
  - 136 images (7.3%) contain 4 shapes (2 distinct tumours, typical of multiple osteochondromas).
  - 104 images (5.6%) contain $\ge 6$ shapes (extensive multi-focal exostoses, up to 19 shapes).
- **Pathological Label Concordance:**
  - Shape labels directly match Excel pathology classifications: `osteochondroma` (1,612), `multiple osteochondromas` (1,288), `osteosarcoma` (596), `simple bone cyst` (412), `other bt` (252), `giant cell tumor` (188), `synovial osteochondroma` (110), `osteofibroma` (92), `other mt` (92).
  - Label concordance between JSON shape labels and `dataset.xlsx` is **99.95%** (1,866 of 1,867 cases). The sole divergence (`IMG001276.jpeg`) represents a subtype refinement (`synovial osteochondroma` vs `osteochondroma`).
- **Clinical Suitability:** The annotations are of high scientific quality and suitable for:
  - Weakly-supervised lesion localization.
  - Class-activation map (Grad-CAM) IoU alignment auditing.
  - Object detection and boundary segmentation experiments.
  - *Distinction:* Class 0 normal radiographs have no annotations (absence of lesion).

---

## 8. Image Quality, Resolution, and Modality Audit

### 8.1 Image Geometry & Spatial Dimensions
Inspection of all 3,746 images revealed substantial anatomical diversity:
- **Width:** Min 153 px, Max 3,594 px, **Median 1,616 px**.
- **Height:** Min 311 px, Max 4,881 px, **Median 2,264 px**.
- **Aspect Ratio ($W/H$):** Min 0.218, Max 2.856, **Median 0.762** (approximately $3:4$ portrait).
  - Long bone views (tibia, fibula, femur, humerus) are narrow and elongated ($W/H \approx 0.25 - 0.45$).
  - Pelvis and bilateral joint radiographs are wide and landscape ($W/H \approx 1.2 - 2.5$).

### 8.2 Color Channels & Dynamic Range
- **Channels:** 2,895 images are single-channel grayscale (`mode: L`); 851 images are 3-channel (`mode: RGB`) where $R = G = B$.
- **Dynamic Range:** All images are 8-bit unsigned integers with values strictly bounded in $[0, 255]$.
- **Mean Pixel Intensity:** $70.8 \pm 25.8$ (Standard radiographic exposure).

### 8.3 Hospital Center Technical Differences

| Hospital Center | Image Count | Color Mode | Median Dimensions | Mean Intensity ($\mu \pm \sigma$) | Technical System Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Center 1** | 2,934 | 100% `L` (Grayscale) | $669 \times 815$ px | $72.21 \pm 24.40$ | Standard digitized computed radiography (CR) |
| **Center 2** | 549 | 27% `L`, 73% `RGB` | $1274 \times 1600$ px | $67.21 \pm 26.93$ | High-resolution modern digital radiography (DR) |
| **Center 3** | 259 | 100% `RGB` | $848 \times 955$ px | $71.51 \pm 27.67$ | Compressed DR exports with 3-channel broadcast |

### 8.4 Preprocessing Pipeline Recommendations
> [!IMPORTANT]
> **RADIOGRAPHIC PREPROCESSING SPECIFICATION**  
> 1. **Do NOT reuse the 16-bit scintigraphy windowing module:** Scintigraphy uses 16-bit counts with bladder clipping. BTXRD images are standard 8-bit radiographs.
> 2. **Do NOT stretch images into square dimensions:** Forcing a tall tibial radiograph ($1:4$ aspect ratio) into a square without padding causes severe geometric distortion of bone trabeculae and lesion margins.
> 3. **Preserve Aspect Ratio via Letterboxing:** Resize the longest image dimension to target resolution (e.g., 512 px) and pad the shorter dimension symmetrically with zero (black) padding to yield a uniform $512 \times 512$ square tensor.
> 4. **Channel Normalization:** Convert single-channel `L` images to 3 identical channels ($C=3$) to allow seamless initialization from ImageNet pre-trained backbones.

---

## 9. Class Imbalance and Training Feasibility Analysis

### 9.1 Imbalance Severity
In the clean dataset ($N = 3,742$), the three classes exhibit significant imbalance:
- Class 0 (Normal): 1,877 (50.16%)
- Class 1 (Benign): 1,523 (40.70%)
- Class 2 (Malignant): 342 (9.14%)

Primary malignant bone tumours (sarcomas) represent an extreme minority class (~$9.1\%$, an imbalance ratio of **$5.49 : 1$** relative to normal radiographs and **$4.45 : 1$** relative to benign tumours).

### 9.2 Training Feasibility & Failure Modes
Without deliberate methodological intervention, deep neural networks will optimize for the 90.9% majority classes (Normal + Benign), achieving high raw accuracy (~91%) while suffering catastrophic failure to detect primary malignant sarcomas.

#### Mandatory Architectural Countermeasures:
1. **Multi-Class Focal Loss:**
   $$\mathcal{L}_{\text{focal}} = -\sum_{c=0}^2 \alpha_c (1 - p_c)^\gamma \log(p_c)$$
   with focusing parameter $\gamma = 2.0$ and inverse-frequency class balancing weights:
   $$\alpha = [0.66, 0.82, 3.65]$$
2. **Stratified Batch Sampling:** Employing a weighted random sampler during training to guarantee that every mini-batch contains an adequate representation of malignant cases.
3. **Transfer Learning:** Fine-tuning from ImageNet pre-trained weights (ResNet-50 / ResNet-34) to provide robust low-level edge and texture representations.
4. **Hardware & VRAM Compatibility:**
   - Architecture: ResNet-50 (25.6M parameters) or ResNet-34 (21.8M parameters).
   - Input Resolution: $512 \times 512 \times 3$.
   - Mixed Precision: FP16 (`torch.cuda.amp.autocast`).
   - Batch Size: 16 (Train) / 32 (Inference).
   - Expected Peak VRAM: $\sim 1.8 - 2.2 \text{ GB}$ on the NVIDIA GeForce RTX 3050 A (4 GB VRAM capacity), leaving $> 45\%$ safety headroom.

---

## 10. Experimental Design and Evaluation Protocol

### 10.1 Diagnostic Task Formulation
A closed-world, three-class categorical classification task:
$$\hat{y} \in \{0, 1, 2\}$$
- **$y = 0$:** Normal Bone Radiograph
- **$y = 1$:** Benign Primary Bone Tumour
- **$y = 2$:** Primary Malignant Bone Tumour

### 10.2 Mandatory Evaluation Metrics
> [!CAUTION]
> **OVERALL ACCURACY MUST NEVER BE USED AS THE SOLE PERFORMANCE METRIC**  
> Due to the 9.1% malignant prevalence, a naive classifier predicting only normal or benign achieves >90% accuracy while failing 100% of cancer cases.

The evaluation protocol must compute and report:
1. **Macro-Averaged F1-Score:** Unweighted harmonic mean of precision and recall across all three classes (Primary Summary Metric).
2. **Balanced Accuracy:** Arithmetic mean of class-specific recalls:
   $$\text{BalAcc} = \frac{\text{Sens}_0 + \text{Sens}_1 + \text{Sens}_2}{3}$$
3. **Per-Class Clinical Diagnostic Metrics:**
   - Sensitivity (Recall) for Class 0, Class 1, and Class 2.
   - Specificity for Class 0, Class 1, and Class 2.
   - Precision (PPV) and Negative Predictive Value (NPV).
   - Class-specific F1-Score.
4. **Discriminative Capacity:**
   - One-vs-Rest (OvR) ROC-AUC (Macro-averaged and per-class).
   - Per-Class Precision-Recall AUC (PR-AUC) compared against empirical prevalence baselines ($0.5016, 0.4070, 0.0914$).
5. **Confusion Matrix:** Full $3 \times 3$ contingency table.
6. **Calibration:** Multi-class Brier Calibration Score.

---

## 11. Automated Test Verification Results

A dedicated automated test suite (`ml_service/tests/test_stage_15_btxrd_audit.py`) was implemented and executed:

```
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\archi\AppData\Local\venvs\osteoguard-gpu\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\archi\OneDrive\Desktop\osteophorosis-main
configfile: pyproject.toml
plugins: anyio-4.16.0, asyncio-1.5.0
collected 12 items

ml_service/tests/test_stage_15_btxrd_audit.py::test_01_raw_archive_and_file_counts PASSED          [  8%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_02_image_integrity_and_modes PASSED           [ 16%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_03_authoritative_class_counts PASSED           [ 25%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_04_annotation_concordance PASSED               [ 33%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_05_absence_of_patient_identifiers PASSED      [ 41%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_06_duplicate_hashes_and_quarantined_conflicts PASSED [ 50%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_07_protocol_a_stratified_split_integrity PASSED [ 58%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_08_protocol_b_multicenter_split_integrity PASSED [ 66%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_09_center_3_class_distribution PASSED          [ 75%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_10_target_class_semantics PASSED              [ 83%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_11_previous_stage_artifacts_unaltered PASSED   [ 91%]
ml_service/tests/test_stage_15_btxrd_audit.py::test_12_environment_and_frontend_integrity PASSED [100%]

======================== 12 passed, 1 warning in 5.87s ========================
```

All 12 integrity assertions passed without errors.

---

## 12. Explicit Regulatory & Clinical Governance Disclaimer

> [!CAUTION]
> **DEVELOPMENTAL RESEARCH CLASSIFICATION ONLY — NOT FOR CLINICAL DIAGNOSTIC USE**  
> The models, architectures, and evaluation protocols designed herein are academic research prototypes of the OsteoGuard AI framework. They have not been evaluated, cleared, or approved by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA), CDSCO, or any international medical regulatory agency.
> 
> **Explicit Scientific Limitations:**
> 1. **No Patient Identifiers:** Patient-level grouping cannot be established within Center 1. Internal performance metrics must be interpreted as image-level benchmarks. True generalization must rely on Center 2 external validation.
> 2. **2D Plain Radiography Constraints:** Planar radiography provides transmission superposition without 3D depth. Subtle primary bone tumours, intra-cortical lesions, or osteoid osteomas can be masked by overlying cortex and soft tissue. Definitive diagnosis requires cross-sectional CT, MRI, and biopsy histopathology.
> 3. **Non-Equivalence with Bone Metastasis:** Primary bone tumours are mesenchymal malignancies (osteosarcomas) and benign exostoses. They must never be conflated with osteoblastic carcinoma metastases evaluated in Stages 11–14.

---

## 13. Stage 15 Final Verdict

**FINAL STATUS: PASS**

The BTXRD bone tumour dataset, metadata manifests, annotation schemas, and partition boundaries have been thoroughly audited, verified, and established with zero cross-split leakage. Quarantined conflicting records are isolated, multi-center distributions are quantified, and class-imbalanced evaluation protocols are formalized.

*Per explicit project instructions, execution stops here. Model training for Stage 16 is NOT initiated.*
