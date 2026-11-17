# OsteoGuard AI — Dataset Quality Audit and Patient-Level Partitioning

## 1. Overview and Architectural Purpose

Medical machine learning models trained on radiologic data face catastrophic failure modes if patient data leaks between training and evaluation splits, or if diagnostic labels contain unverified or conflicting diagnoses. 

**OsteoGuard AI Stage 06** establishes an institutional-grade dataset audit, label verification, duplicate detection, and patient-level partitioning subsystem adhering strictly to **Rule 4** (*Never invent dataset labels or patient records*) and **Rule 5** (*Never use generated synthetic images as real training data*).

```
Raw Medical Imagery (CT / X-ray / DICOM / NIfTI)
                       │
                       ▼
         ┌───────────────────────────┐
         │       DatasetAuditor      │
         ├───────────────────────────┤
         │ • Diagnostic Taxonomy Map │
         │ • Salted De-identification│
         │ • Exact Hash (SHA-256)    │
         │ • Perceptual Hash (dHash) │
         │ • Conflicting Label Guard │
         │ • Quality & Mask Check    │
         └─────────────┬─────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │     Quality Audit Artifacts   │
       │ • dataset_quality_report.json │
       │ • class_distribution.csv      │
       │ • duplicate_audit.csv         │
       └───────────────┬───────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │    PatientLevelPartitioner    │
       ├───────────────────────────────┤
       │ • Patient-grouped Splits      │
       │ • 70% Train / 15% Val / 15% T │
       │ • External-site Holdout       │
       │ • Zero-leakage Assertion      │
       │ • Training-only Fit Guard     │
       └───────────────┬───────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │   split_manifest.csv    │
          └─────────────────────────┘
```

---

## 2. Target Diagnostic Taxonomy

All incoming imaging cohorts are mapped deterministically to the canonical 4-class differential diagnostic taxonomy:

| Target Class Code | Canonical Disease Entity | Clinical Modality | Diagnostic Window |
|---|---|---|---|
| `metastasis` | Secondary Bone Metastasis | CT / X-ray | Lytic, blastic, and mixed secondary osteolytic lesions |
| `paget` | Paget's Disease of Bone | CT / X-ray | Trabecular coarsening, cortical thickening, osteosclerosis |
| `oi` | Osteogenesis Imperfecta | CT / X-ray | Cortical thinning, osteopenia, gracile bone deformaties |
| `healthy` | Healthy Skeletal Control | CT / X-ray | Intact bone mineral density and normal trabecular microarchitecture |

---

## 3. Dataset Integrity & Quality Audit Engine (`auditor.py`)

The `DatasetAuditor` performs comprehensive static analysis of raw and processed clinical directories before training clearance:

### 3.1 Exact Duplicate Detection (SHA-256)
- Computes cryptographic SHA-256 digests across image byte streams.
- Identical files (e.g. copied slices across folders or identical downloads) are detected and flagged in `duplicate_audit.csv`.

### 3.2 Near-Duplicate Detection (Perceptual Difference Hashing - dHash)
- Medical imaging datasets frequently contain repeated consecutive CT slices or minor brightness-adjusted variants.
- The auditor computes a 64-bit difference hash (`dHash` via `imagehash`) by scaling images to $9 \times 8$ grayscale, calculating horizontal pixel gradients ($8 \times 8 = 64$ bits), and computing the bitwise Hamming distance:
  $$\text{HammingDistance}(h_1, h_2) = \sum_{i=0}^{63} (h_{1,i} \oplus h_{2,i})$$
- Pairs with $\text{HammingDistance} \le 3$ are flagged as `NEAR_PERCEPTUAL` duplicates for review.

### 3.3 Conflicting Label Detection
- Cross-references pseudonymous patient identifiers across diagnostic cohorts.
- If a patient ID is found in multiple disease classes (e.g. `PT_001` in both `metastasis` and `healthy`), the auditor raises a conflict blocker and marks the dataset as `NOT_READY`.

### 3.4 Missing Annotations & Uncertain Category Flagging
- **Missing Masks:** Evaluates whether corresponding segmentations (`{stem}_mask.*`) exist for radiomic extraction.
- **Uncertainty Guard:** Scans filenames and metadata for clinical ambiguity keywords: `uncertain`, `indeterminate`, `borderline`, `equivocal`, `atypical`, `suspicious`, `unconfirmed`, `rule out`, `r/o`, `pending`, `inconclusive`. Any matching studies are isolated for clinician review.

---

## 4. Patient-Level Partitioning Strategy (`partitioner.py`)

### 4.1 Zero Patient Leakage Guarantee
Random slice-level splitting in medical imaging causes catastrophic data leakage because slices from the same CT volume or repeated radiographs of the same patient share morphological features, resulting in inflated evaluation metrics.
- The `PatientLevelPartitioner` groups records strictly by pseudonymous `patient_id`.
- All slices, series, and timepoints for a patient are assigned exclusively to **one** split.

### 4.2 Stratified Partitioning Ratios
Internal cohorts are stratified across classes at the patient level:
- **Training (70%):** Model weights and representation optimization.
- **Validation (15%):** Model selection, hyperparameter tuning, and early stopping.
- **Test (15%):** Final unbiased generalizability assessment.

### 4.3 Multi-Site External Evaluation Strategy
When datasets from distinct institutions or scanning hardware exist (identified by `site_id` or `source_id`), the partitioner isolates designated external sites into an `external_test` split. This enables out-of-domain evaluation to verify algorithmic robustness across hospital scanner variances.

### 4.4 Training Data Isolation Guard (`assert_training_only_fit`)
To prevent data snooping:
- Statistical normalization parameters (e.g., ImageNet Z-score mean and standard deviation).
- Radiomic feature scalers (StandardScaler / RobustScaler).
- IBSI feature selectors (variance thresholds, LASSO).

Must be fitted **EXCLUSIVELY** on the `train` split. The partitioner provides `assert_training_only_fit(split)` which throws `TrainingIsolationError` if fitting is attempted on validation or test splits.

---

## 5. Current Zero-Dataset Status & Generated Artifacts

Because zero real clinical datasets currently reside on local disk:
1. `dataset_quality_report.json` records `status: "NOT_READY"`, `total_samples: 0`, `total_patients: 0`, and flags all 4 missing target classes in `blockers`.
2. `class_distribution.csv` contains the canonical headers with zero samples.
3. `duplicate_audit.csv` contains the audit schema with zero duplicate rows.
4. `split_manifest.csv` contains the standard manifest headers with zero rows.

All 4 artifacts are generated and tracked in `ml_service/data/splits/` and mirrored to `ml_artifacts/reports/`.

