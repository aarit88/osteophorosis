# STAGE 06 STAGE REPORT: DATASET LABEL AUDIT AND PARTITIONING

- **Project:** OsteoGuard AI — A Unified Deep Learning and Radiomic Framework for Differential Diagnosis and Biomechanical Fracture Risk Stratification
- **Stage:** STAGE 06 — DATASET LABEL AUDIT AND PARTITIONING
- **Status:** **PASS**
- **Date:** 2026-11-15
- **Lead ML Architect / Lead Python Engineer:** Antigravity AI (Google DeepMind Team)

---

## 1. Executive Summary

Stage 06 implements institutional-grade dataset validation, label integrity auditing, cryptographic and perceptual duplicate detection, and patient-level stratified partitioning for OsteoGuard AI.

In accordance with clinical machine learning rigor, this stage prevents data leakage by ensuring that medical image partitioning strictly groups by pseudonymous patient ID rather than individual image slices or series. Furthermore, strict statistical isolation is enforced: data transformations, feature selection, and normalization parameters are mathematically restricted to fitting exclusively on the `train` split.

Given the current reality that **zero real clinical datasets reside on local disk**, the audit engine strictly marks the dataset quality state as **`NOT_READY`** with zero samples and 4 missing classes. In adherence to **Rule 4** (*Never invent dataset labels or patient records*) and **Rule 5** (*Never use generated synthetic images as real training data*), zero synthetic images were manufactured to balance the four classes.

All 74 Python/PyTorch unit tests (including 10 dedicated Stage 06 tests) and all 6 frontend Vitest tests passed with 100% success.

---

## 2. Key Modules Implemented

### 2.1 Dataset Integrity & Label Auditor (`ml_service/datasets/auditor.py`)
- **Canonical 4-Class Taxonomy Alignment:** Maps raw file cohorts strictly to `metastasis`, `paget`, `oi`, and `healthy`.
- **Exact Duplicate Detection (SHA-256):** Computes cryptographic hash digests of every image; exact duplicates are tracked in `duplicate_audit.csv`.
- **Near-Duplicate Detection (dHash Perceptual Hashing):** Computes 64-bit difference hashes via `imagehash` and calculates bitwise Hamming distances. Pairs with $\text{HammingDistance} \le 3$ are flagged as near-duplicates for clinical oversight.
- **Conflicting Label Guard:** Tracks patients across disease folders; if a single pseudonymous patient ID is found assigned to multiple conflicting classes, a blocking error is raised.
- **Uncertain Category & Annotation Auditing:** Automatically flags filenames and metadata containing clinical ambiguity terms (`uncertain`, `indeterminate`, `borderline`, `suspicious`, `rule out`, etc.) and tracks missing segmentation masks.
- **Distribution Profiling:** Generates sample-, patient-, and study-level counts across modalities (`ct`, `xray`) and diagnostic classes.

### 2.2 Patient-Level Partitioner & Leakage Guard (`ml_service/datasets/partitioner.py`)
- **Zero-Leakage Patient Grouping:** Partitions records strictly by pseudonymous patient ID (`PT_{id}`). All slices, series, and timepoints for a patient remain exclusively in a single partition.
- **Stratified Split Ratios:** Initial configuration of **70% Train, 15% Validation, 15% Test** stratified across diagnostic classes at the patient level.
- **Multi-Site External Evaluation Strategy:** Supports designating institution/scanner identifiers (`site_id` / `source_id`) as out-of-domain holdouts routed to `external_test`.
- **Leakage Verifier Function:** `verify_patient_leakage(manifest)` mathematically computes set intersections across splits:
  $$\text{Overlap}(S_a, S_b) = \text{Patients}(S_a) \cap \text{Patients}(S_b) = \emptyset$$
  Throws `PatientLeakageError` if any overlap is detected.
- **Training-Only Fit Guard:** Implements `assert_training_only_fit(split)` to ensure statistical scalers, ImageNet normalizers, and radiomic selectors throw `TrainingIsolationError` if fit is attempted on validation or test splits.

---

## 3. Generated Quality and Partitioning Artifacts

The four mandatory Stage 06 artifacts have been generated in `ml_service/data/splits/` and mirrored to `ml_artifacts/reports/`:

1. **`dataset_quality_report.json`**:
   - `status`: `"NOT_READY"`
   - `total_samples`: 0
   - `total_patients`: 0
   - `missing_classes`: `["metastasis", "paget", "oi", "healthy"]`
   - `blockers`: Detailed listing of zero samples and missing clinical cohorts.
2. **`class_distribution.csv`**:
   - Standard columns: `target_class`, `samples`, `patients`, `studies`.
   - Records 0 samples across all 4 target classes.
3. **`duplicate_audit.csv`**:
   - Standard columns: `duplicate_type`, `file1`, `file2`, `hamming_distance`, `sha256`.
   - 0 duplicate rows on current zero-dataset state.
4. **`split_manifest.csv`**:
   - Standard columns: `dataset_id`, `patient_id`, `source_id`, `site_id`, `target_class`, `modality`, `file_path`, `mask_path`, `split`, `sha256_hash`.
   - 0 rows on current zero-dataset state.

---

## 4. Test & Verification Results

### 4.1 Pytest Test Suite (`ml_service/tests` and `tests_ml`)
- **Command:** `python -m pytest ml_service/tests tests_ml -v`
- **Result:** **74 passed in 29.77s (100% PASS)**
- **Stage 06 Unit Tests (`ml_service/tests/test_auditor_and_partitioning.py`):**
  - `test_exact_duplicate_detection_sha256`: PASSED
  - `test_near_duplicate_detection_dhash`: PASSED
  - `test_conflicting_label_detection`: PASSED
  - `test_uncertain_keywords_and_missing_masks_flagging`: PASSED
  - `test_zero_patient_leakage_partitioning`: PASSED
  - `test_patient_leakage_verifier_catches_leak`: PASSED
  - `test_external_site_holdout_strategy`: PASSED
  - `test_training_isolation_guard`: PASSED
  - `test_zero_dataset_audit_state`: PASSED
  - `test_artifact_export_integrity`: PASSED

### 4.2 Frontend Vitest Suite (`osteophorosis-main`)
- **Command:** `npm test --prefix osteophorosis-main -- --pool=threads`
- **Result:** **3 test files passed, 6/6 tests passed in 30.73s**
- **Existing Frontend Integrity:** 100% preserved. Zero modifications to React components, styling, or routing.

---

## 5. Files Created and Modified

| File Path | Action | Description |
|---|---|---|
| `ml_service/datasets/auditor.py` | Created | DatasetAuditor, SHA-256 duplicate detection, dHash perceptual hashing, conflicting label detection, and quality exporter |
| `ml_service/datasets/partitioner.py` | Created | PatientLevelPartitioner, zero patient leakage verification, external-site holdout, training-only fitting guard |
| `ml_service/datasets/__init__.py` | Modified | Cleanly exported Stage 06 auditor and partitioner symbols |
| `ml_service/datasets/generate_split_artifacts.py` | Created | Automated script to execute audit and export all 4 split artifacts |
| `ml_service/tests/test_auditor_and_partitioning.py` | Created | 10 comprehensive unit tests for duplicate detection, perceptual hashing, leakage assertions, and audit exports |
| `ml_service/data/splits/dataset_quality_report.json` | Created | Quality report JSON reflecting current zero-dataset state |
| `ml_service/data/splits/class_distribution.csv` | Created | Canonical class distribution CSV table |
| `ml_service/data/splits/duplicate_audit.csv` | Created | Duplicate audit log table |
| `ml_service/data/splits/split_manifest.csv` | Created | Patient-level split manifest CSV |
| `ml_artifacts/reports/*` | Created | Mirrored copies of the 4 split and quality artifacts |
| `docs/DATASET_AUDIT_AND_PARTITIONING.md` | Created | In-depth technical guide for dataset audit and patient-level splitting |
| `ml_artifacts/stage_reports/STAGE_06_AUDIT_PARTITION_REPORT.md` | Created | Stage 06 verification and audit report |
| `requirements.txt` / `ml_service/requirements.txt` / `pyproject.toml` | Modified | Added `imagehash>=4.3.0` and `PyWavelets>=1.10.0` dependencies |

---

## 6. Strict Execution Rule & Policy Adherence

1. **Rule 1 (Frontend Preservation):** No frontend code modified. Vitest tests passed (6/6).
2. **Rule 4 (No Fake Labels/Patients):** Zero synthetic patient IDs or fake diagnostic labels manufactured.
3. **Rule 5 (No Synthetic Images as Real Training Data):** Zero synthetic images added to data directories.
4. **Rule 6 (No False Training Claims):** Model states strictly report `NOT_CONFIGURED` / `AWAITING_DATA`.
5. **Rule 7 (No Fabricated Metrics):** Dataset status strictly marked `NOT_READY` with 0 samples.
6. **Execution Stop Rule:** Stage 06 is fully verified and STOPPED. Training is not initiated.

