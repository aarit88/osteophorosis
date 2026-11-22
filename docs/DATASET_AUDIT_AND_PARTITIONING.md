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

---

## 6. Stage 33 — Track D Clinical Dataset Intake, Modality Isolation & Readiness Gate

Stage 33 extends the foundational Stage 06 audit engine with a dedicated, pure-stdlib **Track D Clinical Intake Pipeline** (`ml_service/datasets/track_d_intake.py` + `ml_service/scripts/run_track_d_intake.py`) operating directly inside AI Studio without requiring PyTorch weights or external GPU services.

### 6.1 Standardized Modality-Separated Directory Structure (`ml_service/data/`)
Modalities (`xray`, `bone_scan`, `ct`) are strictly separated and never assumed interchangeable:
```
ml_service/data/
├── raw/{xray,bone_scan,ct}/{metastasis,paget,oi,healthy}/
├── interim/{xray,bone_scan,ct}/{metastasis,paget,oi,healthy}/
├── processed/{xray,bone_scan,ct}/{metastasis,paget,oi,healthy}/
├── quarantine/{xray,bone_scan,ct}/{metastasis,paget,oi,healthy}/
├── manifests/
│   ├── track_d_manifest.schema.json
│   ├── track_d_manifest.example.json   (non-ingested schema template)
│   └── track_d_manifest.example.csv    (non-ingested schema template)
├── splits/
│   └── track_d_split_manifest.csv / track_d_split_summary.json
└── reports/
    ├── track_d_quality_report.json
    ├── track_d_quarantine_report.json
    └── track_d_readiness_report.json
```

### 6.2 Machine-Readable Manifest (16 Required Fields & PHI Prohibition)
Every clinical sample manifest entry must supply 16 verified fields:
`sample_id`, `patient_id` (pseudonymized `PT_*` / `SUBJ_*` only), `diagnostic_class`, `imaging_modality`, `image_path`, `image_format`, `image_dimensions`, `image_bit_depth`, `anatomical_region`, `institution_id`, `dataset_source`, `license_permissions`, `ethics_approval_ref`, `deidentification_status`, `qc_status`, and `split_assignment`.

Direct PHI keys (`patient_name`, `mrn`, `email`, `ssn`, `address`, `dob`, `phone`) and unpseudonymized patterns are automatically blocked and quarantined.

### 6.3 Quality Validation, Negative Mapping & External Holdout Protection
- **Image Header & Stream Verification**: Inspects PNG (`IHDR` + zlib `IDAT`), JPEG (`SOF`), TIFF (`IFD`), BMP, DICOM (`DICM` tags), and NIfTI headers (`>= 32x32` px, non-zero size, format consistency).
- **Duplicate & Conflict Checks**: SHA-256 exact duplicate detection, 64-bit grayscale `dHash` near-duplicate detection (`Hamming <= 3`), and cross-record patient diagnosis conflict blocking.
- **Uncertain Diagnosis Non-Conversion**: Equivocal/borderline/rule-out studies are isolated for clinical review and **never** converted into another class.
- **Protected External Holdout Isolation**: Blocks and quarantines any sample matching **BTXRD Center 2 (`N=549`)**, **BTXRD Center 3 (`N=259`)**, or **FracAtlas burned test holdouts** by hash, sample ID, or site identifier.

### 6.4 Readiness Gate & Honest Zero-Data Reporting
Readiness passes (`READY_FOR_TRAINING`) only when all 7 structural gates succeed: documented provenance, verified permissions/DUA, verified de-identification, documented diagnostic label evidence, patient-level zero-leakage split integrity, image quality pass rate, and 4-class coverage. No arbitrary minimum clinical sample size is fabricated in software; statistical cohort power requires documented biostatistical and IEC approval. When zero real cohorts exist, both the CLI (`run_track_d_intake.py --all`) and `GET /api/v1/datasets/readiness` report **`NOT_AVAILABLE`** with **`0`** sample counts.

---

## 7. Stage 34 — Clinical Label Ontology, Diagnostic Adjudication & Dataset Safety Audit

### 7.1 Diagnostic Mapping Audit Findings
An exhaustive audit across `ml_service/datasets/`, manifest schemas, readiness endpoints, and UI code confirmed that no positive equivalence rules converted `Osteoporosis -> OI`, `Osteoarthritis -> Paget's`, `Contralateral -> Healthy`, or `Primary bone malignancy -> Metastasis` (`registry_verifier.py` already enforced negative exclusion Rules 2.1–2.4). However, Stage 33 had five structural gaps remediated in Stage 34:
1. Inline `CLASS_ALIASES` lacked a single centralized clinical ontology with explicit `OUT_OF_SCOPE` / `UNCERTAIN` / `REQUIRES_ADJUDICATION` tracking.
2. `diagnostic_evidence` was optional during validation.
3. `healthy` lacked positive eligibility checks (contralateral screening, confirmed radiology report, non-generic `anatomical_region`, cohort `control_selection_criteria`, and documented `patient_skeletal_history`).
4. `track_d_manifest.schema.json` lacked formal provenance & adjudication metadata (`source_report_ref`, `diagnostic_evidence_category`, `histopathology_confirmation_status`, `radiology_report_confirmation_status`, `clinical_adjudication_status`, `reviewer_pseudo_id`, `review_date`, `conflicting_diagnosis_indicator`, `label_provenance`, `label_ontology_version`).
5. The Governance Dashboard and `/api/v1/datasets/readiness` did not expose ontology adjudication breakdown counts.

### 7.2 Centralized Ontology & Adjudication Engine (`clinical_ontology.py` & `adjudication_engine.py`)
All label normalization, status management (`ACCEPTED`, `OUT_OF_SCOPE`, `UNCERTAIN`, `REQUIRES_ADJUDICATION`, `REJECTED`), and chronological label provenance tracking are centralized under **`OSTEO_ONTOLOGY_V1.0_STAGE34`** (`ml_service/datasets/clinical_ontology.py` + `ml_service/app/services/adjudication_engine.py`):
- **Canonical Classes**: `SECONDARY_BONE_METASTASIS`, `PAGETS_DISEASE`, `OSTEOGENESIS_IMPERFECTA`, `HEALTHY_SKELETAL_CONTROL`.
- **Adjudication Outcomes**: `ACCEPTED`, `OUT_OF_SCOPE`, `UNCERTAIN`, `REQUIRES_ADJUDICATION`, `REJECTED`.
- **Zero Unsupported Equivalences**:
  - `Osteoporosis`, `osteopenia`, `rickets`, `osteomalacia` $\rightarrow$ `OUT_OF_SCOPE` (never `OI`).
  - `Osteoarthritis`, `degenerative sclerosis`, `fibrous dysplasia`, `osteopetrosis` $\rightarrow$ `OUT_OF_SCOPE` (never `Paget's`).
  - `Osteosarcoma`, `Ewing sarcoma`, `chondrosarcoma`, `multiple myeloma`, `primary bone malignancy` $\rightarrow$ `OUT_OF_SCOPE` (never `Secondary Metastasis`).
  - `Contralateral` / `fellow limb` / `unaffected limb` / `no fracture detected` / `incomplete medical history` $\rightarrow$ `REJECTED` (never `Healthy`).
  - `Uncertain` / `equivocal` / `borderline` / `rule out` $\rightarrow$ `UNCERTAIN` (excluded from train/val/test splits).
  - Conflicting patient diagnoses or unconfirmed reviews $\rightarrow$ `REQUIRES_ADJUDICATION`.

### 7.3 Recursive Dataset Validator CLI (`ml_service/scripts/validate_dataset.py`)
`python3 ml_service/scripts/validate_dataset.py` recursively scans `ml_service/data/raw/`, verifies image format/size/dimensions/SHA-256 digests, checks Stage 34 metadata & provenance completeness, detects duplicate patient IDs, conflicting patient diagnoses, and duplicate/near-duplicate images, and outputs `track_d_adjudication_audit.json` alongside readiness reports via `run_track_d_intake.py --all`.

---

## 8. Stage 35 — Institutional Dataset Acquisition, Research Governance & Cohort Readiness Management

### 8.1 Centralized Dataset Source Registry (`ml_service/data/manifests/dataset_source_registry.json`)
Registers all four required clinical cohorts (`metastasis`, `paget`, `oi`, `healthy`) while strictly separating **proposed institutional partnerships** (`IDENTIFIED`, `url_verified: false`), **verified external/in-tree candidates** (`VERIFICATION_REQUIRED`), and **disqualified sources** (`REJECTED`):
- **Secondary Bone Metastasis**: Proposed institutional PACS radiograph cohort; verified TCIA `Spine-Mets-CT-SEG` (`ct` only, 55 pts / 108 series pending TCIA DUA); audited in-tree IICS-UNA bone scintigraphy cohort (`bone_scan` only, 60 mets pts / 120 paired views; staging negatives excluded as healthy controls); BTXRD Class 2 primary sarcomas permanently `REJECTED`.
- **Paget's Disease of Bone**: Proposed hospital/UK-EU MSK radiology PACS collaboration (`IDENTIFIED`); Radiopaedia / MedPix educational web figures permanently `REJECTED`.
- **Osteogenesis Imperfecta (OI)**: Verified controlled-access candidate RDCRN BBDC Protocol 7701 (`VERIFICATION_REQUIRED`, no public download in PMC11324408); proposed tertiary pediatric orthopedic genetics registry (`IDENTIFIED`); Figshare murine *oim* micro-CT `10.6084/m9.figshare.c.3795019` permanently `REJECTED`.
- **Healthy Skeletal Controls**: Stanford AIMI MURA upper-extremity comparator (`VERIFICATION_REQUIRED`, strictly upper extremity only); proposed anatomically & age-matched hospital PACS normal controls (`IDENTIFIED`); FracAtlas emergency non-fractured trauma cohort permanently `REJECTED`.

### 8.2 Institutional Governance & Acquisition Workflow State Machine (`acquisition_governance.py` & `dataset-governance.ts`)
- **Non-Trivial Governance Verification**: Never marks governance fields verified merely because strings are non-empty; requires valid approval status (`APPROVED`/`EXECUTED`/`VERIFIED`/`ACTIVE`), non-placeholder protocol/DUA/signoff reference IDs, active non-expired `expiry_date`, evidence reference, and valid pseudonymous reviewer signoff (`REV_*`/`ADJ_*`/`RAD_*`/`CLIN_*`) with authorized `GovernanceRole`.
- **10-Stage Sequential Workflow**: `SOURCE_IDENTIFIED` $\rightarrow$ `SOURCE_VERIFIED` $\rightarrow$ `ETHICS_PENDING` $\rightarrow$ `DUA_PENDING` $\rightarrow$ `ACCESS_APPROVED` $\rightarrow$ `DATA_RECEIVED` $\rightarrow$ `DEIDENTIFICATION_VERIFIED` $\rightarrow$ `LABEL_ADJUDICATION` $\rightarrow$ `QUALITY_VALIDATION` $\rightarrow$ `COHORT_ACCEPTED` (plus controlled exception states `REJECTED`, `EXPIRED`, `REQUIRES_REVIEW`, `BLOCKED`).
- **Four Cohort-Specific Acquisition Protocols (`cohort_acquisition_protocols.json`)**: Enforces `xray`/`bone_scan`/`ct` separation, metastasis biopsy vs PET-CT/radiology concordance tiering, Paget's clinical/radiological triad + serum ALP/BSAP, OI `COL1A1/COL1A2` genetic vs Sillence clinical separation + pediatric consent/IEC waiver, and Stage 34 healthy-control eligibility.

### 8.3 Cohort Planning vs Actual Counts & Draft Request Templates
- **Cohort Planning (`cohort_planning_manifest.json`)**: Documents `940` planned patients (`2,270` planned scans) across 6 cohort $\times$ modality streams with `70/15/15` patient-level splits and `statistical_sample_size_justification_ref: "PENDING_BIOSTATISTICAL_AND_IEC_REVIEW"`, while actual on-disk validated counts remain **`0`**.
- **Seven Draft Templates (`ml_service/data/templates/01_*.md` – `07_*.md`)**: Standardized institutional collaboration, dataset access, ethics submission, DUA, de-identification, adjudication SOP, and transfer/provenance templates, each explicitly marked **`DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`**.
- **Governance APIs & Dashboard**: Exposed via `/api/v1/datasets/{sources,acquisition-status,governance-status,cohort-planning}` and `POST /api/v1/datasets/workflow/transition` and surfaced in the Governance Dashboard.

