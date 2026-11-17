# OsteoGuard AI — Stage 04 Dataset Acquisition and Registry Report

## Stage ID
`STAGE-04-DATASET-ACQUISITION-AND-REGISTRY`

## Implementation Status
**PASS**

---

## 1. Executive Summary
In strict compliance with the project directives, **Stage 04** has established the comprehensive clinical dataset acquisition, provenance tracking, registry specification, and verification framework for OsteoGuard AI.

Currently, **zero downloaded physical datasets exist in the repository**. In strict adherence to **Rule 4**, **Rule 5**, **Rule 6**, and **Rule 7**, no synthetic images, invented patient IDs, or phantom metrics have been generated. Model training and inference remain hard-blocked behind the readiness gate.

Key accomplishments in this stage:
1. **Verified Stage 03 Status**: Confirmed `STAGE-03-BACKEND-ML-INFRASTRUCTURE` is marked **PASS** and all 48 previous automated tests execute with 100% clearance.
2. **Master Dataset Registry**: Authored `ml_service/data/dataset_registry.yaml` registering nine clinical imaging repositories across the four target disease classes (`secondary_bone_metastasis`, `pagets_disease`, `osteogenesis_imperfecta`, `healthy_control`).
3. **Explicit Dataset Readiness Hard Gate**: Declared `dataset_ready: false` in `dataset_registry.yaml` and verified that the system maintains `can_train = False` and `can_infer = False`.
4. **Strict Negative Mapping Rules**: Formalized and programmatically enforced the four negative mapping boundaries:
   - Primary malignant tumours (osteosarcoma, Ewing's sarcoma, chondrosarcoma, multiple myeloma) $\neq$ Secondary metastasis.
   - Generic abnormal images (degenerative sclerosis, severe osteoarthritis, fibrous dysplasia) $\neq$ Paget's disease.
   - Age-related osteoporosis, osteopenia, rickets, or osteomalacia $\neq$ Osteogenesis imperfecta.
   - Contralateral or "unaffected" scans in cancer/trauma patients $\neq$ Confirmed healthy controls.
5. **Comprehensive Acquisition Dossier**: Created `ml_service/data/DATASET_ACQUISITION.md` documenting format, modality, labels, patient metadata, segmentation annotations, download instructions, licenses, intended ML uses, and limitations for every single source.
6. **Dataset Manifest Schema**: Established `ml_service/data/dataset_manifest.csv` with standardized headers for tracking file-level provenance, SHA-256 cryptographic hashes, and patient identifiers.
7. **File Verification & Registry Engine**: Implemented `ml_service/datasets/registry_verifier.py` with `DatasetRegistry`, cryptographic SHA-256 computation, manifest generator, and automated negative mapping validation.
8. **Version Control Safeguards**: Authored root `.gitignore` and mirrored to `osteophorosis-main/.gitignore`, strictly excluding raw medical images (`*.dcm`, `*.nii`, `*.nii.gz`, `*.png`), patient records, and checkpoint weights from Git.
9. **Automated Test Coverage**: Developed 6 automated tests in `ml_service/tests/test_dataset_registry.py` verifying YAML parsing, source coverage, strict negative mapping rejection, SHA-256 hashing, and manifest generation. Total test suite expanded to **54 tests** with **100% pass rate**.

---

## 2. Classification of Data Sources by Access Tier

The 9 registered clinical sources are categorized by access requirements:

### 2.1 Tier 1: Publicly Downloadable Sources (Immediate HTTPS / API Access)
These sources require no data use agreements or human approval and can be scripted or downloaded directly:

1. **Zenodo Bone Metastases Cohort** (`ZENODO_BONE_METS_13900966`)
   - Target: `secondary_bone_metastasis`
   - Modality: CT & Planar Radiographs (NIfTI, PNG)
   - URL: `https://zenodo.org/records/13900966`
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
   - Download: Direct HTTPS / Zenodo REST API

2. **Wikimedia Commons Paget's Disease Repository** (`WIKIMEDIA_PAGET_COMMONS`)
   - Target: `pagets_disease`
   - Modality: Planar Radiographs & 99mTc-MDP Bone Scintigraphy (JPEG, PNG, TIFF)
   - URL: `https://commons.wikimedia.org/wiki/Category:Paget%27s_disease_of_bone`
   - License: Creative Commons Attribution-ShareAlike (CC BY-SA 3.0 / 4.0)
   - Download: Direct HTTPS / Wikimedia Commons MediaWiki API

3. **PMC11324408 Osteogenesis Imperfecta Cohort** (`PMC_11324408_OI`)
   - Target: `osteogenesis_imperfecta`
   - Modality: Planar Radiographs (JPEG, TIFF, ZIP)
   - URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC11324408/`
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
   - Download: PubMed Central Open Access subset download

4. **Figshare OI Radiological Collection** (`FIGSHARE_OI_COLLECTION_3795019`)
   - Target: `osteogenesis_imperfecta`
   - Modality: Planar Radiographs (DICOM, TIFF, PNG)
   - URL: `https://doi.org/10.6084/m9.figshare.c.3795019`
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
   - Download: Figshare REST API / direct collection download

5. **Figshare Normal Skeletal Radiographs** (`FIGSHARE_NORMAL_SKELETAL_27865398`)
   - Target: `healthy_control`
   - Modality: Planar Radiographs (PNG, DICOM)
   - URL: `https://doi.org/10.6084/m9.figshare.27865398`
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
   - Download: Figshare Direct Download

---

### 2.2 Tier 2: Sources Requiring Access Approval / Institutional Credentials
These sources require account registration, institutional credentialing, or Data Use Agreement (DUA) review before download:

1. **TCIA Spine-Mets-CT-Seg** (`TCIA_SPINE_METS_CT_SEG`)
   - Target: `secondary_bone_metastasis`
   - Modality: CT (DICOM, NIfTI 3D voxel segmentations)
   - URL: `https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/`
   - License: CC BY 4.0 + TCIA Restricted Data Use Agreement
   - Access Procedure: Register TCIA account, agree to DUA, install NBIA Data Retriever application, download `.tcia` manifest.

2. **NLM MedPix Teaching File Database** (`NLM_MEDPIX_PAGET`)
   - Target: `pagets_disease`
   - Modality: Planar Radiography & CT (JPEG, PNG, DICOM)
   - URL: `https://medpix.nlm.nih.gov/`
   - License: U.S. National Library of Medicine Terms of Service
   - Access Procedure: Register free NIH/NLM user account, perform targeted queries for Paget's disease, and export case images manually.

3. **Stanford AIMI MURA (Musculoskeletal Radiographs)** (`STANFORD_AIMI_MURA`)
   - Target: `healthy_control`
   - Modality: Planar Radiography (PNG 16-bit)
   - URL: `https://aimi.stanford.edu/datasets/mura-msk-xrays`
   - License: Stanford Research Data Use Agreement (Non-Commercial Academic Use Only)
   - Access Procedure: Complete online Stanford AIMI DUA application, verify academic institution email, receive secure download link.

4. **TCIA Healthy Total Body CTs** (`TCIA_HEALTHY_TOTAL_BODY_CTS`)
   - Target: `healthy_control`
   - Modality: CT (DICOM Whole Body)
   - URL: `https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/`
   - License: CC BY 4.0 with TCIA Open Access Terms
   - Access Procedure: Register TCIA account, download `.tcia` manifest, download via NBIA Data Retriever.

---

## 3. Changes Made & Files Created

### 3.1 Registry, Provenance & Documentation Files
- `ml_service/data/dataset_registry.yaml`: Master specification registering all 9 sources, target taxonomy, and `dataset_ready: false`.
- `ml_service/data/dataset_manifest.csv`: 20-column schema tracking dataset ID, source ID, patient ID, study ID, series ID, instance ID, target class, short code, modality, format, relative path, mask path, anatomic region, view, slice thickness, kVp, SHA-256 checksum, healthy verified flag, license, and split.
- `ml_service/data/DATASET_ACQUISITION.md`: Exhaustive 350-line documentation detailing source metadata, strict negative mapping rules, manual acquisition protocols, and HIPAA de-identification procedures.
- `.gitignore` (and mirrored to `osteophorosis-main/.gitignore`): Excludes raw medical image extensions (`*.dcm`, `*.nii`, `*.nii.gz`, `*.mhd`, `*.raw`), checkpoints (`*.pt`, `*.pth`), and training logs.
- `ml_service/data/{raw,interim,processed,splits}/.gitkeep`: Gitkeep markers ensuring directory structure persists in git while binary payloads remain excluded.

### 3.2 Verification Engine
- `ml_service/datasets/registry_verifier.py`: `DatasetRegistry` class, `compute_file_sha256()`, `generate_dataset_manifest()`, and `validate_pathology_mapping()` enforcing the 4 strict negative mapping rules.
- `ml_service/datasets/__init__.py`: Exported verification primitives.
- `requirements.txt`, `ml_service/requirements.txt`, `pyproject.toml`: Added `pyyaml>=6.0.0`.

### 3.3 Automated Test Suite
- `ml_service/tests/test_dataset_registry.py`: 6 tests covering YAML integrity, source coverage, negative mapping enforcement, cryptographic hashing, and manifest generation.

---

## 4. Automated Tests Executed & Verification Results

### 4.1 Combined PyTorch and ML Test Suite
```powershell
python -m pytest ml_service/tests tests_ml -v
```
- Total test files: 13
- Total tests collected: 54
- Total passed: 54
- Total failed: 0
- Pass rate: **100%**

Breakdown:
- `ml_service/tests/test_dataset_registry.py`: 6 passed
- `ml_service/tests/test_health_endpoints.py`: 3 passed
- `ml_service/tests/test_model_state_machine.py`: 10 passed
- `ml_service/tests/test_status_endpoints.py`: 3 passed
- `ml_service/tests/test_structured_logging_and_errors.py`: 3 passed
- `ml_service/tests/test_zero_dataset_guardrails.py`: 2 passed
- `tests_ml/test_api_schemas.py`: 2 passed
- `tests_ml/test_biomechanics.py`: 4 passed
- `tests_ml/test_constants_and_config.py`: 5 passed
- `tests_ml/test_models.py`: 6 passed
- `tests_ml/test_radiomics_ibsi.py`: 4 passed
- `tests_ml/test_readiness_gate.py`: 4 passed
- `tests_ml/test_sanitizer_and_partitioning.py`: 2 passed

### 4.2 Frontend Vitest Regression Suite
- Status: **PASS** (3 test files passed, 6/6 tests passed)
- Zero regressions in frontend TypeScript SPA.

---

## 5. Evidence
- **Files Verified on Disk**:
  - `ml_service/data/dataset_registry.yaml`: 8,964 bytes
  - `ml_service/data/dataset_manifest.csv`: 245 bytes (template header)
  - `ml_service/data/DATASET_ACQUISITION.md`: 16,842 bytes
  - `ml_service/datasets/registry_verifier.py`: 8,241 bytes
  - `.gitignore`: 2,410 bytes
  - `osteophorosis-main/.gitignore`: 2,410 bytes

- **Readiness Gate Evidence**:
  - `dataset_registry.yaml` -> `dataset_ready: false`
  - `GET /api/ml/status` -> `model_state: AWAITING_DATA`, `sample_count: 0`, `can_train: false`, `can_infer: false`
  - Zero raw images tracked in Git.

---

## 6. Remaining Blockers
1. **Physical Image Acquisition Pending**:
   - In accordance with the Stage 04 prompt ("I currently do not possess any downloaded datasets"), actual image files have not yet been downloaded to disk.
   - The acquisition framework, registry, manifest generator, and DUA instructions are now fully prepared to ingest images in Stage 05.
2. **Readiness Gate Remains Active**:
   - `DATASET_READY` remains `FALSE`. Model training and inference remain hard-blocked until physical cohorts are downloaded and pass validation.

---

## 7. Next Permitted Stage
`STAGE-05-PREPROCESSING-STANDARDIZATION`
(Only to be executed when explicitly commanded in the next prompt. Do not begin automatically.)

