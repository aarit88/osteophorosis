# OsteoGuard AI — Stage 02 Architecture Design Report

## Stage ID
`STAGE-02-ML-ARCHITECTURE-DESIGN`

## Implementation Status
**PASS**

---

## 1. Executive Summary
In strict compliance with the project directives, **Stage 02** has completed the formal system architecture and phased engineering implementation plan for OsteoGuard AI.

Key accomplishments in this stage:
1. Re-verified `docs/REPOSITORY_AUDIT.md` and confirmed frontend compatibility with the existing TypeScript codebase.
2. Formulated **`docs/ML_ARCHITECTURE.md`**, detailing the dual-stream deep learning (ResNet-101 / Swin-T) and radiomic framework, modality-specific data pipelines, U-Net skeletal segmentation, 614-D late fusion, Focal Loss ($\gamma = 2.0$), Mirels' continuous fracture risk calculation, and Grad-CAM/SHAP explainability.
3. Formulated **`docs/ML_IMPLEMENTATION_PLAN.md`**, outlining the component responsibilities, API contracts, data flows, and phased milestone roadmap.
4. Resolved and mathematically documented the **Research Document Discrepancy** regarding the 102 radiomic features (18 First-Order + 26 GLCM + 16 GLRLM + 16 GLSZM + 26 GLDM/NGTDM = 102).
5. Adapted and established the proposed **`ml_service/`** backend structure in the repository alongside the untouched React frontend.
6. Enforced the **zero model training** rule, maintaining readiness gate hard-blocks until real clinical datasets are supplied.

---

## 2. Changes Made & Specifications Produced

### 2.1 Core Architectural Artifacts Created
- **`docs/ML_ARCHITECTURE.md`** (mirrored to `osteophorosis-main/docs/ML_ARCHITECTURE.md`):
  - Comprehensive end-to-end Mermaid architectural flow diagram.
  - Complete 4-class target disease taxonomy (`metastasis`, `paget`, `oi`, `healthy`).
  - Isolated modality pipelines: CT Hounsfield Units standardization ($\text{WL} = +400\text{ HU}, \text{WW} = 1500\text{ HU}$), X-ray CLAHE ($\beta = 2.0$, $8 \times 8$ grid), and Scintigraphy count density percentile normalization.
  - Mathematical formulation of U-Net Soft Dice Loss, ResNet-101 512-D projection, 102-D IBSI radiomic descriptors, 614-D Late Fusion, Multiclass Focal Loss ($\gamma = 2.0$), continuous fracture risk equation ($R_{\text{fracture}}$), Grad-CAM, TreeSHAP, and Shannon entropy abstention ($H > 0.60$).
- **`docs/ML_IMPLEMENTATION_PLAN.md`** (mirrored to `osteophorosis-main/docs/ML_IMPLEMENTATION_PLAN.md`):
  - Component responsibilities, input/output data contracts, and verification gates.
  - Phased multi-stage roadmap from current zero-dataset state to clinical deployment.
  - Strict compliance matrix mapping the 10 execution rules to implementation mechanisms.

### 2.2 Research Document Discrepancy Documented & Resolved
- **Discrepancy**: The Research Paper specifies 102 features across 5 families ($18 + 26 + 16 + 16 + 26 = 102$), whereas the KSCST proposal's summary text mentions only 4 matrix families while still asserting 102 features ($18 + 26 + 16 + 16 = 76$).
- **Resolution**: Documented that the KSCST text was an editorial abbreviation. The complete 5-family IBSI formulation (incorporating the 26 GLDM/NGTDM descriptors) is strictly implemented, ensuring exact compliance with the 102-D vector and the 614-D late fusion representation ($512 + 102 = 614$) without fabricating or duplicating features.

### 2.3 Backend Directory Structure Scaffolding Created
Created directory layout for `ml_service/`:
```
ml_service/
├── app/ (api/, schemas/, services/)
├── config/
├── data/ (raw/, interim/, processed/, splits/)
├── datasets/
├── preprocessing/
├── segmentation/
├── models/
├── radiomics/
├── fusion/
├── training/
├── evaluation/
├── explainability/
├── inference/
├── utils/
└── tests/
```

---

## 3. Automated Tests Executed & Verification Results

### 3.1 Python ML Regression Test Suite
```powershell
python -m pytest tests_ml -v
```
**Output Summary:**
- `tests_ml/test_api_schemas.py`: 2 passed (Health & Analyze OpenAPI contracts)
- `tests_ml/test_biomechanics.py`: 4 passed (LCR, polar moment degradation, Mirels' formula)
- `tests_ml/test_constants_and_config.py`: 5 passed (Taxonomy integrity, 102-D count, configs)
- `tests_ml/test_models.py`: 6 passed (U-Net, ResNet-101 512-D, Late Fusion 614-D, Focal Loss)
- `tests_ml/test_radiomics_ibsi.py`: 4 passed (Deterministic 102-D extraction, no NaNs)
- `tests_ml/test_readiness_gate.py`: 4 passed (Strict zero-dataset hard block enforcement)
- `tests_ml/test_sanitizer_and_partitioning.py`: 2 passed (HIPAA PHI removal, zero patient leakage)

**Result:** `27 passed in 29.08s` (100% pass rate).

### 3.2 Frontend Vitest Test Suite
```powershell
cd osteophorosis-main; npm test
```
**Result:** `3 test files passed, 6 tests passed in 11.04s` (100% pass rate).

### 3.3 OpenAPI 3.1 Validation
```powershell
cd osteophorosis-main; npm run openapi:validate
```
**Result:** `openapi.yaml: validated in 46ms. Woohoo! Your API description is valid. 🎉`

---

## 4. Evidence
- **Files Verified on Disk**:
  - `docs/ML_ARCHITECTURE.md`: 34,659 bytes
  - `docs/ML_IMPLEMENTATION_PLAN.md`: 16,106 bytes
  - `osteophorosis-main/docs/ML_ARCHITECTURE.md`: 34,659 bytes
  - `osteophorosis-main/docs/ML_IMPLEMENTATION_PLAN.md`: 16,106 bytes
- **Untouched Frontend Guarantee**: `src/` components, pages, routes, and styles remain completely untouched and verified passing all Vitest tests.

---

## 5. Remaining Blockers
- **Zero Real Clinical Datasets Downloaded**:
  - Model training is intentionally blocked by design until authentic, verified clinical cohorts (DICOM/PNG) are provided.
  - No synthetic data was used, and no model training was executed in this stage.

---

## 6. Next Permitted Stage
**STAGE-03: Clinical Dataset Ingestion, Manifest Pipeline & Patient-Level Partitioning.**
*(Per stage management instructions, execution is stopped. Awaiting explicit user command before beginning Stage 03).*
