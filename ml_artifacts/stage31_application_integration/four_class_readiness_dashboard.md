# Four-Class Readiness Dashboard: Clinical Gap and Acquisition Analysis

## 1. Executive Diagnostic Readiness Status
- **Unified Diagnostic System**: OsteoGuard AI
- **Proposed Objective**: 4-Class Differential Diagnosis & Biomechanical Fracture Risk Stratification
- **Current Operational Status**: `BLOCKED — APPROVED CLINICAL DATA AND GOVERNANCE REQUIRED`
- **Clinical Inference Readiness**: `0.0%` (Zero models trained on 4-class clinical data)
- **Mathematical / Engineering Engine Readiness**: `100.0%` (All 4 modular components verified on synthetic digital phantoms)

---

## 2. Four-Class Cohort Status Register

### Class 1: Secondary Bone Metastasis
- **Target Anatomy**: Pelvis, Spine, Proximal Femur, Ribs, Calvarium
- **Candidate Modalities**: Plain Radiography (X-Ray), Bone Scintigraphy, CT
- **Current Cohort Availability**:
  - *Bone Scintigraphy*: Screening cohort available from Stage 11–14 (breast cancer metastasis staging).
  - *Plain Radiographs*: Unacquired; existing public datasets (e.g., BTXRD) contain *primary* bone tumours (Osteosarcoma, Ewing Sarcoma), which are biologically and radiographically distinct.
- **Acquisition Blocker**:
  - Requires institutional plain radiographs of biopsy-proven or PET-CT-confirmed secondary metastases from common primary malignancies (breast, prostate, lung, kidney, thyroid).
- **Ethics & Regulatory Dependency**:
  - Institutional Ethics Committee (IEC) review required for retrospective radiograph cohort extraction.

---

### Class 2: Paget's Disease of Bone
- **Target Anatomy**: Pelvis, Femur, Tibia, Lumbar Spine, Skull
- **Candidate Modalities**: Plain Radiography
- **Current Cohort Availability**:
  - `0 images`. Comprehensive audit in Stages 22 and 23 confirmed that no open-access repository of human Paget's radiographs exists globally.
- **Acquisition Blocker**:
  - Paget's disease has low epidemiological prevalence in South Asia.
  - Requires international collaborative outreach to European / UK clinical registries (e.g., Paget's Association UK, PRISM clinical trial archive).
- **Ethics & Regulatory Dependency**:
  - International collaborative Data Use Agreement (DUA).
  - Indian Council of Medical Research (ICMR) Health Ministry's Screening Committee (HMSC) clearance for cross-border data transfer.

---

### Class 3: Osteogenesis Imperfecta (OI)
- **Target Anatomy**: Long Bones (Femur, Tibia, Humerus), Spine
- **Candidate Modalities**: Plain Radiography
- **Current Cohort Availability**:
  - `0 human radiographs`. Forensic re-audit of candidate study PMC11324408 confirmed that it evaluated micro-CT in murine models and retrospective clinical charts without open radiograph access.
- **Acquisition Blocker**:
  - Rare genetic connective tissue disorder.
  - Requires multi-center collaborative protocol with specialized pediatric orthopedic hospitals (e.g., AIIMS, Indira Gandhi Institute of Child Health, Brittle Bone Society).
- **Ethics & Regulatory Dependency**:
  - Dedicated Pediatric Institutional Ethics Committee approval with parental consent waivers for de-identified retrospective data.

---

### Class 4: Healthy Skeletal Controls
- **Target Anatomy**: Anatomically matched sites (Pelvis, Spine, Femur)
- **Candidate Modalities**: Plain Radiography
- **Current Cohort Availability**:
  - Existing "non-fractured" images from trauma centers (e.g., FracAtlas negative class) are confounded by unknown metabolic status, age-related osteopenia, or unrecorded systemic disease.
- **Acquisition Blocker**:
  - Requires age- and sex-matched asymptomatic controls with documented normal Bone Mineral Density (BMD / DEXA T-score > -1.0).
- **Ethics & Regulatory Dependency**:
  - Institutional Ethics approval for asymptomatic control cohort matching.

---

## 3. Four-Class Multi-Modal Architecture Status

| Architectural Block | Planned Specification | Current Implementation | Verification Status | Unblocking Prerequisite |
|---|---|---|---|---|
| **Deep Feature Extractor** | DenseNet-121 (512-D) | Architecture specified | Untrained for 4-class task | Acquisition of all 4 clinical cohorts |
| **Radiomics Feature Extractor** | IBSI Radiomics (102-D) | 81 verified / 21 unverified | `NOT_VERIFIED` (Proposal 102-D spec) | Resolution of remaining 21 IBSI features |
| **Multimodal Fusion Layer** | 512 + 102 = 614-D Concatenation | Tensor contract defined | `ARCHITECTURALLY_SPECIFIED` | Training of deep backbone and clinical validation |
| **Biomechanical Risk Modifier** | Cross-Sectional Geometry + Mirels | Fully implemented | Verified on synthetic phantoms | Correlation with clinical fracture registry |
| **Uncertainty & Abstention Gate** | Shannon Entropy ($H_{\text{norm}} \ge 0.60$) | Fully implemented | Verified on synthetic distributions | Clinical calibration on patient validation cohort |
| **Four-Class Output Layer** | Softmax over 4 classes | Schema defined | `BLOCKED` (Zero weights) | Complete clinical dataset intake and training |

---

## 4. Institutional Action Plan
1. **Submit Institutional Ethics Protocol**: Finalize and submit ICMR IEC Form A prepared in Stage 24 to the Institutional Ethics Committee.
2. **Execute Collaborative DUAs**: Formalize data transfer agreements with participating orthopedic research centers.
3. **Register Prospective Study**: Register clinical validation protocol on the Clinical Trial Registry of India (CTRI).
4. **Maintain Gated Prototype**: Keep the four-class inference endpoint strictly blocked until the four clinical cohorts are acquired, harmonized, and verified.

