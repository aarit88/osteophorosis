# Institutional Research Protocol: OsteoGuard AI Multi-Class Differential Diagnosis

**Protocol Identifier**: `OG-PROT-2026-04C-DRAFT`  
**Protocol Version**: 1.0 (Draft for Institutional Review)  
**Date**: November 16, 2026  
**Investigational System**: OsteoGuard AI Multi-Modal Deep Learning and Radiomic Framework  

> [!IMPORTANT]
> **GOVERNANCE STATUS**:  
> `[DRAFT FOR INSTITUTIONAL RESEARCH COMMITTEE REVIEW — PENDING ETHICS COMMITTEE APPROVAL]`  
> `[REQUIRING CLINICIAN INPUT — RADIOLOGY AND ORTHOPAEDIC CHAIR REVIEW REQUIRED]`  
> `[INSTITUTIONAL REVIEW REQUIRED — DATA PROTECTION OFFICER AND LEGAL COUNSEL SIGN-OFF NEEDED]`  
> *This protocol is a non-clinical research planning document. No patient data has been collected, no ethical approval numbers have been fabricated, and no clinical interventions will be performed under this draft.*

---

## 1. Project Background and Clinical Rationale

Musculoskeletal disorders exhibiting altered bone density, architectural distortion, and pathological fracture risk represent a critical diagnostic challenge. In routine clinical practice, focal and diffuse bone abnormalities may present with overlapping radiographic features:
1. **Secondary bone metastases** (osteolytic, osteoblastic, or mixed lesions arising from primary carcinomas such as breast, prostate, lung, and renal cell).
2. **Paget's disease of bone (Osteitis Deformans)** (characterized by disordered bone remodeling with cortical thickening, trabecular coarsening, and skeletal expansion).
3. **Osteogenesis imperfecta (OI)** (a genetic collagen-related dysplasia leading to severe skeletal fragility, cortical thinning, recurrent fractures, and bone deformities).
4. **Healthy skeletal variants and age-related controls** (requiring robust discrimination to eliminate false-positive triage).

Previous phases of OsteoGuard AI developed and evaluated single-modality models:
- **Stages 11–14**: Evaluated a dual-view bone scintigraphy ensemble for secondary metastases, achieving high sensitivity but constrained strictly to planar nuclear scintigraphy.
- **Stages 15–21**: Evaluated a three-class primary bone tumour model on BTXRD (Normal vs. Benign vs. Primary Sarcoma). External evaluations at Center 2 and Center 3 demonstrated severe domain shift (macro-F1 dropping from 0.8103 to 0.4431 and 0.4566), proving that primary bone sarcomas (osteosarcoma, chondrosarcoma, Ewing sarcoma) have distinct biology, radiological signs, and demographics that **must not be equated with secondary bone metastases**.
- **Stages 22–23**: Audited four-class dataset readiness and verified candidate sources, discovering that candidate online OI datasets were non-human murine micro-CT, while Zenodo metastasis datasets were in-tree duplicates.

Consequently, developing a true four-class differential diagnostic AI requires prospective institutional partnerships to acquire ethically cleared, verified human clinical imaging cohorts.

---

## 2. Research Objectives

### Primary Objective
To construct, train, and validate a multi-class, modality-decoupled machine learning framework capable of differentiating between:
- **Class 1**: Secondary bone metastasis
- **Class 2**: Paget's disease of bone
- **Class 3**: Osteogenesis imperfecta (human clinical)
- **Class 4**: Healthy skeletal control

### Secondary Objectives
1. Quantify biomechanical fracture risk stratification utilizing radiomic texture metrics and architectural descriptors.
2. Establish a clinically interpretable late-fusion decision architecture integrating plain radiography, planar bone scintigraphy, and volumetric computed tomography (CT).
3. Validate diagnostic generalization across multiple independent academic medical centers without patient data leakage.

---

## 3. Cohort Inclusion and Exclusion Criteria

```
+--------------------------------------------------------------------------------------------------+
|                                    INCLUSION & EXCLUSION MATRIX                                  |
+------------------------------+----------------------------------+--------------------------------+
| Target Class                 | Inclusion Criteria               | Exclusion Criteria             |
+------------------------------+----------------------------------+--------------------------------+
| 1. Secondary Bone Metastasis | - Histopathologically confirmed  | - Primary bone sarcoma         |
|                              |   solid primary malignancy       |   (osteosarcoma, Ewing, etc.)  |
|                              | - Documented skeletal metastasis | - Multiple myeloma / lymphoma  |
|                              |   via biopsy, PET-CT, or serial  |   (unless strictly segregated) |
|                              |   radiological progression       | - Unconfirmed primary tumor    |
|                              | - Plain X-ray, Scintigraphy, CT  | - Corrupt / truncated images   |
+------------------------------+----------------------------------+--------------------------------+
| 2. Paget's Disease of Bone   | - Radiologist-confirmed Paget's  | - Isolated educational images  |
|                              |   triad: cortical thickening,    |   without clinical provenance  |
|                              |   trabecular coarsening, expan-  | - Metastatic lesions mimicking |
|                              |   sion (osteomegaly)             |   Paget's (e.g. prostate)      |
|                              | - Elevated serum alkaline phos.  | - Fibrous dysplasia / osteo-   |
|                              | - High-res skeletal radiographs  |   myelitis confounders         |
+------------------------------+----------------------------------+--------------------------------+
| 3. Osteogenesis Imperfecta   | - Verified human diagnosis       | - Animal models (murine micro- |
|                              | - Sillence Types I, II, III, IV  |   CT, rodent oim tibiae)       |
|                              | - Documented COL1A1/COL1A2 mut.  | - Non-OI osteopenia (juvenile  |
|                              |   or clinical fragility triad    |   idiopathic osteoporosis)     |
|                              | - Pediatric or adult radiographs | - Non-accidental trauma        |
+------------------------------+----------------------------------+--------------------------------+
| 4. Healthy Skeletal Control  | - Normal bone density & cortex   | - Emergency trauma radiographs |
|                              | - Verified radiologist sign-off: |   with acute/occult fractures  |
|                              |   "No significant bone abnormality"| - Metabolic bone disease     |
|                              | - Anatomically matched to cases  | - Systemic active malignancy   |
+------------------------------+----------------------------------+--------------------------------+
```

---

## 4. Required Imaging Modalities and Technical Specifications

### Modality Pathway A: Plain Digital Radiography (X-Ray)
- **Target Applications**: Paget's disease, Osteogenesis Imperfecta, Healthy skeletal controls, and plain-film visible metastases.
- **Detector Types**: Direct Digital Radiography (DR) or Computed Radiography (CR).
- **Spatial Resolution**: Minimum $1024 \times 1024$ pixels; preferred $\ge 2048 \times 2048$ pixels.
- **Bit Depth**: 12-bit to 16-bit grayscale DICOM; lossless PNG export acceptable for secondary deep learning pipelines.
- **Anatomical Projections**: Standard orthogonal views (Anteroposterior [AP] and Lateral); pelvis AP; skull AP/lateral; bilateral extremity surveys.

### Modality Pathway B: Planar Bone Scintigraphy
- **Target Applications**: Secondary bone metastases and whole-body metabolic controls.
- **Radiotracer**: $^{99\text{m}}\text{Tc}$-Methylene Diphosphonate ($^{99\text{m}}\text{Tc}$-MDP).
- **View Configuration**: Paired Anterior and Posterior whole-body planar scans.
- **Matrix Dimensions**: Standard $1024 \times 256$ or $512 \times 512$ acquisition matrix.

### Modality Pathway C: Volumetric Computed Tomography (CT)
- **Target Applications**: High-resolution 3D vertebral and axial skeletal metastasis assessment.
- **Reconstruction Kernel**: Bone algorithm (high-spatial-frequency edge enhancement).
- **Slice Thickness**: $\le 1.5\text{ mm}$ for axial volumetric series; bone windowing (Window Width $\approx 2000\text{ HU}$, Window Level $\approx 400\text{ HU}$).

---

## 5. Clinical and Demographic Metadata Requirements

Every imaging study ingested under this protocol must be accompanied by structured, de-identified clinical metadata:
1. **Subject Hash**: Cryptographically salted irreversible pseudonym (e.g., `SUBJ_PAG_0042`).
2. **Demographics**: Age at acquisition (in years; categorized $\ge 89$ per HIPAA guidelines), Biological Sex.
3. **Anatomical Site**: Skeletal region (Skull, Cervical Spine, Thoracic Spine, Lumbar Spine, Pelvis, Femur, Tibia, Humerus, Forearm, Hand, Foot).
4. **Diagnostic Ground Truth**:
   - Reference Standard Tier: Tier 1 (Histopathology/Biopsy), Tier 2 (Multi-modal Imaging Consensus + Clinical Follow-up), Tier 3 (Biochemical Marker + Radiologist Consensus).
   - Primary Malignancy Site (if metastatic): Breast, Prostate, Lung, Kidney, Thyroid, Other.
   - Sillence Classification (if OI): Type I, II, III, IV, or Undifferentiated.
   - Serum Total Alkaline Phosphatase / BSAP (if Paget's): IU/L and reference range.

---

## 6. Patient De-Identification and Privacy Governance

To ensure uncompromising patient privacy, all imaging and clinical data must undergo dual-stage de-identification compliant with the **HIPAA Privacy Rule Safe Harbor Method (45 CFR § 164.514(b)(2))** and the **European Union General Data Protection Regulation (GDPR Article 89(1))**:

1. **DICOM Header Cleansing**:
   - Application of the DICOM PS 3.15 Annex E Basic Application Level Confidentiality Profile.
   - Complete purging of all 18 HIPAA direct identifiers: Patient Name (`(0010,0010)`), Patient ID (`(0010,0020)`), Patient Birth Date (`(0010,0030)`), Institution Name (`(0008,0080)`), Physician Names, Accession Numbers, Device Serial Numbers.
   - Date shifting: Study dates shifted by a random patient-specific negative offset (1 to 365 days) while strictly preserving chronological intervals between serial studies.
2. **Pixel Burn-in Scrubbing**:
   - Automated high-sensitivity OCR scanning across all image margins to detect and redact burned-in patient demographic text, institutional watermarks, and acquisition timestamps.
   - Secondary manual verification by a certified imaging technician.
3. **Pseudonymization Key Custody**:
   - Re-identification mapping keys remain under the exclusive physical and administrative control of the contributing hospital's Honest Broker / Data Trustee. The machine learning research team shall have zero access to the re-identification key.

---

## 7. Patient-Level Data Partitioning Governance

To eliminate patient-level data leakage (intra-patient correlation across multi-view, longitudinal, or bilateral scans):
- **Unit of Partitioning**: The unique **Patient/Subject ID** is the atomic partition unit.
- **Split Proportions**: 70% Training, 15% Tuning/Validation, 15% Internal Test.
- **All-or-Nothing Rule**: All images, views, anatomical regions, and timepoints belonging to a single subject must reside exclusively within one partition.
- **External Multi-Center Isolation**: External hospital cohorts are strictly reserved as holdout test sets and shall never be exposed to model development, feature selection, hyperparameter tuning, or threshold calibration.

---

## 8. Data Storage, Security, and Access Controls

- **Encryption Standards**: AES-256 encryption for all data at rest on secure network-attached storage (NAS); TLS 1.3 encryption for all data in transit.
- **Physical Isolation**: GPU training and analysis conducted on air-gapped institutional workstations or dedicated virtual private clouds (VPC) with disabled public endpoints.
- **Role-Based Access Control (RBAC)**: Access granted strictly to authorized project investigators who have executed institutional data-use agreements and completed human research subject protections certification (e.g., CITI Program).
- **Audit Logging**: Immutable, append-only access and modification logs recording user ID, timestamp, and accessed image files.

---

## 9. Data Retention and Deletion Policy

- **Retention Duration**: Research data retained for 5 years following formal publication of research results, in accordance with standard institutional research data retention guidelines.
- **Cryptographic Purging**: Upon protocol termination or formal request by a contributing institution, data must undergo permanent, irreversible cryptographic wiping compliant with **NIST Special Publication 800-88 Revision 1 (Guidelines for Media Sanitization)**.
- **No Commercial Retention**: Explicit prohibition of dataset transfer, sale, sublicensing, or incorporation into commercial foundation models without independent, explicit ethics board approval.

---

## 10. Research-Only Intent and Clinical Safety Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL RESEARCH USE ONLY**:  
> The OsteoGuard AI software system, algorithms, feature extractors, and predictive models are experimental research artifacts designed exclusively for academic evaluation and clinical research feasibility studies.  
> 1. The software is **NOT** a certified medical device and has **NOT** been evaluated, approved, or cleared by the United States Food and Drug Administration (FDA), the European Medicines Agency (EMA), the Central Drugs Standard Control Organisation (CDSCO), or any other regulatory body.  
> 2. The software must **NEVER** be used as a primary diagnostic tool, autonomous triage system, or clinical decision support system for live clinical patient management.  
> 3. Licensed, board-certified radiologists, orthopaedic surgeons, and treating physicians retain sole and absolute clinical responsibility for patient diagnosis, treatment planning, and outcomes.

---

## 11. Ethical Approvals and Governance Placeholders

- **Institutional Review Board / Research Ethics Committee**:  
  `[PENDING ETHICS COMMITTEE APPROVAL - PROTOCOL TO BE FORMALLY SUBMITTED TO THE INSTITUTIONAL REVIEW BOARD / ETHICS COMMITTEE OF COLLABORATING INSTITUTIONS]`
- **Principal Investigator Endorsement**:  
  `[PRINCIPAL INVESTIGATOR SIGNATURE AND DATE PENDING FORMAL REVIEW]`
- **Departmental Reviewers**:  
  `[DEPARTMENT CHAIR OF RADIOLOGY SIGN-OFF PENDING]`  
  `[DEPARTMENT CHAIR OF ORTHOPAEDICS SIGN-OFF PENDING]`  
  `[DATA PROTECTION OFFICER / LEGAL COUNSEL APPROVAL PENDING]`
