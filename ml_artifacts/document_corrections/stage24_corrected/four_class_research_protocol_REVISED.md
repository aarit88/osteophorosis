# Institutional Research Protocol (Revised Draft): OsteoGuard AI Multi-Class Differential Diagnosis

**Protocol Identifier**: `OG-PROT-2026-04C-REVISED`  
**Protocol Version**: 2.0 (Revised Draft for Academic Guidance and Institutional Review)  
**Date**: November 16, 2026  
**Investigational System**: OsteoGuard AI Multi-Modal Deep Learning and Radiomic Framework  

> [!IMPORTANT]
> **GOVERNANCE & ETHICAL REVIEW STATUS**:  
> `[DRAFT FOR INSTITUTIONAL ETHICS COMMITTEE (IEC) REVIEW — PENDING ETHICAL CLEARANCE]`  
> `[PRINCIPAL INVESTIGATOR: ROLE TO BE FORMALLY APPOINTED BY THE INSTITUTION]`  
> `[STUDENT RESEARCHER / TRAINEE: PROPOSING INVESTIGATOR UNDER FACULTY ADVISORY OVERSIGHT]`  
> `[PRIMARY ETHICAL FRAMEWORK: ICMR NATIONAL ETHICAL GUIDELINES FOR BIOMEDICAL RESEARCH (2017)]`  
> *This document is an academic research protocol draft prepared for institutional guidance and preliminary review. No patient data has been collected, no ethical approval numbers have been created, and no clinical interventions will be performed under this draft.*

---

## 1. Project Background and Clinical Rationale

Musculoskeletal disorders exhibiting altered bone density, architectural distortion, and pathological fracture risk present significant diagnostic challenges. In routine clinical practice, focal and diffuse bone abnormalities may present with overlapping radiographic features:
1. **Secondary bone metastases** (osteolytic, osteoblastic, or mixed lesions arising from primary epithelial carcinomas such as breast, prostate, lung, and renal cell).
2. **Paget's disease of bone (*Osteitis Deformans*)** (characterized by disordered bone remodeling with cortical thickening, trabecular coarsening, and skeletal expansion).
3. **Osteogenesis imperfecta (OI)** (a genetic collagen-related dysplasia causing bone fragility, cortical thinning, recurrent low-trauma fractures, and skeletal deformities).
4. **Healthy skeletal variants and age-matched controls** (requiring robust discrimination to avoid false-positive triage).

### Findings and Limitations from Prior Project Stages
- **Stages 11–14**: Evaluated a dual-view bone scintigraphy ensemble for secondary metastases (IICS-UNA cohort, 291 patients). While demonstrating high internal sensitivity, this model is constrained strictly to planar nuclear scintigraphy ($^{99\text{m}}\text{Tc}$-MDP) and does not classify plain radiographs.
- **Stages 15–21**: Trained and evaluated a three-class primary bone tumour model on the BTXRD radiographic archive:
  - Target classes: `0 = Normal bone`, `1 = Benign primary bone tumour`, `2 = Primary malignant bone tumour`.
  - **Internal Development Validation (Center 1, N=588)**: Achieved Macro-F1 of **0.8103**, Balanced Accuracy of **0.8290**, Overall Accuracy of **0.8180**, Macro ROC-AUC of **0.9114**, and Brier Score of **0.3168**.
  - **Primary External Benchmark (Center 2, N=549)**: Evaluated on an independent external cohort without retraining. Macro-F1 dropped to **0.5069** (95% CI: [0.4648, 0.5494]), representing an **absolute decrease of 0.3034 (-30.34 percentage points)** and a **relative decrease of 37.44%**. Balanced Accuracy was 0.5374 and Overall Accuracy was 0.5173.
  - **Secondary External Stress Test (Center 3, N=259)**: Evaluated on a specialized oncology referral cohort with severe class imbalance (only 27 normal controls [10.42%], 201 benign lesions [77.61%], and 31 malignant tumours [11.97%]). Macro-F1 dropped to **0.3592** (95% CI: [0.3007, 0.4183]), representing an **absolute decrease of 0.4511 (-45.11 percentage points)** and a **relative decrease of 55.67%** compared to Center 1.
  - **Cohort Independence Notice**: Center 2 and Center 3 represent distinct clinical referral populations and were evaluated separately; their results must not be pooled.
  - **Diagnostic NPV Interpretation**: High negative predictive value (NPV) observed in certain classes (e.g., Center 1 malignant NPV of 99.26%) is mathematically driven by low class prevalence (7.0% malignant prevalence in validation) and must not be interpreted as evidence of safe clinical exclusion or autonomous rule-out capability.
  - **Model Scope Boundary**: The BTXRD model classifies primary bone tumours (sarcomas such as osteosarcoma, chondrosarcoma, Ewing sarcoma). It does **NOT** predict secondary bone metastasis, Paget's disease, or Osteogenesis Imperfecta. Primary sarcomas differ fundamentally in genomics, histopathology, periosteal reactions, and epidemiology from secondary metastases, and the BTXRD model must never be presented as a completed four-class diagnostic tool.
- **Stages 22–23**: Audited candidate four-class datasets and discovered that online candidate OI datasets (e.g., Figshare) were non-human murine micro-CT scans, while candidate metastasis Zenodo uploads were duplicates of existing in-tree scintigraphy data.

Consequently, establishing a true four-class differential diagnostic model requires prospective institutional collaboration to acquire verified, ethically cleared human clinical imaging cohorts.

---

## 2. Research Objectives

### Primary Objective
To construct, train, and independently validate a modality-decoupled machine learning framework to evaluate four discrete target classes:
- **Class 1**: Secondary bone metastasis (from known primary epithelial solid tumours)
- **Class 2**: Paget's disease of bone (*Osteitis Deformans*)
- **Class 3**: Osteogenesis imperfecta (verified human clinical cases)
- **Class 4**: Healthy skeletal controls (anatomically and demographically matched)

### Secondary Objectives
1. Investigate radiomic texture metrics and architectural descriptors for quantitative assessment of biomechanical fracture vulnerability.
2. Establish a clinically interpretable late-fusion decision architecture integrating plain digital radiography, planar bone scintigraphy, and volumetric CT, without forcing incompatible physical modalities into a single encoder.
3. Quantify domain shift across independent clinical centers through rigorous external evaluation.

---

## 3. Ethical and Legal Governance Framework (Indian Context)

This research protocol is designed within the Indian academic research framework and adheres to the following ethical and legal standards:

### 3.1 Primary Ethical Review Route
- Primary ethical review and oversight will be conducted by a recognized **Institutional Ethics Committee (IEC)** registered with the relevant Indian regulatory authorities (e.g., Department of Health Research / CDSCO where applicable).
- The study protocol will follow the **ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017)**, specifically Section 5 (Ethical Review Procedures) and Section 11 (Biological Materials, Biobanking and Data).
- The **Principal Investigator (PI)** role will be formally designated and appointed by the collaborating academic/clinical institution. Student researchers serve as research trainees under authorized faculty and PI mentorship.

### 3.2 Consent and Waiver of Consent Governance
- Consent or waiver of consent, where applicable, will be determined by the competent Institutional Ethics Committee under the applicable ethical framework and institutional requirements.
- Under ICMR Guidelines (2017), Section 5 (Box 5.2), an Institutional Ethics Committee may consider granting a waiver of informed consent for retrospective studies involving existing, anonymized/de-identified data or records, provided the research involves no more than minimal risk, the waiver will not adversely affect the rights and welfare of participants, and the research cannot practicably be conducted without the waiver.
- **No waiver is assumed or guaranteed.** Application for ethical review and waiver consideration will be formally submitted to the IEC.

### 3.3 Indian Digital Personal Data Protection Act (DPDP Act, 2023) and Rules
- The **Digital Personal Data Protection Act, 2023** provides the statutory framework for the processing of digital personal data in India.
- The Act and its associated Rules are subject to **phased commencement** by the Central Government via official Gazette notifications.
- Institutional legal and privacy personnel will verify which specific provisions of the DPDP Act and Rules apply to the proposed data processing at the time of study initiation and determine institutional compliance requirements.
- This protocol does not provide definitive legal advice; all data processing arrangements remain subject to institutional legal sign-off.

### 3.4 Conditional Applicability of Foreign Privacy Regulations (HIPAA / GDPR)
- **HIPAA (United States)**: Does not automatically govern Indian domestic research. HIPAA Privacy Rule Safe Harbor principles (45 CFR § 164.514(b)(2)) apply conditionally only when accessing US-based repositories (e.g., TCIA) or collaborating with US covered entities where required by contractual Data Use Agreements. Scrubbing 18 DICOM identifiers alone does not confer legal HIPAA compliance.
- **GDPR (European Union)**: Applies conditionally only where research involves personal data of individuals located within the European Economic Area or where European collaborative agreements stipulate GDPR adherence (Articles 6, 9, and 89).
- Compliance will be governed primarily by the applicable Indian statutory requirements, institutional data policies, and repository-specific Data Use Agreements.

---

## 4. Cohort Inclusion and Exclusion Criteria

```
+--------------------------------------------------------------------------------------------------+
|                                    INCLUSION & EXCLUSION MATRIX                                  |
+------------------------------+----------------------------------+--------------------------------+
| Target Class                 | Inclusion Criteria               | Exclusion Criteria             |
+------------------------------+----------------------------------+--------------------------------+
| 1. Secondary Bone Metastasis | - Histopathologically confirmed  | - Primary bone sarcoma         |
|                              |   solid primary epithelial tumor |   (osteosarcoma, Ewing, etc.)  |
|                              | - Documented skeletal metastasis | - Multiple myeloma / lymphoma  |
|                              |   via biopsy, cross-sectional    |   (unless segregated)          |
|                              |   imaging or clinical follow-up  | - Unconfirmed primary tumor    |
|                              | - Modality A (X-ray), B (Scan),  | - Severe motion artifact or    |
|                              |   or C (CT)                      |   corrupted DICOM files        |
+------------------------------+----------------------------------+--------------------------------+
| 2. Paget's Disease of Bone   | - Radiologist-confirmed Paget's  | - Isolated educational cases   |
|                              |   triad: cortical thickening,    |   without clinical provenance  |
|                              |   trabecular coarsening, expan-  | - Sclerotic metastases         |
|                              |   sion (osteomegaly)             |   mimicking Paget's            |
|                              | - Corroborated by serum alkaline | - Fibrous dysplasia or chronic |
|                              |   phosphatase where available    |   osteomyelitis mimics         |
|                              | - High-resolution digital X-rays |                                |
+------------------------------+----------------------------------+--------------------------------+
| 3. Osteogenesis Imperfecta   | - Verified human clinical case   | - Animal models (murine micro- |
|                              | - Sillence Types I, III, or IV   |   CT, rodent oim tibiae)       |
|                              | - Molecular COL1A1/COL1A2 status | - Non-OI pediatric osteopenia  |
|                              |   or clinical fragility triad    |   (hypophosphatasia, etc.)     |
|                              | - Plain skeletal radiographs     | - Non-accidental trauma cases  |
+------------------------------+----------------------------------+--------------------------------+
| 4. Healthy Skeletal Controls | - Verified radiologist sign-off: | - Emergency trauma radiographs |
|                              |   "No significant bone abnormal."|   with acute/healing fractures |
|                              | - Age- and site-matched to cases | - Known metabolic bone disease |
|                              | - Normal cortical thickness and  | - Active systemic malignancy   |
|                              |   intact trabecular architecture | - Severe osteoarthritis mimics |
+------------------------------+----------------------------------+--------------------------------+
```

---

## 5. De-Identification, Data Governance, and Security Controls

### 5.1 De-Identification Protocol
- De-identification must follow the data-providing institution's approved protocol under honest-broker supervision.
- Technical guidance: Application of DICOM PS 3.15 Basic Application Level Confidentiality Profile where feasible.
- Mandatory dual inspection:
  1. Automated cleansing of direct patient identifiers from DICOM metadata tags.
  2. Visual and OCR inspection of pixel arrays to detect and redact burned-in demographic text or institutional markings.
- Patient grouping: A deterministic, one-way pseudonymous grouping key (e.g., `SUBJ_PAG_0042`) will be assigned by the data provider's honest broker to preserve multi-view and longitudinal linkages.
- Key Custody: The re-identification key mapping will be maintained exclusively by the data provider's authorized custodian; research team members will have zero access to the key.
- Re-identification Risk: Technical de-identification alone does not constitute ethical or legal authorization; data transfer requires formal Institutional Ethics Committee review and a signed Data Use Agreement.

### 5.2 Proposed Institutional Data Security Controls
*(Subject to Institutional IT and Ethics Committee Approval Prior to Implementation)*
- **Data at Rest**: Proposed AES-256 encryption on dedicated institutional network storage.
- **Data in Transit**: Proposed TLS 1.3 encrypted secure file transfer protocols.
- **Access Control**: Role-based access control (RBAC) restricted to authorized research personnel who have completed human research ethics training.
- **Physical Environment**: GPU training conducted on institutional research compute infrastructure or verified dedicated workstations with restricted external connectivity.
- **Audit Logging**: Immutable access logging recording timestamp, user identity, and accessed datasets.

### 5.3 Data Retention and Disposal Policy
- Retention duration will NOT follow an arbitrary fixed commitment; it will strictly adhere to:
  1. The collaborating institution's research data retention policy.
  2. The terms approved by the Institutional Ethics Committee.
  3. The specific provisions of executed Data Use Agreements.
  4. Applicable statutory requirements.
- Upon completion of the approved research or termination of the agreement, data disposal will follow the media sanitization guidelines specified by the collaborating institution (e.g., NIST SP 800-88 Rev. 1 compliant cryptographic purge).

---

## 6. Patient-Level Data Partitioning Governance

To prevent data leakage caused by intra-patient correlation:
- **Unit of Partitioning**: The unique, pseudonymous **Patient/Subject ID** is the atomic partition unit.
- **Split Proportions**: Approximately 70% Training, 15% Validation, 15% Internal Test.
- **Strict Disjointness**: All images, anatomical projections, views, and longitudinal follow-up studies belonging to a single subject must reside exclusively within one partition.
- **External Multi-Center Isolation**: Independent hospital cohorts designated for external evaluation will remain completely untouched during model architecture design, feature extraction, hyperparameter tuning, and threshold selection.

---

## 7. Decoupled Modality Architecture Strategy

Due to fundamental differences in physical signal generation (X-ray photon transmission attenuation vs. gamma scintigraphic emission vs. volumetric CT Hounsfield Units), forcing multiple modalities into a single shared encoder is strictly rejected:
- **Pathway A (Plain Radiography)**: Evaluates 2D cortical and trabecular patterns (Paget's, OI, skeletal controls, plain film metastases).
- **Pathway B (Planar Bone Scintigraphy)**: Evaluates whole-body osteoblastic metabolic distribution (secondary metastases, metabolic controls) using dual-stream anterior/posterior architectures.
- **Pathway C (Volumetric CT)**: Evaluates 3D axial vertebral osteolysis and biomechanical stability.
- **Clinical Late-Fusion Layer**: Modality-specific posterior probabilities are integrated with standardized clinical covariates (age, sex, anatomical site, alkaline phosphatase) through a calibrated Bayesian meta-model. Missing modalities are handled via Bayesian marginalization.

---

## 8. Authorship and Scientific Contributions

- Contributor roles and co-authorship for any resulting academic publications will strictly follow the **International Committee of Medical Journal Editors (ICMJE)** criteria and applicable institutional guidelines:
  1. Substantial contributions to the conception or design of the work; or the acquisition, analysis, or interpretation of data.
  2. Drafting the work or revising it critically for important intellectual content.
  3. Final approval of the version to be published.
  4. Agreement to be accountable for all aspects of the work.
- No co-authorship or high-impact publication is guaranteed in advance; all contributions will be recognized based on actual scientific participation.

---

## 9. Investigational Use and Clinical Safety Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL RESEARCH USE ONLY**:  
> The OsteoGuard AI software system, algorithms, models, and documentation are experimental research artifacts developed exclusively for academic feasibility evaluation.  
> 1. The software is **NOT** a certified medical device and has **NOT** been evaluated, approved, or cleared by the Central Drugs Standard Control Organisation (CDSCO), the US FDA, the European EMA, or any statutory health authority.  
> 2. The software must **NEVER** be used as an autonomous diagnostic system, triage mechanism, or clinical decision tool for patient care.  
> 3. Licensed medical practitioners retain full and independent clinical responsibility for all diagnostic and therapeutic decisions.
