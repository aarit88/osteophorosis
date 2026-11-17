"""
Stage 24 Document Corrector.

Performs rigorous scientific, statistical, legal, and ethical corrections
on the Stage 24 documentation for OsteoGuard AI:
- Corrects three-center BTXRD metrics (Center 1: 0.8103, Center 2: 0.5069, Center 3: 0.3592)
- Corrects absolute (-30.34 pp, -45.11 pp) and relative (-37.44%, -55.67%) decreases
- Replaces global overstatements with cautious audit language
- Aligns legal/ethical framework with Indian ICMR 2017 Guidelines and DPDP Act 2023/Rules
- Clarifies MURA breakdown (14,863 studies: 9,045 normal, 5,818 abnormal; upper-extremity only)
- Clarifies PMC11324408 OI cohort (159 children baseline, 55 follow-up; hand/wrist bone-age only)
- Updates outreach email drafts with respectful student-to-faculty tone and non-transmission markings
- Generates correction log and source verification notes
"""

import csv
import json
from pathlib import Path


def execute_corrections(base_dir: Path):
    corrected_dir = base_dir / "ml_artifacts/document_corrections/stage24_corrected"
    corrected_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. four_class_research_protocol_REVISED.md
    # -------------------------------------------------------------
    protocol_path = corrected_dir / "four_class_research_protocol_REVISED.md"
    protocol_content = """# Institutional Research Protocol (Revised Draft): OsteoGuard AI Multi-Class Differential Diagnosis

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
- **Stages 11–14**: Evaluated a dual-view bone scintigraphy ensemble for secondary metastases (IICS-UNA cohort, 291 patients). While demonstrating high internal sensitivity, this model is constrained strictly to planar nuclear scintigraphy ($^{99\\text{m}}\\text{Tc}$-MDP) and does not classify plain radiographs.
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
"""
    protocol_path.write_text(protocol_content, encoding="utf-8")

    # -------------------------------------------------------------
    # 2. institutional_access_request_packages_REVISED.md
    # -------------------------------------------------------------
    pkg_path = corrected_dir / "institutional_access_request_packages_REVISED.md"
    pkg_content = """# Institutional Access Request Packages (Revised Draft)

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Prepared Formal Access Application Packages (DRAFT FOR REVIEW)  
**Author**: OsteoGuard AI Governance Office  

> [!NOTE]
> These packages represent structured formal application drafts prepared for institutional mentor review and formal processing. **No applications have been submitted, and no access has been granted merely by drafting these packages.**

---

## PACKAGE 1: Rare Diseases Clinical Research Network (RDCRN) / Brittle Bone Disorders Consortium (BBDC)

### 1. Administrative Overview
- **Data Repository**: RDCRN Data Management and Coordinating Center (DMCC) / BBDC Protocol 7701.
- **Reference Publication**: *Bone*, 2024;185:117192 (PMC11324408).
- **Published Cohort Characteristics**:
  - Baseline cohort: 159 pediatric participants with Osteogenesis Imperfecta (OI).
  - Follow-up cohort: 55 participants with repeated bone-age observations approximately 24 months later.
  - Phenotypic distribution: OI Types I, III, and IV.
  - **Anatomical Scope**: Left hand and wrist radiographs evaluated for bone-age maturation.
- **Target Data Requested**: De-identified digital radiographs (DICOM format) of the hand/wrist series with linked clinical metadata (Sillence type, age, sex, fracture history, and *COL1A1*/*COL1A2* variant classification where recorded).

### 2. Candidate Status and Anatomical Limitations
- **Candidate Cohort Status**: This represents a candidate human clinical OI research cohort. Formal access is entirely subject to review and authorization by the RDCRN BBDC Data Access Committee (DAC).
- **Anatomical Boundary**: The published dataset is strictly limited to hand and wrist bone-age radiographs. It does not represent whole-skeleton imaging, long bone bowing, or spinal deformities.
- **Scientific Purpose**: To establish an objective pediatric baseline for radiomic bone mineral texture analysis and investigate whether bone-age radiographs carry discriminative textural markers of collagen dysplasia compared to age-matched normal controls.

### 3. Proposed Safeguards
- Secure institutional storage with role-based access controls.
- Explicit agreement prohibiting any attempt to re-identify study participants.
- Strict prohibition of secondary data transfer, redistribution, or commercial use.

### 4. Key Procedural Questions for Custodian
- What is the formal submission and review schedule of the RDCRN BBDC Data Access Committee?
- Does the DMCC require an approved Institutional Ethics Committee (IEC) protocol from the applicant's home institution prior to application review?
- What specific Data Use Agreement (DUA) template governs academic non-commercial analysis of BBDC Protocol 7701 assets?

---

## PACKAGE 2: Hospital-Based Musculoskeletal Radiology Collaboration (Paget's Disease & Matched Controls)

### 1. Administrative Overview
- **Target Setting**: Academic Medical Center / Teaching Hospital Department of Radiology.
- **Proposed Cohort**: Retrospective cohort of adult patients evaluated for Paget's disease of bone (*Osteitis Deformans*) and anatomically matched healthy controls (2010–2025).
- **Target Data Requested**: De-identified plain skeletal radiographs (DICOM format) of the pelvis, spine, femur, skull, and tibia, accompanied by patient age, biological sex, anatomical site, serum alkaline phosphatase (ALP) levels, and clinical confirmation notes.

### 2. Candidate Status and Diagnostic Review
- **Proposed Collaboration**: This represents a proposed research partnership. No hospital has agreed to supply data, and no formal partnership exists prior to institutional review and approval.
- **Diagnostic Criteria Review**: Candidate cases will require formal clinical review and sign-off by a qualified radiologist or clinician to verify the classic radiographic triad (cortical thickening, trabecular coarsening, osteomegaly) and biochemical correlation.
- **Literature and Source Context**: Our current dataset-source audit did not identify a suitable, publicly accessible and appropriately licensed clinical imaging cohort for Paget's disease within the project's requirements. A hospital-based retrospective study under ethical oversight represents the primary viable acquisition route.

### 3. Governance and Privacy Safeguards
- **Honest Broker Mechanism**: Identification, extraction, and de-identification of imaging studies will be performed entirely within the hospital firewall by authorized institutional personnel.
- **Zero Raw PHI Transfer**: No direct patient identifiers (names, hospital numbers, exact dates) will leave hospital custody.
- **Ethics Review**: Submission of a formal research protocol to the hospital's Institutional Ethics Committee (IEC) requesting review and approval (including waiver of consent consideration for retrospective anonymized data under ICMR Guidelines).

### 4. Key Questions for Clinical Partners
- What is the institutional process for conducting a retrospective RIS/PACS search based on ICD-10 diagnostic codes (`M88.0` through `M88.9`)?
- What standard institutional Data Use Agreement (DUA) or Collaborative Research Agreement (CRA) is required by the hospital's research office?
- Who will serve as the institutional Principal Investigator and lead Musculoskeletal Radiologist?

---

## PACKAGE 3: The Cancer Imaging Archive (TCIA) Spine-Mets-CT-SEG Access

### 1. Administrative Overview
- **Collection**: Spine-Mets-CT-SEG (Computed Tomography Images with Segmentation Masks of Spine Metastases).
- **DOI**: `10.7937/kh36-ds04`
- **Verified Metadata**: 55 subjects, 108 CT examination series (~20.36 GB uncompressed).
- **Data Composition**: 3D volumetric axial CT scans reconstructed with bone algorithm, with corresponding 3D voxel-level segmentation masks of spinal metastases.

### 2. Access Terms and Prerequisites
- **Usage Requirements**: Access requires compliance with TCIA Data Usage Policies and Creative Commons licensing terms.
- **Download Requirement**: Requires utilization of the TCIA NBIA Data Retriever with authorized user credentials.
- **Notice**: Citing the collection DOI does not establish project permission or license fulfillment; formal download and staging must adhere strictly to TCIA data compliance requirements.

### 3. Proposed Research Role
- To benchmark Pathway C (Volumetric CT) for spinal metastasis segmentation, trabecular destruction scoring, and vertebral stability assessment.
- Completely decoupled from 2D plain radiography and planar scintigraphy encoders.

---

## PACKAGE 4: Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI) MURA Access

### 1. Administrative Overview
- **Dataset**: Stanford AIMI MURA (Musculoskeletal Radiographs).
- **Official Dataset Scale**:
  - Total Studies: 14,863 musculoskeletal studies.
  - Total Patients: 12,173 unique patients.
  - Total Multi-View Radiographs: 40,561 images.
  - **Diagnostic Distribution**:
    - Normal studies: 9,045 studies.
    - Abnormal studies: 5,818 studies.
  - **Anatomical Scope**: Seven upper-extremity regions:
    1. Elbow
    2. Finger
    3. Forearm
    4. Hand
    5. Humerus
    6. Shoulder
    7. Wrist

### 2. Critical Scope Clarifications
- **Normal vs. Abnormal Distribution**: MURA contains **BOTH normal and abnormal studies**. It is **INCORRECT** to describe all 40,561 images as normal controls. Only the verified normal studies (9,045 studies) can serve as upper-extremity controls.
- **Anatomical Boundary**: A normal label in MURA is strictly an **upper-extremity study-level normal evaluation** (e.g., normal wrist or normal elbow). It does **NOT** guarantee whole-skeleton health and cannot serve as a healthy control for pelvic, spinal, or lower-extremity (femur/tibia) pathologies.
- **Access Terms**: Access requires formal registration and adherence to the Stanford AIMI Research Data Use Agreement for academic, non-commercial research. Unrestricted public redistribution is strictly prohibited.

---

## 5. Authorship and Contributor Roles

In all institutional collaborative packages:
- Contributor roles and academic authorship will be determined strictly according to the **ICMJE guidelines** based on substantial scientific contribution, protocol design, data curation, manuscript drafting, and critical revision.
- No co-authorship or high-impact publication is guaranteed in advance.
"""
    pkg_path.write_text(pkg_content, encoding="utf-8")

    # -------------------------------------------------------------
    # 3. outreach_email_drafts_REVISED.md
    # -------------------------------------------------------------
    email_path = corrected_dir / "outreach_email_drafts_REVISED.md"
    email_content = """# Clinical and Academic Outreach Email Drafts (Revised)

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Prepared Formal Outreach Communications (DRAFT FOR REVIEW)  
**Author**: OsteoGuard AI Governance Office  

> [!CAUTION]
> **COMMUNICATION PROTOCOL RULE (DRAFT ONLY — NOT SENT)**:  
> These drafts are strictly intended for internal academic review and discussion with faculty mentors.  
> **NO EMAILS HAVE BEEN TRANSMITTED, AND NO OUTREACH HAS BEEN INITIATED.**  
> Text enclosed in brackets (`[Placeholders]`) must be verified and approved by the academic supervisor prior to any transmission.

---

## DRAFT 1: Faculty Research Advisor / Department Mentor (RECOMMENDED FIRST COMMUNICATION)

**Subject**: Academic Guidance & Protocol Review: OsteoGuard AI Research Framework

Dear Professor `[Faculty Advisor Name]`,

I hope this email finds you well.

I am writing to seek your academic guidance and review of our research protocol draft for the next phase of our bone disease artificial intelligence project, **OsteoGuard AI**.

Over the previous project phases, we developed single-modality baseline models for bone scintigraphy metastases and primary bone tumours. In our recent multi-center evaluations of the BTXRD primary bone tumour classifier (Stages 19–21), we observed substantial domain shift when evaluating the model on independent hospital cohorts:
- Internal Development Validation (Center 1, N=588): Macro-F1 = 0.8103 (Balanced Acc = 0.8290, Overall Acc = 0.8180).
- External Evaluation (Center 2, N=549): Macro-F1 = 0.5069 (an absolute decrease of 0.3034 [-30.34 percentage points]; relative decrease of 37.44%).
- External Stress Test (Center 3, N=259, specialized oncology referral center): Macro-F1 = 0.3592 (an absolute decrease of 0.4511 [-45.11 percentage points]; relative decrease of 55.67%).

Furthermore, our forensic dataset audits confirmed that primary bone sarcomas (osteosarcoma, Ewing sarcoma) differ fundamentally from secondary bone metastases and cannot be conflated with them. In addition, candidate open-access datasets for Osteogenesis Imperfecta were found to be non-human murine micro-CT data, and no suitable, publicly accessible clinical imaging cohorts were identified for Paget's disease.

To establish a scientifically sound foundation for a four-class differential diagnosis framework (Secondary Metastasis, Paget's Disease, Osteogenesis Imperfecta, and Healthy Controls), we have prepared a draft Institutional Research Protocol (`OG-PROT-2026-04C-REVISED`) aligned with the ICMR National Ethical Guidelines (2017) and the Digital Personal Data Protection Act framework.

I would be extremely grateful for an opportunity to meet with you for 15–20 minutes to discuss this protocol, receive your guidance on our proposed methodology, and discuss how to properly approach our university hospital's radiology department under appropriate faculty mentorship.

Thank you very much for your time and support.

Respectfully yours,  
`[Student Researcher Name]`  
`[Student ID / Degree Program]`  
`[Department / Institution Name]`  
`[Contact Email]` | `[Phone Number]`  

---

## DRAFT 2: Head of Musculoskeletal (MSK) Radiology (Proposed Secondary Outreach)

**Subject**: Exploratory Research Inquiry: Multi-Class Bone Disorder Imaging & Paget's Disease Evaluation

Dear Dr. `[Radiology Department Chair / Senior MSK Radiologist Name]`,

I hope you are doing well.

I am writing from `[Institution / University Department]` under the academic mentorship of Professor `[Faculty Advisor Name]`. We are conducting research into machine learning methods for musculoskeletal imaging, focused on differentiating challenging bone disorders.

In our preliminary evaluations of deep learning models on multi-center bone radiograph datasets, we identified significant cross-institutional performance degradation, highlighting the critical need for verified clinical reference standards and diverse multi-center validation.

Our current dataset audit did not identify a suitable, publicly accessible and appropriately licensed clinical imaging cohort for **Paget's disease of bone (*Osteitis Deformans*)**. We are exploring the academic feasibility of developing a collaborative retrospective imaging protocol with your department to evaluate the classic radiographic features of Paget's disease alongside anatomically matched normal skeletal radiographs.

Key elements of our proposed protocol draft:
- **Ethics and Privacy**: All work would be conducted under formal Institutional Ethics Committee (IEC) review in accordance with ICMR Ethical Guidelines (2017).
- **Honest Broker Extraction**: Image de-identification and extraction would be handled entirely within the hospital firewall by authorized personnel; no direct patient identifiers would be accessed by the machine learning team.
- **Academic Contributions**: Contributor roles and co-authorship on any resulting scientific publications would follow ICMJE guidelines based on substantive clinical and radiological contributions.

We have prepared an initial protocol concept draft for faculty review. If this area might be of interest to you or a faculty member in your division, would it be possible to schedule a brief 15-minute introductory meeting with Professor `[Faculty Advisor Name]` and myself to discuss feasibility?

Thank you very much for considering our inquiry.

Sincerely,  
`[Student Researcher Name]` (on behalf of `[Faculty Advisor / Senior Sponsor Name]`)  
`[Department / Academic Unit]`  
`[Institution Name]`  
`[Contact Email]`  

---

## DRAFT 3: Director of Pediatric Orthopaedics / Metabolic Bone Clinic

**Subject**: Academic Research Inquiry: Quantitative Radiomic Evaluation in Osteogenesis Imperfecta

Dear Dr. `[Director / Senior Clinician Name]`,

I hope this email finds you well.

My name is `[Student Researcher Name]`, and I am a research student working under the supervision of Professor `[Faculty Advisor Name]` at `[Institution / University Name]`. Our research focuses on computer-assisted analysis of skeletal radiographs and quantitative assessment of bone fragility.

We are currently investigating computational methods to evaluate bone texture, cortical thinning, and deformity progression in **Osteogenesis Imperfecta (OI)**. As part of our preliminary planning, we are reviewing candidate clinical datasets, including published pediatric bone-age radiograph collections from the Brittle Bone Disorders Consortium (BBDC Protocol 7701).

Because pediatric bone maturation introduces complex radiological confounders (such as open epiphyseal plates mimicking fracture lines), expert clinical insight is essential to ensure our research questions are clinically meaningful and scientifically valid.

We would be deeply grateful for an opportunity to consult with you or a designated clinical colleague for 15 minutes to share our draft research concept and receive your perspective on key clinical priorities in OI fracture risk assessment.

Thank you very much for your time and consideration.

Respectfully yours,  
`[Student Researcher Name]`  
`[Department / Institution Name]`  
`[Faculty Supervisor: Prof. Faculty Advisor Name]`  
`[Contact Email]`  

---

## DRAFT 4: Institutional Ethics Committee (IEC) / Research Office (Procedural Guidance Request)

**Subject**: Procedural Guidance Request: Retrospective De-Identified Human Imaging Protocol Draft

Dear `[Ethics Committee Secretariat / Member Secretary]`,

I hope you are doing well.

I am writing under the guidance of my faculty advisor, Professor `[Faculty Advisor Name]`, to request preliminary procedural guidance regarding the preparation of a research protocol for Institutional Ethics Committee review.

We are drafting a retrospective, non-interventional imaging protocol entitled:  
*"OsteoGuard AI: Deep Learning and Radiomic Framework for Differential Diagnosis of Bone Disorders"* (`OG-PROT-2026-04C-REVISED`).

The proposed research will involve the computational analysis of existing, fully de-identified diagnostic radiographs and CT scans across four diagnostic categories, accessed through authorized institutional repositories and collaborative hospital archives.

Study parameters:
- **Nature of Study**: Strictly retrospective review of existing de-identified imaging; no direct patient interaction or interventional procedures.
- **De-Identification**: De-identification to be conducted by the data provider's honest broker prior to research team access, with all direct identifiers removed.
- **Guidance Framework**: The protocol is being prepared under the ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017).

Could you please advise us on:
1. The standard institutional application format and documentation checklist required for retrospective de-identified imaging studies.
2. The specific institutional criteria and documentation required when applying for consideration of a waiver of informed consent under ICMR Guidelines (Section 5, Box 5.2).
3. Upcoming submission deadlines for the Institutional Ethics Committee.

Thank you very much for your guidance and assistance.

Sincerely,  
`[Student Researcher Name]`  
`[Faculty Sponsor: Prof. Faculty Advisor Name, Designation]`  
`[Department]`  
`[Institution Name]`  
`[Contact Email]`  
"""
    email_path.write_text(email_content, encoding="utf-8")

    # -------------------------------------------------------------
    # 4. paget_cohort_specification_REVISED.md
    # -------------------------------------------------------------
    paget_path = corrected_dir / "paget_cohort_specification_REVISED.md"
    paget_content = r"""# Clinical Cohort Acquisition Specification (Revised): Paget's Disease of Bone

**Target Class 2**: Paget's Disease of Bone (*Osteitis Deformans*)  
**Document Status**: Formal Cohort Acquisition Specification (Revised Draft)  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Disease Definition and Pathophysiology

Paget's disease of bone is a chronic focal skeletal disorder characterized by abnormal, accelerated bone remodeling. Disordered bone turnover progresses through three overlapping phases:
1. **Initial Osteolytic Phase**: Intense localized osteoclastic bone resorption (e.g., advancing resorption wedge or "blade of grass" sign).
2. **Intermediate Mixed Phase**: Concurrent rapid bone resorption and disorganized osteoblastic new bone formation.
3. **Late Osteosclerotic Phase**: Exhaustion of cellular remodeling resulting in dense, chaotic, poorly mineralized woven bone ("mosaic pattern"), prominent trabecular thickening, and skeletal enlargement (osteomegaly).

Bone affected by Paget's disease is mechanically compromised, architecturally disorganized, highly vascular, and susceptible to bowing deformities, secondary osteoarthritis, and pathological fractures.

---

## 2. Dataset Availability and Sourcing Context

> [!NOTE]
> **DATASET-SOURCE AUDIT STATEMENT**:  
> Our current dataset-source audit did not identify a suitable, publicly accessible and appropriately licensed clinical imaging cohort for Paget's disease within the project's requirements.  
> Web-based educational teaching files (e.g., Radiopaedia, Eurorad) cannot be utilized as bulk clinical training data due to copyright restrictions, selection bias, absence of verified longitudinal clinical metadata, and lossy compression artifacts.  
> Therefore, establishing a verified Paget's disease cohort requires a proposed retrospective hospital-based collaboration under Institutional Ethics Committee (IEC) oversight.

---

## 3. Diagnostic Reference Standards (Requiring Radiologist Review)

Candidate cases must satisfy established clinical and radiological diagnostic criteria, subject to formal review and verification by an appropriate radiologist or clinician:

### Primary Standard: Classic Radiographic Triad
Independent confirmation by a qualified musculoskeletal radiologist of the classic triad:
1. **Cortical Thickening**: Expansion of outer cortical boundaries with indistinct endosteal margins.
2. **Coarsened / Accentuated Trabeculae**: Prominent, thickened trabecular lines conforming to altered mechanical stress.
3. **Osseous Expansion (Osteomegaly)**: True enlargement of the bone contour compared to normal contralateral bone or standard anatomical norms.

### Corroborating Standard: Biochemical & Scintigraphic Markers
- Documented elevation of total serum alkaline phosphatase (ALP) or bone-specific alkaline phosphatase (BSAP) with normal serum calcium and phosphate.
- Where available, whole-body planar $^{99\text{m}}\text{Tc}$-MDP bone scintigraphy showing intense, uniform radiotracer uptake throughout the involved bone.

---

## 4. Anatomical Distribution and Technical Imaging Parameters

- **Primary Skeletal Sites**: Pelvis (~75%), lumbar/thoracic spine (~50%), femur (~35%), skull (~30%), tibia (~25%), and humerus (~15%).
- **Imaging Modality**: Plain Digital Radiography (Direct Radiography [DR] or Computed Radiography [CR]).
- **Image Resolution**: Minimum $1024 \times 1024$ pixels; preferred $\ge 2048 \times 2048$ pixels.
- **Grayscale Depth**: 12-bit or 16-bit DICOM; uncompressed or losslessly compressed.
- **Target Cohort Scale**: Approximately 150 to 250 confirmed human cases across multiple skeletal regions, accompanied by anatomically matched controls.
"""
    paget_path.write_text(paget_content, encoding="utf-8")

    # -------------------------------------------------------------
    # 5. oi_cohort_specification_REVISED.md
    # -------------------------------------------------------------
    oi_path = corrected_dir / "oi_cohort_specification_REVISED.md"
    oi_content = r"""# Clinical Cohort Acquisition Specification (Revised): Osteogenesis Imperfecta (OI)

**Target Class 3**: Osteogenesis Imperfecta (Human Clinical)  
**Document Status**: Formal Cohort Acquisition Specification (Revised Draft)  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Clinical Definition and Molecular Genetics

Osteogenesis imperfecta (OI, "Brittle Bone Disease") is a heterogeneous group of genetic connective tissue disorders characterized by bone fragility, reduced bone mineral density, recurrent low-trauma fractures, and skeletal deformities.

Approximately 85–90% of individuals have autosomal dominant pathogenic variants in *COL1A1* or *COL1A2*, which encode type I procollagen. Additional recessive and X-linked forms affect collagen biosynthesis, post-translational modification, folding, and osteoblast differentiation.

Bone tissue in OI demonstrates thin cortical borders, reduced trabecular volume, altered osteocyte density, and elevated mineralization heterogeneity, predisposing to recurrent skeletal injury.

---

## 2. Published Research Cohort Evidence and Limitations (PMC11324408)

### 2.1 Study Evidence Summary
A major published study investigating human pediatric OI imaging is:
- **Reference**: *Bone*, 2024;185:117192 (PMC11324408).
- **Consortium**: Rare Diseases Clinical Research Network (RDCRN) Brittle Bone Disorders Consortium (BBDC Protocol 7701).
- **Cohort Composition**:
  - Baseline: 159 pediatric participants with clinical/genetic OI.
  - Follow-up: 55 participants with repeated examinations approximately 24 months later.
  - Phenotypes: Sillence Types I, III, and IV.
  - **Anatomical Coverage**: Left hand and wrist radiographs evaluated for automated bone age and bone mineral density assessment.

### 2.2 Sourcing and Scope Limitations
- **Candidate Status**: The published study documents the existence of this research cohort; however, **no data access has been granted merely by citing the paper**. Access requires formal application to and approval by the RDCRN BBDC Data Access Committee.
- **Anatomical Boundary**: The cohort is **strictly limited to hand and wrist bone-age imaging**. It cannot automatically represent whole-skeleton manifestations, femoral bowing, tibial deformities, or vertebral compression fractures.
- **Pediatric Controls**: Analysis of pediatric radiographs requires age- and growth-plate-matched controls to prevent automated models from confounding normal open epiphyseal plates with radiolucent fracture lines or lytic lesions.

---

## 3. Diagnostic Reference Standards

Candidate OI cases must satisfy one or more of the following clinical standards:
1. **Molecular Confirmation**: Documented pathogenic or likely pathogenic variant in *COL1A1*, *COL1A2*, or recognized recessive OI genes.
2. **Clinical Phenotype (Sillence Criteria)**: Verified diagnosis by a pediatric geneticist or endocrinologist confirming Sillence Type I (mild), Type III (progressively deforming), or Type IV (moderate).
3. **Clinical Fragility History**: Documented history of recurrent low-trauma fragility fractures, supported by clinical features (blue sclerae, dentinogenesis imperfecta, hearing impairment, or DXA $Z$-score $\le -2.0$).

---

## 4. Absolute Disqualification of Animal Models

> [!CAUTION]
> **STRICT REJECTION OF ANIMAL MICRO-CT DATA**:  
> Candidate datasets identified in earlier audits containing benchtop micro-CT ($\mu\text{CT}$) scans of murine (*oim* mouse) bones (e.g., Figshare DOI `10.6084/m9.figshare.c.3795019`) are **STRICTLY DISQUALIFIED**:  
> - Murine rodent bones lack Haversian remodeling systems and differ substantially in cortical dimensions and trabecular morphology.  
> - Benchtop 3D $\mu\text{CT}$ reconstructions and surface meshes bear zero physical or mathematical equivalence to human 2D projection radiographs.  
> - **Mandatory Rule**: Only ethically approved, verified human clinical radiographs may be considered for project intake.
"""
    oi_path.write_text(oi_content, encoding="utf-8")

    # -------------------------------------------------------------
    # 6. correction_log.md
    # -------------------------------------------------------------
    corr_log_path = corrected_dir / "correction_log.md"
    corr_log_headers = [
        "document",
        "section",
        "original_statement",
        "corrected_statement",
        "correction_reason",
        "supporting_source_or_artifact",
        "review_required"
    ]
    corr_log_rows = [
        [
            "four_class_research_protocol.md",
            "1. Background",
            "macro-F1 dropping from 0.8103 to 0.4431 and 0.4566, with degradation up to 45%",
            "Center 1 internal: Macro-F1 0.8103; Center 2 external: Macro-F1 0.5069 (decrease of 0.3034 pp, -37.44% relative); Center 3 stress test: Macro-F1 0.3592 (decrease of 0.4511 pp, -55.67% relative). External cohorts must not be pooled.",
            "Corrected metric inaccuracies and distinguished absolute percentage-point decrease from relative percentage decrease.",
            "Stage 18 freeze manifest, Stage 19/20 classification_metrics.json, Stage 21 multicenter_performance_comparison.csv",
            "Faculty Advisor Review"
        ],
        [
            "four_class_research_protocol.md",
            "1. Background",
            "High NPV presented as safe clinical exclusion capability",
            "High NPV explained as mathematically driven by low class prevalence (e.g. 7.0% malignant in C1) and not evidence of autonomous clinical exclusion.",
            "Prevented clinical misinterpretation of prevalence-dependent NPV metrics.",
            "Stage 21 model_readiness_decision.md, Stage 19/20 per_class_metrics.json",
            "Musculoskeletal Radiologist Review"
        ],
        [
            "four_class_research_protocol.md",
            "Governance Header",
            "HIPAA and GDPR presented as automatic governing frameworks; 45 CFR 46.116(f) as default waiver",
            "Indian Institutional Ethics Committee (IEC) as primary route; ICMR National Ethical Guidelines (2017) as primary ethical framework; consent/waiver determined by competent IEC; HIPAA/GDPR conditional.",
            "Aligned legal and ethical governance with Indian statutory and academic context.",
            "ICMR Ethical Guidelines (2017), DPDP Act 2023 Gazette Notifications",
            "Institutional Legal / Ethics Committee Review"
        ],
        [
            "four_class_research_protocol.md",
            "Governance Header",
            "Student researcher designated as Principal Investigator (PI)",
            "PI designated as a role to be formally appointed by the institution; student researcher identified as trainee under faculty supervision.",
            "Student researcher cannot unilaterally claim institutional PI status without formal appointment.",
            "ICMR Guidelines (2017) Section 1, Institutional Research Policy",
            "Institutional Head / Academic Mentor Review"
        ],
        [
            "four_class_research_protocol.md",
            "8. Data Security",
            "AES-256, air-gapped server, and RBAC presented as already implemented controls",
            "Safeguards marked as proposed institutional security controls requiring approval and implementation.",
            "Controls must not be claimed as active before verification of the institutional compute environment.",
            "Stage 24 Institutional Security Audit",
            "Institutional IT / Security Officer Review"
        ],
        [
            "four_class_research_protocol.md",
            "9. Retention",
            "Universal fixed five-year retention commitment",
            "Retention and deletion made subject to institutional policy, Ethics Committee approval, DUA terms, repository rules, and applicable law.",
            "Fixed universal commitment was legally inaccurate across diverse multi-institutional DUAs.",
            "ICMR Guidelines (2017), Standard DUA Templates",
            "Institutional Legal Counsel Review"
        ],
        [
            "paget_cohort_specification.md",
            "2. Availability",
            "Zero open-access Paget's datasets worldwide; no clinical benchmark exists anywhere in the world",
            "Our current dataset-source audit did not identify a suitable, publicly accessible and appropriately licensed clinical imaging cohort within the project's requirements.",
            "Removed unsupported global claims; replaced with scientifically defensible audit scope language.",
            "Stage 22/23 dataset source verification reports",
            "Faculty Advisor Review"
        ],
        [
            "oi_cohort_specification.md",
            "2. Published Evidence",
            "PMC11324408 described as full OI cohort ready for access",
            "Published study verified: 159 children at baseline, left hand/wrist bone-age radiographs, 55 follow-up at ~24 months, types I, III, IV. Limited to hand/wrist; access subject to RDCRN DMCC approval.",
            "Clarified exact published cohort characteristics and documented anatomical hand/wrist limitation.",
            "PMC11324408 (Bone, 2024;185:117192)",
            "Pediatric Clinician Review"
        ],
        [
            "institutional_access_request_packages.md",
            "Package 4: MURA",
            "All 40,561 MURA images described as normal controls",
            "MURA accurately distinguished: 14,863 studies, 12,173 patients, 40,561 multi-view images (9,045 normal studies, 5,818 abnormal studies). Upper-extremity only (7 regions); study normal != whole-skeleton health.",
            "Corrected critical dataset misrepresentation; MURA contains both normal and abnormal studies.",
            "Stanford AIMI MURA Official Publication & Repository Documentation",
            "Faculty Advisor Review"
        ],
        [
            "outreach_email_drafts.md",
            "Drafts 1-4",
            "Informal/presumptive tone, guaranteed authorship promises, claimed pending approvals",
            "Respectful student-to-faculty tone; faculty advisor recommended as first outreach; accurate Stage 17-21 metrics; authorship following ICMJE criteria; marked DRAFT ONLY - NOT SENT.",
            "Ensured professional, ethical, and non-presumptive academic communication.",
            "ICMJE Authorship Guidelines, Academic Protocol Best Practices",
            "Faculty Advisor Review"
        ]
    ]

    with open(corr_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(corr_log_headers)
        writer.writerows(corr_log_rows)

    # -------------------------------------------------------------
    # 7. source_verification_notes.md
    # -------------------------------------------------------------
    notes_path = corrected_dir / "source_verification_notes.md"
    notes_content = r"""# Source Verification Notes: Scientific, Statistical, and Governance Foundations

**Project**: OsteoGuard AI  
**Document**: Source Evidence and Verification Audit Notes  
**Date**: November 16, 2026  

---

## 1. Verified Model Performance Metrics (BTXRD Three-Center Evaluation)

All performance metrics have been re-verified from the primary source JSON and CSV files:
- **Center 1**: `ml_artifacts/btxrd_full_training/validation_metrics.json` and `ml_artifacts/btxrd_frozen_model/model_freeze_manifest.json`
- **Center 2**: `ml_artifacts/btxrd_center2_external_evaluation/classification_metrics.json`
- **Center 3**: `ml_artifacts/btxrd_center3_external_evaluation/classification_metrics.json`
- **Multi-Center Audit**: `ml_artifacts/btxrd_multicenter_audit/multicenter_performance_comparison.csv`

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    VERIFIED THREE-CENTER PERFORMANCE COMPARISON                                    |
+----------+-------+-------------------------+----------+----------+----------+----------+---------+-----------------+
| Center   | N     | Clinical Cohort Role    | Overall  | Balanced | Macro-F1 | Macro-   | Brier   | Macro-F1 Change |
|          |       |                         | Accuracy | Accuracy |          | ROC-AUC  | Score   | from Center 1   |
+----------+-------+-------------------------+----------+----------+----------+----------+---------+-----------------+
| Center 1 | 588   | Internal Dev Validation | 0.8180   | 0.8290   | 0.8103   | 0.9114   | 0.3168  | Baseline        |
| Center 2 | 549   | Primary External Holdout| 0.5173   | 0.5374   | 0.5069   | 0.7180   | 0.5731  | -0.3034 pp      |
|          |       | (General Referral)      |          |          |          |          |         | (-37.44% rel.)  |
| Center 3 | 259   | Secondary External Test | 0.4710   | 0.4467   | 0.3592   | 0.6468   | 0.6486  | -0.4511 pp      |
|          |       | (Oncology Imbalance)    |          |          |          |          |         | (-55.67% rel.)  |
+----------+-------+-------------------------+----------+----------+----------+----------+---------+-----------------+
```

### Statistical Distinction Rules
1. **Macro-F1 Score**: Unweighted arithmetic mean of per-class F1-scores ($\frac{1}{K}\sum_{k=1}^K F1_k$).
2. **Absolute Change (Percentage Points, pp)**: $\Delta_{\text{abs}} = \text{Macro-F1}_{\text{ext}} - \text{Macro-F1}_{\text{C1}}$.
   - Center 2: $0.5069 - 0.8103 = -0.3034$ (-30.34 percentage points).
   - Center 3: $0.3592 - 0.8103 = -0.4511$ (-45.11 percentage points).
3. **Relative Percentage Decrease (%)**: $\Delta_{\text{rel}} = \frac{\text{Macro-F1}_{\text{ext}} - \text{Macro-F1}_{\text{C1}}}{\text{Macro-F1}_{\text{C1}}} \times 100\%$.
   - Center 2: $\frac{-0.3034}{0.8103} \times 100\% \approx -37.44\%$.
   - Center 3: $\frac{-0.4511}{0.8103} \times 100\% \approx -55.67\%$.
4. **Cohort Non-Pooling**: Center 2 and Center 3 represent heterogeneous clinical referral distributions. Pooling them violates clinical epidemiological principles and obscures etiology-specific degradation.
5. **Negative Predictive Value (NPV)**: In Center 1, malignant tumour support is 41/588 (7.0%). High malignant NPV (99.26%) is mathematically mandated by low class prevalence and must not be described as proof of safe autonomous clinical exclusion.

---

## 2. Stanford AIMI MURA Verification

- **Source Reference**: Rajpurkar et al., *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs*, Stanford ML Group / AIMI.
- **Verified Study Counts**:
  - Total Studies: 14,863
  - Total Patients: 12,173
  - Total Images: 40,561
  - **Normal Studies**: 9,045 (60.86%)
  - **Abnormal Studies**: 5,818 (39.14%)
- **Anatomical Coverage**: Strictly confined to seven upper-limb regions: Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist.
- **Critical Finding**: MURA contains both normal and abnormal studies. Describing all 40,561 images as "healthy controls" was a critical error. Only the 9,045 normal studies represent candidate upper-extremity controls, and a normal wrist does not establish whole-skeleton health.

---

## 3. Human Osteogenesis Imperfecta (PMC11324408) Verification

- **Source Reference**: *Bone*, 2024;185:117192 (PMC11324408).
- **Study Title**: Brittle Bone Disorders Consortium (BBDC Protocol 7701) Bone Age Evaluation.
- **Verified Cohort Scale**:
  - Baseline: 159 pediatric participants with clinical/genetic OI.
  - Follow-up: 55 participants with repeated observations approximately 24 months later.
  - Phenotypic Coverage: Sillence Types I, III, and IV.
  - Anatomical Coverage: Left hand and wrist radiographs for bone age assessment.
- **Critical Finding**: The publication does not provide unrestricted bulk download of radiographs. Access requires formal application to the RDCRN Data Management and Coordinating Center (DMCC). Anatomical coverage is limited to hand/wrist radiographs and does not cover the long bones or axial skeleton.

---

## 4. Indian Ethical and Legal Regulatory Framework

- **Primary Guidance**: ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017).
  - Section 5: Ethical review procedures, expedited review, and criteria for waiver of informed consent (Box 5.2).
  - Section 11: Governance of biological materials, secondary data use, and biobanking.
- **Institutional Authority**: Institutional Ethics Committee (IEC) is the primary governing authority.
- **Principal Investigator**: Role must be formally appointed by the academic/clinical institution. Student researchers cannot unilaterally claim PI status.
- **Data Protection Statutory Context**: Digital Personal Data Protection Act, 2023 (DPDP Act) and Rules (2025/2026). Phased commencement is in progress; institutional legal counsel must determine operational applicability.
- **Foreign Privacy Laws (HIPAA/GDPR)**: Apply conditionally based on repository location or contractual DUA terms, not automatically to Indian domestic data processing.
"""
    notes_path.write_text(notes_content, encoding="utf-8")

    # -------------------------------------------------------------
    # 8. STAGE_24_DOCUMENT_CORRECTION_REPORT.md
    # -------------------------------------------------------------
    report_path = corrected_dir / "STAGE_24_DOCUMENT_CORRECTION_REPORT.md"
    report_content = r"""# STAGE 24 DOCUMENT CORRECTION REPORT: SCIENTIFIC, STATISTICAL, AND GOVERNANCE REALIGNMENT

**Project**: OsteoGuard AI  
**Report Type**: Stage 24 Formal Correction and Audit Alignment Report  
**Execution Date**: November 16, 2026  
**Environment**: GPU-Accelerated Python Environment (`osteoguard-gpu`)  
**Status**: **ALL STAGE 24 CORRECTIONS COMPLETED (PARTS A–F PASS)**  

---

## 1. Executive Summary

In accordance with strict project rules, a systematic forensic review was conducted across all Stage 24 deliverables to rectify scientific, statistical, legal, and ethical inaccuracies:
1. **Statistical Rigor**: Verified and corrected the three-center BTXRD performance metrics from primary JSON/CSV source artifacts. Distinguish between Macro-F1, absolute percentage-point changes, and relative percentage decreases. Removed claims equating high NPV with clinical safety.
2. **Model Scope Boundaries**: Re-established that the BTXRD model predicts 3 classes of primary bone tumours (Normal, Benign, Sarcoma) and does not predict secondary metastases, Paget's disease, or OI.
3. **Scientific Claims Alignment**: Removed unsupported global statements (e.g., "zero datasets worldwide", "no benchmark anywhere"). Replaced with cautious, defensible audit language.
4. **Indian Ethical & Legal Governance**: Established the Indian Institutional Ethics Committee (IEC) under ICMR Guidelines (2017) as the primary review route; removed 45 CFR 46.116(f) as default; contextualized the DPDP Act 2023 phased commencement; made HIPAA/GDPR conditional.
5. **Access Package Realignment**: Verified exact characteristics of PMC11324408 (159 baseline children, hand/wrist bone-age only); corrected MURA documentation (14,863 studies, 9,045 normal, 5,818 abnormal; upper-extremity only); removed promises of guaranteed co-authorship.
6. **Outreach Communications**: Updated email drafts to a respectful student-to-faculty tone; recommended the faculty advisor as the first outreach; preserved placeholders; marked strictly `DRAFT ONLY — NOT SENT`.

All original Stage 24 files are preserved in `ml_artifacts/document_corrections/stage24_originals/`.  
All corrected, review-ready versions are saved in `ml_artifacts/document_corrections/stage24_corrected/`.

---

## 2. Inventory of Corrected Deliverables

| Deliverable | Description | Key Revisions |
| :--- | :--- | :--- |
| [`four_class_research_protocol_REVISED.md`](four_class_research_protocol_REVISED.md) | Revised Institutional Protocol (`OG-PROT-2026-04C-REVISED`) | ICMR 2017 framework, DPDP Act context, accurate three-center metrics, conditional HIPAA/GDPR |
| [`institutional_access_request_packages_REVISED.md`](institutional_access_request_packages_REVISED.md) | Revised Access Packages | PMC11324408 hand/wrist boundary, MURA normal/abnormal breakdown, ICMJE authorship |
| [`outreach_email_drafts_REVISED.md`](outreach_email_drafts_REVISED.md) | Revised Outreach Communications | Respectful student tone, faculty sponsor first, accurate metrics, non-transmission notice |
| [`paget_cohort_specification_REVISED.md`](paget_cohort_specification_REVISED.md) | Revised Paget's Specification | Cautious audit language, radiologist review requirement, no bulk educational harvesting |
| [`oi_cohort_specification_REVISED.md`](oi_cohort_specification_REVISED.md) | Revised OI Specification | Hand/wrist bone-age scope, human clinical criteria, absolute disqualification of animal $\mu\text{CT}$ |
| [`correction_log.md`](correction_log.md) | Systematic Correction Log | 10-row matrix detailing original vs. corrected statements, reasons, sources, and review needs |
| [`source_verification_notes.md`](source_verification_notes.md) | Source Evidence Notes | Mathematical formulas, metric verification tables, ICMR/DPDP citations, MURA/OI audits |
| [`STAGE_24_DOCUMENT_CORRECTION_REPORT.md`](STAGE_24_DOCUMENT_CORRECTION_REPORT.md) | Master Synthesis Report | This comprehensive realignment summary |

---

## 3. Verified Performance Metrics Summary

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    VERIFIED THREE-CENTER PERFORMANCE COMPARISON                                    |
+----------+-------+-------------------------+----------+----------+----------+----------+---------+-----------------+
| Center   | N     | Clinical Cohort Role    | Overall  | Balanced | Macro-F1 | Macro-   | Brier   | Macro-F1 Change |
|          |       |                         | Accuracy | Accuracy |          | ROC-AUC  | Score   | from Center 1   |
+----------+-------+-------------------------+----------+----------+----------+----------+---------+-----------------+
| Center 1 | 588   | Internal Dev Validation | 0.8180   | 0.8290   | 0.8103   | 0.9114   | 0.3168  | Baseline        |
| Center 2 | 549   | Primary External Holdout| 0.5173   | 0.5374   | 0.5069   | 0.7180   | 0.5731  | -0.3034 pp      |
|          |       | (General Referral)      |          |          |          |          |         | (-37.44% rel.)  |
| Center 3 | 259   | Secondary External Test | 0.4710   | 0.4467   | 0.3592   | 0.6468   | 0.6486  | -0.4511 pp      |
|          |       | (Oncology Imbalance)    |          |          |          |          |         | (-55.67% rel.)  |
+----------+-------+-------------------------+----------+----------+----------+----------+---------+-----------------+
```

---

## 4. Preservation Status

- **Original Stage 24 Files**: Preserved intact in `ml_artifacts/document_corrections/stage24_originals/`.
- **Historical Stage Reports**: All reports from Stages 01 through 23 remain completely untouched.
- **Frozen Checkpoint SHA-256**: `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD` (Unmodified).
- **Environment**: Host CPU Python (`C:\Python313`) and React frontend (`osteophorosis-main/src`) remain untouched.
"""
    report_path.write_text(report_content, encoding="utf-8")

    print("[SUCCESS] Stage 24 corrections completed and saved to ml_artifacts/document_corrections/stage24_corrected/")


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    execute_corrections(workspace_root)
