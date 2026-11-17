# Institutional Access Request Packages (Revised Draft)

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
