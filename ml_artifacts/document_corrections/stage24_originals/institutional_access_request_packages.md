# Institutional Access Request Packages

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Prepared Formal Access Request Packages  
**Author**: OsteoGuard AI Regulatory and Clinical Governance Office  

> [!NOTE]
> These packages represent completed, structured formal application templates. They are ready for institutional sponsor sign-off and formal submission. **No applications have been automatically submitted or transmitted.**

---

## PACKAGE 1: Rare Diseases Clinical Research Network (RDCRN) / Brittle Bone Disorders Consortium (BBDC)

### 1. Administrative Overview
- **Data Repository**: RDCRN Data Management and Coordinating Center (DMCC) / BBDC Protocol 7701.
- **Reference Publication**: *Bone*, 2024;185:117192 (PMC11324408).
- **Target Target Cohort**: Longitudinal human pediatric and adult Osteogenesis Imperfecta radiographic and clinical registry.
- **Requested Target Data**: De-identified plain skeletal radiographs (DICOM format) accompanied by Sillence classification, age, sex, fracture history, and *COL1A1*/*COL1A2* mutation status where available.

### 2. Research Purpose and Scientific Justification
To develop an objective, non-invasive deep learning and radiomic classifier capable of distinguishing Osteogenesis Imperfecta from secondary bone metastases, Paget's disease, and normal skeletal architecture. The research aims to establish automated quantitative biomarkers for bone fragility and low-trauma fracture risk stratification to assist pediatric radiologists and endocrinologists.

### 3. Proposed Data Safeguards
- Data will be stored on an air-gapped, encrypted research cluster (AES-256 at rest).
- Access restricted exclusively to named investigators who have completed human research subject ethics training.
- Zero linkage attempts to re-identify participants.
- Prohibition of data redistribution, secondary transfer, or public repository posting.

### 4. Key Questions to Resolve Prior to Submission
- What is the current turnaround time for RDCRN BBDC Data Access Committee (DAC) protocol review?
- Does the DMCC supply raw 16-bit DICOM pixel arrays, or are radiographs archived as secondary captures?
- Is institutional ethics committee (IRB) approval from the receiving institution required prior to DAC submission, or is provisional DAC review possible?

---

## PACKAGE 2: Hospital-Based Musculoskeletal Radiology Collaboration (Paget's Disease & Matched Controls)

### 1. Administrative Overview
- **Target Institution**: Academic Medical Center / University Hospital Department of Radiology.
- **Target Cohort**: Retrospective cohort of patients diagnosed with Paget's disease of bone (2010–2025) and anatomically matched healthy skeletal controls.
- **Requested Target Data**: De-identified digital radiographs (DICOM format) of the pelvis, spine, femur, skull, and tibia, linked with age, sex, serum alkaline phosphatase levels, and clinical confirmation notes.

### 2. Research Purpose and Scientific Justification
Paget's disease represents a critical diagnostic dilemma that can mimic sclerotic metastases (e.g., from prostate cancer) or lytic lesions. No public machine learning benchmark currently exists for Paget's disease. This collaboration will establish the first scientifically validated, multi-center AI evaluation benchmark for automated recognition of pagetic bone remodeling.

### 3. Collaborative Framework and Co-Authorship
- Departmental radiologists will serve as co-investigators, adjudicating ground truth labels and co-authoring high-impact clinical and machine learning publications.
- The hospital's Honest Broker will perform local extraction and pseudonymization; zero unprotected health information (PHI) will leave the hospital firewall.

### 4. Key Questions to Resolve Prior to Submission
- What specific PACS query tools are available to retrieve historical DICOM studies by ICD-10 code (`M88.0` to `M88.9`)?
- Does the institutional ethics committee grant a waiver of informed consent for retrospective anonymized imaging studies?
- What standard institutional Data Use Agreement (DUA) or Material Transfer Agreement (MTA) template does the hospital require?

---

## PACKAGE 3: The Cancer Imaging Archive (TCIA) Spine-Mets-CT-SEG Access

### 1. Administrative Overview
- **Collection Name**: Spine-Mets-CT-SEG (Computed Tomography Images with Segmentation Masks of Spine Metastases).
- **DOI**: `10.7937/kh36-ds04`
- **Data Volume**: 55 subjects, 108 CT series, ~20.36 GB uncompressed.
- **Requested Target Data**: Volumetric axial CT DICOM series with accompanying 3D voxel-level segmentation masks of spinal metastases.

### 2. Research Purpose and Scientific Justification
To benchmark the volumetric CT pathway (Pathway C) of OsteoGuard AI for automated detection, segmentation, and biomechanical fracture vulnerability scoring in metastatic spine disease.

### 3. TCIA Data Compliance and Attribution
- Strictly adhere to the TCIA Data Usage Policy and Creative Commons Attribution license terms.
- Provide full bibliographic citation to the collection authors and TCIA in all resultant publications.
- Use the TCIA NBIA Data Retriever with secure token-based authentication.

### 4. Key Questions to Resolve Prior to Submission
- Are the segmentation masks stored as standard DICOM Segmentation Objects (DICOM-SEG) or binary NIfTI (`.nii.gz`) files?
- Are contrast-enhanced and non-contrast CT series explicitly differentiated in the collection metadata?

---

## PACKAGE 4: Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI) MURA Access

### 1. Administrative Overview
- **Dataset**: Stanford AIMI MURA (Musculoskeletal Radiographs).
- **Cohort Scale**: 9,045 verified normal upper-extremity studies (40,561 multi-view radiographs).
- **Requested Target Data**: Verified normal studies across elbow, finger, forearm, hand, humerus, shoulder, and wrist.

### 2. Research Purpose and Scientific Justification
To supply the upper-extremity healthy skeletal control cohort for OsteoGuard AI, providing a rigorous normal baseline to prevent false-positive anomaly detections on arm and shoulder radiographs.

### 3. Compliance and Data Use Terms
- Execution of the Stanford AIMI Research Data Use Agreement.
- Strict restriction to non-commercial academic research.
- Maintenance of local secure storage with no public redistribution.

### 4. Key Questions to Resolve Prior to Submission
- Are study-level demographic variables (age, sex) fully linked in the latest release?
- What is the verified resolution and image compression applied to the downloadable release?
