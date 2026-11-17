# Dataset Access Request Templates and Data Governance Agreements

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 23 — Verified Dataset Sources, Access and Acquisition Gate  
**Artifact Path:** `ml_artifacts/dataset_source_verification/dataset_access_request_templates.md`  

---

## 1. Executive Notice & Ethical Directives

> [!CAUTION]
> **GOVERNANCE MANDATE — DRAFTS ONLY:**  
> In strict compliance with Stage 23 instructions:
> 1. These communication templates are **DRAFTS ONLY**.
> 2. **NO EMAILS HAVE BEEN SENT.**
> 3. **NO DATA USE AGREEMENTS HAVE BEEN SIGNED OR EXECUTED.**
> 4. **NO PROTECTED REPOSITORIES HAVE BEEN ACCESSED OR SCRAPED.**
> All communication requires explicit human investigator authorization, institutional legal counsel review, and relevant Institutional Review Board (IRB) approvals before transmission.

---

## 2. Request Template 1: Human Osteogenesis Imperfecta (RDCRN / BBDC)

### Target Institution / Portal
- **Consortium:** Brittle Bone Disorders Consortium (BBDC), Protocol 7701 (Natural History Study).
- **Data Coordinating Center:** Rare Diseases Clinical Research Network (RDCRN) Data Management and Coordinating Center (DMCC).
- **Contact Portal:** [https://www.rarediseasesnetwork.org/](https://www.rarediseasesnetwork.org/) / BBDC Steering Committee.

### Formal Data Request Draft

```text
Subject: Data Access Request: BBDC Protocol 7701 Radiographic Imaging for Academic Deep Learning Research

To: Brittle Bone Disorders Consortium (BBDC) Steering Committee & RDCRN DMCC
From: [Principal Investigator Name], [Academic Institution / Lab]
Date: [Date]

Dear BBDC Steering Committee and RDCRN Data Access Committee,

We are writing to formally submit a scientific data access request for de-identified radiographic imaging series collected under BBDC Protocol 7701 (as described in Bone 2024;185:117192, PMCID: PMC11324408).

1. Research Scope and Scientific Purpose:
Our academic research project, OsteoGuard AI, focuses on developing an open-source, non-commercial computer vision framework for automated differential diagnosis and biomechanical fracture risk stratification in metabolic and genetic skeletal disorders. Specifically, we are investigating convolutional neural representations to distinguish pediatric and adult osteogenesis imperfecta (Sillence Types I, III, IV) from healthy controls and focal bone neoplasms.

2. Requested Data Assets:
- De-identified planar skeletal radiographs (hand/wrist bone age, femur, tibia, spine) in native DICOM or lossless format.
- Associated minimal de-identified clinical covariates: subject ID, chronological age at scan, biological sex, confirmed genetic mutation (COL1A1/COL1A2), and Sillence classification.

3. Regulatory, IRB, and Data Security Compliance:
- Institutional Ethics: Our research protocol has been reviewed/exempted by [Institution] Institutional Review Board (Protocol #[IRB-XXXX]).
- Data Protection: All data will be housed on encrypted, access-restricted institutional servers behind multi-factor authentication. No data will be shared with commercial third parties.
- Research Data Use Agreement: We are prepared to execute the standard RDCRN Data Use Agreement (DUA) and commit to all required authorship attribution, consortium acknowledgments, and publication review timelines.

We appreciate your consideration of this request to advance computational radiomics for rare brittle bone disorders.

Sincerely,

[Principal Investigator Name, Title]
[Department / Institution]
[Institutional Contact Information]
```

---

## 3. Request Template 2: Paget's Disease of Bone (Academic Hospital MSK Radiology)

### Target Institution / Portal
- **Entity:** Academic Medical Center Department of Musculoskeletal Radiology / Metabolic Bone Clinic.
- **Reference Cohort:** Retrospective PACS search for confirmed Paget's disease (`ICD-10 M88.0`–`M88.9`).

### Formal Data Collaboration Request Draft

```text
Subject: Research Collaboration Request: Retrospective DICOM Radiography Cohort of Paget's Disease of Bone

To: Department of Musculoskeletal Radiology / Clinical Research Chair
From: [Principal Investigator Name], [Academic Institution / Lab]
Date: [Date]

Dear Dr. [Radiology Chair / Collaborator Name],

We are contacting you to propose an academic research collaboration between [Your Institution] and [Partner Medical Center] to establish a validated radiographic cohort of Paget's disease of bone (osteitis deformans) for deep learning differential diagnosis.

1. Background & Clinical Need:
Currently, zero standardized public machine-learning benchmarks exist for Paget's disease. Available online teaching files are compressed, non-standardized, and legally barred from bulk computational reuse. Consequently, clinical AI algorithms frequently misclassify Pagetic cortical thickening and trabecular remodeling as osteoblastic metastases or primary osteosarcomas.

2. Proposed Collaborative Protocol:
- Study Design: Retrospective, de-identified PACS cohort study.
- Cohort Definition: 100–200 adult patients with confirmed Paget's disease (biopsy or pathognomonic radiologist consensus plus elevated serum alkaline phosphatase) and matched normal skeletal controls.
- Imaging Modality: Native 16-bit DICOM planar radiographs (pelvis, skull, tibia, femur, lumbar spine).
- Metadata: De-identified patient key, age, sex, anatomic site, monostotic vs. polyostotic involvement, history of bisphosphonate therapy.

3. Governance, Authorship & Ethics:
- All patient scans will be de-identified at source using standard DICOM PS 3.15 Annex E anonymization pipelines prior to transfer.
- Our team will secure formal IRB protocol approval and execute an institutional Data Transfer Agreement (DTA).
- Contributing clinical faculty and radiologists will be included as co-authors on resulting peer-reviewed publications.

We would welcome an opportunity to schedule a brief teleconference to discuss protocol feasibility and data governance.

Warm regards,

[Principal Investigator Name, Title]
[Department / Institution]
[Institutional Contact Information]
```

---

## 4. Request Template 3: Stanford AIMI MURA Data Use Agreement Submission

### Target Institution / Portal
- **Institution:** Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI).
- **Portal:** [https://aimi.stanford.edu/datasets/mura-msk-xrays](https://aimi.stanford.edu/datasets/mura-msk-xrays).

### Submission Registration Specification

```text
Dataset Requested: MURA: Musculoskeletal Radiographs (v1.1)
Requested By: [Investigator Name], [Institution]
User Category: Academic Researcher (Verified .edu / Academic Email)

1. Proposed Non-Commercial Project Description:
OsteoGuard AI is an investigational academic research initiative developing multi-class differential diagnostic benchmarks for skeletal conditions. We request access to the MURA normal cohort (9,045 normal upper-extremity studies) to serve as a verified negative comparator for upper-extremity plain radiography.

2. Terms of Agreement Confirmation:
- The data will be utilized strictly for non-commercial academic research.
- No attempt will be made to re-identify any patient from study metadata.
- Data will not be redistributed, sublicensed, or uploaded to public repositories.
- Stanford University and the Stanford Machine Learning Group will be explicitly cited in all academic papers and presentations.
```

---

## 5. Technical Checksum and Secure Storage Protocol

Prior to ingesting any newly approved external cohort:
1. **Transfer Protocol:** Transfer exclusively via encrypted TLS 1.3 / SFTP or HTTPS authenticated tokens.
2. **Immediate Digest Computation:** Automatically compute SHA-256 hashes upon download completion:
   $$\text{SHA256}(f) = H(f)$$
3. **Verification against Manifest:** Validate the computed digest against the official source manifest before unzipping or parsing headers.
4. **Permanent Burned Holdout Sealing:** Newly acquired cohorts will be immediately assigned to training, validation, and sealed holdout partitions under `GroupKFold` by `patient_id`.

