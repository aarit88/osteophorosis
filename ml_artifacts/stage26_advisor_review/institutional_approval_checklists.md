# Institutional Approval and Data Access Checklists

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Operational Governance Checklists  
**Date**: November 16, 2026  

---

## CHECKLIST 1: Hospital Collaboration (Paget's Disease & Matched Healthy Controls)

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    HOSPITAL COLLABORATION CHECKLIST                                                |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| #  | Required Document / Step    | Responsible Role      | Dependencies            | Current Status                |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 1  | Protocol Finalization       | Faculty Advisor / PI  | Research Protocol       | REVISED DRAFT PREPARED        |
|    |                             | & Student Researcher  | (OG-PROT-2026-04C)      | (Awaiting Faculty Review)     |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 2  | Clinical Co-I Engagement    | Lead Musculoskeletal  | Faculty Outreach        | OUTREACH DRAFT PREPARED       |
|    |                             | Radiologist           | (Email Draft 2)         | (Draft Only - Not Transmitted)|
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 3  | Institutional Ethics (IEC)  | Appointed PI /        | Signed Protocol,        | PENDING FACULTY PI            |
|    | Application Submission      | Faculty Advisor       | Waiver Request Form     | APPOINTMENT                   |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 4  | Waiver of Informed Consent  | Institutional Ethics  | ICMR Guidelines (2017)  | AWAITING IEC REVIEW           |
|    | Consideration               | Committee (IEC)       | Section 5, Box 5.2      |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 5  | Data Use Agreement (DUA) /  | Hospital Legal / R&D  | IEC Approval Letter,    | PENDING IEC CLEARANCE         |
|    | Research Agreement          | Office & Institute    | Institutional DUA Form  |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 6  | Honest Broker PACS Search & | Hospital Radiology IT | Executed DUA,           | BLOCKED PENDING DUA           |
|    | ICD-10 Extraction (M88.x)   | / Honest Broker       | ICD-10 Search Protocol  |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 7  | DICOM PS 3.15 Cleansing &   | Hospital Honest Broker| Honest Broker Extraction| BLOCKED PENDING EXTRACTION    |
|    | Pixel Burn-in Redaction     | & Radiology QC        | Protocol                |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 8  | Radiologist Ground-Truth    | Musculoskeletal       | De-Identified DICOM     | BLOCKED PENDING DE-IDENTIF.   |
|    | Review & Sign-Off Form      | Radiologist Co-I      | Series                  |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
```

---

## CHECKLIST 2: RDCRN / Brittle Bone Disorders Consortium Access (Human Pediatric OI)

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    RDCRN BBDC PROTOCOL 7701 ACCESS CHECKLIST                                       |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| #  | Required Document / Step    | Responsible Role      | Dependencies            | Current Status                |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 1  | Research Plan Preparation   | Faculty Advisor / PI  | BBDC Protocol 7701 Scope| PREPARED IN PACKAGE 1         |
|    |                             | & Student Researcher  | (Hand/Wrist Bone Age)   | (Awaiting Faculty Review)     |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 2  | Home Institution IEC        | Appointed PI /        | Home Institution Ethics | PENDING FACULTY PI            |
|    | Clearance / Waiver Letter   | Faculty Advisor       | Review Protocol         | APPOINTMENT                   |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 3  | RDCRN DMCC Data Access      | Appointed PI /        | Endorsed Research Plan, | FORMAL DRAFT PREPARED         |
|    | Committee Application       | Faculty Advisor       | CITI Ethics Certs       | (Draft Only - Not Submitted)  |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 4  | BBDC Data Access Committee  | RDCRN BBDC Data Access| Submitted Application   | AWAITING SUBMISSION           |
|    | (DAC) Formal Review         | Committee             | Dossier                 |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 5  | RDCRN Data Use Agreement    | Institutional Legal   | DAC Approval Letter     | PENDING DAC APPROVAL          |
|    | Execution                   | Counsel & RDCRN DMCC  |                         |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 6  | Secure Transfer & Staging   | Lead ML Researcher /  | Executed RDCRN DUA,     | BLOCKED PENDING DUA           |
|    | to Air-Gapped GPU Cluster   | Institutional IT      | Encrypted Storage       |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
```

---

## CHECKLIST 3: Stanford AIMI MURA Access (Upper-Extremity Healthy Controls)

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    STANFORD AIMI MURA ACCESS CHECKLIST                                             |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| #  | Required Document / Step    | Responsible Role      | Dependencies            | Current Status                |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 1  | Academic User Registration  | Lead ML Researcher /  | Institutional Email,    | READY FOR FACULTY             |
|    |                             | Faculty Advisor       | Advisor Supervision     | ENDORSEMENT                   |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 2  | Stanford AIMI Research DUA  | Appointed PI /        | Non-Commercial Academic | PREPARED IN PACKAGE 4         |
|    | Terms Acceptance            | Faculty Advisor       | Research Commitment     | (Awaiting Registration)       |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 3  | Normal Study Subset Filter  | Lead ML Researcher    | MURA Metadata CSV       | SPECIFICATION READY           |
|    | (9,045 Normal Studies Only) |                       | (Study-Level Labels)    | (Abnormal Studies Excluded)   |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 4  | Upper-Extremity Scoping     | Lead ML Researcher    | Multi-View Directory    | SPECIFICATION READY           |
|    | (7 Anatomical Regions Only) |                       | Layout                  | (Upper Extremity Only)        |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 5  | Data Staging & Checksum     | Lead ML Researcher /  | Approved Registration,  | BLOCKED PENDING AIMI DUA      |
|    | Verification                | Institutional IT      | Secure Local Storage    |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
```

---

## CHECKLIST 4: The Cancer Imaging Archive (TCIA) Access (Spine-Mets-CT-SEG)

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    TCIA SPINE-METS-CT-SEG ACCESS CHECKLIST                                         |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| #  | Required Document / Step    | Responsible Role      | Dependencies            | Current Status                |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 1  | TCIA Account Registration   | Lead ML Researcher    | Institutional Academic  | READY FOR REGISTRATION        |
|    |                             |                       | Credentials             |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 2  | NBIA Data Retriever Client  | Lead ML Researcher /  | Java Runtime / NBIA     | CONFIGURATION SPECIFIED       |
|    | Configuration               | System Admin          | CLI Package             |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 3  | TCIA Data Use Terms & CC    | Appointed PI /        | Collection DOI          | PREPARED IN PACKAGE 3         |
|    | License Acknowledgment      | Faculty Advisor       | (10.7937/kh36-ds04)     | (Awaiting Setup)              |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 4  | Volumetric Series Download  | Lead ML Researcher    | Verified .tcia Manifest | BLOCKED PENDING SETUP         |
|    | (55 Subjects, ~20.36 GB)    |                       | (Restricted Staging)    |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
| 5  | NIfTI / DICOM-SEG Lesion    | Lead ML Researcher    | CT Axial Reconstruction | BLOCKED PENDING DOWNLOAD      |
|    | Mask Verification           |                       | Series                  |                               |
+----+-----------------------------+-----------------------+-------------------------+-------------------------------+
```
