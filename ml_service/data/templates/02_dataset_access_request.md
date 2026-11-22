# Controlled Dataset Access Request (Template 02)

> **STATUS**: `DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`  
> **Target Repositories**: RDCRN DMCC / BBDC Protocol 7701 (OI), TCIA Spine-Mets-CT-SEG (`10.7937/kh36-ds04`), Stanford AIMI MURA (Upper-Extremity Healthy Comparator), or Institutional PACS Registry

---

## 1. Applicant & Protocol Metadata
- **Target Dataset / Registry ID**: `[e.g., SRC_OI_RDCRN_BBDC_7701 | SRC_METS_TCIA_SPINE_CT | SRC_HEALTHY_STANFORD_MURA]`
- **Applicant Investigator & Role**: `[Name, Pseudonymous Track ID e.g. REV_*, Institution]`
- **Local IEC/IRB Determination Ref**: `[PENDING — Attach Approval / Exemption Letter]`
- **Requested Modality Stream**: `[xray | bone_scan | ct]` *(Single modality per access scope)*

## 2. Specific Data Elements Requested
1. De-identified imaging files (`DICOM` / lossless `PNG` / `NIfTI` per repository standard).
2. Pseudonymous `patient_id` linking multi-view and longitudinal scans to prevent train/test patient leakage.
3. Documented diagnostic reference standard category:
   - Metastasis: Histopathology vs multimodal PET-CT/radiology concordance kept distinct.
   - Paget's: Radiographic triad + serum ALP/BSAP or biopsy.
   - OI: Genetic `COL1A1`/`COL1A2` status vs clinical Sillence Type kept distinct; pediatric age strata.
   - Healthy Controls: Verified radiologist normal confirmation, specific `anatomical_region`, explicit `control_selection_criteria`.

## 3. Technical Security & Non-Redistribution Attestation
- AES-256 encrypted storage at rest on access-controlled institutional compute; TLS 1.3 in transit.
- Strictly non-commercial academic research; zero redistribution, re-hosting, or re-identification attempts.
- **Exclusion Attestation**: Will never merge primary sarcomas into metastasis, trauma contralateral limbs into healthy controls, osteoporosis into OI, or animal micro-CT into human cohorts.

## 4. Requested Repository DAC Sign-Off
- **Data Access Committee Decision**: `[PENDING SUBMISSION]`
- **Signed Bilateral DUA Attached**: `[PENDING]`
