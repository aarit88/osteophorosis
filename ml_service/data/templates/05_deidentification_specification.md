# De-Identification & Privacy Specification (Template 05)

> **STATUS**: `DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`  
> **Compliance Baselines**: HIPAA Privacy Rule Safe Harbor (45 CFR 164.514(b)(2)), DICOM PS 3.15 Annex E, GDPR Art. 89(1), DPDP Act 2023

---

## 1. Pseudonymous Identifiers Required at Intake
- **`patient_id`**: Strictly formatted `PT_<SITE>_<HASH>` or `SUBJ_<SITE>_<HASH>` (regex: `^(?:PT|SUBJ)_[A-Za-z0-9_-]{2,}$`). Never raw MRN, hospital ID, SSN, email, or name.
- **`reviewer_pseudo_id`**: Strictly formatted `REV_*`, `ADJ_*`, `RAD_*`, or `CLIN_*`.
- **Key Custody**: Re-identification linkage keys never leave the contributing institution's Honest Broker environment.

## 2. DICOM Tag Cleansing Checklist (DICOM PS 3.15 Annex E)
Scrub all 18 HIPAA direct identifiers prior to export:
- `(0010,0010)` Patient's Name -> PURGED
- `(0010,0020)` Patient ID -> REPLACED WITH SALTED PSEUDONYM
- `(0010,0030)` Patient's Birth Date -> PURGED (store age band or pediatric strata only; aggregate `>= 89`)
- `(0008,0080)` Institution Name & `(0008,0050)` Accession Number -> PURGED / TOKENIZED
- `(0008,0090)` Referring Physician & Performing/Operator Names -> PURGED
- `(0018,1000)` Device Serial Number & Private Vendor Tags -> PURGED
- Study Dates -> Shifted by random patient-specific negative offset preserving intra-patient intervals.

## 3. Pixel Burn-In Redaction & Secondary Quality Check
- Run automated OCR margin scan + secondary technician visual inspection on 100% of radiographs and bone scans to mask burned-in names, DOBs, accession numbers, and institution overlays.

## 4. Automated Intake Rejection Guarantees (`track_d_intake.py`)
Any manifest or image containing `patient_name`, `mrn`, `email`, `ssn`, `phone`, `dob`, unmasked date patterns, or unverified `deidentification_status != VERIFIED_PSEUDONYMIZED` is automatically **quarantined** and blocked from training.
