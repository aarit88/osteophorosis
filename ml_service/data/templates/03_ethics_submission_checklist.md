# Institutional Ethics (IRB / IEC) Submission Checklist (Template 03)

> **STATUS**: `DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`  
> **Applicable Regulations**: Institutional Ethics Committee (IEC/IRB), HIPAA Safe Harbor (45 CFR 164.514), GDPR Art. 89(1), DPDP Act 2023, ICMR-HMSC (where cross-border)

---

## Mandatory Pre-Submission Checklist
- [ ] **1. Protocol Package Complete**: Draft protocol `OG-PROT-2026-04C-DRAFT` reviewed by Radiology, Orthopaedics, and Biostatistician.
- [ ] **2. Biostatistical Sample-Size Justification**: Formal statistical power / sample-size calculation attached (`statistical_sample_size_justification_ref` — never fabricated in software).
- [ ] **3. Informed Consent / Retrospective Waiver Determination**:
  - [ ] Adult retrospective anonymized imaging waiver rationale documented; OR
  - [ ] Prospective written informed consent forms finalized.
- [ ] **4. Pediatric / Vulnerable Subject Safeguards (Mandatory for OI Cohort)**:
  - [ ] Parental/guardian informed consent + minor assent procedures (when age < 18) or explicit pediatric IEC waiver justification.
  - [ ] Growth-plate maturation stratification & age-matched normal pediatric control justification.
- [ ] **5. De-Identification & Honest Broker Workflow**:
  - [ ] Re-identification lookup table stays strictly within contributing hospital firewall under Honest Broker custody.
  - [ ] DICOM PS 3.15 Annex E + pixel OCR burn-in scrubbing verified (Template 05).
- [ ] **6. Cross-Border / Consortium Clearances (If Applicable)**:
  - [ ] ICMR-HMSC or national health data transfer clearance documented for international registries (e.g., Paget's UK/EU or RDCRN US).
- [ ] **7. Data Retention & Cryptographic Purge Schedule**:
  - [ ] Approved retention duration (e.g., 5 years post-publication) and NIST SP 800-88 Rev 1 destruction plan.
- [ ] **8. Evidence & Expiry Tracking**:
  - [ ] Signed IEC approval letter reference, approval date, and annual renewal/expiry date recorded in `dataset_source_registry.json`.
