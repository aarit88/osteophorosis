# Bilateral Data Use Agreement (DUA) Checklist (Template 04)

> **STATUS**: `DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`  
> **Notice**: Legal counsel & Data Protection Officer review required before execution. Zero data transfer may proceed before an active, non-expired DUA is verified.

---

## Mandatory Contractual Clauses Checklist
- [ ] **1. Named Data Provider & Recipient Institutions**: Legal entity names, signatory officers, and Principal Investigators identified.
- [ ] **2. Explicit Cohort & Modality Scope**: Enumerates diagnostic cohort (`metastasis | paget | oi | healthy`), separated modality (`xray | bone_scan | ct`), and anatomical coverage (e.g., Stanford AIMI MURA upper-extremity boundary).
- [ ] **3. Permitted Research Purposes**: Strictly limited to non-commercial academic differential diagnosis, radiomic/biomechanical research, and model safety evaluation.
- [ ] **4. Redistribution & Re-Identification Prohibitions**:
  - [ ] Zero public posting, sublicensing, or transfer to unauthorized third parties or external commercial LLM/foundation APIs.
  - [ ] Explicit covenant against attempting re-identification or linkage with external clinical records.
- [ ] **5. De-Identification Warranties**: Provider confirms Safe Harbor / DICOM PS 3.15 de-identification; Recipient commits to quarantine any inadvertent PHI discovered during validation.
- [ ] **6. Security Safeguards**: AES-256 encryption at rest, TLS 1.3 in transit, role-based access control, immutable audit logs.
- [ ] **7. Agreement Effective Date, Expiry Date & Renewal**: Explicit `approval_date` and `expiry_date`; expired DUAs automatically block workflow advancement.
- [ ] **8. Post-Termination Destruction**: NIST SP 800-88 Rev 1 sanitization attestation upon expiry or withdrawal.
- [ ] **9. Publication Attribution & Co-Authorship**: Citation requirements (e.g., TCIA, RDCRN BBDC, Stanford AIMI, or hospital co-authors).
