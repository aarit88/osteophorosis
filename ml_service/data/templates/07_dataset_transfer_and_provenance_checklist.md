# Dataset Transfer, Verification & Provenance Checklist (Template 07)

> **STATUS**: `DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`  
> **Pipeline Entrypoint**: `python3 ml_service/scripts/run_track_d_intake.py --all`

---

## Pre-Transfer & Ingestion Sign-Off Steps
1. [ ] **Pre-Transfer Governance Gate Cleared**: Workflow state has reached `ACCESS_APPROVED` with verified, non-expired IEC/IRB approval, signed DUA, and evidence references.
2. [ ] **Modality Directory Isolation**: Stage files strictly into their separate modality folders under `ml_service/data/raw/{xray|bone_scan|ct}/{metastasis|paget|oi|healthy}/` — never mix modalities.
3. [ ] **Encrypted Transfer & Integrity Digest**: Transfer over TLS 1.3 / institutional SFTP / NBIA Data Retriever; record SHA-256 manifest checksums upon receipt.
4. [ ] **Manifest Schema v2.0.0 Completeness**: Populate `ml_service/data/manifests/track_d_manifest.json` matching `track_d_manifest.schema.json` (16 core fields + Stage 34 provenance & confirmation fields).
5. [ ] **De-Identification & PHI Verification**: Run intake audit (`validate_dataset.py`) confirming `0` PHI violations and `deidentification_status = VERIFIED_PSEUDONYMIZED`.
6. [ ] **Protected Holdout Leakage Audit**: Confirm zero SHA-256 or ID collisions with **BTXRD Center 2 (`N=549`)**, **BTXRD Center 3 (`N=259`)**, or **FracAtlas test (`N=614`)**.
7. [ ] **Deduplication & Quality Validation**: Inspect `track_d_quality_report.json` & `track_d_quarantine_report.json` for exact SHA-256 duplicates, perceptual `dHash` (`<=3`) duplicates, and header/bit-depth checks.
8. [ ] **Patient-Level Partition Generation**: Verify zero patient overlap across `train`, `val`, `test`, and `external_test` in `track_d_split_manifest.csv`.
9. [ ] **Final Reviewer Sign-Off**: Record pseudonymous reviewer decision transitioning workflow state to `COHORT_ACCEPTED`.
