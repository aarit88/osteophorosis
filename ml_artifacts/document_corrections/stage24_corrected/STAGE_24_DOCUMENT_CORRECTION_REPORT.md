# STAGE 24 DOCUMENT CORRECTION REPORT: SCIENTIFIC, STATISTICAL, AND GOVERNANCE REALIGNMENT

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
