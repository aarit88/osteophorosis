# Stage 25 Patient Grouping and Partition Independence Audit

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Stage**: 25 — Controlled Dataset Intake & Provenance Verification  
**Date**: November 16, 2026  

---

## 1. Patient Grouping in Existing In-Tree Cohorts

To prevent data leakage caused by intra-patient correlation across multiple anatomical projections, serial examinations, or contralateral views, patient-level grouping is audited across all existing assets:

```
+--------------------------------------------------------------------------------------------------+
|                                    EXISTING PATIENT GROUPING AUDIT                               |
+--------------------------+-----------------------+-----------------------+-----------------------+
| In-Tree Dataset          | Patient ID Format     | Intra-Patient Leakage | Preservation Rule     |
|                          |                       | Risk in Prior Stages  |                       |
+--------------------------+-----------------------+-----------------------+-----------------------+
| IICS-UNA Bone Scan       | Explicit Patient IDs  | ZERO LEAKAGE          | All 582 images paired |
| (Metastasis)             | (`Patient 1` to 291)  | Paired views locked to| by patient ID; single |
|                          |                       | single patient entity | partition allocation. |
+--------------------------+-----------------------+-----------------------+-----------------------+
| BTXRD Center 1           | Documented in         | MANAGED IN STAGE 17   | Evaluated in 3-class  |
| (Primary Tumours)        | acquisition manifest  | Partitioned strictly  | model development.    |
|                          |                       | for C1 training/val.  |                       |
+--------------------------+-----------------------+-----------------------+-----------------------+
| BTXRD Center 2 & 3       | Documented hospital   | PERMANENTLY BURNED    | Strictly external;    |
| (External Holdouts)      | patient cohorts       | Zero overlap with C1. | NEVER reusable for    |
|                          |                       |                       | training or tuning.   |
+--------------------------+-----------------------+-----------------------+-----------------------+
| FracAtlas Trauma Cohort  | Image-level numbering | HIGH POTENTIAL BIAS   | Confounded trauma     |
|                          | with clinical tags    | Not patient-grouped   | screen; excluded from |
|                          |                       | across visits.        | pristine controls.    |
+--------------------------+-----------------------+-----------------------+-----------------------+
```

---

## 2. Patient Grouping Rules for Prospective Four-Class Intake

Before any newly acquired dataset can enter the project staging directory, it must satisfy mandatory grouping constraints:
1. **Deterministic Pseudonymous ID**: Every image must link to an immutable, one-way pseudonym (e.g., `SUBJ_PAG_0042`) assigned by the data provider's honest broker.
2. **Multi-View Integrity**: All views (AP, lateral, oblique) and longitudinal examinations belonging to a single individual must share the same subject identifier.
3. **Strict Disjoint Partitioning**: No patient's data may be split across training, validation, or holdout test partitions.
4. **Holdout Hospital Isolation**: Entire institutions or hospital systems must be set aside as independent external holdouts to assess clinical domain generalization.
