# Mandatory Data Acceptance Gate and Verification Checklist

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Enforced Operational Data Gate  
**Author**: OsteoGuard AI Quality Assurance and Regulatory Office  

> [!IMPORTANT]
> **ABSOLUTE INGESTION RULE**:  
> No dataset shall enter the project training, validation, or test pipeline simply because it is downloadable, publicly accessible, or supplied by a collaborator.  
> Every prospective dataset must pass all 10 mandatory checks documented below. Any failure results in immediate classification as `REJECTED` or `CONDITIONALLY ACCEPTED`.

---

## 1. The 10-Point Mandatory Acceptance Gate

```
+--------------------------------------------------------------------------------------------------+
|                                    10-POINT DATA ACCEPTANCE GATE                                 |
+----+-----------------------------+------------------------------------+--------------------------+
| #  | Gate Dimension              | Verification Requirement           | Pass Criteria            |
+----+-----------------------------+------------------------------------+--------------------------+
| 1  | Species & Clinical Status   | Verified human clinical imaging    | Homo sapiens ONLY. Zero  |
|    |                             |                                    | animal / micro-CT data.  |
+----+-----------------------------+------------------------------------+--------------------------+
| 2  | Diagnostic Label Fidelity   | Conforms strictly to target class  | Verified pathology, bio- |
|    |                             | definition                         | chemical, or consensus.  |
+----+-----------------------------+------------------------------------+--------------------------+
| 3  | Modality & Projection       | Conforms to designated pathway     | Digital X-ray (A), Scin- |
|    |                             | requirements                       | tigraphy (B), CT (C).    |
+----+-----------------------------+------------------------------------+--------------------------+
| 4  | Legal & Licensing Rights    | Documented research authorization  | Approved DUA, IRB waiver,|
|    |                             |                                    | or open research license.|
+----+-----------------------------+------------------------------------+--------------------------+
| 5  | Patient-Level Identity      | Discrete, immutable subject IDs    | True patient separation; |
|    |                             | available for partitioning         | zero view leakage.       |
+----+-----------------------------+------------------------------------+--------------------------+
| 6  | Technical Image Quality     | Diagnostic bit depth & resolution  | Resolution >= 1024x1024; |
|    |                             | without lossy compression artifacts| no degraded 8-bit JPGs.  |
+----+-----------------------------+------------------------------------+--------------------------+
| 7  | Deduplication Verification  | Cryptographic & perceptual check   | Zero hash collision with |
|    |                             |                                    | existing in-tree assets. |
+----+-----------------------------+------------------------------------+--------------------------+
| 8  | Reference Standard Proof    | Documented diagnostic provenance   | Explicit Tier 1, 2, or 3 |
|    |                             | level                              | reference standard.      |
+----+-----------------------------+------------------------------------+--------------------------+
| 9  | Clinical QA Review          | Musculoskeletal radiologist audit  | Signed radiologist QA    |
|    |                             | of candidate cohort sample         | approval form.           |
+----+-----------------------------+------------------------------------+--------------------------+
| 10 | Modality Pathway Assignment | Assigned to appropriate single-    | Decoupled pathway lock;  |
|    |                             | modality feature encoder           | no monolithic sharing.   |
+----+-----------------------------+------------------------------------+--------------------------+
```

---

## 2. Ingestion Disposition Classifications

Upon evaluation against the 10-point checklist, every prospective candidate must be assigned exactly one formal disposition:

### 1. `ACCEPTED FOR RESEARCH INGESTION`
- **Definition**: All 10 acceptance checks have been rigorously verified, documented, and cryptographically recorded.
- **Action**: Dataset is assigned an official catalog identifier, ingested into `ml_service/data/raw/`, and entered into the patient-level partitioning pipeline.

### 2. `CONDITIONALLY ACCEPTED`
- **Definition**: The dataset has high clinical validity but exhibits minor remediable deficiencies (e.g., missing anatomical view tags, pending pixel burn-in redaction, or unformatted clinical metadata).
- **Action**: Quarantined in staging storage. Data cannot enter training until all conditional requirements are fully remediated and signed off.

### 3. `AWAITING PERMISSION`
- **Definition**: Scientific suitability and clinical value verified, but formal institutional permissions, IRB approval, or bilateral DUA execution are pending.
- **Action**: Zero data transfer permitted. Documented in project dependency tracker awaiting formal authorization.

### 4. `REJECTED`
- **Definition**: Failed one or more non-negotiable gates (e.g., non-human animal imaging, unresolvable diagnostic ambiguity, predatory copyright violation, or duplicate of existing in-tree assets).
- **Action**: Permanently barred from ingestion. Forensic disqualification report logged in project repository.
