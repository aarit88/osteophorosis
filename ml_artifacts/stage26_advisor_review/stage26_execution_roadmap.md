# Stage 26 Dependency-Based Execution Roadmap

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Official Phased Implementation Roadmap  
**Date**: November 16, 2026  

---

## 1. Five-Tier Dependency Architecture

To ensure continuous research productivity without violating ethical or acquisition boundaries, all remaining project activities are structured into five discrete dependency tiers:

```
+--------------------------------------------------------------------------------------------------+
|                                  FIVE-TIER DEPENDENCY ARCHITECTURE                               |
+--------+------------------------------------+----------------------------+-----------------------+
| Tier   | Category Description               | Core Activities            | Blocking Dependency   |
+--------+------------------------------------+----------------------------+-----------------------+
| TIER 1 | Independent Technical Work         | - IBSI Radiomics testing   | ZERO DATA BLOCK       |
|        | (No new clinical data required)    | - Digital phantom analysis | Can proceed           |
|        |                                    | - Fracture engine equations| immediately           |
|        |                                    | - API contracts & frontend |                       |
+--------+------------------------------------+----------------------------+-----------------------+
| TIER 2 | Faculty Advisor Endorsement        | - Protocol sign-off        | Faculty Advisor       |
|        | (Academic review & sponsorship)    | - PI formal appointment    | Review & Meeting      |
|        |                                    | - Outreach authorization   |                       |
+--------+------------------------------------+----------------------------+-----------------------+
| TIER 3 | Institutional Ethics & Legal       | - Hospital IEC submission  | Institutional IEC     |
|        | Governance                         | - Retrospective waiver req.| Review Cycle & Legal  |
|        |                                    | - Bilateral DUA drafting   | DUA Approval          |
|        |                                    | - RDCRN DAC application    |                       |
+--------+------------------------------------+----------------------------+-----------------------+
| TIER 4 | Controlled Clinical Acquisition    | - Honest broker PACS query | Executed DUAs &       |
|        | & De-Identification                | - DICOM PS 3.15 scrubbing  | Ethics Clearances     |
|        |                                    | - Pixel OCR burn-in redaction                      |
|        |                                    | - Checksum staging storage |                       |
+--------+------------------------------------+----------------------------+-----------------------+
| TIER 5 | Unified 4-Class Deep Learning      | - Multi-pathway training   | 10-Point Data         |
|        | Development & Evaluation           | - Late-fusion calibration  | Acceptance Gate       |
|        |                                    | - Cross-center holdout eval| Formal Sign-Off       |
+--------+------------------------------------+----------------------------+-----------------------+
```

---

## 2. Detailed Work Breakdown by Tier

### TIER 1: Work That Can Continue Immediately (Zero Data Required)
The research team can actively advance the following engineering and validation tasks without accessing new patient data:
1. **IBSI Radiomic Unit Test Validation**: Expand mathematical unit tests for all 102 radiomic features in `osteoguard/radiomics/engine.py` using standardized digital phantom arrays (e.g., IBSI digital phantom benchmark values).
2. **Biomechanical Fracture Equation Calibration**: Perform parametric sensitivity analyses on the continuous Mirels' lesion-to-cortex ratio equation ($R_{\text{fracture}}$) across synthetic ranges of cortical defect angles ($\theta_{\text{defect}}$) and bone diameters.
3. **Multi-Pathway Architecture Code Implementation**: Implement the decoupled multi-pathway network skeleton (Pathway A: DenseNet/ConvNeXt, Pathway B: Siamese ResNet-34, Pathway C: 3D-UNet, and Late-Fusion Bayesian module) in `osteoguard/models/` without training weights.
4. **FastAPI & React Integration Testing**: Build mock analysis endpoints and verify end-to-end data schemas between `osteoguard/api/` and the React frontend dashboard (`osteophorosis-main/src/`).

### TIER 2: Work Requiring Faculty Advisor Approval
1. Review and formal endorsement of revised protocol `OG-PROT-2026-04C-REVISED`.
2. Formal institutional appointment of the Principal Investigator (PI) role.
3. Authorization to initiate exploratory contact with the hospital Musculoskeletal Radiology department using Draft 2.

### TIER 3: Work Requiring IEC Approval and Executed DUAs
1. Formal submission of research protocol to the Institutional Ethics Committee (IEC).
2. Review and granting of a waiver of informed consent for retrospective anonymized imaging under ICMR Guidelines (2017).
3. Negotiation and institutional execution of bilateral Data Use Agreements (DUA).
4. Submission of formal application to the RDCRN BBDC Data Access Committee.

### TIER 4: Work Requiring Actual Clinical Dataset Acquisition
1. PACS database query by hospital honest broker using ICD-10 codes (`M88.x` for Paget's disease; asymptomatic trauma screening for normal controls).
2. DICOM PS 3.15 de-identification and pixel OCR burn-in text removal.
3. Secure transfer to isolated institutional GPU storage.

### TIER 5: Work Strictly Blocked Until Acceptance Gate is Satisfied
1. Four-class multi-modal deep learning model training.
2. Cross-center external validation on holdout hospital cohorts.
3. Clinical performance auditing and diagnostic reporting.
