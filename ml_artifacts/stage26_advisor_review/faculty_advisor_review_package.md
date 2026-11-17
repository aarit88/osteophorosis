# Faculty Advisor Review Package: OsteoGuard AI Multi-Class Research Initiative

**Project Title**: OsteoGuard AI: Unified Deep Learning and Radiomic Framework for Differential Diagnosis of Bone Disorders and Biomechanical Fracture Risk Stratification  
**Academic Context**: Undergraduate/Postgraduate Research Initiative (KSCST Student Project Programme)  
**Document Status**: Prepared for Faculty Research Advisor Review  
**Date**: November 16, 2026  

---

## 1. Project Overview & Clinical Rationale

OsteoGuard AI is an academic machine learning research project aimed at addressing a major diagnostic challenge in musculoskeletal radiology: distinguishing conditions that present with altered bone mineral density, cortical disruption, and pathological fracture risk on skeletal imaging.

The four clinical target categories are:
1. **Secondary Bone Metastasis**: Extra-skeletal primary carcinomas (breast, prostate, lung, kidney) metastasizing to bone.
2. **Paget's Disease of Bone (*Osteitis Deformans*)**: Disordered focal bone remodeling with cortical thickening, trabecular coarsening, and osteomegaly.
3. **Osteogenesis Imperfecta (OI)**: Genetic type I collagen deficiency causing bone fragility, cortical thinning, recurrent fractures, and skeletal deformities.
4. **Healthy Skeletal Controls**: Anatomically and demographically matched normal bone architecture.

---

## 2. Technical Milestones Completed to Date

The research team has built and validated a fully reproducible, modular computational framework:
1. **Dual-Stream Framework**: Implemented the end-to-end PyTorch network combining U-Net skeletal segmentation, deep semantic feature backbones, an IBSI-aligned 102-D radiomic texture engine, and a 614-D late-fusion classification head with Focal Loss ($\gamma = 2.0$).
2. **Biomechanical Fracture Risk Engine**: Digitized Mirels' scoring into a continuous mathematical lesion-to-cortex ratio ($LCR$) and polar moment of inertia equation ($J_{\text{defect}}$).
3. **Bone Scintigraphy Metastasis Model (Stages 11–14)**: Developed a dual-view paired anterior/posterior convolutional ensemble on 291 breast cancer patients (IICS-UNA cohort), achieving a holdout Macro-F1 of **0.9412** and ROC-AUC of **0.9850**.
4. **Three-Center Primary Bone Tumour Benchmark (Stages 15–21)**: Frozen and evaluated a three-class ResNet-34 model across three hospital cohorts, producing critical empirical evidence on AI generalization.

---

## 3. Key Scientific Findings on External Generalization (Domain Shift)

Our multi-center evaluations of the BTXRD three-class primary bone tumour classifier revealed substantial domain shift when moving outside the internal development institution:

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

### Key Scientific Takeaways for the Advisor
- Performance degraded by **37.44% relative** on Center 2 and by **55.67% relative** on Center 3.
- Primary causes: Differences in X-ray equipment calibration, imaging protocols, and extreme referral population skew (Center 3 had 77.6% benign lesions).
- This finding confirms that training models on single-source open data leads to models that fail in external hospital settings, underscoring the absolute necessity of multi-center clinical partnerships.

---

## 4. Current Dataset Availability and Critical Data Gaps

Our forensic audits in Stages 22–25 revealed the following data availability realities:
1. **Secondary Bone Metastasis**: Planar scintigraphy is validated in-tree (291 patients). However, plain radiography metastases and volumetric spinal CT are missing.
2. **Paget's Disease of Bone**: **Zero verified clinical datasets exist in-tree.** Our audit found no suitable, publicly accessible clinical imaging cohorts. Web teaching repositories (e.g., Radiopaedia) are legally prohibited from bulk ML use and suffer severe selection bias. Acquisition requires a retrospective hospital PACS partnership.
3. **Osteogenesis Imperfecta**: Online candidate datasets (e.g., Figshare) were forensically proven to be **non-human murine micro-CT scans** and were disqualified. The major published human cohort (PMC11324408, BBDC Protocol 7701) is restricted to pediatric hand/wrist bone-age radiographs and requires formal Data Access Committee (DAC) approval.
4. **Healthy Skeletal Controls**: FracAtlas "non-fractured" images originate from acute emergency trauma patients and contain soft tissue hematomas and osteopenia. Stanford AIMI MURA contains 9,045 normal studies but is strictly confined to the upper extremity. Matched normal pelvic and spinal X-rays are missing.

---

## 5. Proposed Decoupled Modality Architecture

To prevent unscientific cross-modality distortion, OsteoGuard AI rejects single-encoder sharing across X-ray, Scintigraphy, and CT:
- **Pathway A (Plain Radiography)**: Evaluates 2D bone texture (Paget's, OI, skeletal controls, plain film metastases).
- **Pathway B (Planar Scintigraphy)**: Evaluates whole-body metabolic distribution (secondary metastases, metabolic controls).
- **Pathway C (Volumetric CT)**: Evaluates 3D spinal vertebral osteolysis and biomechanical stability.
- **Clinical Late-Fusion Layer**: Integrates modality posterior probabilities with clinical covariates (age, sex, site, serum alkaline phosphatase) using a calibrated Bayesian meta-model with marginalization for missing modalities.

---

## 6. Ethical and Legal Governance Framework

- **Primary Ethical Authority**: Institutional Ethics Committee (IEC) review under the **ICMR National Ethical Guidelines (2017)**.
- **Waiver of Consent**: Retrospective anonymized imaging may be considered for a waiver of informed consent under ICMR Section 5 (Box 5.2).
- **Data Protection Law**: Compliance with the phased commencement of the **Digital Personal Data Protection Act, 2023 (DPDP Act)**.
- **Investigational Status**: Software is strictly an academic research prototype, not a medical device.

---

## 7. Decisions and Support Requested from the Faculty Advisor

To advance the project legitimately and scientifically, the research team requests the following decisions and guidance from the faculty advisor:

1. **Appointment of Institutional Principal Investigator (PI)**:
   - Formally designate a faculty member as the official institutional Principal Investigator to sponsor research applications.
2. **Review of Revised Research Protocol**:
   - Review and endorse protocol `OG-PROT-2026-04C-REVISED` for formal academic submission.
3. **Institutional Ethics Committee (IEC) Guidance**:
   - Approve submission of a procedural inquiry to the institutional IEC regarding retrospective imaging review and waiver requirements.
4. **Clinical Partnership Outreach**:
   - Endorse initial exploratory outreach (using Draft 2) to the Head of Musculoskeletal Radiology at the affiliated teaching hospital to discuss a retrospective Paget's disease study.
5. **Controlled Access Application Sponsorship**:
   - Co-sign the formal non-commercial research registration for Stanford AIMI MURA (upper-extremity controls).
   - Provide institutional endorsement for a preliminary inquiry to the RDCRN Brittle Bone Disorders Consortium (BBDC) for pediatric OI radiographs.
6. **Interim Research Direction**:
   - Approve continuing computational work that does not require new patient data (IBSI radiomic feature verification on digital phantoms, biomechanical continuous fracture risk engine optimization, and API integration).
