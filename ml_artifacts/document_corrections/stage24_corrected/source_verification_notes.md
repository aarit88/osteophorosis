# Source Verification Notes: Scientific, Statistical, and Governance Foundations

**Project**: OsteoGuard AI  
**Document**: Source Evidence and Verification Audit Notes  
**Date**: November 16, 2026  

---

## 1. Verified Model Performance Metrics (BTXRD Three-Center Evaluation)

All performance metrics have been re-verified from the primary source JSON and CSV files:
- **Center 1**: `ml_artifacts/btxrd_full_training/validation_metrics.json` and `ml_artifacts/btxrd_frozen_model/model_freeze_manifest.json`
- **Center 2**: `ml_artifacts/btxrd_center2_external_evaluation/classification_metrics.json`
- **Center 3**: `ml_artifacts/btxrd_center3_external_evaluation/classification_metrics.json`
- **Multi-Center Audit**: `ml_artifacts/btxrd_multicenter_audit/multicenter_performance_comparison.csv`

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

### Statistical Distinction Rules
1. **Macro-F1 Score**: Unweighted arithmetic mean of per-class F1-scores ($\frac{1}{K}\sum_{k=1}^K F1_k$).
2. **Absolute Change (Percentage Points, pp)**: $\Delta_{\text{abs}} = \text{Macro-F1}_{\text{ext}} - \text{Macro-F1}_{\text{C1}}$.
   - Center 2: $0.5069 - 0.8103 = -0.3034$ (-30.34 percentage points).
   - Center 3: $0.3592 - 0.8103 = -0.4511$ (-45.11 percentage points).
3. **Relative Percentage Decrease (%)**: $\Delta_{\text{rel}} = \frac{\text{Macro-F1}_{\text{ext}} - \text{Macro-F1}_{\text{C1}}}{\text{Macro-F1}_{\text{C1}}} \times 100\%$.
   - Center 2: $\frac{-0.3034}{0.8103} \times 100\% \approx -37.44\%$.
   - Center 3: $\frac{-0.4511}{0.8103} \times 100\% \approx -55.67\%$.
4. **Cohort Non-Pooling**: Center 2 and Center 3 represent heterogeneous clinical referral distributions. Pooling them violates clinical epidemiological principles and obscures etiology-specific degradation.
5. **Negative Predictive Value (NPV)**: In Center 1, malignant tumour support is 41/588 (7.0%). High malignant NPV (99.26%) is mathematically mandated by low class prevalence and must not be described as proof of safe autonomous clinical exclusion.

---

## 2. Stanford AIMI MURA Verification

- **Source Reference**: Rajpurkar et al., *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs*, Stanford ML Group / AIMI.
- **Verified Study Counts**:
  - Total Studies: 14,863
  - Total Patients: 12,173
  - Total Images: 40,561
  - **Normal Studies**: 9,045 (60.86%)
  - **Abnormal Studies**: 5,818 (39.14%)
- **Anatomical Coverage**: Strictly confined to seven upper-limb regions: Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist.
- **Critical Finding**: MURA contains both normal and abnormal studies. Describing all 40,561 images as "healthy controls" was a critical error. Only the 9,045 normal studies represent candidate upper-extremity controls, and a normal wrist does not establish whole-skeleton health.

---

## 3. Human Osteogenesis Imperfecta (PMC11324408) Verification

- **Source Reference**: *Bone*, 2024;185:117192 (PMC11324408).
- **Study Title**: Brittle Bone Disorders Consortium (BBDC Protocol 7701) Bone Age Evaluation.
- **Verified Cohort Scale**:
  - Baseline: 159 pediatric participants with clinical/genetic OI.
  - Follow-up: 55 participants with repeated observations approximately 24 months later.
  - Phenotypic Coverage: Sillence Types I, III, and IV.
  - Anatomical Coverage: Left hand and wrist radiographs for bone age assessment.
- **Critical Finding**: The publication does not provide unrestricted bulk download of radiographs. Access requires formal application to the RDCRN Data Management and Coordinating Center (DMCC). Anatomical coverage is limited to hand/wrist radiographs and does not cover the long bones or axial skeleton.

---

## 4. Indian Ethical and Legal Regulatory Framework

- **Primary Guidance**: ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017).
  - Section 5: Ethical review procedures, expedited review, and criteria for waiver of informed consent (Box 5.2).
  - Section 11: Governance of biological materials, secondary data use, and biobanking.
- **Institutional Authority**: Institutional Ethics Committee (IEC) is the primary governing authority.
- **Principal Investigator**: Role must be formally appointed by the academic/clinical institution. Student researchers cannot unilaterally claim PI status.
- **Data Protection Statutory Context**: Digital Personal Data Protection Act, 2023 (DPDP Act) and Rules (2025/2026). Phased commencement is in progress; institutional legal counsel must determine operational applicability.
- **Foreign Privacy Laws (HIPAA/GDPR)**: Apply conditionally based on repository location or contractual DUA terms, not automatically to Indian domestic data processing.
