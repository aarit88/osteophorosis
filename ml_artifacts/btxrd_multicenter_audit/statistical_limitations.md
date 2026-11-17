# Statistical and Clinical Interpretation Limitations Audit

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 21 — BTXRD Multi-Center Generalization Audit  
**Artifact Path:** `ml_artifacts/btxrd_multicenter_audit/statistical_limitations.md`  

---

## 1. Executive Statistical Notice

> [!CAUTION]
> **INVESTIGATIONAL RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL USE.**  
> OsteoGuard AI and its associated deep learning models are investigational research prototypes. The evaluations documented herein do not constitute a clinical validation study, have not received clearance or approval from regulatory authorities (such as the US FDA, CE-IVD, or PMDA), and must **never** be used to guide, inform, or alter patient diagnosis, triage, biopsy decisions, or therapeutic regimens.

---

## 2. Confidence Interval Methodology and Resampling Boundary

Across Stages 19 and 20, non-parametric empirical 95% confidence intervals were generated using **1,000 bootstrap iterations**:

$$\text{Bootstrap Estimator: } \hat{\theta}^*_b = f(D^*_b), \quad b = 1, \dots, 1000$$

Where $D^*_b$ represents sampling with replacement from the evaluated cohort.

### Methodological Limitations of Image-Level Resampling:
1. **Absence of Patient Accession Identifiers:**
   - The BTXRD dataset repository does not provide anonymized patient IDs, encounter IDs, or accession numbers.
   - Consequently, bootstrap resampling was conducted strictly over **individual image observations** rather than patient clusters.
2. **Clustering & Intra-Patient Correlation Risk:**
   - In orthopedic oncology, patients frequently undergo multiple radiographic views (e.g., anteroposterior, lateral, oblique) or serial follow-up studies.
   - If multiple radiographs from a single individual exist across partitions or within an evaluation set, image-level bootstrap violates the assumption of independent and identically distributed (i.i.d.) observations.
   - This may result in artificially narrow confidence intervals and unmodeled cluster-level variance.
3. **True Patient Independence Unverified:**
   - Neither Center 1 development validation, Center 2 holdout, nor Center 3 stress test can definitively claim patient-level independence. All conclusions must be understood as image-level generalizability benchmarks.

---

## 3. Prevalence Dependence of Predictive Values (PPV & NPV)

A critical statistical vulnerability observed across Centers 1, 2, and 3 is the profound mathematical coupling between Bayesian predictive values and institutional disease prevalence.

By Bayes' Theorem:
$$\text{PPV} = \frac{\text{Sensitivity} \times \text{Prevalence}}{\text{Sensitivity} \times \text{Prevalence} + (1 - \text{Specificity}) \times (1 - \text{Prevalence})}$$

$$\text{NPV} = \frac{\text{Specificity} \times (1 - \text{Prevalence})}{\text{Specificity} \times (1 - \text{Prevalence}) + (1 - \text{Sensitivity}) \times \text{Prevalence}}$$

### Empirical Cross-Center Reality:

| Cohort | Malignant Prevalence | Malignant Sensitivity | Malignant Specificity | Malignant Precision (PPV) | Malignant NPV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Center 1 (Internal Val)** | 6.97% (41 / 588) | 90.24% | 97.81% | 75.51% | 99.26% |
| **Center 2 (External Holdout)** | 13.84% (76 / 549) | 59.21% | 84.36% | 37.82% | 92.79% |
| **Center 3 (External Stress Test)** | 11.97% (31 / 259) | 67.74% | 69.30% | 23.08% | 94.05% |

### Critical Clinical Interpretation Warning:
> [!WARNING]
> **High NPV Must NOT Be Presented as Proof of Safe Malignancy Rule-Out.**  
> In Center 3, Malignant NPV is `94.05%`. However, this is largely driven by the high non-malignant proportion (88.03% of Center 3 cases are not primary malignant tumours). In absolute terms, **6 out of 31 primary malignant bone tumours (19.35%) were completely misdiagnosed as normal controls**, and another **4 out of 31 (12.90%) were misdiagnosed as benign lesions**. In total, **32.26% (10 / 31)** of malignant cases failed primary oncological detection. Claiming that a 94.05% NPV demonstrates clinical safety would dangerously obscure these life-threatening false negatives.

---

## 4. Center 3 Small Normal Cohort Distortion ($N=27$)

Center 3 reflects a specialized tertiary referral hospital where only 27 normal control radiographs were cataloged (10.42% prevalence):

1. **Extreme Metric Volatility:**
   - With $N=27$, a single misclassified case shifts sensitivity by $\sim 3.7\%$.
   - Normal Sensitivity: **`18.52%`** (5 / 27 correct; 95% CI: `[5.00%, 34.48%]`).
   - Normal Precision (PPV): **`9.09%`** (5 / 55 calls correct; 95% CI: `[2.17%, 17.19%]`).
2. **Wide Confidence Intervals:**
   - The 95% bootstrap CI for normal sensitivity spans nearly 30 percentage points, demonstrating severe sample size instability.
3. **Interaction with Training Loss Prior:**
   - The model was trained in Stage 17 using Focal Loss with class weights $\alpha = [0.3333, 0.4667, 2.1850]$ to emphasize the rare malignant class in Center 1.
   - When deployed to Center 3 (where 89.58% of patients have true bone lesions), the aggressive suspicion bias resulted in 22 of 27 normal radiographs being classified as pathological (13 benign, 9 malignant).

---

## 5. Raw Softmax Probability Calibration Deficiencies

All probability outputs generated and archived are **raw, uncalibrated softmax activations**:

$$p_c = \frac{\exp(z_c)}{\sum_{j=0}^2 \exp(z_j)}$$

### Cross-Center Brier Score Progression:
- **Center 1 Internal Val:** `0.3168` (moderately calibrated under training distribution)
- **Center 2 External Holdout:** `0.5731` (+80.9% increase in probability error)
- **Center 3 External Stress Test:** `0.6486` (+104.7% increase in probability error)

### Methodological Rule:
Post-hoc calibration techniques (such as Platt scaling, Temperature Scaling, or Isotonic Regression) were **strictly prohibited** on Center 2 and Center 3. Fitting calibration maps directly on external holdout sets constitutes test-set data leakage. Consequently, the reported probabilities exhibit significant overconfidence and entropy inflation under out-of-distribution conditions.

---

## 6. Summary of Regulatory & Methodological Boundaries

1. **Not a Clinical Trial or Diagnostic Accuracy Study:** No prospectively enrolled patients, blinded radiological re-reads, or clinical outcome measurements were conducted.
2. **No Post-Hoc Threshold Manipulation:** All evaluations maintained the unadjusted $\arg\max$ decision rule.
3. **Zero Data Pooling:** Center 2 and Center 3 evaluations are independent and must remain unpooled.
4. **Research Prototype Boundary:** This model is an experimental benchmark for differential feature learning and must not be integrated into clinical workflows.

