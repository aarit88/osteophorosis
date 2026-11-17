# Model Disposition and Readiness Decision

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 21 — BTXRD Multi-Center Generalization Audit  
**Artifact Path:** `ml_artifacts/btxrd_multicenter_audit/model_readiness_decision.md`  
**Model Identifier:** `OsteoGuard-AI-BTXRD-ResNet34` (SHA-256: `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`)  

---

## 1. Formal Disposition Decision Matrix

| Readiness Category | Formal Audit Verdict | Primary Rationale & Empirical Justification |
| :--- | :---: | :--- |
| **1. Research Prototype Readiness** | **CONDITIONAL PASS** | Established proof-of-concept for deep radiomic feature extraction on plain radiographs; achieves strong internal benchmark metrics (Macro-F1: 0.8103, Malignant ROC-AUC: 0.9868) and retains meaningful ranking discrimination externally (Center 2 Malignant AUC: 0.8198, Center 3: 0.7490). Suitable solely for in silico experimental investigations. |
| **2. Cross-Center Generalization Readiness** | **FAIL** | Severe generalization penalty across institutional boundaries. Macro-F1 dropped by **37.4%** in Center 2 (0.5069) and by **55.7%** in Center 3 (0.3592). Normal sensitivity collapsed to 18.52% in Center 3. Probability calibration degraded severely (Brier score rose from 0.3168 to 0.6486). The model is not domain-invariant. |
| **3. Clinical Deployment Readiness** | **STRICTLY PROHIBITED (FAIL)** | Massive safety risks due to false-negative malignant predictions: **40.79%** of malignant cases missed in Center 2 (10 called normal, 21 called benign) and **32.26%** missed in Center 3 (6 called normal, 4 called benign). Missing primary bone malignancies as normal controls poses life-threatening diagnostic delay. Lack of patient identifiers and lack of clinical trial validation strictly bar clinical use. |
| **4. Integration into Final Classifier** | **UNSUITABLE IN CURRENT FORM** | The current ResNet-34 model is an isolated 3-class primary bone tumour model (Normal vs Benign vs Malignant). It does **not** classify secondary bone metastasis, Paget's disease, or osteogenesis imperfecta. It cannot be directly dropped in as the final OsteoGuard AI system without architectural expansion and domain harmonization. |

---

## 2. Category-by-Category Deep Audit Analysis

### 2.1 Research Prototype Readiness: CONDITIONAL PASS
- **Supporting Evidence:**
  1. The ResNet-34 architecture successfully converged on Center 1 development data over 20 epochs, with Epoch 18 delivering Macro-F1 of 0.8103 and Malignant Sensitivity of 90.24%.
  2. In external evaluation, the model retains significant area under the ROC curve for detecting primary bone malignancy (Center 2: 0.8198, Center 3: 0.7490), well above chance (0.50).
  3. In Center 3, total neoplasm detection reached 80.65% (25 of 31 malignant cases flagged as abnormal bone lesions).
  4. The model architecture, training pipeline, and evaluation harness are fully reproducible, deterministic, and cryptographically verified.
- **Conditions of Research Use:**
  - Must remain strictly within computational experimentation pipelines.
  - Must not be interpreted as validated medical device software.

---

### 2.2 Cross-Center Generalization Readiness: FAIL
- **Supporting Evidence:**
  1. **Steep Degradation of Balanced Accuracy:**
     - Center 1: `82.90%`
     - Center 2: `53.74%` ($\Delta = -29.16\%$)
     - Center 3: `44.67%` ($\Delta = -38.23\%$)
  2. **Collapse of Normal Detection Under Skewed Referral Priors:**
     - In Center 3, where normal cases constitute only 10.42% of incoming patients, Normal Sensitivity plummeted to `18.52%` (5 / 27). The classifier generated 22 false alarms out of 27 healthy scans.
  3. **Domain Uncertainty & Overconfidence:**
     - Multiclass Brier score increased from `0.3168` (Center 1) to `0.5731` (Center 2) and `0.6486` (Center 3), indicating that raw softmax activations become uncalibrated under novel hospital acquisition conditions.

---

### 2.3 Clinical Deployment Readiness: STRICTLY PROHIBITED (FAIL)
- **Supporting Evidence:**
  1. **Unacceptable Oncological False Negative Rate:**
     - In Center 2, 10 primary bone malignancies were labeled as normal, and 21 were labeled as benign (total miss rate: 31 / 76 = 40.79%).
     - In Center 3, 6 primary bone malignancies were labeled as normal, and 4 were labeled as benign (total miss rate: 10 / 31 = 32.26%).
     - In oncology, misdiagnosing an aggressive osteosarcoma, Ewing sarcoma, or chondrosarcoma as a normal bone radiograph can directly cause diagnostic delay, pathological fracture, metastatic seeding, and avoidable mortality.
  2. **Uncalibrated Probability Scores:**
     - The network outputs raw softmax scores that convey pseudo-certainty without formal uncertainty quantification or conformal prediction intervals.
  3. **Absence of Regulatory & Multi-Center Clinical Verification:**
     - No clinical trial has evaluated patient outcomes, reader-assistant interaction, or institutional demographic subgroup fairness.

---

### 2.4 Suitability for Integration into Final Diagnostic Classifier: UNSUITABLE IN CURRENT FORM

> [!IMPORTANT]
> **CRITICAL ARCHITECTURAL DISTINCTION — SCOPE OF CLASSIFICATION:**
> 
> The current BTXRD classifier predicts exactly three clinical classes:
> - **Class 0:** Normal bone radiograph
> - **Class 1:** Benign primary bone tumour (e.g., osteoma, chondroma, non-ossifying fibroma)
> - **Class 2:** Primary malignant bone tumour (e.g., osteosarcoma, chondrosarcoma, Ewing sarcoma)
> 
> **It does NOT classify:**
> 1. **Secondary Bone Metastasis** (disseminated carcinoma from breast, prostate, lung, kidney, thyroid, etc. — addressed in the Bone Scintigraphy module).
> 2. **Paget's Disease of Bone** (metabolic/osteoclastic dysregulation).
> 3. **Osteogenesis Imperfecta** (genetic collagen dysplasia and fragility).
> 
> **Scientific Integrity Requirement:**
> Completing the three-class BTXRD benchmark (Stages 15–20) does **not** satisfy the final four-class OsteoGuard AI differential diagnostic objective. Attempting to directly use this model as the core engine for differential metabolic or secondary metastatic diagnosis would be scientifically invalid.

---

## 3. Final Summary Verdict

The frozen ResNet-34 classifier represents an informative **research milestone** that illuminates the real-world challenge of radiographic domain shift and institutional referral bias. It is officially **DECOMMISSIONED from standalone clinical consideration** and marked as **INVESTIGATIONAL ONLY**, establishing clear empirical requirements for subsequent multi-modal architectural evolution.

