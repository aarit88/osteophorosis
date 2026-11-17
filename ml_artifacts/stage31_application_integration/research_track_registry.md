# Research Track Registry: Strict Segregation of Sub-Track Models

## 1. Principles of Track Segregation
To uphold scientific integrity and prevent invalid diagnostic claims, OsteoGuard AI maintains **four strictly segregated research tracks**. 

Preliminary models developed during early research stages addressed narrow sub-problems (e.g., binary fracture identification, radionuclide uptake screening, primary tumour differentiation). These preliminary sub-tracks **must never be repurposed, remapped, or conflated with the final four-class differential diagnosis**.

---

## 2. Research Track Specifications

### Track A: FracAtlas Fracture Research
- **Track Identifier**: `track_a`
- **Clinical Task**: Binary Fracture Detection (Fractured vs. Non-Fractured)
- **Modality**: Plain Radiography (X-Ray)
- **Target Output Classes**:
  1. Fractured
  2. Non-fractured
- **Model Checkpoint**: Frozen (Stage 09 / Stage 10)
- **Architecture**: ResNet-101 (Pretrained IMAGENET1K_V2, 42.5M parameters)
- **Historical Evidence (Holdout Test N=614)**:
  - ROC-AUC: 0.9117 (95% CI: [0.8838, 0.9370])
  - PR-AUC: 0.7537 (95% CI: [0.6775, 0.8171])
  - Sensitivity (Recall at θ=0.31): 90.00% (99/110)
  - Specificity (at θ=0.31): 74.40% (375/504)
  - Balanced Accuracy: 82.20%
  - Overall Accuracy: 77.20% (θ=0.31) | 88.27% (θ=0.50)
  - F1-Score: 0.5858 (θ=0.31) | 0.6436 (θ=0.50)
  - NPV: 97.15% (θ=0.31)
- **Critical Limitations & Governance Disclosures**:
  - *Absence of Patient Identifiers*: Patient IDs were not provided in the original FracAtlas dataset. Consequently, patient-level independence between train and test partitions cannot be rigorously guaranteed.
  - *Narrow Task Scope*: Differentiates fractured bone from non-fractured bone only; does NOT assess metabolic bone disease, osteolysis, Paget's disease, or bone metastasis.
  - *Holdout Partition*: Test split evaluated in Stage 10 (N=614) is burned and must not be re-used.

---

### Track B: Bone Scintigraphy Metastasis Research
- **Track Identifier**: `track_b`
- **Clinical Task**: Binary Skeletal Metastasis Assessment
- **Modality**: Tc-99m Whole-Body Planar Bone Scintigraphy (Anterior & Posterior)
- **Target Output Classes**:
  1. Metastasis-positive
  2. No detected bone metastasis in breast cancer staging cohort
- **Model Checkpoint**: Frozen (Stage 13 / Stage 14)
- **Architecture**: Patient-Aware 5-Fold ResNet-18 Ensemble with Clinical Max-Pooling (55.9M parameters)
- **Historical Evidence (Holdout Test N=44 Patients, 88 Scans)**:
  - ROC-AUC: 0.8317 (95% CI: [0.6214, 0.9825])
  - PR-AUC: 0.7257 (95% CI: [0.4285, 0.9286])
  - Overall Accuracy (θ=0.60): 84.09%
  - Balanced Accuracy (θ=0.60): 77.62%
  - Sensitivity (θ=0.60): 66.67% (6/9) | Sensitivity (θ=0.45): 77.78% (7/9)
  - Specificity (θ=0.60): 88.57% (31/35)
  - Precision (PPV): 60.00%
  - Negative Predictive Value: 91.18%
  - F1-Score: 0.6316
  - Brier Calibration Score: 0.1791
- **Critical Limitations & Governance Disclosures**:
  - *Negative Class is Not a Healthy Control*: The negative cohort consists of breast cancer staging patients with no detectable metastatic bone involvement; it is NOT an asymptomatic normal control population.
  - *Radionuclide Specificity*: Bone scintigraphy visualizes osteoblastic metabolic activity; uptake is frequently confounded by osteoarthritis, trauma, and degenerative remodeling.
  - *Holdout Partition*: Test split evaluated in Stage 14 (44 patients, 88 scans) is burned and must not be re-evaluated.

---

### Track C: BTXRD Primary Bone Tumour Research
- **Track Identifier**: `track_c`
- **Clinical Task**: 3-Class Primary Tumour Classification
- **Modality**: Planar Radiography (X-Ray)
- **Target Output Classes**:
  1. Normal / non-tumour bone
  2. Benign primary bone tumour
  3. Primary malignant bone tumour
- **Model Checkpoint**: Cryptographically Frozen (Stage 18)
  - SHA-256: `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`
- **Architecture**: ResNet-34 Aspect-Ratio Letterboxed (512x512) + Focal Loss (gamma=2.0)
- **Historical Multi-Center Evidence**:
  - Center 1 (Internal Val, N=588): Overall Acc: 81.80%, Balanced Acc: 82.90%, Macro-F1: 0.8103, Malignant Sens: 90.24%
  - Center 2 (External Holdout, N=549): Overall Acc: 51.73%, Balanced Acc: 53.74%, Macro-F1: 0.5069, Malignant Sens: 59.21%
  - Center 3 (External Stress Test, N=259): Overall Acc: 47.10%, Balanced Acc: 44.67%, Macro-F1: 0.3592, Malignant Sens: 67.74%
- **Critical Limitations & Governance Disclosures**:
  - *Catastrophic Multi-Center Generalization Collapse*: Cross-center testing revealed a drop of 30.07 to 34.70 percentage points in accuracy and drop of 37.4% to 55.7% in Macro-F1 due to radiographic beam quality, post-processing variations, and population shift.
  - *Pathological Distinction*: Primary malignant bone tumours (Osteosarcoma, Ewing's) differ fundamentally in etiology, histopathology, and imaging appearance from secondary bone metastases.
  - *Burned External Partitions*: Center 2 (N=549) and Center 3 (N=259) are burned external validation sets. Under good machine learning practice, they cannot be incorporated into training.
  - *Clinical Suitability*: Not suitable for clinical deployment.

---

### Track D: Final Four-Class OsteoGuard AI Diagnostic System
- **Track Identifier**: `track_d`
- **Clinical Task**: 4-Class Differential Diagnosis & Biomechanical Fracture Risk Stratification
- **Modality**: Multimodal (Plain Radiography + Scintigraphy + CT)
- **Intended Target Classes**:
  1. Secondary bone metastasis
  2. Paget's disease of bone
  3. Osteogenesis imperfecta (OI)
  4. Healthy skeletal control
- **System Status**: `BLOCKED — APPROVED CLINICAL DATA AND GOVERNANCE REQUIRED`
- **Model Training Status**: `ZERO MODELS TRAINED` (0 / 4 approved cohorts acquired)
- **Execution Safeguard**:
  - Calling `POST /api/v1/diagnostic/four-class` is permanently intercepted and returns `HTTP 503 SERVICE UNAVAILABLE` with diagnostic blockers.
- **Prerequisites for Unblocking**:
  - Institutional Ethics Committee (IEC/IRB) approval (ICMR Form A).
  - Executed bilateral Data Use Agreements (DUA).
  - Clinical acquisition of verified human Paget's disease radiographs ($N \ge 100$).
  - Acquisition of pediatric/adult human OI radiographs ($N \ge 100$).
  - Matched asymptomatic healthy control cohort ($N \ge 250$).
  - Resolution of remaining 21 unverified radiomics features against IBSI standards.

