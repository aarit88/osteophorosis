# KSCST Project Alignment and Architectural Status Audit

**Project**: OsteoGuard AI: Unified Deep Learning and Radiomic Framework for Differential Diagnosis and Biomechanical Fracture Risk Stratification  
**Originating Academic Grant**: Karnataka State Council for Science and Technology (KSCST) Student Project Programme  
**Document Status**: Formal Architectural and Milestone Alignment Audit  
**Date**: November 16, 2026  

---

## 1. Executive Alignment Summary

This audit compares the original KSCST project proposal, research paper specifications, and Phase-I architectural blueprints against the current implementation status following the Stage 25 Controlled Dataset Intake.

The core research objectives and architectural foundations of OsteoGuard AI remain **100% faithful to the original proposal**:
1. **Target Diagnostic Taxonomy**: The intended four-class differential diagnosis remains strictly unchanged:
   - **Class 0 / 1**: Secondary bone metastasis
   - **Class 1 / 2**: Paget's disease of bone (*Osteitis Deformans*)
   - **Class 2 / 3**: Osteogenesis imperfecta (OI)
   - **Class 3 / 4**: Healthy skeletal controls
2. **Dual-Stream Framework**: Deep semantic convolutional/Transformer stream combined with a 102-D IBSI-compliant radiomic texture stream via 614-D late fusion.
3. **Biomechanical Fracture Engine**: Continuous Mirels' lesion-to-cortex ratio ($LCR$) and polar moment of inertia degradation ($J_{\text{defect}}$).
4. **Explainability**: Spatial Grad-CAM heatmaps and radiomic SHAP counterfactual explanations.

---

## 2. Component-by-Component Milestone Tracking

```
+--------------------------------------------------------------------------------------------------+
|                                    KSCST MILESTONE TRACKING MATRIX                               |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| #  | Proposed Component          | Architectural Status  | Current Milestone | Data Block Status |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 1  | 4-Class Diagnostic Taxonomy | Implemented in        | Preserved         | BLOCKED on Paget, |
|    | (Mets, Paget, OI, Healthy)  | osteoguard/constants  | Canonical Spec    | human OI, matched |
|    |                             |                       |                   | normal X-rays     |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 2  | U-Net Skeletal Segmentation | Implemented in        | Code Complete &   | Ready for training|
|    | Masking Stream              | osteoguard/models/unet| Synthetically Unit| on clinical DICOM |
|    |                             |                       | Tested            |                   |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 3  | Deep Semantic Feature       | Implemented in        | ResNet-34 Evaluat.| 4-Class Unified   |
|    | Extractor (ResNet / Swin-T) | osteoguard/models/    | in BTXRD 3-Class  | Model Blocked on  |
|    |                             | backbones             | Benchmark (C1-C3) | Data Intake       |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 4  | IBSI Radiomics Engine       | Implemented in        | Code Complete &   | Ready for feature |
|    | (102 Features, 5 Families)  | osteoguard/radiomics  | Mathematically    | extraction on raw |
|    |                             |                       | Verified          | radiographs       |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 5  | 614-D Late-Fusion Head      | Implemented in        | Code Complete     | Awaiting joint    |
|    | (512-D Sem + 102-D Rad)     | osteoguard/models/    | (MLP Head + Focal | semantic-radiomic |
|    |                             | fusion                | Loss gamma=2.0)   | training data     |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 6  | Biomechanical Fracture      | Implemented in        | Mathematical Eq.  | Ready for clinical|
|    | Risk Engine (Mirels' LCR)   | osteoguard/biomechan. | Fully Formulated  | calibration       |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 7  | Explainability Protocols    | Implemented in        | Code Complete &   | Verified on       |
|    | (Grad-CAM & SHAP)           | osteoguard/explainab. | Unit Tested       | synthetic batches |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 8  | FastApi Inference Backend   | Implemented in        | Endpoints Defined | Production-ready  |
|    | & React Web Dashboard       | osteoguard/api/ & src | (/api/v1/analyze) | scaffolding       |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 9  | Bone Scintigraphy Metastasis| Evaluated in          | Stages 11-14 PASS | Single-modality;  |
|    | Dual-View Ensemble          | ml_artifacts/bone_scan| Holdout Macro-F1  | planar nuclear    |
|    |                             |                       | = 0.9412          | scans only        |
+----+-----------------------------+-----------------------+-------------------+-------------------+
| 10 | BTXRD Three-Center Primary  | Evaluated in          | Stages 15-21 PASS | Interim benchmark;|
|    | Bone Tumour Benchmark       | ml_artifacts/btxrd    | Severe domain     | Primary sarcomas  |
|    |                             |                       | shift demonstrated| != metastases     |
+----+-----------------------------+-----------------------+-------------------+-------------------+
```

---

## 3. Clear Distinction: Interim BTXRD Benchmark vs. Final 4-Class Goal

A crucial scientific distinction must be maintained for faculty and external reviewers:
- **BTXRD Benchmark (Stages 15–21)**:
  - An interim technical feasibility benchmark evaluating a deep convolutional classifier on 3 classes of primary bone lesions (`0 = Normal`, `1 = Benign primary tumour`, `2 = Primary malignant sarcoma`).
  - Proved that single-center models suffer catastrophic domain shift on external holdouts (Center 1: 0.8103, Center 2: 0.5069 [-37.44% relative], Center 3: 0.3592 [-55.67% relative]).
  - Proved that primary mesenchymal sarcomas (osteosarcoma, Ewing sarcoma) have distinct biology, clinical presentation, and radiological features and **cannot substitute for secondary bone metastases**.
- **Final OsteoGuard AI 4-Class Research Goal**:
  - The true objective of the KSCST proposal is a multi-modal differential diagnostic framework for metabolic, metastatic, genetic, and normal bone conditions:
    1. Secondary bone metastasis
    2. Paget's disease of bone
    3. Osteogenesis imperfecta
    4. Healthy skeletal controls
  - The BTXRD model is **NOT** the final four-class model. The final four-class model cannot be trained until ethical institutional data acquisition is completed.
