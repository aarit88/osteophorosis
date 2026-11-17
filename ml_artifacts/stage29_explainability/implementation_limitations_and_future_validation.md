# Implementation Limitations & Future Clinical Validation Requirements

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Date**: November 16, 2026  
**Status**: **PRE-CLINICAL RESEARCH SAFEGUARDS**

---

## 1. Current Implementation Limitations

1. **Third-Party Dependency Absence**: The `shap` Python package is not installed in the GPU virtual environment. Production KernelSHAP and TreeSHAP adapters remain pending library installation.
2. **Uncalibrated on Clinical Data**: All explainability testing in Stage 29 was conducted on synthetic digital CNN fixtures and controlled mathematical equations.
3. **Absence of 4-Class Trained Weights**: No model in the repository has been trained on the final 4-class taxonomy (`metastasis`, `paget`, `oi`, `healthy`).

---

## 2. Requirements for Future Clinical Validation

Before any explainability output may be presented to medical professionals:
1. **Institutional Ethical Approval**: Formal sign-off by the Institutional Ethics Committee (IEC).
2. **Clinical Image Pointing Game**: Evaluation of Grad-CAM localization accuracy against blinded radiologist bounding boxes and segmentation ground-truths.
3. **Radiologist Reader Studies**: Controlled user trials assessing whether heatmaps improve diagnostic accuracy or induce automation bias.
4. **Computational Faithfulness Audits**: Sensitivity-N testing, deletion/insertion curve evaluation, and sanity checks on real clinical backbones.
