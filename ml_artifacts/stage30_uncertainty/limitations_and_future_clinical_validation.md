# Limitations & Future Clinical Calibration Requirements

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Date**: November 16, 2026  
**Status**: **FORMAL RESEARCH GOVERNANCE LIMITATIONS**

---

## 1. Current Engineering Limitations

1. **Uncalibrated on Clinical Outcomes**: All calibration, temperature scaling, and ECE values in Stage 30 were computed on synthetic mathematical distributions.
2. **Threshold Candidate Status**: The $H > 0.60$ threshold is an engineering design parameter and has not been tuned against clinical diagnostic sensitivity/specificity.
3. **Absence of 4-Class Trained Weights**: No trained 4-class neural network exists in the repository.

---

## 2. Requirements for Future Clinical Validation

Before deploying selective prediction in clinical settings:
1. **Institutional Approval**: Approval of the selective prediction protocol by the Institutional Review Board (IRB) / Institutional Ethics Committee (IEC).
2. **Clinical Calibration Study**: Temperature scaling fitted on a verified multicenter clinical validation cohort.
3. **Clinical Utility Trial**: Radiologist reader studies assessing whether abstention reduces diagnostic errors and prevents overreliance on automated suggestions.
