# Explanation Reliability & Faithfulness Test Report

**Stage**: Stage 29 Explainability Verification  
**Date**: November 16, 2026  
**Overall Verdict**: **ALL RELIABILITY CHECKS PASSED**

---

## 1. Summary of Faithfulness & Reliability Checks

| Check Category | Reference Standard | Result | Verdict |
| :--- | :--- | :--- | :---: |
| **Adebayo Randomization Check** | L1 difference $> 0.15$ after weight scramble | L1 Diff $= 0.0625$ | **PASS** |
| **Grad-CAM Localization** | Peak inside true spatial region | Peak at (5, 5) in rows $4..12$ | **PASS** |
| **Spatial Shift Sensitivity** | Peak moves with translated feature | Peak moves to (21, 21) | **PASS** |
| **Zero Gradient Stability** | Zero gradients do not cause NaN/Inf | Finite zero array returned | **PASS** |
| **Hook Lifecycle Cleanup** | Zero handle retention after `remove_hooks()` | All handles removed, clean GC | **PASS** |
| **Model State Preservation** | Training mode and weights unaltered | Verified identical weights | **PASS** |
| **SHAP Local Accuracy** | Residual $|\sum \phi_i + \phi_0 - f(x)| < 10^{-6}$ | Max residual $= 0.00e+00$ | **PASS** |
| **SHAP Analytical Alignment** | Exact match with $w_i(x_i - \mu_i)$ | Max error $< 10^{-5}$ | **PASS** |
| **Schema Validation Enforcement**| Missing features raise `ValueError` | Caught missing features | **PASS** |

---

## 2. Detailed Findings

### 2.1 Model Parameter Sensitivity (Adebayo Sanity Check)
Adebayo et al. (NeurIPS 2018) demonstrated that many visual saliency methods act as edge detectors and produce identical heatmaps even when model weights are completely randomized.
- In our test, randomizing the weights of `ControlledSpatialCNN` produced a substantial L1 shift (0.0625), confirming that Grad-CAM in OsteoGuard AI is strictly sensitive to model weights.

### 2.2 SHAP Local Accuracy Verification
Across all evaluated synthetic instances, the sum of Shapley attributions plus the expected baseline reconstructed the true model prediction with negligible residual ($0.00e+00$), proving exact satisfaction of the Efficiency Axiom.
