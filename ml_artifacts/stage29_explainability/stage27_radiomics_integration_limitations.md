# Stage 27 Radiomics Integration Limitations

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Date**: November 16, 2026  
**Status**: **FORMAL INTEGRATION BLOCKER NOTICE**

---

## 1. Discrepancy Status from Stage 27

Stage 27 evaluated the 102-dimensional IBSI radiomic feature vector against exact analytical digital phantoms:
- **81 Features Analytically Verified**:
  - First-Order Statistics: 18 features
  - GLCM (Gray-Level Co-occurrence): 26 features
  - GLRLM (Gray-Level Run-Length): 16 features
  - GLSZM (Gray-Level Size-Zone): 16 features
  - NGTDM (Neighborhood Gray-Tone Difference): 5 features
- **21 Features Unverified**:
  - Identified as an engineering padding heuristic to satisfy the legacy 102-dimensional vector count.
  - Overall Radiomics Status: **PARTIAL / SPECIFICATION UNRESOLVED**.

---

## 2. Impact on Stage 29 Explainability

> [!CAUTION]
> **Prohibition on Fabricated Radiomic SHAP Explanations**:
> 1. We strictly prohibit generating SHAP feature attributions on the unverified 21 GLDM padding features.
> 2. The 614-dimensional fused representation ($512\text{ semantic} + 102\text{ radiomic}$) CANNOT be validated for SHAP attribution until the radiomics feature specification is formally resolved and signed off by the faculty advisor.
> 3. Stage 29 therefore validates SHAP exclusively on an independent 5-feature synthetic tabular schema (`SYNTHETIC_FEATURE_NAMES`).
> 4. A modular registration mechanism (`TabularShapAdapter.validate_schema`) is implemented so that the approved feature vector can be connected seamlessly once authorized.
