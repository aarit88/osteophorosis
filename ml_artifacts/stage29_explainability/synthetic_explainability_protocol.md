# Deterministic Synthetic Explainability Testing Protocol

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Module**: `osteoguard.explainability.synthetic_fixtures`  
**Date**: November 16, 2026  
**Status**: **DETERMINISTIC FIXTURES VERIFIED**

---

## 1. Visual Explainability Protocol (Grad-CAM)

### 1.1 Controlled Spatial CNN Architecture
- **Model**: `ControlledSpatialCNN`
- **Layers**:
  - `conv1`: `Conv2d(1, 2, kernel_size=3, padding=1, bias=False)`
  - `relu`: `ReLU()`
  - `pool`: `AdaptiveAvgPool2d((1, 1))`
  - `fc`: `Linear(2, 2, bias=False)`
- **Deterministic Fixed Weights**:
  - Filter 0: Positive uniform patch detector ($w_{0, 0} > 0$).
  - Filter 1: Negative inverted patch detector ($w_{1, 0} < 0$).
  - Classifier Head: Diagonal routing matrix $[[1.0, 0.0], [0.0, 1.0]]$.

### 1.2 Five Synthetic Input Fixtures ($32 \times 32$)
1. **Bright Spot Top-Left**: Localized square ($1.0$) at rows $4..12$, cols $4..12$. Background $0.0$.
   - *Result*: Peak at (5, 5), Top-Left energy = 0.8562, Bottom-Right energy = 0.0000.
2. **Shifted Spot Bottom-Right**: Identical square moved to rows $20..28$, cols $20..28$.
   - *Result*: Peak moved to (21, 21), Bottom-Right energy = 0.8562.
3. **Uniform Flat Image**: Constant array ($0.5$).
   - *Result*: Verified safe execution without NaNs, Infs, or divide-by-zero errors.
4. **Dark Spot Center**: Dark square ($0.0$) inside bright field ($1.0$).
5. **Two Separate Regions**: Independent patches in Top-Left and Bottom-Right.

---

## 2. Tabular Feature Attribution Protocol (SHAP)

### 2.1 Controlled Additive Linear Model
$$f(x) = 3.0 x_0 - 2.0 x_1 + 1.5 x_2 - 0.5 x_3 + 4.0 x_4 + 10.0$$

### 2.2 Registered Synthetic Feature Schema
1. `cortical_thickness_mm` ($w_0 = 3.0$)
2. `trabecular_attenuation_hu` ($w_1 = -2.0$)
3. `endosteal_scalloping_ratio` ($w_2 = 1.5$)
4. `calcium_mass_mg_cm3` ($w_3 = -0.5$)
5. `microfracture_density_idx` ($w_4 = 4.0$)

### 2.3 Deterministic Background Dataset
A 20-sample reference matrix $X_{\text{bg}} \in \mathbb{R}^{20 \times 5}$ generated via fixed mathematical progressions.
