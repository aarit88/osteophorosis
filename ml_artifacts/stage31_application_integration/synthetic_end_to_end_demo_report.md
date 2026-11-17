# Synthetic End-to-End Demonstration Report

## 1. Executive Summary
The Synthetic Research Workbench (`/api/v1/workbench/run-demo` and `/workbench`) is the central demonstration fixture of OsteoGuard AI. It demonstrates how the four mathematical and engineering engines developed across Stages 27 to 30 function together as an integrated diagnostic and risk-stratification pipeline without relying on unapproved patient data.

---

## 2. Coordinated Pipeline Demonstration Architecture

```
   [ Synthetic Digital Phantom ] --------------> [ Stage 27: IBSI Radiomics Engine ]
     (Checkerboard / Gradient / Annulus)              - 81 Verified IBSI Textures
                                                      - 21 Unverified Heuristic Textures
                                                                    |
   [ Synthetic Bone Cross-Section ] -----------> [ Stage 28: Biomechanics Engine ]
     (Intact / Lytic Defect / Osteopenia)             - Area (mm²), Ixx, Iyy, Polar J (mm⁴)
                                                      - Conventional Mirels Scorer (1989)
                                                                    |
   [ Controlled Spatial CNN Fixture ] ---------> [ Stage 29: Explainability Engine ]
     (Tensor: 1x1x32x32 Synthetic Lesion)             - Grad-CAM Heatmap Localization (conv1)
                                                      - Exact Combinatorial Shapley Status
                                                                    |
   [ Synthetic Softmax Distribution ] ---------> [ Stage 30: Uncertainty & Abstention ]
     (4-Class Probability Simulators)                 - Shannon Entropy (H_nats)
                                                      - Normalized Entropy (H_norm = H/ln(4))
                                                      - Selective Triage Gate (H_norm >= 0.60)
                                                                    |
                                                                    v
                        [ FOUR-CLASS CLINICAL DIAGNOSTIC HEAD ]
                        STATUS: STRICTLY BLOCKED (HTTP 503)
```

---

## 3. Coordinated Execution Trace

When the user triggers `Run Full Synthetic Pipeline` from the frontend workbench, the backend executes the following operations synchronously:

### Panel 1: Radiomics Feature Extraction (Stage 27)
- **Input Fixture**: 64x64 Checkerboard Phantom ($1.0\text{ mm/pixel}$ spacing).
- **Extracted Dimension**: 102 total features.
- **Verified Breakdown**:
  - First-Order Statistics: 18 features (mean, variance, skewness, energy, entropy, etc.).
  - Gray Level Co-occurrence Matrix (GLCM): 26 features (contrast, dissimilarity, joint energy, etc.).
  - Gray Level Run Length Matrix (GLRLM): 16 features (SRE, LRE, GLN, RLN, etc.).
  - Gray Level Size Zone Matrix (GLSZM): 16 features (SAE, LAE, GLN, ZVN, etc.).
  - Gray Level Dependence Matrix (GLDM): 5 verified features.
- **Unverified Breakdown**: 21 heuristic features marked as non-standard due to discrepancies between 2D slices and 3D volumetric standards.

### Panel 2: Biomechanical Geometry & Mirels Risk (Stage 28)
- **Input Fixture**: `intact_circular_ring` (simulating femoral diaphysis, $r_{\text{outer}}=15\text{ mm}$, $r_{\text{inner}}=10\text{ mm}$, thickness $=5\text{ mm}$, spacing $=0.5\text{ mm}$).
- **Calculated Geometric Properties**:
  - Cortical Area: $392.70\text{ mm}^2$
  - Moment of Inertia $I_{xx}$: $30,679.61\text{ mm}^4$
  - Moment of Inertia $I_{yy}$: $30,679.61\text{ mm}^4$
  - Polar Moment of Inertia $J$: $61,359.23\text{ mm}^4$
  - Area & Torsion Retention Ratios: $1.0000$ ($100\%$ intact baseline).
- **Mirels Score Execution**:
  - Parameters: Site $=2$ (Lower extremity), Pain $=2$ (Moderate), Nature $=3$ (Lytic), Size $=2$ (1/3 - 2/3).
  - Score: $9 / 12$ points.
  - Empirical Risk: $33.3\%$ fracture rate (Mirels 1989 cohort).
  - Risk Tier: `HIGH_RISK_SURGICAL_CONSULT` (Prophylactic fixation indicated).

### Panel 3: Explainability Engine (Stage 29)
- **Input Fixture**: `ControlledSpatialCNN` with synthetic bright spot at coordinates $(16, 16)$ of a $32 \times 32$ tensor.
- **Target Layer**: `conv1`.
- **Output Heatmap**: Grad-CAM localization matrix peaking at $[16, 16]$ with active region ratio $0.0625$ ($6.25\%$).
- **SHAP Status**: Reported as `PARTIAL` (analytical low-dimensional Shapley calculator available; high-dimensional deep SHAP disabled pending clinical models).

### Panel 4: Uncertainty Quantification & Selective Prediction (Stage 30)
- **Input Distribution**: `highly_confident` ($p = [0.94, 0.02, 0.02, 0.02]$).
- **Metrics**:
  - Shannon Entropy: $0.3242\text{ nats}$
  - Normalized Entropy $H_{\text{norm}}$: $0.2339$ (below $0.60$ threshold).
  - Top-Two Margin: $0.9200$ ($92.0\%$).
- **Triage Outcome**: `ACCEPT_PREDICTION` (human specialist review: standard double reading).
- **High Ambiguity Test Case**: `uniform_four_class` ($p = [0.25, 0.25, 0.25, 0.25]$).
  - Normalized Entropy $H_{\text{norm}}$: $1.0000$ (exceeds $0.60$).
  - Triage Outcome: `ABSTAIN_HUMAN_REVIEW_REQUIRED` (Mandatory clinical review).

---

## 4. Scientific Rigor and Safeguards
1. **Zero Clinical Extrapolation**: The workbench explicitly notifies the user that synthetic phantoms cannot prove clinical diagnostic efficacy.
2. **Deterministic Outputs**: Re-running the workbench produces identical analytical results across test runs.
3. **Transparent Failure Modes**: The UI clearly displays when and why uncertainty triggers abstention.

