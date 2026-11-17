# Radiomics Specification Discrepancy Register

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Component**: Radiomics Texture Engine (102-Dimensional Feature Vector)  
**Date**: November 16, 2026  
**Status**: **EXPLICIT DISCREPANCY REGISTERED — SPECIFICATION MARKED AS PARTIALLY VERIFIED**  

---

## 1. Discrepancy Overview and Source Conflict

An exhaustive review of the originating research documentation reveals a mathematical and taxonomic discrepancy regarding the exact definition of the **102-dimensional radiomic feature vector**:

```
+--------------------------------------------------------------------------------------------------+
|                                    SPECIFICATION CONFLICT MATRIX                                 |
+--------------------------+-----------------------+-----------------------+-----------------------+
| Document Source          | Documented Feature    | Documented Feature    | Internal Arithmetic   |
|                          | Families              | Vector Length         | Consistency           |
+--------------------------+-----------------------+-----------------------+-----------------------+
| 1. KSCST Proposal        | 4 Families Listed:    | Asserts exactly       | DISCREPANT:           |
|    Summary Section       | - First-Order (18)    | 102 Dimensions        | 18 + 26 + 16 + 16     |
|                          | - GLCM (26)           | (512 + 102 = 614)     | = 76 features         |
|                          | - GLRLM (16)          |                       | (26 features missing) |
|                          | - GLSZM (16)          |                       |                       |
+--------------------------+-----------------------+-----------------------+-----------------------+
| 2. OsteoGuard AI         | 5 Families Listed:    | Asserts exactly       | ARITHMETICALLY        |
|    Research Paper        | - First-Order (18)    | 102 Dimensions        | CONSISTENT:           |
|                          | - GLCM (26)           | (512 + 102 = 614)     | 18 + 26 + 16 + 16 + 26|
|                          | - GLRLM (16)          |                       | = 102 features        |
|                          | - GLSZM (16)          |                       |                       |
|                          | - GLDM/NGTDM (26)     |                       |                       |
+--------------------------+-----------------------+-----------------------+-----------------------+
| 3. Official IBSI         | NGTDM defines 5 feats.| Standard definitions  | STANDARDIZATION       |
|    Standard Benchmark    | GLDM defines 14 feats.| do not natively yield | GAP: Blended 26-feat  |
|    (Zwanenburg et al.)   | (in PyRadiomics / IBSI| a 26-feature family   | family is an engineer-|
|                          | extension)            |                       | ing heuristic.        |
+--------------------------+-----------------------+-----------------------+-----------------------+
```

---

## 2. Detailed Root-Cause Analysis

### Discrepancy A: The 76 vs. 102 KSCST Proposal Abbreviation
In the KSCST student grant summary text, the proposal explicitly detailed four standard matrices:
$$18\text{ (First-Order)} + 26\text{ (GLCM)} + 16\text{ (GLRLM)} + 16\text{ (GLSZM)} = 76\text{ features}$$
However, in the system architecture section of the same proposal, the late-fusion vector is mathematically defined as:
$$v_{\text{fusion}} = [\hat{v}_{\text{sem}} \oplus \hat{v}_{\text{rad}}] \in \mathbb{R}^{614} \quad \text{where } \hat{v}_{\text{sem}} \in \mathbb{R}^{512}, \hat{v}_{\text{rad}} \in \mathbb{R}^{102}$$
This indicates that the proposal narrative abbreviated the feature family list, inadvertently omitting the 26-feature fifth category.

### Discrepancy B: The GLDM / NGTDM 26-Feature Composition
In the technical code (`osteoguard/constants.py` and `osteoguard/radiomics/engine.py`), the 26-feature category was implemented as `IBSI:26DV` ("Neighboring Gray Tone Difference / Dependence Matrix"), combining:
- **5 NGTDM Features**: Coarseness (IBSI:QCDE), Contrast (IBSI:65HE), Busyness (IBSI:NQ30), Complexity (IBSI:HDEZ), and Strength (IBSI:1X9X). These 5 conform strictly to standard formulas.
- **21 Padded GLDM Features**: Small/Large Dependence Emphasis, Variance, Entropy, Energy, Homogeneity, and arbitrary multiplier variants (`entropy_m = de * 0.8`, `contrast_m = contrast * 1.2`).
- **Critical Audit Finding**: These 21 padded features represent an **engineering heuristic** designed to bridge the dimensionality to exactly 102, rather than canonical IBSI-standardized descriptors.

---

## 3. Formal Disposition and Governance Action

In strict accordance with Stage 27 rules:
1. **No Silent Modification**: The 102-dimensional feature vector has **NOT** been silently modified, replaced, or truncated.
2. **Feature Status Classification**:
   - First-Order (18 features): **ANALYTICALLY VERIFIED (IBSI:Q3RU)**
   - GLCM (26 features): **ANALYTICALLY VERIFIED (IBSI:P88C)**
   - GLRLM (16 features): **ANALYTICALLY VERIFIED (IBSI:9SA3)**
   - GLSZM (16 features): **ANALYTICALLY VERIFIED (IBSI:4NB3)**
   - NGTDM (5 features): **ANALYTICALLY VERIFIED (IBSI:26DV)**
   - GLDM Padded (21 features): **NOT_VERIFIED (ENGINEERING HEURISTIC)**
3. **Master Engine Verdict**:
   - **Overall 102-D Radiomic Output Status**: **PARTIAL / NOT FULLY VERIFIED**
   - The 614-D fusion contract ($512 + 102 = 614$) is **TECHNICALLY READY** in code, but the formal standardization of the 21 GLDM features requires faculty advisor and clinical PI sign-off.
