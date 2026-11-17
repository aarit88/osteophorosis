# Implementation Limitations and Unresolved Decisions

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Component**: Radiomics Texture Extraction Engine  
**Date**: November 16, 2026  

---

## 1. Technical Limitations of Current Engine

1. **2D Projection Assumption**: The radiomics engine currently operates exclusively on 2D digital radiograph pixel arrays $(H, W)$. It does not extract 3D voxel-level spherical radiomics for volumetric CT (which is reserved for Pathway C).
2. **Fixed Bin Count Discretization**: The engine implements uniform fixed bin count ($N_g = 64$). In certain radiomics protocols, fixed bin size (bin width $W$) is preferred for calibrated CT Hounsfield Units.
3. **GLDM Standardization Gap**: 21 of the 102 features represent heuristic GLDM padding. While mathematically stable, they should be reviewed and aligned with official PyRadiomics GLDM-14 standards.

---

## 2. Unresolved Decisions Requiring Faculty Advisor Review

To transition from the current Stage 27 PARTIAL verification to full clinical compliance:
1. **Decision on 102-D Feature Vector Composition**:
   - *Option A (Preserve 102-D Contract)*: Retain the current 102-D vector to strictly preserve the $512 + 102 = 614$ late-fusion MLP architecture from the KSCST proposal, documenting the 21 GLDM features as an internal heuristic.
   - *Option B (Align with Canonical IBSI)*: Re-specify the radiomic vector to canonical IBSI standards ($18 + 26 + 16 + 16 + 5\text{ NGTDM} + 14\text{ GLDM} = 95\text{ features}$), adjusting the fusion dimension from 614 to $512 + 95 = 607$.
2. **Decision on PyRadiomics Dependency**:
   - Evaluate whether to retain the pure lightweight NumPy/SciPy engine or bind the pipeline to the official PyRadiomics package.
