# Comprehensive Radiomics Feature Specification (102 Features)

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Component**: Radiomics Feature Extraction Specification  
**Date**: November 16, 2026  

---

## 1. Feature Family Summary

```
+--------------------------------------------------------------------------------------------------+
|                                    RADIOMICS FEATURE FAMILY SUMMARY                              |
+----+-----------------------------+-----------+---------------+-----------------+-----------------+
| #  | Feature Family              | Count     | IBSI Code     | Proposal Status | Verified Status |
+----+-----------------------------+-----------+---------------+-----------------+-----------------+
| 1  | First-Order Statistics      | 18        | IBSI:Q3RU     | In Original Prop| ANALYTICALLY VER|
| 2  | Gray Level Co-occurrence    | 26        | IBSI:P88C     | In Original Prop| ANALYTICALLY VER|
| 3  | Gray Level Run-Length       | 16        | IBSI:9SA3     | In Original Prop| ANALYTICALLY VER|
| 4  | Gray Level Size Zone        | 16        | IBSI:4NB3     | In Original Prop| ANALYTICALLY VER|
| 5a | Neighboring Gray Tone Diff. | 5         | IBSI:26DV     | In Paper Arch   | ANALYTICALLY VER|
| 5b | Gray Level Dependence (Pad) | 21        | NON_STANDARD  | Engineering Pad | NOT VERIFIED    |
+----+-----------------------------+-----------+---------------+-----------------+-----------------+
|    | TOTAL FEATURE COUNT         | 102       | —             | 512+102=614 Dim | PARTIALLY VERIF.|
+----+-----------------------------+-----------+---------------+-----------------+-----------------+
```

---

## 2. Mathematical Formulations and Specifications

### Family 1: First-Order Intensity Statistics (18 Features, IBSI:Q3RU)
- **Input Properties**: Continuous intensity array within binary mask $\mathbf{X} = \{I(x, y) \mid M(x, y) = 1\}$, cardinality $N$.
- **Discretization**: Raw continuous intensities for moments; $N_g = 64$ bins for entropy/uniformity.
- **Features**:
  1. `first_order_mean`: $\mu = \frac{1}{N} \sum_{i=1}^N X_i$
  2. `first_order_variance`: $\sigma^2 = \frac{1}{N} \sum_{i=1}^N (X_i - \mu)^2$
  3. `first_order_skewness`: $\gamma_1 = \frac{1}{N \sigma^3} \sum_{i=1}^N (X_i - \mu)^3$
  4. `first_order_kurtosis`: $\beta_2 = \frac{1}{N \sigma^4} \sum_{i=1}^N (X_i - \mu)^4$
  5. `first_order_median`: 50th percentile of $\mathbf{X}$
  6. `first_order_min`: $\min(\mathbf{X})$
  7. `first_order_max`: $\max(\mathbf{X})$
  8. `first_order_iqr`: $P_{75} - P_{25}$
  9. `first_order_range`: $\max(\mathbf{X}) - \min(\mathbf{X})$
  10. `first_order_mad`: $\frac{1}{N} \sum_{i=1}^N |X_i - \mu|$
  11. `first_order_rms`: $\sqrt{\frac{1}{N} \sum_{i=1}^N X_i^2}$
  12. `first_order_energy`: $\sum_{i=1}^N X_i^2$
  13. `first_order_total_energy`: $\frac{1}{N} \sum_{i=1}^N X_i^2 \times (\Delta y \cdot \Delta x)$
  14. `first_order_uniformity`: $\sum_{k=1}^{N_g} p(k)^2$
  15. `first_order_entropy`: $-\sum_{k=1}^{N_g} p(k) \log_2(p(k) + \epsilon)$
  16. `first_order_p10`: 10th percentile
  17. `first_order_p90`: 90th percentile
  18. `first_order_rmad`: Mean absolute deviation within $[P_{10}, P_{90}]$

### Family 2: Gray Level Co-occurrence Matrix (26 Features, IBSI:P88C)
- **Input Properties**: Discretized matrix $\mathbf{I}_{\text{disc}} \in [0, N_g - 1]$.
- **Angles & Distances**: Averaged across 4 planar angles $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ at distance $\delta = 1$.
- **Features**: ASM, Contrast, Dissimilarity, Homogeneity, Correlation, Sum Average, Sum Variance, Sum Entropy, Difference Average, Difference Variance, Difference Entropy, Inverse Difference, Inverse Difference Normalized, Inverse Difference Moment Normalized, Cluster Shade, Cluster Prominence, Cluster Tendency, Maximum Probability, IMC1, IMC2, Autocorrelation, Joint Variance, Sum Squares.

### Family 3: Gray Level Run-Length Matrix (16 Features, IBSI:9SA3)
- **Input Properties**: Run lengths of collinear voxels with identical gray level.
- **Features**: SRE, LRE, GLNU, RLNU, RP, LGLRE, HGLRE, SRLGLE, SRHGLE, LRLGLE, LRHGLE, GLV, RLV, RE, GLNUN, RLNUN.

### Family 4: Gray Level Size Zone Matrix (16 Features, IBSI:4NB3)
- **Input Properties**: Connected 8-neighborhood zones of identical gray level.
- **Features**: SAE, LAE, GLNU, ZSNU, ZP, LGLZE, HGLZE, SALGLE, SAHGLE, LALGLE, LAHGLE, GLV, ZSV, ZSE, LAEN, SAEN.

### Family 5: Blended NGTDM & GLDM (26 Features, IBSI:26DV)
- **Standard NGTDM (5 Features)**: Coarseness, Contrast, Busyness, Complexity, Strength.
- **Heuristic GLDM (21 Features)**: SDE, LDE, GLV, DV, DE, LGLE, HGLE, SDLGLE, SDHGLE, LDLGLE, LDHGLE, DNU, DNUN, DCP, GLNU_M, GLNUN_M, Variance_M, Entropy_M, Energy_M, Homogeneity_M, Contrast_M.
