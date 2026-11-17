# Deterministic Synthetic Uncertainty Validation Protocol

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Module**: `osteoguard.uncertainty.synthetic_fixtures`  
**Date**: November 16, 2026  
**Status**: **ALL SYNTHETIC PROTOCOLS VERIFIED**

---

## 1. Four-Class Target Space

1. `metastasis`: Secondary bone metastasis
2. `paget`: Paget's disease of bone
3. `oi`: Osteogenesis imperfecta
4. `healthy`: Healthy skeletal control

---

## 2. Canonical Synthetic Probability Distributions

1. **`highly_confident`**: $[0.94, 0.02, 0.02, 0.02]$ ($H_{\text{norm}} = 0.2113$)
2. **`moderately_confident`**: $[0.65, 0.15, 0.10, 0.10]$ ($H_{\text{norm}} = 0.7394$)
3. **`ambiguous_two_class`**: $[0.48, 0.48, 0.02, 0.02]$ ($H_{\text{norm}} = 0.6211$)
4. **`uniform_four_class`**: $[0.25, 0.25, 0.25, 0.25]$ ($H_{\text{norm}} = 1.0000$)
5. **`near_equal`**: $[0.28, 0.26, 0.24, 0.22]$ ($H_{\text{norm}} = 0.9971$)
6. **`one_hot_zero_entropy`**: $[1.00, 0.00, 0.00, 0.00]$ ($H_{\text{norm}} = 0.0000$)
7. **`extremely_small_probs`**: $[0.9999997, 10^{-7}, 10^{-7}, 10^{-7}]$
