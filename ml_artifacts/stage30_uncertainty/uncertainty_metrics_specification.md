# Uncertainty Metrics Technical Specification

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI: Uncertainty Engine  
**Module**: `osteoguard.uncertainty.metrics`  
**Date**: November 16, 2026  
**Status**: **MATHEMATICALLY VERIFIED**

---

## 1. Mathematical Formulations

For a $K$-class discrete probability vector $p = [p_1, \dots, p_K]$ satisfying $\sum_{i=1}^K p_i = 1$ and $p_i \ge 0$:

### 1.1 Raw Shannon Predictive Entropy
$$H(p) = -\sum_{i=1}^K p_i \ln(p_i) \quad [\text{nats}]$$
*Zero Handling Protocol*: $0 \cdot \ln(0) \equiv 0.0$ via $\lim_{x \to 0^+} x \ln(x) = 0$.

### 1.2 Normalized Shannon Entropy
$$H_{\text{norm}}(p) = \frac{H(p)}{\ln(K)} \quad \in [0.0, 1.0]$$

### 1.3 Maximum Class Probability & Top-Two Margin
$$p_{(1)} = \max_{i} p_i$$
$$\text{Margin} = p_{(1)} - p_{(2)}$$

---

## 2. Analytical Reference Values ($K = 4$)

| Distribution | $p$ | Raw Entropy ($H$) | Normalized ($H_{\text{norm}}$) | Margin | Analytical Formula |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Uniform** | $[0.25, 0.25, 0.25, 0.25]$ | $\ln(4) \approx 1.3863$ | $1.0000$ | $0.0000$ | $\ln(K)$ |
| **Deterministic** | $[1.00, 0.00, 0.00, 0.00]$ | $0.0000$ | $0.0000$ | $1.0000$ | $0$ |
| **Two-Class Tie** | $[0.48, 0.48, 0.02, 0.02]$ | $0.8604$ | $0.6206$ | $0.0000$ | $-2(0.48\ln 0.48 + 0.02\ln 0.02)$ |
| **High Confidence** | $[0.94, 0.02, 0.02, 0.02]$ | $0.2928$ | $0.2112$ | $0.9200$ | $-(0.94\ln 0.94 + 3 \cdot 0.02\ln 0.02)$ |

---

## 3. Strict Input Validation Standards
- Probabilities must sum to $1.0 \pm 10^{-4}$.
- Out-of-bounds probabilities ($< 0.0$ or $> 1.0 + 10^{-4}$) raise `ValueError`.
- NaN or Infinite values raise `ValueError`.
- The engine **NEVER silently normalizes** invalid vectors.
