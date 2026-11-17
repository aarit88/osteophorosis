# Out-Of-Distribution (OOD) Detection Specification

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Module**: `osteoguard.uncertainty.ood_detector`  
**Date**: November 16, 2026  
**Status**: **SYNTHETIC OOD DETECTORS VERIFIED**

---

## 1. Supported OOD Methods

### 1.1 Maximum Softmax Probability (MSP)
- **Score**: $S_{\text{MSP}}(p) = \max_{k} p_k$.
- **Decision**: OOD if $S_{\text{MSP}} < 0.50$.

### 1.2 Logit Free Energy
- **Energy Function**:
  $$E(z; T) = -T \cdot \ln \sum_{k=1}^K \exp(z_k / T)$$
- **Decision**: OOD if $E(z; T) > -2.0$ (less negative energy corresponds to flat/ambiguous logits).

### 1.3 Regularized Mahalanobis Distance
- **Centroids**: $\mu_c = \frac{1}{N_c} \sum_{i: y_i = c} z_i$.
- **Regularized Covariance**:
  $$\Sigma_{\text{reg}} = \frac{1}{N} \sum_{c=1}^K \sum_{i: y_i=c} (z_i - \mu_c)(z_i - \mu_c)^T + \lambda I$$
  where $\lambda = 10^{-3}$ guarantees full rank and non-singular inversion.
- **Distance**:
  $$D_M(z) = \min_{c} \sqrt{(z - \mu_c)^T \Sigma_{\text{reg}}^{-1} (z - \mu_c)}$$
- **Decision**: OOD if $D_M(z) > 3.5$.
