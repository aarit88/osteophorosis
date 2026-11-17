# Comprehensive Audit of the Proposed Entropy Threshold ($H > 0.60$)

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI: Uncertainty & Selective Prediction Subsystem  
**Date**: November 16, 2026  
**Status**: **MATHEMATICALLY RESOLVED & RIGOROUSLY AUDITED**

---

## 1. Background & Proposal Review

The original OsteoGuard KSCST research proposal and architectural documentation (`docs/ML_ARCHITECTURE.md`, Section 10.3) specify a confidence-gated human-in-the-loop review mechanism:
> *"Shannon entropy uncertainty scoring ($H$) that triggers automated clinical abstention whenever diagnostic ambiguity exceeds safe limits ($H > 0.60$ or Out-Of-Distribution heuristic)."*

However, in computational statistics and information theory, the symbol $H$ is applied to several distinct mathematical formulations:
1. **Raw Shannon Entropy in Nats** (base $e$): $H(p) = -\sum_{i=1}^K p_i \ln(p_i) \in [0, \ln(K)]$.
2. **Raw Shannon Entropy in Bits** (base $2$): $H_2(p) = -\sum_{i=1}^K p_i \log_2(p_i) \in [0, \log_2(K)]$.
3. **Normalized Shannon Entropy** (base $K$): $H_{\text{norm}}(p) = \frac{H(p)}{\ln(K)} = -\sum_{i=1}^K p_i \log_K(p_i) \in [0.0, 1.0]$.

---

## 2. Forensic Resolution of the $H > 0.60$ Ambiguity

### 2.1 Codebase & Schema Forensic Evidence
In `docs/ML_ARCHITECTURE.md`, line 314 explicitly writes:
$$H = -\sum_{c=1}^4 \hat{y}_c \log_4(\hat{y}_c) \quad \in [0.0, 1.0]$$
Furthermore, `osteophorosis-main/src/api/types.ts` (line 24) and `osteoguard/api/schemas.py` (line 35) explicitly annotate:
`entropy: float = Field(..., ge=0.0, le=1.0, description="Normalized Shannon entropy")`

### 2.2 Mathematical Comparison for Four Diagnostic Classes ($K = 4$)

| Scale Definition | Maximum Entropy Value | Threshold $\tau = 0.60$ Meaning | Fraction of Max Uncertainty | Implied Clinical Consequence |
| :--- | :---: | :---: | :---: | :--- |
| **Normalized Entropy ($H_{\text{norm}}$)** | **$1.0000$** | **$H_{\text{norm}} > 0.6000$** | **$60.0\%$** | **Balanced Triage**: Abstains when probability distribution is widely dispersed (e.g. top probability $< 0.65$). |
| **Raw Entropy in Nats ($H_{\text{nats}}$)** | $\ln(4) \approx 1.3863$ | $H_{\text{nats}} > 0.6000$ | $0.60 / 1.3863 \approx 43.3\%$ | **Excessively Conservative**: Triggers abstention on predictions with $> 75\%$ confidence! |
| **Raw Entropy in Bits ($H_{\text{bits}}$)** | $\log_2(4) = 2.0000$ | $H_{\text{bits}} > 0.6000$ | $0.60 / 2.0000 = 30.0\%$ | **Extreme Inoperability**: Triggers abstention on nearly all non-deterministic predictions. |

### 2.3 Formal Decision & Architectural Seam
1. The **canonical engineering implementation** applies the threshold strictly to **Normalized Shannon Entropy**:
   $$\text{Abstain if: } H_{\text{norm}}(p) > 0.60$$
2. On the raw natural log scale, this is mathematically equivalent to:
   $$\text{Abstain if: } H(p) > 0.8318 \text{ nats}$$
3. The comparison operator is strictly greater than ($>$). If $H_{\text{norm}} = 0.6000$, the prediction is accepted.
4. **Research Status**: $H > 0.60$ is an **unvalidated candidate engineering threshold**. It has not been optimized on real clinical outcomes.
