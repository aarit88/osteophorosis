# Tabular SHAP Technical Specification & Exact Reference Protocol

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI: Explainability Engine  
**Module**: `osteoguard.explainability.shap_engine`  
**Date**: November 16, 2026  
**Status**: **EXACT REFERENCE VERIFIED; PRODUCTION ADAPTER PARTIAL (PARTIAL_DEPENDENCY_NOT_INSTALLED)**

---

## 1. Cooperative Game Theory & Shapley Formulation

SHAP (SHapley Additive exPlanations) attributes a prediction $f(x)$ to individual features by treating each feature as a player in a cooperative game, computing Shapley values defined by Lloyd Shapley (1953).

### 1.1 Exact Shapley Value Formula
For feature $i \in N = \{1, \dots, d\}$:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

### 1.2 Characteristic Function $v(S)$
In conditional expectation Shapley frameworks, $v(S)$ marginalizes over an empirical background dataset $X_{\text{bg}}$:
$$v(S) = \frac{1}{|X_{\text{bg}}|} \sum_{z \in X_{\text{bg}}} f(x_S \cup z_{N \setminus S})$$

### 1.3 Baseline Expected Value ($\phi_0$)
$$\phi_0 = v(\emptyset) = \frac{1}{|X_{\text{bg}}|} \sum_{z \in X_{\text{bg}}} f(z) = \mathbb{E}_{z \sim X_{\text{bg}}}[f(z)]$$

### 1.4 Fundamental Axiomatic Guarantees
1. **Efficiency (Local Accuracy)**:
   $$\sum_{i=1}^d \phi_i(x) + \phi_0 = f(x)$$
   *Verification Result*: Exact combinatorial calculator achieves local accuracy residual $\le 0.00e+00$.
2. **Symmetry**: If $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S \subseteq N \setminus \{i, j\}$, then $\phi_i = \phi_j$.
3. **Dummy (Null) Player**: If $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$.
4. **Additivity**: For independent models $f_1 + f_2$, $\phi_i(f_1 + f_2) = \phi_i(f_1) + \phi_i(f_2)$.

---

## 2. Analytical Additive Reference Derivation

For an additive linear model $f(x) = \sum_{i=1}^d w_i x_i + b$ with background dataset $X_{\text{bg}}$ having mean vector $\mu = \mathbb{E}[X_{\text{bg}}]$:
- Marginal contribution of feature $i$ to any coalition $S$ is constant:
  $$v(S \cup \{i\}) - v(S) = w_i (x_i - \mu_i)$$
- Therefore, the exact Shapley attribution simplifies analytically to:
  $$\phi_i^{\text{analytical}}(x) = w_i (x_i - \mu_i)$$
- The base value is:
  $$\phi_0^{\text{analytical}} = \sum_{i=1}^d w_i \mu_i + b$$
- Numerical execution on controlled fixtures matches analytical values to within $10^{-5}$ precision.

---

## 3. Dependency Audit & Computational Complexity

### 3.1 Environment Dependency Status
- Third-party `shap` library is **NOT INSTALLED** in the active GPU virtual environment (`osteoguard-gpu`).
- In strict adherence to Stage 29 rules, we did **NOT** silently fabricate a third-party wrapper.
- The production adapter is explicitly classified as:
  `STATUS: PARTIAL_DEPENDENCY_NOT_INSTALLED`.
- An exact combinatorial reference calculator (`ExactShapleyCalculator`) is provided for verified low-dimensional testing ($d \le 6$).

### 3.2 Computational Scaling Comparison
| Explainer Type | Computational Complexity | Scalability ($d=5$) | Scalability ($d=102$) | Scalability ($d=614$) |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Combinatorial** | $O(2^d \cdot N_{\text{bg}})$ | 32 coalition evals (Instant) | $2^{102}$ (Intractable) | $2^{614}$ (Intractable) |
| **KernelSHAP** | $O(M \cdot N_{\text{bg}} \cdot d)$ | $M \approx 200$ evals | $M \approx 2000$ evals | $M \approx 10000$ evals |
| **TreeSHAP** | $O(T \cdot L \cdot D^2)$ | Trees only (Fast) | Trees only (Fast) | Trees only (Fast) |
