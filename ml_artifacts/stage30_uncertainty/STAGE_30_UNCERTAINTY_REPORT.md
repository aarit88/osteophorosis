# STAGE 30 — UNCERTAINTY QUANTIFICATION AND SELECTIVE PREDICTION (ABSTENTION) ENGINE VALIDATION REPORT

**Project**: OsteoGuard AI: Unified Deep Learning and Radiomic Framework  
**Stage**: 30 — Uncertainty Quantification and Selective Prediction Engine Validation  
**Execution Timestamp**: November 16, 2026  
**Environment**: GPU-Accelerated Python Environment (`osteoguard-gpu`)  
**Overall Verification Status**: **PASS (ENGINEERING & MATHEMATICAL VALIDATION COMPLETE; CLINICAL CALIBRATION UNVERIFIED)**  

---

## 1. Executive Summary

Stage 30 engineered, mathematically audited, and verified the uncertainty quantification and selective prediction components of the OsteoGuard AI framework using deterministic synthetic digital fixtures.

In strict accordance with research governance:
- **Zero patient datasets downloaded, scraped, or ingested.**
- **Zero neural network models trained or fine-tuned.**
- **BTXRD Center 2 and Center 3 remain untouched, unaccessed external holdouts.**
- **Frozen BTXRD ResNet-34 checkpoint SHA-256 (`21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`) remains strictly intact.**
- **The proposed $H > 0.60$ entropy threshold was forensically audited and resolved as Normalized Shannon Entropy ($H_{\text{norm}} \in [0.0, 1.0]$).**
- **No claim is made that the system is currently calibrated on real clinical patient outcomes.**

---

## 2. Key Accomplishments

### 2.1 Uncertainty Metrics Engine (`osteoguard.uncertainty.metrics`)
- Implemented numerically safe Shannon predictive entropy ($H$), Normalized Shannon entropy ($H_{\text{norm}}$), Maximum Class Probability, Top-Two Margin, and full distribution diagnostics.
- Analytical ground truths verified:
  - Uniform 4-class: $H = \ln(4) \approx 1.3863$ nats, $H_{\text{norm}} = 1.0000$, Margin $= 0.0000$.
  - Deterministic one-hot: $H = 0.0000$ nats, $H_{\text{norm}} = 0.0000$, Margin $= 1.0000$.
- Verified class permutation invariance and strict rejection of negative, NaN, infinite, and unnormalized inputs.

### 2.2 Forensic Audit of $H > 0.60$ (`entropy_threshold_audit.md`)
- Resolved the architectural ambiguity: `docs/ML_ARCHITECTURE.md` (line 314) and `api/types.ts` (line 24) explicitly define $H = -\sum p_i \log_4(p_i) \in [0.0, 1.0]$, which is **Normalized Shannon Entropy**.
- On the natural log scale, $H_{\text{norm}} > 0.60$ corresponds to $H > 0.8318$ nats.
- Formally classified as an **unvalidated candidate engineering threshold** requiring future clinical ROC/utility optimization.

### 2.3 Selective Prediction & Triage Engine (`osteoguard.uncertainty.selective_prediction`)
- Implemented tri-state triage: `ACCEPT_PREDICTION`, `ABSTAIN_HUMAN_REVIEW_REQUIRED`, and `INVALID_INPUT`.
- Evaluated strict boundary transitions:
  - $H_{\text{norm}} \le 0.60$ and `ood_flag == false`: `ACCEPT_PREDICTION`.
  - $H_{\text{norm}} > 0.60$: `ABSTAIN_HUMAN_REVIEW_REQUIRED` (Reason: `ELEVATED_ENTROPY`).
  - `ood_flag == true`: `ABSTAIN_HUMAN_REVIEW_REQUIRED` (Reason: `OUT_OF_DISTRIBUTION`).
  - Both triggers active: `COMBINED_ENTROPY_AND_OOD`.
- Mandatory invariant enforced: Abstention is never represented as a negative diagnosis.

### 2.4 Epistemic & Aleatoric Decomposition (`osteoguard.uncertainty.epistemic_aleatoric`)
- Decomposed predictive uncertainty via Mutual Information:
  $$I(Y, W \mid x) = H(\bar{p}) - \mathbb{E}[H(p)] \ge 0$$
- Verified: Agreeing ensemble yields $\text{MI} = 0.0000$, disagreeing ensemble yields high MI ($> 0.50$ nats).
- Validated Monte Carlo Dropout adapter across 8 stochastic passes with controlled seeding.

### 2.5 OOD Detectors & Probabilistic Calibration
- Implemented and tested MSPDetector, LogitEnergyDetector, and regularized MahalanobisFeatureDetector.
- Evaluated NLL, Brier score, ECE, and fitted TemperatureScaler ($T = 0.1000$) on synthetic partitions.
- Verified Risk-Coverage curve: rejecting uncertain cases monotonically improves selective accuracy.

---

## 3. Deliverables Summary (`ml_artifacts/stage30_uncertainty/`)

1. [`STAGE_30_UNCERTAINTY_REPORT.md`](STAGE_30_UNCERTAINTY_REPORT.md): Master report (mirrored to `ml_artifacts/stage_reports/STAGE_30_UNCERTAINTY.md`).
2. [`uncertainty_metrics_specification.md`](uncertainty_metrics_specification.md): Mathematical formulations, limits, and zero-handling rules.
3. [`selective_prediction_specification.md`](selective_prediction_specification.md): Tri-state triage architecture and abstention reason codes.
4. [`entropy_threshold_audit.md`](entropy_threshold_audit.md): Forensic audit of the $H > 0.60$ normalized vs raw entropy specification.
5. [`ood_detection_specification.md`](ood_detection_specification.md): MSP, Logit Free Energy, and regularized Mahalanobis distance algorithms.
6. [`calibration_and_risk_coverage_report.md`](calibration_and_risk_coverage_report.md): Calibration metrics and monotonic risk-coverage curve.
7. [`multimodal_uncertainty_architecture.md`](multimodal_uncertainty_architecture.md): Blueprint for multimodal uncertainty and missing-modality rules.
8. [`synthetic_validation_protocol.md`](synthetic_validation_protocol.md): Definitions of the 7 valid and 6 invalid 4-class synthetic fixtures.
9. [`limitations_and_future_clinical_validation.md`](limitations_and_future_clinical_validation.md): Pre-clinical safeguards and requirements for IRB trials.
10. [`reproducibility_manifest.json`](reproducibility_manifest.json): Cryptographic freeze verification and zero patient data declaration.
11. [`test_execution_report.json`](test_execution_report.json): Complete execution statistics across all uncertainty experiments.

---

## 4. Scientific Verification Status

| Component | Status | Clinical Recommendation |
| :--- | :---: | :--- |
| **Uncertainty Metrics Engine** | **PASS** | Mathematically verified on canonical distributions. |
| **Entropy Threshold Audit ($H > 0.60$)** | **PASS** | Resolved as Normalized Shannon Entropy; candidate engineering status. |
| **Selective Prediction Engine** | **PASS** | Tri-state triage and boundary logic verified. |
| **Epistemic/Aleatoric Decomposition** | **PASS** | Mutual Information and MC Dropout verified. |
| **OOD Detection Subsystem** | **PASS** | MSP, Energy, and Mahalanobis detectors verified on synthetic points. |
| **Calibration & Risk-Coverage** | **PASS** | Monotonic risk reduction verified on synthetic partitions. |
| **Clinical Calibration** | **NOT_VERIFIED** | Synthetic engineering results only; no clinical calibration exists. |
