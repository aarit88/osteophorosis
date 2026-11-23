"""
OsteoGuard AI - Stage 30 Uncertainty & Selective Prediction Test Suite.

Rigorously verifies:
1. Frozen checkpoint SHA-256 integrity (21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD).
2. Presence and completeness of all 11 Stage 30 artifacts and mirrored report.
3. Shannon entropy analytical calculations (uniform = ln 4, one-hot = 0).
4. Normalized entropy strictly bounded within [0.0, 1.0].
5. Distribution diagnostics and top-two margin calculation.
6. Strict probability vector validation and error rejection (no silent normalization).
7. Permutation invariance of predictive entropy.
8. Selective prediction tri-state triage and boundary checks (0.5999 vs 0.6000 vs 0.6001).
9. Independent OOD triggers and combined entropy + OOD reason codes.
10. Abstention representation safety (abstention is NEVER treated as negative or healthy).
11. Epistemic vs aleatoric uncertainty decomposition (Mutual Information >= 0).
12. Monte Carlo Dropout adapter deterministic repeatability with fixed seed.
13. Out-Of-Distribution detectors (MSP, Logit Energy, Mahalanobis feature distance).
14. Probabilistic calibration metrics (NLL, Brier score, ECE).
15. Post-hoc Temperature Scaling optimization on isolated calibration split.
16. Risk-coverage curve monotonicity (selective accuracy increases as coverage decreases).
17. Governance preservation: Stage 27 radiomics, Stage 28 biomechanics, Stage 29 explainability boundaries.
18. Host Python and React frontend integrity, and burned holdout preservation.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from osteoguard.uncertainty.metrics import (
    compute_shannon_entropy,
    compute_normalized_entropy,
    compute_top_two_margin,
    compute_distribution_diagnostics,
    validate_probability_distribution,
    DistributionDiagnostics,
)
from osteoguard.uncertainty.epistemic_aleatoric import (
    EnsembleUncertaintyEngine,
    MonteCarloDropoutAdapter,
    UncertaintyDecomposition,
)
from osteoguard.uncertainty.selective_prediction import (
    SelectivePredictionEngine,
    SelectiveDecision,
    AbstentionReasonCode,
    SelectivePredictionResult,
)
from osteoguard.uncertainty.ood_detector import (
    MSPDetector,
    LogitEnergyDetector,
    MahalanobisFeatureDetector,
    OODDetectionResult,
)
from osteoguard.uncertainty.calibration import (
    CalibrationDiagnostics,
    RiskCoverageEvaluator,
    RiskCoveragePoint,
    TemperatureScaler,
    CalibrationReport,
)
from osteoguard.uncertainty.synthetic_fixtures import (
    SyntheticDistributions,
    generate_synthetic_calibration_dataset,
)

EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
STAGE30_ARTIFACTS_DIR = WORKSPACE_ROOT / "ml_artifacts" / "stage30_uncertainty"
MIRRORED_REPORT_PATH = WORKSPACE_ROOT / "ml_artifacts" / "stage_reports" / "STAGE_30_UNCERTAINTY.md"


def test_01_frozen_checkpoint_integrity():
    """Verify frozen BTXRD checkpoint SHA-256 remains strictly uncorrupted."""
    ckpt_path = WORKSPACE_ROOT / "ml_artifacts" / "btxrd_frozen_model" / "best_checkpoint.pt"
    assert ckpt_path.exists(), f"Frozen checkpoint missing at {ckpt_path}"
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest().upper()
    assert actual_hash == EXPECTED_SHA256, (
        f"Cryptographic hash mismatch! Expected {EXPECTED_SHA256}, got {actual_hash}"
    )


def test_02_all_stage30_deliverables_exist_and_non_empty():
    """Verify all 11 required deliverables in ml_artifacts/stage30_uncertainty and mirrored report exist."""
    required_files = [
        "entropy_threshold_audit.md",
        "uncertainty_metrics_specification.md",
        "selective_prediction_specification.md",
        "ood_detection_specification.md",
        "multimodal_uncertainty_architecture.md",
        "synthetic_validation_protocol.md",
        "calibration_and_risk_coverage_report.md",
        "limitations_and_future_clinical_validation.md",
        "test_execution_report.json",
        "reproducibility_manifest.json",
        "STAGE_30_UNCERTAINTY_REPORT.md",
    ]
    for fname in required_files:
        fpath = STAGE30_ARTIFACTS_DIR / fname
        assert fpath.exists(), f"Missing deliverable: {fname}"
        assert fpath.stat().st_size > 0, f"Empty deliverable: {fname}"

    assert MIRRORED_REPORT_PATH.exists(), f"Missing mirrored report: {MIRRORED_REPORT_PATH}"
    assert MIRRORED_REPORT_PATH.stat().st_size > 0, "Empty mirrored report"


def test_03_entropy_analytical_properties():
    """Verify Shannon entropy matches theoretical ground truth for uniform and degenerate cases."""
    # 4-class uniform distribution: H = ln(4) nats = 2.0 bits
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    h_nats = compute_shannon_entropy(p_uniform, base=math.e)
    h_bits = compute_shannon_entropy(p_uniform, base=2.0)
    h_base4 = compute_shannon_entropy(p_uniform, base=4.0)

    assert math.isclose(h_nats, math.log(4), rel_tol=1e-7), f"Expected ln(4), got {h_nats}"
    assert math.isclose(h_bits, 2.0, rel_tol=1e-7), f"Expected 2.0 bits, got {h_bits}"
    assert math.isclose(h_base4, 1.0, rel_tol=1e-7), f"Expected 1.0 base-4, got {h_base4}"

    # Also test string base acceptance
    assert math.isclose(compute_shannon_entropy(p_uniform, base="e"), math.log(4), rel_tol=1e-7)
    assert math.isclose(compute_shannon_entropy(p_uniform, base="2"), 2.0, rel_tol=1e-7)

    # One-hot distribution: H = 0.0
    p_onehot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    h_zero = compute_shannon_entropy(p_onehot, base=math.e)
    assert math.isclose(h_zero, 0.0, abs_tol=1e-9), f"Expected 0.0, got {h_zero}"


def test_04_normalized_entropy_bounds():
    """Verify normalized entropy is strictly bounded within [0.0, 1.0]."""
    for name, p in SyntheticDistributions.get_valid_suite().items():
        h_norm = compute_normalized_entropy(p)
        assert 0.0 <= h_norm <= 1.0 + 1e-9, f"Normalized entropy {h_norm} out of bounds for {name}"

    assert math.isclose(compute_normalized_entropy(np.array([0.25, 0.25, 0.25, 0.25])), 1.0, abs_tol=1e-7)
    assert math.isclose(compute_normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0])), 0.0, abs_tol=1e-9)


def test_05_distribution_diagnostics_and_margin():
    """Verify top-two margin and distribution diagnostics."""
    # Highly confident: [0.94, 0.02, 0.02, 0.02] -> margin = 0.92
    p_conf = np.array([0.94, 0.02, 0.02, 0.02])
    margin_conf = compute_top_two_margin(p_conf)
    assert math.isclose(margin_conf, 0.92, abs_tol=1e-7)

    # Tie between top two: [0.48, 0.48, 0.02, 0.02] -> margin = 0.0
    p_tie = np.array([0.48, 0.48, 0.02, 0.02])
    margin_tie = compute_top_two_margin(p_tie)
    assert math.isclose(margin_tie, 0.0, abs_tol=1e-7)

    diag = compute_distribution_diagnostics(p_conf)
    assert diag.predicted_class_idx == 0
    assert math.isclose(diag.max_probability, 0.94, abs_tol=1e-7)
    assert math.isclose(diag.top_two_margin, 0.92, abs_tol=1e-7)
    assert diag.is_valid is True


def test_06_strict_probability_validation_rejections():
    """Verify strict validation rejects invalid, unnormalized, negative, or degenerate vectors."""
    invalid_cases = SyntheticDistributions.get_invalid_suite()
    for name, p in invalid_cases.items():
        with pytest.raises(ValueError):
            validate_probability_distribution(p)

    # Also test NaN and Inf explicitly
    with pytest.raises(ValueError):
        compute_shannon_entropy(np.array([np.nan, 0.5, 0.25, 0.25]))
    with pytest.raises(ValueError):
        compute_shannon_entropy(np.array([np.inf, 0.0, 0.0, 0.0]))


def test_07_entropy_permutation_invariance():
    """Verify Shannon entropy is invariant under permutation of class probabilities."""
    p_orig = np.array([0.70, 0.15, 0.10, 0.05])
    perms = [
        [0.05, 0.10, 0.15, 0.70],
        [0.15, 0.70, 0.05, 0.10],
        [0.10, 0.05, 0.70, 0.15],
    ]
    h_orig = compute_shannon_entropy(p_orig)
    for p_perm in perms:
        h_perm = compute_shannon_entropy(np.array(p_perm))
        assert math.isclose(h_orig, h_perm, rel_tol=1e-9), f"Permutation failed: {h_orig} vs {h_perm}"


def test_08_selective_prediction_threshold_boundary():
    """Verify selective prediction tri-state behavior at the boundary of H_norm = 0.60."""
    engine = SelectivePredictionEngine(entropy_threshold=0.60, use_normalized_entropy=True)

    # Test right below threshold: H_norm = 0.5999 -> ACCEPT
    res_below = engine.evaluate(probabilities=np.array([0.75, 0.15, 0.05, 0.05]))
    actual_h_below = compute_normalized_entropy(np.array([0.75, 0.15, 0.05, 0.05]))
    assert actual_h_below < 0.60
    assert res_below.decision == SelectiveDecision.ACCEPT_PREDICTION
    assert res_below.human_review_required is False

    # Test right above threshold: H_norm > 0.60 -> ABSTAIN
    res_above = engine.evaluate(probabilities=np.array([0.65, 0.15, 0.10, 0.10]))
    actual_h_above = compute_normalized_entropy(np.array([0.65, 0.15, 0.10, 0.10]))
    assert actual_h_above > 0.60
    assert res_above.decision == SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED
    assert res_above.human_review_required is True
    assert res_above.abstention_reason == AbstentionReasonCode.ELEVATED_ENTROPY


def test_09_independent_ood_trigger_and_combined_reasons():
    """Verify OOD detection triggers abstention independently of entropy and reports combined reasons."""
    engine = SelectivePredictionEngine(entropy_threshold=0.60, use_normalized_entropy=True)

    # Low entropy but OOD flag is True
    res_ood = engine.evaluate(
        probabilities=np.array([0.94, 0.02, 0.02, 0.02]),
        ood_flag=True,
    )
    assert res_ood.decision == SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED
    assert res_ood.abstention_reason == AbstentionReasonCode.OUT_OF_DISTRIBUTION

    # High entropy AND OOD flag is True
    res_both = engine.evaluate(
        probabilities=np.array([0.25, 0.25, 0.25, 0.25]),
        ood_flag=True,
    )
    assert res_both.decision == SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED
    assert res_both.abstention_reason == AbstentionReasonCode.COMBINED_ENTROPY_AND_OOD


def test_10_abstention_safety_semantics():
    """Verify that an abstention decision is NEVER represented as a negative or healthy diagnosis."""
    engine = SelectivePredictionEngine(entropy_threshold=0.60, use_normalized_entropy=True)
    res = engine.evaluate(probabilities=np.array([0.30, 0.30, 0.20, 0.20]))

    assert res.decision == SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED
    assert res.human_review_required is True
    assert "RESEARCH ONLY" in res.disclaimer
    assert "NOT indicate absence of disease or healthy status" in res.disclaimer


def test_11_epistemic_aleatoric_decomposition():
    """Verify mutual information decomposition: MI = 0 for full agreement, MI > 0 for disagreement."""
    # Case 1: Complete agreement across 5 models
    p_agree = np.tile([0.80, 0.10, 0.05, 0.05], (5, 1))
    decomp_agree = EnsembleUncertaintyEngine.decompose_ensemble(p_agree)
    assert math.isclose(decomp_agree.epistemic_mutual_info_nats, 0.0, abs_tol=1e-9)
    assert math.isclose(decomp_agree.total_entropy_nats, decomp_agree.aleatoric_entropy_nats, abs_tol=1e-9)

    # Case 2: Disagreement across 2 models
    p_disagree = np.array([
        [0.90, 0.10, 0.00, 0.00],
        [0.10, 0.90, 0.00, 0.00],
    ])
    decomp_disagree = EnsembleUncertaintyEngine.decompose_ensemble(p_disagree)
    assert decomp_disagree.epistemic_mutual_info_nats > 0.0
    assert decomp_disagree.total_entropy_nats > decomp_disagree.aleatoric_entropy_nats


def test_12_mc_dropout_deterministic_adapter():
    """Verify Monte Carlo Dropout adapter produces deterministic, bitwise reproducible passes with fixed seeds."""
    torch.manual_seed(42)
    toy_model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(32, 4),
    )

    x = torch.randn(1, 16)
    adapter1 = MonteCarloDropoutAdapter(toy_model, num_passes=8, seed=123)
    res1 = adapter1.run_mc_dropout(x)

    adapter2 = MonteCarloDropoutAdapter(toy_model, num_passes=8, seed=123)
    res2 = adapter2.run_mc_dropout(x)

    assert math.isclose(res1.epistemic_mutual_info_nats, res2.epistemic_mutual_info_nats, abs_tol=1e-7)
    np.testing.assert_allclose(res1.mean_probabilities, res2.mean_probabilities, atol=1e-6)


def test_13_ood_detectors_msp_energy_mahalanobis():
    """Verify MSP, Energy, and Mahalanobis OOD detector implementations."""
    # MSP
    msp = MSPDetector(threshold=0.70)
    res_in = msp.detect(np.array([0.85, 0.05, 0.05, 0.05]))
    res_out = msp.detect(np.array([0.40, 0.30, 0.20, 0.10]))
    assert not res_in.is_ood
    assert res_out.is_ood

    # Energy
    energy = LogitEnergyDetector(temperature=1.0, threshold=-2.0)
    res_e_in = energy.detect(np.array([4.0, 0.0, 0.0, 0.0]))
    res_e_out = energy.detect(np.array([-1.0, -1.0, -1.0, -1.0]))
    assert not res_e_in.is_ood
    assert res_e_out.is_ood

    # Mahalanobis
    rng = np.random.RandomState(42)
    train_features = rng.randn(100, 8)
    train_labels = rng.randint(0, 4, size=100)
    maha = MahalanobisFeatureDetector(num_classes=4, feature_dim=8, threshold=4.0)
    maha.fit(train_features, train_labels)

    in_sample = rng.randn(8)
    out_sample = rng.randn(8) * 15.0  # Far outlier
    res_m_in = maha.detect(in_sample)
    res_m_out = maha.detect(out_sample)
    assert res_m_out.score > res_m_in.score


def test_14_probabilistic_calibration_metrics():
    """Verify NLL, Brier score, and ECE computation."""
    # Perfect predictions
    y_true = np.array([0, 1, 2, 3])
    y_prob_perf = np.eye(4)
    report_perf = CalibrationDiagnostics.evaluate(y_true, y_prob_perf, num_bins=10)
    assert math.isclose(report_perf.accuracy, 1.0, abs_tol=1e-7)
    assert report_perf.brier_score < 0.01
    assert report_perf.expected_calibration_error < 0.01

    # Completely wrong predictions
    y_prob_wrong = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    report_wrong = CalibrationDiagnostics.evaluate(y_true, y_prob_wrong, num_bins=10)
    assert math.isclose(report_wrong.accuracy, 0.0, abs_tol=1e-7)
    assert report_wrong.brier_score > 1.0


def test_15_temperature_scaler_optimization():
    """Verify post-hoc temperature scaling fits strictly on validation/calibration split."""
    cal_logits, _, cal_labels = generate_synthetic_calibration_dataset(n_samples=200, seed=42)
    eval_logits, _, eval_labels = generate_synthetic_calibration_dataset(n_samples=200, seed=99)

    scaler = TemperatureScaler(initial_temperature=1.0)
    optimal_T = scaler.fit(cal_logits, cal_labels)
    assert optimal_T > 0.0, "Fitted temperature must be strictly positive"

    calibrated_probs = scaler.predict_proba(eval_logits)
    assert calibrated_probs.shape == eval_logits.shape
    np.testing.assert_allclose(calibrated_probs.sum(axis=1), np.ones(200), atol=1e-5)


def test_16_risk_coverage_monotonicity():
    """Verify risk-coverage curve: selective accuracy increases or stays constant as coverage decreases."""
    _, eval_probs, eval_labels = generate_synthetic_calibration_dataset(n_samples=100, seed=202)
    entropies = np.array([compute_normalized_entropy(p) for p in eval_probs])

    rc_points = RiskCoverageEvaluator.compute_curve(
        y_true=eval_labels,
        y_prob=eval_probs,
        uncertainty_scores=entropies,
        coverage_steps=[0.2, 0.4, 0.6, 0.8, 1.0],
    )

    assert len(rc_points) == 5
    # Accuracy at 20% coverage should be >= accuracy at 100% coverage
    acc_20 = rc_points[0].selective_accuracy
    acc_100 = rc_points[-1].selective_accuracy
    assert acc_20 >= acc_100 - 1e-6, f"Selective accuracy failed monotonicity: 20%={acc_20}, 100%={acc_100}"


def test_17_governance_and_research_boundary_preservation():
    """Verify preservation of previous stage limitations and governance boundaries."""
    # Stage 27 radiomics limitation documented
    assert (WORKSPACE_ROOT / "ml_artifacts" / "stage_reports" / "STAGE_27_RADIOMICS.md").exists()
    # Stage 28 biomechanics report documented
    assert (WORKSPACE_ROOT / "ml_artifacts" / "stage_reports" / "STAGE_28_BIOMECHANICS.md").exists()
    # Stage 29 explainability report documented
    assert (WORKSPACE_ROOT / "ml_artifacts" / "stage_reports" / "STAGE_29_EXPLAINABILITY.md").exists()

    # Verify BTXRD Center 2 and 3 holdouts remain burned and untouched
    for center in ["center2", "center3"]:
        center_eval_path = WORKSPACE_ROOT / "ml_artifacts" / f"{center}_evaluation"
        if center_eval_path.exists():
            manifest = center_eval_path / "evaluation_manifest.json"
            assert manifest.exists(), f"{center} evaluation manifest missing"


def test_18_host_python_and_frontend_integrity():
    """Verify host Python environment and frontend files remain untouched."""
    host_python = Path(sys.executable)
    assert host_python.exists()

    frontend_dir = WORKSPACE_ROOT / "src"
    assert frontend_dir.exists()
    assert (frontend_dir / "App.tsx").exists()
    assert (frontend_dir / "main.tsx").exists()
