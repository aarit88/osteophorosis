"""
OsteoGuard AI - Uncertainty Quantification and Selective Prediction Subsystem.

Contains:
    1. metrics.py: Shannon entropy (raw & normalized), margin, distribution diagnostics.
    2. epistemic_aleatoric.py: Ensemble uncertainty, mutual information, Monte Carlo dropout.
    3. selective_prediction.py: Triage gate implementing H > 0.60 abstention rule.
    4. ood_detector.py: MSP, Free Energy, and Mahalanobis distance detectors.
    5. calibration.py: NLL, Brier score, ECE, Risk-coverage curves, Temperature scaling.
    6. synthetic_fixtures.py: Deterministic 4-class synthetic distributions and calibration sets.
"""

from osteoguard.uncertainty.metrics import (
    DistributionDiagnostics,
    validate_probability_distribution,
    compute_shannon_entropy,
    compute_normalized_entropy,
    compute_top_two_margin,
    compute_distribution_diagnostics,
)
from osteoguard.uncertainty.epistemic_aleatoric import (
    UncertaintyDecomposition,
    EnsembleUncertaintyEngine,
    MonteCarloDropoutAdapter,
)
from osteoguard.uncertainty.selective_prediction import (
    SelectiveDecision,
    AbstentionReasonCode,
    SelectivePredictionResult,
    SelectivePredictionEngine,
)
from osteoguard.uncertainty.ood_detector import (
    OODDetectionResult,
    MSPDetector,
    LogitEnergyDetector,
    MahalanobisFeatureDetector,
)
from osteoguard.uncertainty.calibration import (
    CalibrationReport,
    RiskCoveragePoint,
    CalibrationDiagnostics,
    RiskCoverageEvaluator,
    TemperatureScaler,
)
from osteoguard.uncertainty.synthetic_fixtures import (
    SyntheticDistributions,
    generate_synthetic_calibration_dataset,
)

__all__ = [
    # Metrics
    "DistributionDiagnostics",
    "validate_probability_distribution",
    "compute_shannon_entropy",
    "compute_normalized_entropy",
    "compute_top_two_margin",
    "compute_distribution_diagnostics",
    # Epistemic / Aleatoric
    "UncertaintyDecomposition",
    "EnsembleUncertaintyEngine",
    "MonteCarloDropoutAdapter",
    # Selective Prediction
    "SelectiveDecision",
    "AbstentionReasonCode",
    "SelectivePredictionResult",
    "SelectivePredictionEngine",
    # OOD
    "OODDetectionResult",
    "MSPDetector",
    "LogitEnergyDetector",
    "MahalanobisFeatureDetector",
    # Calibration
    "CalibrationReport",
    "RiskCoveragePoint",
    "CalibrationDiagnostics",
    "RiskCoverageEvaluator",
    "TemperatureScaler",
    # Synthetic Fixtures
    "SyntheticDistributions",
    "generate_synthetic_calibration_dataset",
]

