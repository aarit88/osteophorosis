"""
OsteoGuard AI - Probabilistic Calibration and Selective Risk Engine.

Implements reusable statistical evaluation metrics:
1. Negative Log-Likelihood (NLL / Log-Loss)
2. Multiclass Brier Score (Mean Squared Probability Error)
3. Expected Calibration Error (ECE) with adaptive/uniform reliability binning
4. Risk-Coverage Curve (Selective Prediction Accuracy Profile)
5. Post-hoc Temperature Scaling Optimizer (Guo et al., ICML 2017)

RESEARCH INTEGRITY SAFEGUARD:
    - These metrics are evaluated exclusively on synthetic labelled distributions.
    - No patient data or historical validation holdouts are accessed.
"""

from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
from pydantic import BaseModel, Field


class CalibrationReport(BaseModel):
    """Container for probabilistic calibration metrics."""
    n_samples: int = Field(..., ge=1)
    num_classes: int = Field(..., ge=2)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    negative_log_likelihood: float = Field(..., ge=0.0)
    brier_score: float = Field(..., ge=0.0, le=2.0)
    expected_calibration_error: float = Field(..., ge=0.0, le=1.0)
    max_calibration_error: float = Field(..., ge=0.0, le=1.0)
    bin_statistics: List[Dict[str, float]] = Field(default_factory=list)


class RiskCoveragePoint(BaseModel):
    """Single operating point on a risk-coverage curve."""
    coverage: float = Field(..., ge=0.0, le=1.0, description="Fraction of predictions retained")
    threshold: float = Field(..., description="Entropy/uncertainty cutoff used")
    retained_count: int = Field(..., ge=0)
    selective_risk: float = Field(..., ge=0.0, le=1.0, description="Error rate on retained predictions")
    selective_accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy on retained predictions")


class CalibrationDiagnostics:
    """
    Evaluator computing NLL, Brier score, ECE, and reliability diagrams.
    """

    @staticmethod
    def compute_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
        """Negative Log-Likelihood: - (1/N) * sum_i ln(p_{i, y_i})."""
        y = np.asarray(y_true, dtype=np.int64)
        P = np.asarray(y_prob, dtype=np.float64)
        N = len(y)
        P_clipped = np.clip(P, eps, 1.0)
        true_class_probs = P_clipped[np.arange(N), y]
        nll = -float(np.mean(np.log(true_class_probs)))
        return float(round(nll, 6))

    @staticmethod
    def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Multiclass Brier Score: (1/N) * sum_i sum_k (p_{i, k} - one_hot(y_i)_k)^2."""
        y = np.asarray(y_true, dtype=np.int64)
        P = np.asarray(y_prob, dtype=np.float64)
        N, K = P.shape

        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y] = 1.0

        brier = float(np.mean(np.sum((P - Y_one_hot) ** 2, axis=1)))
        return float(round(brier, 6))

    @classmethod
    def compute_ece(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_bins: int = 10,
    ) -> Tuple[float, float, List[Dict[str, float]]]:
        """
        Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        """
        y = np.asarray(y_true, dtype=np.int64)
        P = np.asarray(y_prob, dtype=np.float64)
        N = len(y)

        preds = np.argmax(P, axis=1)
        confs = np.max(P, axis=1)
        accuracies = (preds == y).astype(np.float64)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_stats = []

        for b in range(num_bins):
            bin_lower = bin_boundaries[b]
            bin_upper = bin_boundaries[b + 1]

            if b == num_bins - 1:
                in_bin = (confs >= bin_lower) & (confs <= bin_upper)
            else:
                in_bin = (confs >= bin_lower) & (confs < bin_upper)

            prop_in_bin = float(np.mean(in_bin))

            if prop_in_bin > 0:
                acc_in_bin = float(np.mean(accuracies[in_bin]))
                conf_in_bin = float(np.mean(confs[in_bin]))
                gap = abs(acc_in_bin - conf_in_bin)

                ece += prop_in_bin * gap
                mce = max(mce, gap)

                bin_stats.append({
                    "bin_lower": round(bin_lower, 2),
                    "bin_upper": round(bin_upper, 2),
                    "count": int(np.sum(in_bin)),
                    "fraction": round(prop_in_bin, 4),
                    "observed_acc": round(acc_in_bin, 4),
                    "predicted_conf": round(conf_in_bin, 4),
                    "calibration_gap": round(gap, 4),
                })

        return float(round(ece, 6)), float(round(mce, 6)), bin_stats

    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_bins: int = 10,
    ) -> CalibrationReport:
        """Full calibration report."""
        y = np.asarray(y_true, dtype=np.int64)
        P = np.asarray(y_prob, dtype=np.float64)
        N, K = P.shape

        preds = np.argmax(P, axis=1)
        acc = float(np.mean(preds == y))
        nll = cls.compute_nll(y, P)
        brier = cls.compute_brier_score(y, P)
        ece, mce, bin_stats = cls.compute_ece(y, P, num_bins=num_bins)

        return CalibrationReport(
            n_samples=N,
            num_classes=K,
            accuracy=round(acc, 4),
            negative_log_likelihood=nll,
            brier_score=brier,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            bin_statistics=bin_stats,
        )


class RiskCoverageEvaluator:
    """
    Evaluator computing risk-coverage curves for selective prediction triage.
    """

    @staticmethod
    def compute_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        uncertainty_scores: np.ndarray,
        coverage_steps: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    ) -> List[RiskCoveragePoint]:
        """
        Sorts instances in ascending order of uncertainty (lowest uncertainty first).
        Calculates selective error rate (risk = 1 - accuracy) at given coverage tiers.
        """
        y = np.asarray(y_true, dtype=np.int64)
        P = np.asarray(y_prob, dtype=np.float64)
        unc = np.asarray(uncertainty_scores, dtype=np.float64)
        N = len(y)

        preds = np.argmax(P, axis=1)
        correct = (preds == y)

        # Sort ascending by uncertainty
        sorted_indices = np.argsort(unc)

        points = []
        for cov in coverage_steps:
            clamped_cov = min(1.0, max(0.01, float(cov)))
            k_retain = max(1, int(round(clamped_cov * N)))
            retained_idx = sorted_indices[:k_retain]

            sel_acc = float(np.mean(correct[retained_idx]))
            sel_risk = 1.0 - sel_acc
            threshold_val = float(unc[retained_idx[-1]])

            points.append(
                RiskCoveragePoint(
                    coverage=round(clamped_cov, 3),
                    threshold=round(threshold_val, 4),
                    retained_count=k_retain,
                    selective_risk=round(sel_risk, 4),
                    selective_accuracy=round(sel_acc, 4),
                )
            )

        return points


class TemperatureScaler:
    """
    Post-hoc temperature scaling optimizer.
    Finds scalar T > 0 that minimizes NLL on a held-out calibration set:
      softmax(logits / T)
    """

    def __init__(self, initial_temperature: float = 1.0):
        self.temperature: float = initial_temperature
        self.is_fitted: bool = False

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Finds optimal temperature via gradient descent on NLL.
        """
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.int64)
        N, K = z.shape

        T = float(self.temperature)

        for _ in range(max_iter):
            # Softmax with current T
            scaled = z / T
            max_s = np.max(scaled, axis=1, keepdims=True)
            exp_s = np.exp(scaled - max_s)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

            # NLL loss
            # Gradient d(NLL)/dT = (1 / T^2) * (1/N) * sum_i sum_k (probs_{i, k} - one_hot_k) * z_{i, k}
            Y_one_hot = np.zeros((N, K), dtype=np.float64)
            Y_one_hot[np.arange(N), y] = 1.0

            diff = probs - Y_one_hot
            grad_T = float(np.mean(np.sum(diff * z, axis=1))) / (T ** 2)

            T = max(0.1, T - lr * grad_T)

        self.temperature = float(round(T, 4))
        self.is_fitted = True
        return self.temperature

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Applies fitted temperature to transform unscaled logits into calibrated probabilities."""
        z = np.asarray(logits, dtype=np.float64)
        scaled = z / self.temperature
        max_s = np.max(scaled, axis=-1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        return exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Alias for scale_probabilities."""
        return self.scale_probabilities(logits)
