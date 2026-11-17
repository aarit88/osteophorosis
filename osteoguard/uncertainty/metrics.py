"""
OsteoGuard AI - Uncertainty Metrics Engine.

Implements mathematically rigorous, numerically safe calculation of:
- Raw Shannon Predictive Entropy (natural log / nats or arbitrary base)
- Normalized Shannon Entropy (strictly in [0.0, 1.0])
- Maximum Class Probability
- Top-Two Probability Margin
- Distribution diagnostics with strict validation (zero silent normalization)

Reference:
    Shannon, C. E. (1948). "A Mathematical Theory of Communication."
    Bell System Technical Journal, 27(3), 379-423.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
from pydantic import BaseModel, Field


class DistributionDiagnostics(BaseModel):
    """Container for comprehensive prediction distribution diagnostics."""
    num_classes: int = Field(..., ge=2, description="Number of categorical classes")
    predicted_class_idx: int = Field(..., ge=0, description="Argmax class index")
    max_probability: float = Field(..., ge=0.0, le=1.0, description="Highest class probability")
    top_two_margin: float = Field(..., ge=0.0, le=1.0, description="Difference between top two probabilities")
    raw_entropy_nats: float = Field(..., ge=0.0, description="Raw Shannon predictive entropy in nats (base e)")
    normalized_entropy: float = Field(..., ge=0.0, le=1.0, description="Normalized Shannon entropy in [0.0, 1.0]")
    probability_sum: float = Field(..., description="Actual sum of input probability vector")
    is_valid: bool = Field(True, description="True if distribution passes all mathematical integrity checks")


def validate_probability_distribution(
    probabilities: Union[np.ndarray, Sequence[float]],
    tolerance: float = 1e-4,
) -> np.ndarray:
    """
    Validates mathematical integrity of a discrete probability distribution.

    Requirements:
    - Must be a 1D sequence of finite numbers.
    - Every probability must be in [0.0, 1.0].
    - Sum of probabilities must equal 1.0 within numerical tolerance.
    - Does NOT silently normalize invalid vectors (e.g. all zeros or negative values).

    Returns:
        np.ndarray: Validated float64 1D array.

    Raises:
        TypeError: If input cannot be converted to float numpy array.
        ValueError: If array is empty, non-1D, contains NaN/Inf, negative values,
                    or sum deviates from 1.0 beyond tolerance.
    """
    try:
        p = np.asarray(probabilities, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"Probabilities must be convertible to float numpy array, got {type(probabilities)}: {e}")

    if p.ndim != 1:
        raise ValueError(f"Probability vector must be 1-dimensional, got shape {p.shape}")
    if len(p) < 2:
        raise ValueError(f"Probability vector must contain at least 2 classes, got {len(p)}")

    # Check finite
    if not np.all(np.isfinite(p)):
        raise ValueError("Probability distribution contains NaN or Infinite values.")

    # Check lower and upper bounds
    if np.any(p < -1e-8):
        raise ValueError(f"Probabilities cannot be negative. Found minimum value: {float(np.min(p))}")
    if np.any(p > 1.0 + tolerance):
        raise ValueError(f"Probabilities cannot exceed 1.0. Found maximum value: {float(np.max(p))}")

    # Clip slight floating-point underflows
    p_clamped = np.clip(p, 0.0, 1.0)
    prob_sum = float(np.sum(p_clamped))

    # Check sum
    if abs(prob_sum - 1.0) > tolerance:
        raise ValueError(
            f"Probability vector must sum to 1.0 +/- {tolerance}. Actual sum: {prob_sum:.6f}"
        )

    # Normalize slight floating point residual strictly within tolerance
    return p_clamped / prob_sum


def compute_shannon_entropy(
    probabilities: np.ndarray,
    base: Union[float, str] = math.e,
    tolerance: float = 1e-4,
) -> float:
    """
    Computes Shannon predictive entropy:
        H(p) = -sum_{i=1}^K p_i * log_b(p_i)

    Numerically safe: 0 * log(0) evaluates to 0.0 in accordance with standard limit:
        lim_{x -> 0+} x * log(x) = 0

    Args:
        probabilities: Validated 1D probability distribution.
        base: Logarithm base (default math.e for nats, 2 for bits). Accepts float or 'e', '2', '4'.
        tolerance: Sum tolerance for distribution validation.

    Returns:
        float: Non-negative Shannon entropy.
    """
    p = validate_probability_distribution(probabilities, tolerance=tolerance)
    if isinstance(base, str):
        if base.lower() == "e":
            b = math.e
        else:
            b = float(base)
    else:
        b = float(base)

    if b <= 1.0:
        raise ValueError(f"Logarithm base must be greater than 1.0, got {base}")

    # Stable mask for p > 0
    non_zero = p > 0.0
    if not np.any(non_zero):
        return 0.0

    p_active = p[non_zero]
    log_p = np.log(p_active) / math.log(b)
    raw_h = -float(np.sum(p_active * log_p))
    return float(max(0.0, round(raw_h, 8)))


def compute_normalized_entropy(
    probabilities: np.ndarray,
    tolerance: float = 1e-4,
) -> float:
    """
    Computes Normalized Shannon Entropy strictly bounded in [0.0, 1.0]:
        H_norm(p) = H(p) / log(K)

    Where K is the number of classes.
    For K >= 2:
      H_norm = 0.0 for deterministic prediction (one-hot)
      H_norm = 1.0 for completely uniform distribution (maximum ignorance)
    """
    p = validate_probability_distribution(probabilities, tolerance=tolerance)
    k = len(p)
    raw_h = compute_shannon_entropy(p, base=math.e, tolerance=tolerance)
    max_h = math.log(k)

    h_norm = raw_h / max_h
    return float(min(1.0, max(0.0, round(h_norm, 6))))


def compute_top_two_margin(
    probabilities: np.ndarray,
    tolerance: float = 1e-4,
) -> float:
    """
    Computes margin between highest and second-highest class probability:
        Margin = p_(1) - p_(2)

    In [0.0, 1.0]. A larger margin indicates decisive class discrimination.
    """
    p = validate_probability_distribution(probabilities, tolerance=tolerance)
    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - sorted_p[1])
    return float(min(1.0, max(0.0, round(margin, 6))))


def compute_distribution_diagnostics(
    probabilities: np.ndarray,
    tolerance: float = 1e-4,
) -> DistributionDiagnostics:
    """
    Runs full suite of distribution diagnostics across a probability vector.
    """
    p = validate_probability_distribution(probabilities, tolerance=tolerance)
    pred_idx = int(np.argmax(p))
    max_prob = float(p[pred_idx])
    margin = compute_top_two_margin(p, tolerance=tolerance)
    raw_h = compute_shannon_entropy(p, base=math.e, tolerance=tolerance)
    norm_h = compute_normalized_entropy(p, tolerance=tolerance)
    prob_sum = float(np.sum(p))

    return DistributionDiagnostics(
        num_classes=len(p),
        predicted_class_idx=pred_idx,
        max_probability=round(max_prob, 6),
        top_two_margin=round(margin, 6),
        raw_entropy_nats=round(raw_h, 6),
        normalized_entropy=round(norm_h, 6),
        probability_sum=round(prob_sum, 6),
        is_valid=True,
    )
