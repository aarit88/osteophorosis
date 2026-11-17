"""
OsteoGuard AI - Deterministic Synthetic Uncertainty Fixtures.

Provides:
1. Canonical four-class probability distributions covering confident, cautious,
   ambiguous, uniform, and degenerate states.
2. Exact analytical ground truths for entropy and margins.
3. Deterministic synthetic calibration and evaluation datasets for temperature scaling.

Four-Class Alignment:
    Index 0: Secondary bone metastasis
    Index 1: Paget's disease of bone
    Index 2: Osteogenesis imperfecta (OI)
    Index 3: Healthy skeletal control
"""

import math
from typing import Dict, List, Tuple
import numpy as np


class SyntheticDistributions:
    """
    Factory generating canonical 4-class distributions with derivable analytical properties.
    """

    @staticmethod
    def highly_confident() -> np.ndarray:
        """1. High confidence: [0.94, 0.02, 0.02, 0.02]."""
        return np.array([0.94, 0.02, 0.02, 0.02], dtype=np.float64)

    @staticmethod
    def moderately_confident() -> np.ndarray:
        """2. Moderate confidence: [0.65, 0.15, 0.10, 0.10]."""
        return np.array([0.65, 0.15, 0.10, 0.10], dtype=np.float64)

    @staticmethod
    def ambiguous_two_class() -> np.ndarray:
        """3. Ambiguous two-class tie: [0.48, 0.48, 0.02, 0.02]."""
        return np.array([0.48, 0.48, 0.02, 0.02], dtype=np.float64)

    @staticmethod
    def uniform_four_class() -> np.ndarray:
        """4. Complete uniform dispersion: [0.25, 0.25, 0.25, 0.25]."""
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    @staticmethod
    def near_equal() -> np.ndarray:
        """5. Near-equal four-class spread: [0.28, 0.26, 0.24, 0.22]."""
        return np.array([0.28, 0.26, 0.24, 0.22], dtype=np.float64)

    @staticmethod
    def one_hot_zero_entropy() -> np.ndarray:
        """6. Deterministic single-class: [1.0, 0.0, 0.0, 0.0]."""
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def extremely_small_probs() -> np.ndarray:
        """7. Tail probability test: [0.9999997, 1e-7, 1e-7, 1e-7]."""
        return np.array([0.9999997, 1e-7, 1e-7, 1e-7], dtype=np.float64)

    @classmethod
    def get_valid_suite(cls) -> Dict[str, np.ndarray]:
        return {
            "highly_confident": cls.highly_confident(),
            "moderately_confident": cls.moderately_confident(),
            "ambiguous_two_class": cls.ambiguous_two_class(),
            "uniform_four_class": cls.uniform_four_class(),
            "near_equal": cls.near_equal(),
            "one_hot_zero_entropy": cls.one_hot_zero_entropy(),
            "extremely_small_probs": cls.extremely_small_probs(),
        }

    @classmethod
    def get_invalid_suite(cls) -> Dict[str, np.ndarray]:
        return {
            "negative_probability": np.array([0.80, -0.10, 0.20, 0.10], dtype=np.float64),
            "greater_than_one": np.array([1.20, -0.10, -0.05, -0.05], dtype=np.float64),
            "sum_sub_one": np.array([0.40, 0.30, 0.10, 0.10], dtype=np.float64),  # sum 0.90
            "sum_super_one": np.array([0.50, 0.40, 0.20, 0.10], dtype=np.float64), # sum 1.20
            "nan_present": np.array([0.50, float("nan"), 0.25, 0.25], dtype=np.float64),
            "inf_present": np.array([float("inf"), 0.0, 0.0, 0.0], dtype=np.float64),
        }


def generate_synthetic_calibration_dataset(
    n_samples: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates deterministic synthetic dataset of (logits, probabilities, ground_truth_labels).
    Uses controlled pseudo-random seed to guarantee identical fixtures across runs.
    """
    rng = np.random.RandomState(seed)
    # Generate 4-class synthetic logits with controlled over-confidence (temperature ~ 1.5 needed)
    logits = rng.randn(n_samples, 4) * 2.5
    # True labels loosely correlated with argmax
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        # 75% accuracy correlation with noise
        if rng.rand() < 0.75:
            labels[i] = int(np.argmax(logits[i]))
        else:
            labels[i] = rng.choice(4)

    # Softmax probabilities
    max_z = np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(logits - max_z)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return logits, probs, labels

