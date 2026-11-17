"""
OsteoGuard AI - Out-Of-Distribution (OOD) Detection Subsystem.

Implements three mathematically verifiable OOD detection algorithms:
1. Maximum Softmax Probability (MSP) Baseline (Hendrycks & Gimpel, 2017)
2. Free Energy Logit Detector (Liu et al., NeurIPS 2020)
3. Mahalanobis Feature-Space Distance Detector with Covariance Regularization (Lee et al., NeurIPS 2018)

RESEARCH SAFEGUARD NOTICE:
    - These algorithms evaluate synthetic mathematical distributions only.
    - Synthetic distance thresholds do NOT establish real-world generalization
      across unseen scanners, clinical protocols, or patient demographics.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field


class OODDetectionResult(BaseModel):
    """Container for OOD detection results."""
    method: str = Field(..., description="OOD algorithm used")
    score: float = Field(..., description="Computed anomalousness score")
    threshold: float = Field(..., description="Applied decision threshold")
    is_ood: bool = Field(..., description="True if input is classified as out-of-distribution")
    details: Dict[str, float] = Field(default_factory=dict, description="Supplementary statistics")
    disclaimer: str = Field(
        default="RESEARCH ONLY: Evaluated on synthetic representations. Not calibrated on clinical data."
    )


class MSPDetector:
    """
    Maximum Softmax Probability OOD Detector.
    Inputs with low peak probability (max(p) < threshold) are flagged as OOD.
    """

    def __init__(self, threshold: float = 0.50):
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(f"MSP threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold

    def detect(self, probabilities: np.ndarray) -> OODDetectionResult:
        p = np.asarray(probabilities, dtype=np.float64)
        max_p = float(np.max(p))
        is_ood = max_p < self.threshold

        return OODDetectionResult(
            method="MaximumSoftmaxProbability",
            score=round(max_p, 6),
            threshold=self.threshold,
            is_ood=is_ood,
            details={"peak_probability": max_p},
        )


class LogitEnergyDetector:
    """
    Logit Free Energy OOD Detector.
    Computes Helmholtz free energy over unnormalized logits:
      E(x; T) = -T * ln( sum_{c=1}^K exp(z_c / T) )
    In-distribution inputs have negative free energy; OOD inputs have higher (less negative) energy.
    """

    def __init__(self, temperature: float = 1.0, threshold: float = -2.0):
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = temperature
        self.threshold = threshold

    def compute_energy(self, logits: np.ndarray) -> float:
        z = np.asarray(logits, dtype=np.float64)
        scaled = z / self.temperature
        # Log-sum-exp trick for numerical stability
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        energy = -self.temperature * lse
        return float(energy)

    def detect(self, logits: np.ndarray) -> OODDetectionResult:
        energy = self.compute_energy(logits)
        is_ood = energy > self.threshold

        return OODDetectionResult(
            method="LogitFreeEnergy",
            score=round(energy, 6),
            threshold=self.threshold,
            is_ood=is_ood,
            details={"temperature": self.temperature, "free_energy": energy},
        )


class MahalanobisFeatureDetector:
    """
    Feature-space distance detector computing minimum Mahalanobis distance
    to class centroids with Tikhonov/shrinkage regularization:
      D_M(z) = min_c sqrt( (z - mu_c)^T * (Sigma + lambda * I)^(-1) * (z - mu_c) )
    """

    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 8,
        regularization: float = 1e-4,
        threshold: float = 4.0,
    ):
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        if regularization <= 0.0:
            raise ValueError(f"regularization must be strictly positive, got {regularization}")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.regularization = regularization
        self.threshold = threshold

        self.centroids: Optional[np.ndarray] = None  # (K, D)
        self.inv_cov: Optional[np.ndarray] = None     # (D, D)
        self.is_fitted: bool = False

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Fits class centroids and tied regularized covariance matrix.
        """
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(f"Features shape must be (N, {self.feature_dim}), got {X.shape}")
        if len(y) != len(X):
            raise ValueError("Features and labels must have matching length.")

        n_samples = len(X)
        self.centroids = np.zeros((self.num_classes, self.feature_dim), dtype=np.float64)
        residuals = np.zeros_like(X)

        for c in range(self.num_classes):
            mask = (y == c)
            if not np.any(mask):
                raise ValueError(f"No samples provided for class {c}")
            self.centroids[c] = np.mean(X[mask], axis=0)
            residuals[mask] = X[mask] - self.centroids[c]

        # Tied empirical covariance
        empirical_cov = np.dot(residuals.T, residuals) / float(n_samples)

        # Tikhonov regularization to guarantee invertibility even if D > N or rank-deficient
        reg_cov = empirical_cov + self.regularization * np.eye(self.feature_dim)
        self.inv_cov = np.linalg.inv(reg_cov)
        self.is_fitted = True

    def compute_distance(self, feature_vector: np.ndarray) -> float:
        """Computes minimum Mahalanobis distance to any class centroid."""
        if not self.is_fitted or self.centroids is None or self.inv_cov is None:
            raise RuntimeError("MahalanobisFeatureDetector must be fitted before computing distance.")

        z = np.asarray(feature_vector, dtype=np.float64).flatten()
        if len(z) != self.feature_dim:
            raise ValueError(f"Expected feature vector of length {self.feature_dim}, got {len(z)}")

        min_dist = float("inf")
        for c in range(self.num_classes):
            diff = z - self.centroids[c]
            dist_sq = float(np.dot(np.dot(diff, self.inv_cov), diff))
            dist = math.sqrt(max(0.0, dist_sq))
            if dist < min_dist:
                min_dist = dist

        return float(min_dist)

    def detect(self, feature_vector: np.ndarray) -> OODDetectionResult:
        dist = self.compute_distance(feature_vector)
        is_ood = dist > self.threshold

        return OODDetectionResult(
            method="MahalanobisFeatureDistance",
            score=round(dist, 6),
            threshold=self.threshold,
            is_ood=is_ood,
            details={"min_mahalanobis_distance": dist},
        )

