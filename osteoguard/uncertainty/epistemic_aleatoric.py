"""
OsteoGuard AI - Epistemic and Aleatoric Uncertainty Engine.

Decomposes predictive uncertainty into aleatoric (data/observation noise)
and epistemic (model parameter / knowledge gap) components using:
1. Ensemble probability distributions.
2. Monte Carlo Dropout (stochastic forward passes with fixed seed).

Information-Theoretic Decomposition:
    Total Uncertainty (Predictive Entropy):
        H(p_bar) = -sum_c p_bar_c * ln(p_bar_c)
    Aleatoric Uncertainty (Expected Entropy):
        E[H(p)] = (1 / M) * sum_m H(p^(m))
    Epistemic Uncertainty (Mutual Information):
        I(Y, W | x) = H(p_bar) - E[H(p)] >= 0
    (Strictly non-negative by Jensen's inequality on concave entropy)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from osteoguard.uncertainty.metrics import (
    compute_shannon_entropy,
    compute_normalized_entropy,
    validate_probability_distribution,
)


class UncertaintyDecomposition(BaseModel):
    """Container for decomposed uncertainty metrics."""
    ensemble_size: int = Field(..., ge=1, description="Number of models or stochastic passes M")
    mean_probabilities: List[float] = Field(..., description="Averaged predictive probability vector p_bar")
    predicted_class_idx: int = Field(..., ge=0, description="Argmax of mean predictive probabilities")
    total_entropy_nats: float = Field(..., ge=0.0, description="Total predictive entropy H(p_bar)")
    total_normalized_entropy: float = Field(..., ge=0.0, le=1.0, description="Normalized total entropy in [0, 1]")
    aleatoric_entropy_nats: float = Field(..., ge=0.0, description="Expected entropy E[H(p)] (data noise)")
    epistemic_mutual_info_nats: float = Field(..., ge=0.0, description="Mutual information I(Y, W | x) (model knowledge gap)")
    variance_across_models: List[float] = Field(..., description="Per-class variance across ensemble members")
    disclaimer: str = Field(
        default=(
            "RESEARCH ONLY: Synthetic mathematical decomposition. Does not reflect clinical diagnostic risk."
        )
    )


class EnsembleUncertaintyEngine:
    """
    Computes information-theoretic uncertainty decomposition for ensemble predictions.
    """

    @classmethod
    def decompose_ensemble(
        cls,
        member_predictions: Union[np.ndarray, List[List[float]]],
    ) -> UncertaintyDecomposition:
        """
        Decomposes an ensemble of M probability vectors across K classes.

        Args:
            member_predictions: 2D array of shape (M, K) where each row is a valid probability vector.

        Returns:
            UncertaintyDecomposition containing total, aleatoric, and epistemic metrics.
        """
        preds = np.asarray(member_predictions, dtype=np.float64)
        if preds.ndim != 2:
            raise ValueError(f"Ensemble predictions must be a 2D matrix (M, K), got shape {preds.shape}")

        m_models, k_classes = preds.shape
        if m_models < 1:
            raise ValueError("Ensemble must contain at least 1 prediction vector.")
        if k_classes < 2:
            raise ValueError(f"Predictions must contain at least 2 classes, got {k_classes}")

        # Validate each row as a probability distribution
        validated_rows = np.zeros_like(preds)
        individual_entropies = np.zeros(m_models, dtype=np.float64)

        for i in range(m_models):
            validated_rows[i] = validate_probability_distribution(preds[i])
            individual_entropies[i] = compute_shannon_entropy(validated_rows[i])

        # Predictive mean distribution: p_bar = (1 / M) * sum_m p^(m)
        mean_p = np.mean(validated_rows, axis=0)
        mean_p = mean_p / np.sum(mean_p)  # Clean normalization
        pred_idx = int(np.argmax(mean_p))

        # Total uncertainty: H(p_bar)
        total_h = compute_shannon_entropy(mean_p)
        total_norm_h = compute_normalized_entropy(mean_p)

        # Aleatoric uncertainty: E[H(p)] = (1 / M) * sum_m H(p^(m))
        aleatoric_h = float(np.mean(individual_entropies))

        # Epistemic uncertainty: Mutual Information = H(p_bar) - E[H(p)]
        # Guard against minor floating point precision underflow below 0
        epistemic_mi = float(max(0.0, total_h - aleatoric_h))

        # Variance across models per class
        class_variances = [float(round(v, 6)) for v in np.var(validated_rows, axis=0)]

        return UncertaintyDecomposition(
            ensemble_size=m_models,
            mean_probabilities=[float(round(x, 6)) for x in mean_p],
            predicted_class_idx=pred_idx,
            total_entropy_nats=round(total_h, 6),
            total_normalized_entropy=round(total_norm_h, 6),
            aleatoric_entropy_nats=round(aleatoric_h, 6),
            epistemic_mutual_info_nats=round(epistemic_mi, 6),
            variance_across_models=class_variances,
        )


class MonteCarloDropoutAdapter:
    """
    Adapter executing deterministic Monte Carlo Dropout forward passes on PyTorch modules.
    Keeps dropout active at inference time to sample stochastic predictive distributions.
    """

    def __init__(
        self,
        model: nn.Module,
        num_passes: int = 10,
        seed: int = 42,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model)}")
        if num_passes < 2:
            raise ValueError(f"num_passes must be at least 2, got {num_passes}")

        self.model = model
        self.num_passes = num_passes
        self.seed = seed

    def _enable_dropout(self):
        """Forces Dropout layers into train mode while freezing BatchNormalization."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def run_mc_dropout(
        self,
        input_tensor: torch.Tensor,
    ) -> UncertaintyDecomposition:
        """
        Executes M stochastic forward passes with controlled seeding.
        """
        was_training = self.model.training
        torch.manual_seed(self.seed)

        try:
            self._enable_dropout()
            pass_probs: List[np.ndarray] = []

            with torch.no_grad():
                for _ in range(self.num_passes):
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                    pass_probs.append(probs)

            return EnsembleUncertaintyEngine.decompose_ensemble(np.array(pass_probs))

        finally:
            self.model.train(was_training)

