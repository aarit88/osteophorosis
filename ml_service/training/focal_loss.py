"""
OsteoGuard AI - Numerically Stable Multiclass Focal Loss.

Mathematical formulation:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
    - gamma: Focusing parameter (default: 2.0) down-weighting easy examples.
    - alpha_t: Class balancing weight:
        * Class 0 (Non-fractured / negative): alpha_negative = 0.25
        * Class 1 (Fractured / positive):     alpha_positive = 0.75
    - log(p_t): Obtained via log_softmax for optimal numerical stability.
    - p_t = exp(log_softmax(z)[target]).
"""

from typing import Optional, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Numerically stable Focal Loss for multi-class and binary classification.

    Args:
        gamma: Focusing parameter gamma >= 0. Default: 2.0.
        alpha: Class weights tensor or sequence. For binary classification with classes
               [0: Non-fractured, 1: Fractured], default is [0.25, 0.75].
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[Sequence[float], torch.Tensor]] = (0.25, 0.75),
        reduction: str = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"Focal loss gamma must be non-negative, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            else:
                alpha_tensor = alpha.float()
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            logits: Predicted raw logits of shape (B, C).
            targets: Ground-truth class indices of shape (B,) with values in [0, C-1].

        Returns:
            Computed scalar loss (if reduction is 'mean' or 'sum') or per-sample loss tensor (B,).
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits tensor of shape (B, C), got ndim={logits.ndim}")
        if targets.ndim != 1:
            raise ValueError(f"Expected 1D targets tensor of shape (B,), got ndim={targets.ndim}")
        if logits.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: logits has {logits.size(0)} samples, "
                f"targets has {targets.size(0)} samples"
            )

        num_classes = logits.size(1)
        if (targets < 0).any() or (targets >= num_classes).any():
            raise ValueError(f"Target values must be in range [0, {num_classes - 1}]")

        # Compute log probabilities via numerically stable log_softmax
        log_p = F.log_softmax(logits, dim=-1)  # (B, C)

        # Extract log_p of true class: log(p_t)
        targets_unsqueezed = targets.unsqueeze(1)  # (B, 1)
        log_pt = log_p.gather(dim=1, index=targets_unsqueezed).squeeze(1)  # (B,)

        # Compute p_t = exp(log_pt), clamped to [0, 1] for robust subtraction
        pt = torch.exp(log_pt).clamp(min=1e-7, max=1.0)

        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - pt).pow(self.gamma)

        # Apply class balancing alpha_t
        if self.alpha is not None:
            if self.alpha.size(0) != num_classes:
                raise ValueError(
                    f"Alpha tensor length ({self.alpha.size(0)}) does not match "
                    f"number of classes ({num_classes})"
                )
            alpha_device = self.alpha.to(logits.device)
            alpha_t = alpha_device.gather(dim=0, index=targets)  # (B,)
            focal_loss = -alpha_t * modulating_factor * log_pt
        else:
            focal_loss = -modulating_factor * log_pt

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

