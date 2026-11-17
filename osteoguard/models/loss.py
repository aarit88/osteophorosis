"""
OsteoGuard AI - Loss Functions.
Implements multiclass Focal Loss (gamma = 2.0) with class weighting
and Soft Dice Loss for skeletal U-Net segmentation.
"""

from typing import Optional, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from osteoguard.constants import FOCAL_LOSS_GAMMA, NUM_CLASSES


class MulticlassFocalLoss(nn.Module):
    """
    Multiclass Focal Loss addressing diagnostic class imbalance:
      L_FL = - sum_c alpha_c * (1 - p_c)^gamma * log(p_c + eps)
    """

    def __init__(
        self,
        gamma: float = FOCAL_LOSS_GAMMA,
        alpha: Optional[Sequence[float]] = None,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

        if alpha is not None:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) unnormalized predictions
            targets: (B,) ground-truth class indices in {0, ..., num_classes-1}
        Returns:
            Computed scalar or tensor loss.
        """
        probs = F.softmax(logits, dim=-1)
        # Gather predicted probability for target class
        targets = targets.view(-1, 1)
        p_t = probs.gather(1, targets).view(-1)

        log_p_t = torch.log(p_t.clamp(min=self.eps))
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        loss = -focal_weight * log_p_t

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for automated skeletal bone mask segmentation:
      L_Dice = 1 - (2 * sum(p * g) + eps) / (sum(p^2) + sum(g^2) + eps)
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw mask logits
            targets: (B, 1, H, W) binary ground truth masks {0, 1}
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denominator = (probs_flat ** 2).sum(dim=1) + (targets_flat ** 2).sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        loss = 1.0 - dice
        return loss.mean()


class OsteoGuardJointLoss(nn.Module):
    """
    Joint multi-task loss combining classification Focal Loss
    and optional skeletal segmentation Soft Dice Loss.
    """

    def __init__(
        self,
        focal_gamma: float = FOCAL_LOSS_GAMMA,
        class_weights: Optional[Sequence[float]] = None,
        seg_lambda: float = 0.5,
    ):
        super().__init__()
        self.focal_loss = MulticlassFocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.dice_loss = SoftDiceLoss()
        self.seg_lambda = seg_lambda

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_logits: Optional[torch.Tensor] = None,
        seg_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_loss = self.focal_loss(cls_logits, cls_targets)
        if seg_logits is not None and seg_targets is not None:
            dice = self.dice_loss(seg_logits, seg_targets)
            total_loss = total_loss + self.seg_lambda * dice
        return total_loss

