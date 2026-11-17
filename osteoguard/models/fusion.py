"""
OsteoGuard AI - Late Feature Fusion and Classification Module.
Implements calibrated late fusion of the 512-D deep semantic representation
and the 102-D IBSI radiomic texture vector via independent 1D BatchNorm,
concatenating into a 614-D latent vector processed by a dual-stage MLP head.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from osteoguard.constants import (
    FUSION_VECTOR_DIM,
    MLP_DROPOUT_RATE,
    MLP_HIDDEN_DIM_1,
    MLP_HIDDEN_DIM_2,
    NUM_CLASSES,
    RADIOMICS_FEATURE_DIM,
    SEMANTIC_EMBEDDING_DIM,
    TARGET_CLASSES,
)


class LateFusionClassificationHead(nn.Module):
    """
    Dual-stream late fusion module.
    Takes v_sem (512-D) and v_rad (102-D), normalizes them independently,
    concatenates into v_fusion (614-D), and computes class logits for:
      [0: metastasis, 1: paget, 2: oi, 3: healthy]
    """

    def __init__(
        self,
        semantic_dim: int = SEMANTIC_EMBEDDING_DIM,
        radiomics_dim: int = RADIOMICS_FEATURE_DIM,
        fusion_dim: int = FUSION_VECTOR_DIM,
        hidden_1: int = MLP_HIDDEN_DIM_1,
        hidden_2: int = MLP_HIDDEN_DIM_2,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = MLP_DROPOUT_RATE,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.radiomics_dim = radiomics_dim
        self.expected_fusion_dim = semantic_dim + radiomics_dim
        assert self.expected_fusion_dim == fusion_dim, (
            f"Expected fusion_dim {self.expected_fusion_dim} but got {fusion_dim}"
        )

        # Independent BatchNorm1d to prevent dynamic range disparity
        self.bn_sem = nn.BatchNorm1d(semantic_dim)
        self.bn_rad = nn.BatchNorm1d(radiomics_dim)

        # Multi-layer perceptron (MLP) classification head
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_2, num_classes),
        )

    def forward(
        self,
        v_sem: torch.Tensor,
        v_rad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_sem: (B, 512) semantic feature tensor
            v_rad: (B, 102) IBSI radiomics feature tensor
        Returns:
            logits: (B, 4) unnormalized classification logits
            v_fusion: (B, 614) concatenated normalized fusion vector
        """
        # Batch size 1 handling for BatchNorm during single-sample eval
        if v_sem.size(0) == 1 and not self.training:
            norm_sem = v_sem
            norm_rad = v_rad
        else:
            norm_sem = self.bn_sem(v_sem)
            norm_rad = self.bn_rad(v_rad)

        v_fusion = torch.cat([norm_sem, norm_rad], dim=1)
        logits = self.mlp(v_fusion)
        return logits, v_fusion

    def predict_probabilities(
        self,
        v_sem: torch.Tensor,
        v_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Returns softmax probability distribution over 4 classes."""
        logits, _ = self.forward(v_sem, v_rad)
        return F.softmax(logits, dim=-1)

