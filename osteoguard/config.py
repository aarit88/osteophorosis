"""
OsteoGuard AI - System and Pipeline Configuration Modules.
Provides validated, immutable configurations for models, training,
radiomic extraction, and data ingestion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from osteoguard.constants import (
    FOCAL_LOSS_GAMMA,
    FUSION_VECTOR_DIM,
    MLP_DROPOUT_RATE,
    MLP_HIDDEN_DIM_1,
    MLP_HIDDEN_DIM_2,
    NUM_CLASSES,
    RADIOMICS_FEATURE_DIM,
    SEMANTIC_EMBEDDING_DIM,
    TARGET_CLASSES,
    SUPPORTED_MODALITIES,
)


class BackboneType(str):
    RESNET101 = "resnet101"
    SWIN_T = "swin_t"
    SWIN_B = "swin_b"


class ModelConfig(BaseModel):
    """Configuration for OsteoGuard AI model architectures."""
    backbone_name: str = Field(default=BackboneType.RESNET101, description="Primary or comparative backbone")
    semantic_dim: int = Field(default=SEMANTIC_EMBEDDING_DIM, description="Semantic latent dimension")
    radiomics_dim: int = Field(default=RADIOMICS_FEATURE_DIM, description="IBSI radiomics vector length")
    fusion_dim: int = Field(default=FUSION_VECTOR_DIM, description="Concatenated fusion dimension")
    mlp_hidden_1: int = Field(default=MLP_HIDDEN_DIM_1, description="First dense hidden layer dimension")
    mlp_hidden_2: int = Field(default=MLP_HIDDEN_DIM_2, description="Second dense hidden layer dimension")
    dropout_rate: float = Field(default=MLP_DROPOUT_RATE, description="Dropout regularization probability")
    num_classes: int = Field(default=NUM_CLASSES, description="Target diagnostic categories")
    pretrained_backbone: bool = Field(default=True, description="Initialize backbone with pretrained weights")
    enable_unet_masking: bool = Field(default=True, description="Pre-mask images with skeletal U-Net")
    image_size: Tuple[int, int] = Field(default=(512, 512), description="Input spatial resolution (H, W)")

    model_config = ConfigDict(frozen=True)


class RadiomicsConfig(BaseModel):
    """Configuration for IBSI-compliant radiomic texture extraction."""
    num_bins: int = Field(default=64, description="Fixed bin count for intensity discretization")
    bin_width: Optional[float] = Field(default=None, description="Optional fixed bin width")
    distances: List[int] = Field(default_factory=lambda: [1], description="Inter-pixel offsets (delta=1)")
    angles: List[float] = Field(default_factory=lambda: [0.0, 45.0, 90.0, 135.0], description="GLCM directional angles")
    resample_resolution: Optional[Tuple[float, float]] = Field(default=(1.0, 1.0), description="In-plane voxel spacing mm")
    normalize_intensities: bool = Field(default=True, description="Normalize intensity histogram prior to discretization")

    model_config = ConfigDict(frozen=True)


class TrainingConfig(BaseModel):
    """Hyperparameters and execution rules for the training loop."""
    seed: int = Field(default=42, description="RNG seed for deterministic reproducibility")
    batch_size: int = Field(default=16, description="Mini-batch size")
    epochs: int = Field(default=100, description="Maximum training epochs")
    learning_rate: float = Field(default=1e-4, description="AdamW base learning rate")
    weight_decay: float = Field(default=1e-2, description="AdamW decoupled weight decay")
    focal_loss_gamma: float = Field(default=FOCAL_LOSS_GAMMA, description="Focusing parameter gamma")
    early_stopping_patience: int = Field(default=15, description="Epochs without validation gain before early exit")
    gradient_clip_norm: float = Field(default=1.0, description="Max L2 norm for gradient clipping")
    
    # Patient-level partition fractions (must sum to 1.0)
    train_split: float = Field(default=0.70, description="Train cohort fraction")
    val_split: float = Field(default=0.15, description="Validation cohort fraction")
    test_split: float = Field(default=0.15, description="Hold-out test cohort fraction")

    # Artifact storage paths
    artifacts_dir: Path = Field(default=Path("ml_artifacts"), description="Root directory for logs and models")
    checkpoints_dir: Path = Field(default=Path("ml_artifacts/checkpoints"), description="Saved model weights")
    logs_dir: Path = Field(default=Path("ml_artifacts/logs"), description="Training logs and metrics")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ReadinessGateConfig(BaseModel):
    """
    Readiness gate rules protecting against phantom training,
    data fabrication, and invalid dataset structures.
    """
    enforce_real_dataset: bool = Field(default=True, description="Hard block training if dataset is empty or synthetic")
    min_patient_records: int = Field(default=20, description="Minimum distinct patients required per class")
    require_all_classes: bool = Field(default=True, description="Require representation of all 4 target classes")
    allow_synthetic_training: bool = Field(default=False, description="STRICT RULE 5: Never train on synthetic data")
    verify_patient_isolation: bool = Field(default=True, description="Ensure zero patient overlap across splits")
    allowed_modalities: List[str] = Field(
        default_factory=lambda: list(SUPPORTED_MODALITIES),
        description="Supported imaging modalities"
    )

    model_config = ConfigDict(frozen=True)
