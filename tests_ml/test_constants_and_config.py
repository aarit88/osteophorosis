"""
Unit tests for OsteoGuard AI Constants, Taxonomies, and Configurations.
Enforces Rule 9 (Never silently change target classes) and IBSI dimension standards.
"""

import pytest
from osteoguard.constants import (
    CLASS_DISPLAY_NAMES,
    CLASS_TO_IDX,
    FOCAL_LOSS_GAMMA,
    FUSION_VECTOR_DIM,
    IBSI_CATEGORIES,
    IDX_TO_CLASS,
    MLP_DROPOUT_RATE,
    MLP_HIDDEN_DIM_1,
    MLP_HIDDEN_DIM_2,
    NUM_CLASSES,
    RADIOMICS_FEATURE_DIM,
    SEMANTIC_EMBEDDING_DIM,
    SUPPORTED_MODALITIES,
    TARGET_CLASSES,
    TOTAL_RADIOMIC_FEATURES,
)
from osteoguard.config import (
    ModelConfig,
    RadiomicsConfig,
    TrainingConfig,
    ReadinessGateConfig,
)


def test_target_classes_integrity():
    """Verify strictly required 4-class taxonomy."""
    assert len(TARGET_CLASSES) == 4
    assert TARGET_CLASSES == ("metastasis", "paget", "oi", "healthy")
    assert NUM_CLASSES == 4

    # Verify bidirectional mappings
    for idx, cls in enumerate(TARGET_CLASSES):
        assert CLASS_TO_IDX[cls] == idx
        assert IDX_TO_CLASS[idx] == cls
        assert cls in CLASS_DISPLAY_NAMES


def test_ibsi_feature_dimensions():
    """Verify that IBSI texture categories sum to exactly 102 descriptors."""
    assert IBSI_CATEGORIES["first_order"]["count"] == 18
    assert IBSI_CATEGORIES["glcm"]["count"] == 26
    assert IBSI_CATEGORIES["glrlm"]["count"] == 16
    assert IBSI_CATEGORIES["glszm"]["count"] == 16
    assert IBSI_CATEGORIES["gldm_ngtdm"]["count"] == 26
    assert TOTAL_RADIOMIC_FEATURES == 102
    assert RADIOMICS_FEATURE_DIM == 102


def test_network_dimensions():
    """Verify latent representation dimensions matching research paper."""
    assert SEMANTIC_EMBEDDING_DIM == 512
    assert RADIOMICS_FEATURE_DIM == 102
    assert FUSION_VECTOR_DIM == 614  # 512 + 102
    assert MLP_HIDDEN_DIM_1 == 256
    assert MLP_HIDDEN_DIM_2 == 64
    assert MLP_DROPOUT_RATE == 0.4
    assert FOCAL_LOSS_GAMMA == 2.0


def test_supported_modalities():
    """Verify approved medical imaging modalities."""
    assert "xray" in SUPPORTED_MODALITIES
    assert "ct" in SUPPORTED_MODALITIES
    assert "bone_scan" in SUPPORTED_MODALITIES
    # Animal micro-CT must NOT be an approved primary clinical modality
    assert "micro_ct" not in SUPPORTED_MODALITIES


def test_pydantic_configs():
    """Verify configuration immutability and default values."""
    m_cfg = ModelConfig()
    assert m_cfg.semantic_dim == 512
    assert m_cfg.radiomics_dim == 102
    assert m_cfg.fusion_dim == 614

    r_cfg = RadiomicsConfig()
    assert r_cfg.num_bins == 64

    t_cfg = TrainingConfig()
    assert t_cfg.focal_loss_gamma == 2.0
    assert abs(t_cfg.train_split + t_cfg.val_split + t_cfg.test_split - 1.0) < 1e-4

    gate_cfg = ReadinessGateConfig()
    assert gate_cfg.allow_synthetic_training is False
    assert gate_cfg.enforce_real_dataset is True

