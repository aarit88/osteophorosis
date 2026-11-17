"""
Unit tests for IBSI-compliant 102-D Radiomics Texture Extraction Engine.
"""

import numpy as np
import pytest

from osteoguard.constants import RADIOMICS_FEATURE_DIM
from osteoguard.radiomics.engine import IBSIRadiomicsEngine


def test_radiomics_feature_count_and_names():
    """Verify that radiomic engine generates exactly 102 descriptors."""
    engine = IBSIRadiomicsEngine(num_bins=32)
    assert len(engine.feature_names) == RADIOMICS_FEATURE_DIM
    assert len(engine.feature_names) == 102


def test_radiomics_feature_extraction_shape_and_types():
    """Verify extracted array shape, dtype, and absence of NaNs or Infs."""
    engine = IBSIRadiomicsEngine(num_bins=32)

    # Test image with simulated trabecular texture
    np.random.seed(42)
    test_image = np.random.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)
    test_mask = np.ones((64, 64), dtype=np.uint8)

    features = engine.extract_features(test_image, test_mask)

    assert isinstance(features, np.ndarray)
    assert features.shape == (102,)
    assert features.dtype == np.float32
    assert not np.isnan(features).any(), "NaN found in radiomics features"
    assert not np.isinf(features).any(), "Inf found in radiomics features"


def test_radiomics_deterministic_reproducibility():
    """Verify identical inputs yield bitwise identical radiomic vectors."""
    engine = IBSIRadiomicsEngine(num_bins=32)

    np.random.seed(123)
    img = np.random.normal(100.0, 25.0, size=(48, 48)).astype(np.float32)

    run_1 = engine.extract_features(img)
    run_2 = engine.extract_features(img)

    assert np.array_equal(run_1, run_2), "Radiomic extraction is non-deterministic"


def test_radiomics_dictionary_output():
    """Verify named feature dictionary mapping."""
    engine = IBSIRadiomicsEngine(num_bins=16)
    img = np.ones((32, 32), dtype=np.float32) * 50.0

    feat_dict = engine.extract_features_dict(img)
    assert len(feat_dict) == 102
    assert "first_order_mean" in feat_dict
    assert "glcm_contrast" in feat_dict
    assert "glrlm_lre" in feat_dict
    assert "glszm_lae" in feat_dict
    assert "ngtdm_contrast" in feat_dict

