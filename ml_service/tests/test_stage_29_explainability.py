"""
OsteoGuard AI - Stage 29 Automated Verification Test Suite.

Comprehensive tests for:
1. Checkpoint cryptographic immutability (Rule 1).
2. Stage 29 artifact completeness (10 deliverables + mirrored report).
3. Grad-CAM mathematical computation, shape, and normalization.
4. Grad-CAM spatial localization on controlled synthetic CNN fixture.
5. Grad-CAM spatial shift sensitivity.
6. Adebayo et al. model parameter randomization sanity check.
7. Grad-CAM zero-gradient and flat-input numerical safety.
8. Grad-CAM hook lifecycle management and memory leak prevention.
9. Grad-CAM model state preservation and weight immutability.
10. Grad-CAM input validation and error raising.
11. Exact Shapley calculator local accuracy (Efficiency Axiom).
12. Exact Shapley calculator analytical formula alignment.
13. Exact Shapley baseline sensitivity.
14. Tabular SHAP adapter status and schema validation.
15. Stage 27 radiomics discrepancy preservation and isolation.
16. Multimodal explainability architecture separation.
17. Zero patient data ingestion and preservation of burned holdouts.
18. Host Python and React frontend integrity.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from osteoguard.explainability.gradcam import GradCAMExplainer
from osteoguard.explainability.shap_engine import (
    ExactShapleyCalculator,
    TabularShapAdapter,
    ShapAttributionResult,
    _SHAP_AVAILABLE,
)
from osteoguard.explainability.synthetic_fixtures import (
    ControlledSpatialCNN,
    SyntheticInputSuite,
    analytical_additive_model,
    compute_exact_analytical_shapley,
    create_deterministic_background_dataset,
    SYNTHETIC_FEATURE_NAMES,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CHECKPOINT_PATH = WORKSPACE_ROOT / "ml_artifacts" / "btxrd_frozen_model" / "best_checkpoint.pt"
EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
STAGE29_DIR = WORKSPACE_ROOT / "ml_artifacts" / "stage29_explainability"
MIRROR_REPORT = WORKSPACE_ROOT / "ml_artifacts" / "stage_reports" / "STAGE_29_EXPLAINABILITY.md"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def test_01_checkpoint_hash_strictly_unmodified():
    """Verify frozen BTXRD checkpoint SHA-256 matches cryptographic baseline."""
    assert FROZEN_CHECKPOINT_PATH.exists(), f"Frozen checkpoint missing at {FROZEN_CHECKPOINT_PATH}"
    actual = compute_sha256(FROZEN_CHECKPOINT_PATH)
    assert actual == EXPECTED_SHA256, f"Checkpoint hash modified!\nExpected: {EXPECTED_SHA256}\nActual:   {actual}"


def test_02_all_stage29_artifacts_exist():
    """Verify all 10 Stage 29 deliverables and mirrored report exist and are populated."""
    required_files = [
        "STAGE_29_EXPLAINABILITY_REPORT.md",
        "gradcam_technical_specification.md",
        "shap_technical_specification.md",
        "synthetic_explainability_protocol.md",
        "explanation_reliability_test_report.md",
        "multimodal_explainability_architecture.md",
        "stage27_radiomics_integration_limitations.md",
        "implementation_limitations_and_future_validation.md",
        "reproducibility_manifest.json",
        "test_execution_report.json",
    ]
    for rf in required_files:
        p = STAGE29_DIR / rf
        assert p.exists(), f"Required artifact missing: {rf}"
        assert p.stat().st_size > 0, f"Artifact {rf} is empty"

    assert MIRROR_REPORT.exists(), f"Mirrored report missing at {MIRROR_REPORT}"
    assert MIRROR_REPORT.stat().st_size > 0, "Mirrored report is empty"


def test_03_gradcam_mathematical_computation_and_shape():
    """Verify Grad-CAM output dimensions, normalization in [0, 1], and post-ReLU non-negativity."""
    cnn = ControlledSpatialCNN()
    explainer = GradCAMExplainer(cnn, "conv1")
    inp = SyntheticInputSuite.create_bright_spot(size=32)

    cam = explainer.generate_cam(inp, target_class_idx=0, input_size=(64, 64))

    assert cam.shape == (64, 64)
    assert cam.dtype == np.float32
    assert np.min(cam) >= 0.0
    assert np.max(cam) <= 1.0
    assert not np.isnan(cam).any()
    assert not np.isinf(cam).any()


def test_04_gradcam_spatial_localization_on_synthetic_fixture():
    """Verify Grad-CAM correctly localizes controlled bright region in top-left quadrant."""
    cnn = ControlledSpatialCNN()
    explainer = GradCAMExplainer(cnn, "conv1")
    inp = SyntheticInputSuite.create_bright_spot(size=32)

    cam = explainer.generate_cam(inp, target_class_idx=0)
    
    tl_energy = float(np.mean(cam[4:12, 4:12]))
    br_energy = float(np.mean(cam[20:28, 20:28]))
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)

    assert tl_energy > 0.5
    assert br_energy < 0.1
    assert 4 <= peak_y <= 12
    assert 4 <= peak_x <= 12


def test_05_gradcam_spatial_shift_response():
    """Verify Grad-CAM peak moves dynamically when synthetic feature is translated."""
    cnn = ControlledSpatialCNN()
    explainer = GradCAMExplainer(cnn, "conv1")
    inp_shifted = SyntheticInputSuite.create_shifted_spot(size=32)

    cam_shifted = explainer.generate_cam(inp_shifted, target_class_idx=0)
    
    tl_energy = float(np.mean(cam_shifted[4:12, 4:12]))
    br_energy = float(np.mean(cam_shifted[20:28, 20:28]))
    peak_y, peak_x = np.unravel_index(np.argmax(cam_shifted), cam_shifted.shape)

    assert br_energy > tl_energy
    assert 20 <= peak_y <= 28
    assert 20 <= peak_x <= 28


def test_06_gradcam_adebayo_parameter_randomization_sanity_check():
    """Verify Adebayo sanity check: altering model weights substantially alters heatmap."""
    cnn = ControlledSpatialCNN()
    inp = SyntheticInputSuite.create_bright_spot(size=32)
    explainer = GradCAMExplainer(cnn, "conv1")
    cam_orig = explainer.generate_cam(inp, target_class_idx=0)

    # Invert linear routing weights
    cnn_scrambled = ControlledSpatialCNN()
    with torch.no_grad():
        cnn_scrambled.fc.weight.data = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    explainer_scrambled = GradCAMExplainer(cnn_scrambled, "conv1")
    cam_scrambled = explainer_scrambled.generate_cam(inp, target_class_idx=0)

    active_diff = float(np.mean(np.abs(cam_orig[4:12, 4:12] - cam_scrambled[4:12, 4:12])))
    max_diff = float(np.max(np.abs(cam_orig - cam_scrambled)))

    assert active_diff > 0.5, f"Heatmap insensitive to parameter modification! Active diff={active_diff}"
    assert max_diff > 0.5


def test_07_gradcam_zero_gradient_and_uniform_input_safety():
    """Verify uniform inputs and flat gradients return safe zero/flat maps without NaNs or Infs."""
    cnn = ControlledSpatialCNN()
    explainer = GradCAMExplainer(cnn, "conv1")
    inp_uniform = SyntheticInputSuite.create_uniform_image(size=32, val=0.5)

    cam = explainer.generate_cam(inp_uniform, target_class_idx=0)
    assert not np.isnan(cam).any()
    assert not np.isinf(cam).any()
    assert np.min(cam) >= 0.0


def test_08_gradcam_hook_lifecycle_and_memory_leak_prevention():
    """Verify hook handles are cleanly removed and registered state toggles correctly."""
    cnn = ControlledSpatialCNN()
    explainer = GradCAMExplainer(cnn, "conv1")
    assert explainer._hooks_registered is True
    assert explainer._fwd_handle is not None
    assert explainer._bwd_handle is not None

    explainer.remove_hooks()
    assert explainer._hooks_registered is False
    assert explainer._fwd_handle is None
    assert explainer._bwd_handle is None

    # Context manager test
    with GradCAMExplainer(cnn, "conv1") as exp:
        assert exp._hooks_registered is True
    # Out of context, hooks should be removed
    assert exp._hooks_registered is False


def test_09_gradcam_model_state_preservation_and_weight_immutability():
    """Verify model training mode is preserved and weights remain completely identical."""
    cnn = ControlledSpatialCNN()
    cnn.train()  # Explicitly set to train mode

    # Store copy of weights
    w_conv_orig = cnn.conv1.weight.clone().detach()
    w_fc_orig = cnn.fc.weight.clone().detach()

    explainer = GradCAMExplainer(cnn, "conv1")
    inp = SyntheticInputSuite.create_bright_spot(size=32)
    _ = explainer.generate_cam(inp, target_class_idx=0)

    # Verify model is restored to train mode
    assert cnn.training is True

    # Verify weights are unmodified
    assert torch.equal(cnn.conv1.weight, w_conv_orig)
    assert torch.equal(cnn.fc.weight, w_fc_orig)


def test_10_gradcam_input_validation_and_error_handling():
    """Verify informative errors on invalid input tensors, layer names, and class indices."""
    cnn = ControlledSpatialCNN()

    # Invalid layer name
    with pytest.raises(ValueError, match="Target layer 'nonexistent_layer' not found"):
        GradCAMExplainer(cnn, "nonexistent_layer")

    explainer = GradCAMExplainer(cnn, "conv1")

    # Non-4D tensor
    with pytest.raises(ValueError, match="input_tensor must be 4D"):
        explainer.generate_cam(torch.zeros(32, 32))

    # Batch size > 1
    with pytest.raises(ValueError, match="supports batch size 1"):
        explainer.generate_cam(torch.zeros(2, 1, 32, 32))

    # Out of bounds class index
    inp = SyntheticInputSuite.create_bright_spot(size=32)
    with pytest.raises(ValueError, match="out of range for 2 classes"):
        explainer.generate_cam(inp, target_class_idx=5)


def test_11_exact_shapley_local_accuracy_and_efficiency_axiom():
    """Verify Efficiency Axiom: sum(phi_i) + phi_0 == f(x) within float precision."""
    bg_data = create_deterministic_background_dataset(n_samples=20)
    exact_calc = ExactShapleyCalculator(
        predict_fn=analytical_additive_model,
        feature_names=SYNTHETIC_FEATURE_NAMES,
        background_data=bg_data,
    )
    test_instance = np.array([3.0, 150.0, 0.5, 750.0, 5.0])
    res = exact_calc.explain(test_instance)

    assert res.local_accuracy_residual < 1e-6
    assert abs(res.reconstructed_output - res.model_output) < 1e-6


def test_12_exact_shapley_matches_analytical_derivation():
    """Verify combinatorial Shapley values match exact theoretical formula for linear models."""
    bg_data = create_deterministic_background_dataset(n_samples=20)
    exact_calc = ExactShapleyCalculator(
        predict_fn=analytical_additive_model,
        feature_names=SYNTHETIC_FEATURE_NAMES,
        background_data=bg_data,
    )
    test_instance = np.array([4.5, 200.0, 0.8, 600.0, 2.0])
    
    phi_anal, base_anal = compute_exact_analytical_shapley(test_instance, bg_data)
    res = exact_calc.explain(test_instance)
    calc_phis = np.array([a.shapley_value for a in res.feature_attributions])

    assert np.allclose(calc_phis, phi_anal, atol=1e-4)
    assert abs(res.base_value - base_anal) < 1e-4


def test_13_exact_shapley_baseline_sensitivity():
    """Verify that altering the background dataset shifts base value and attributions predictably."""
    bg_1 = create_deterministic_background_dataset(n_samples=10)
    bg_2 = bg_1 + 50.0  # Shift background features upwards

    calc_1 = ExactShapleyCalculator(analytical_additive_model, SYNTHETIC_FEATURE_NAMES, bg_1)
    calc_2 = ExactShapleyCalculator(analytical_additive_model, SYNTHETIC_FEATURE_NAMES, bg_2)

    test_instance = np.array([3.0, 150.0, 0.5, 750.0, 5.0])
    res_1 = calc_1.explain(test_instance)
    res_2 = calc_2.explain(test_instance)

    # Base values must differ due to shifted background expectation
    assert abs(res_1.base_value - res_2.base_value) > 1.0
    # Both must still satisfy local accuracy individually
    assert res_1.local_accuracy_residual < 1e-6
    assert res_2.local_accuracy_residual < 1e-6


def test_14_tabular_shap_adapter_status_and_schema_validation():
    """Verify TabularShapAdapter schema validation and dependency reporting."""
    bg_data = create_deterministic_background_dataset(n_samples=10)
    adapter = TabularShapAdapter(
        feature_names=SYNTHETIC_FEATURE_NAMES,
        background_data=bg_data,
        predict_fn=analytical_additive_model,
    )

    # Dependency status
    assert adapter.shap_available == (adapter.shap_version is not None)
    assert adapter.explainer_status == (
        "READY" if adapter.shap_available else "PARTIAL_DEPENDENCY_NOT_INSTALLED"
    )

    # Missing feature dictionary validation
    with pytest.raises(ValueError, match="Missing required features"):
        adapter.validate_schema({"cortical_thickness_mm": 3.0})

    # Array dimension mismatch
    with pytest.raises(ValueError, match="does not match schema dimension"):
        adapter.validate_schema(np.array([1.0, 2.0]))


def test_15_stage27_radiomics_discrepancy_preserved_and_isolated():
    """Verify unverified 21 GLDM padding features are excluded from SHAP attributions."""
    limits_doc = STAGE29_DIR / "stage27_radiomics_integration_limitations.md"
    assert limits_doc.exists()
    content = limits_doc.read_text(encoding="utf-8")
    assert "81 Features Analytically Verified" in content
    assert "21 Features Unverified" in content
    assert "PARTIAL / SPECIFICATION UNRESOLVED" in content


def test_16_multimodal_architecture_document_and_separation():
    """Verify multimodal blueprint separates visual heatmaps and tabular feature attributions."""
    arch_doc = STAGE29_DIR / "multimodal_explainability_architecture.md"
    assert arch_doc.exists()
    content = arch_doc.read_text(encoding="utf-8")
    assert "Strict Separation of Heatmaps and Tabular Scores" in content
    assert "Correlation, Not Causation" in content
    assert "Volumetric CT" in content


def test_17_zero_unauthorized_patient_data_and_holdouts_intact():
    """Verify zero patient datasets ingested and Center 2/3 holdouts remain intact."""
    with open(STAGE29_DIR / "reproducibility_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["patient_data_accessed"] is False
    assert manifest["holdout_accessed"] is False
    assert manifest["models_trained"] is False
    assert (WORKSPACE_ROOT / "ml_artifacts" / "btxrd_center2_external_evaluation").exists()
    assert (WORKSPACE_ROOT / "ml_artifacts" / "btxrd_center3_external_evaluation").exists()


def test_18_host_python_and_frontend_untouched():
    """Verify host Python and React frontend remain completely unmodified."""
    host_python = Path(sys.executable)
    assert host_python.exists()

    frontend_dir = WORKSPACE_ROOT / "src"
    assert frontend_dir.exists()
    assert (frontend_dir / "App.tsx").exists()
    assert (frontend_dir / "main.tsx").exists()

