"""
OsteoGuard AI - Explainability Module (Grad-CAM & SHAP).

Contains:
    1. GradCAMExplainer: Production PyTorch Grad-CAM engine with automatic hook cleanup.
    2. ExactShapleyCalculator: Combinatorial game-theoretic Shapley engine for low-d models.
    3. TabularShapAdapter: Modular schema-validated tabular SHAP adapter.
    4. ControlledSpatialCNN & SyntheticInputSuite: Deterministic synthetic testing fixtures.
    5. RadiomicShapExplainer: Legacy heuristic explainer (retained for backwards compatibility).
"""

from osteoguard.explainability.gradcam import GradCAMExplainer
from osteoguard.explainability.shap_engine import (
    ExactShapleyCalculator,
    TabularShapAdapter,
    ShapAttributionResult,
    ShapFeatureAttribution,
)
from osteoguard.explainability.synthetic_fixtures import (
    ControlledSpatialCNN,
    SyntheticInputSuite,
    analytical_additive_model,
    compute_exact_analytical_shapley,
    create_deterministic_background_dataset,
    SYNTHETIC_FEATURE_NAMES,
)
from osteoguard.explainability.shap_explainer import (
    RadiomicShapExplainer,
    ShapFeatureContribution,
    CounterfactualExplanation,
)

__all__ = [
    # Grad-CAM
    "GradCAMExplainer",
    # SHAP Engine
    "ExactShapleyCalculator",
    "TabularShapAdapter",
    "ShapAttributionResult",
    "ShapFeatureAttribution",
    # Synthetic Fixtures
    "ControlledSpatialCNN",
    "SyntheticInputSuite",
    "analytical_additive_model",
    "compute_exact_analytical_shapley",
    "create_deterministic_background_dataset",
    "SYNTHETIC_FEATURE_NAMES",
    # Legacy
    "RadiomicShapExplainer",
    "ShapFeatureContribution",
    "CounterfactualExplanation",
]
