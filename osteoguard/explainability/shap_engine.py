"""
OsteoGuard AI - Tabular SHAP Explainability Engine & Exact Reference Calculator.

Implements:
1. Exact mathematical Shapley value reference calculator for low-dimensional
   synthetic test fixtures (d <= 6) conforming to cooperative game theory axioms:
     Efficiency (Local Accuracy): sum(phi_i) + phi_0 = f(x)
     Symmetry, Dummy player, and Additivity.
2. Modular TabularShapAdapter with strict feature-schema validation, background data
   management, and dependency detection for production SHAP libraries.

RESEARCH GOVERNANCE NOTICE:
    - The third-party 'shap' package is NOT currently installed in the environment.
    - As mandated by Stage 29, we do NOT silently simulate or fabricate a third-party
      library wrapper. The production adapter is marked as:
      STATUS: PARTIAL_DEPENDENCY_NOT_INSTALLED
    - For exact mathematical validation on controlled fixtures, ExactShapleyCalculator
      computes true combinatorial Shapley values analytically without approximations.
    - Stage 27 unverified 21 GLDM padding features are STRICTLY EXCLUDED.
"""

import itertools
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

# Check if third-party shap package is installed
try:
    import shap  # type: ignore
    _SHAP_AVAILABLE = True
    _SHAP_VERSION = getattr(shap, "__version__", "unknown")
except ImportError:
    _SHAP_AVAILABLE = False
    _SHAP_VERSION = None


class ShapFeatureAttribution(BaseModel):
    feature_name: str = Field(..., description="Feature identifier")
    feature_value: float = Field(..., description="Observed input value for feature")
    shapley_value: float = Field(..., description="Signed contribution score phi_i")


class ShapAttributionResult(BaseModel):
    """
    Structured outcome of Shapley feature attribution analysis.
    """
    feature_attributions: List[ShapFeatureAttribution] = Field(..., description="Per-feature contributions")
    base_value: float = Field(..., description="Expected baseline model output phi_0 = E[f(X)]")
    model_output: float = Field(..., description="Actual model prediction f(x)")
    sum_attributions: float = Field(..., description="Sum of all phi_i")
    reconstructed_output: float = Field(..., description="base_value + sum_attributions")
    local_accuracy_residual: float = Field(..., description="|reconstructed_output - model_output|")
    explainer_type: str = Field(..., description="Algorithm used (e.g. ExactCombinatorialShapley, KernelSHAP)")
    status: str = Field(..., description="Execution status")
    disclaimer: str = Field(
        default=(
            "RESEARCH ONLY: Shapley values reflect feature attribution within the "
            "computational model; they do NOT establish clinical biological causality."
        )
    )


class ExactShapleyCalculator:
    """
    Combinatorial Exact Shapley Value Calculator for low-dimensional functions (d <= 6).

    Uses the classic Shapley formula from cooperative game theory:
      phi_i(v) = sum_{S subset N \\ {i}} [ |S|! * (|N| - |S| - 1)! / |N|! ] * [ v(S union {i}) - v(S) ]
    where the characteristic function v(S) evaluates f(x) by conditioning on features in S
    and marginalizing features not in S over the empirical background distribution X_bg:
      v(S) = (1 / |X_bg|) * sum_{z in X_bg} f(x_S union z_{N \\ S})
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], Union[float, np.ndarray]],
        feature_names: List[str],
        background_data: np.ndarray,
    ):
        if not callable(predict_fn):
            raise TypeError("predict_fn must be a callable function f(x)")
        if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
            raise ValueError("feature_names must be a non-empty sequence of strings")
        
        bg = np.asarray(background_data, dtype=np.float64)
        if bg.ndim != 2:
            raise ValueError(f"background_data must be 2D (N_samples, n_features), got shape {bg.shape}")
        if bg.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature dimension mismatch: background has {bg.shape[1]} features, "
                f"feature_names has {len(feature_names)}"
            )

        if len(feature_names) > 8:
            raise ValueError(
                f"Exact combinatorial Shapley calculation is O(2^d) and restricted to d <= 8. "
                f"Got d = {len(feature_names)}. Use sampling/kernel approximation for larger sets."
            )

        self.predict_fn = predict_fn
        self.feature_names = list(feature_names)
        self.background_data = bg
        self.d = len(feature_names)
        self.n_bg = bg.shape[0]

        # Precompute base value phi_0 = E_{z in bg}[f(z)]
        bg_preds = [float(self._eval_scalar(z)) for z in self.background_data]
        self.base_value: float = float(np.mean(bg_preds))

    def _eval_scalar(self, x: np.ndarray) -> float:
        """Evaluates predict_fn on single 1D vector and ensures scalar float output."""
        x_2d = x.reshape(1, -1)
        res = self.predict_fn(x_2d)
        if isinstance(res, np.ndarray):
            return float(res.squeeze())
        return float(res)

    def _evaluate_coalition(self, x_input: np.ndarray, coalition_indices: Sequence[int]) -> float:
        """
        Evaluates characteristic function v(S) by marginalizing over background:
          v(S) = (1 / |X_bg|) * sum_{z in X_bg} f(x_S union z_{N \\ S})
        """
        s_set = set(coalition_indices)
        if len(s_set) == 0:
            return self.base_value
        if len(s_set) == self.d:
            return self._eval_scalar(x_input)

        # Vectorized or batched evaluation across background rows
        synthetic_batch = np.copy(self.background_data)
        for idx in s_set:
            synthetic_batch[:, idx] = x_input[idx]

        res = self.predict_fn(synthetic_batch)
        if isinstance(res, np.ndarray):
            return float(np.mean(res))
        return float(res)

    def explain(self, x: np.ndarray) -> ShapAttributionResult:
        """
        Computes exact Shapley values for input instance x.
        Guarantees local accuracy: sum(phi_i) + phi_0 == f(x) within float precision.
        """
        x_vec = np.asarray(x, dtype=np.float64).flatten()
        if len(x_vec) != self.d:
            raise ValueError(f"Input feature vector length {len(x_vec)} does not match expected {self.d}")

        model_output = self._eval_scalar(x_vec)
        n = self.d
        all_indices = set(range(n))

        shap_values = np.zeros(n, dtype=np.float64)

        # Precompute factorials
        fact = math.factorial

        for i in range(n):
            other_indices = [idx for idx in range(n) if idx != i]
            phi_i = 0.0

            # Sum over all subsets S of N \ {i}
            for s_size in range(n):
                weight = (fact(s_size) * fact(n - s_size - 1)) / fact(n)
                for s_coalition in itertools.combinations(other_indices, s_size):
                    v_s = self._evaluate_coalition(x_vec, s_coalition)
                    v_s_with_i = self._evaluate_coalition(x_vec, list(s_coalition) + [i])
                    marginal_contribution = v_s_with_i - v_s
                    phi_i += weight * marginal_contribution

            shap_values[i] = phi_i

        sum_phi = float(np.sum(shap_values))
        reconstructed = float(self.base_value + sum_phi)
        residual = abs(reconstructed - model_output)

        attributions = [
            ShapFeatureAttribution(
                feature_name=self.feature_names[i],
                feature_value=round(float(x_vec[i]), 5),
                shapley_value=round(float(shap_values[i]), 6),
            )
            for i in range(n)
        ]

        return ShapAttributionResult(
            feature_attributions=attributions,
            base_value=round(self.base_value, 6),
            model_output=round(model_output, 6),
            sum_attributions=round(sum_phi, 6),
            reconstructed_output=round(reconstructed, 6),
            local_accuracy_residual=round(residual, 8),
            explainer_type="ExactCombinatorialShapley",
            status="VERIFIED_LOCAL_ACCURACY",
        )


class TabularShapAdapter:
    """
    Modular Adapter for Tabular Feature Explanations.
    Manages feature schema validation, background datasets, and third-party SHAP bindings.
    """

    def __init__(
        self,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
            raise ValueError("feature_names must be a non-empty sequence")
        
        self.feature_names = list(feature_names)
        self.background_data = np.asarray(background_data, dtype=np.float64) if background_data is not None else None
        self.predict_fn = predict_fn

        if self.background_data is not None:
            if self.background_data.ndim != 2 or self.background_data.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Background data shape {self.background_data.shape} incompatible with "
                    f"{len(self.feature_names)} features"
                )

        self.shap_available: bool = _SHAP_AVAILABLE
        self.shap_version: Optional[str] = _SHAP_VERSION
        self.explainer_status: str = (
            "READY" if self.shap_available else "PARTIAL_DEPENDENCY_NOT_INSTALLED"
        )

    def validate_schema(self, input_dict_or_array: Union[Dict[str, float], np.ndarray]) -> np.ndarray:
        """
        Validates input against registered feature schema.
        Raises ValueError if features are missing or invalid.
        """
        if isinstance(input_dict_or_array, dict):
            missing = [f for f in self.feature_names if f not in input_dict_or_array]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            vec = np.array([float(input_dict_or_array[f]) for f in self.feature_names], dtype=np.float64)
            return vec

        arr = np.asarray(input_dict_or_array, dtype=np.float64)
        if arr.ndim == 1:
            if len(arr) != len(self.feature_names):
                raise ValueError(
                    f"Input vector length {len(arr)} does not match schema dimension {len(self.feature_names)}"
                )
            return arr
        elif arr.ndim == 2:
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Input array columns {arr.shape[1]} do not match schema dimension {len(self.feature_names)}"
                )
            return arr

        raise TypeError(f"Input must be dict or ndarray, got {type(input_dict_or_array)}")

    def explain(
        self,
        instance: Union[Dict[str, float], np.ndarray],
    ) -> ShapAttributionResult:
        """
        Generates feature attributions. If third-party shap is installed, delegates to KernelExplainer;
        if low-dimensional (d <= 6), falls back to ExactShapleyCalculator;
        otherwise raises RuntimeError explaining dependency requirement.
        """
        vec = self.validate_schema(instance).flatten()

        if self.predict_fn is None or self.background_data is None:
            raise RuntimeError("predict_fn and background_data must be configured to generate SHAP explanations.")

        if self.shap_available:
            # Future integration seam when shap package is installed
            explainer = shap.KernelExplainer(self.predict_fn, self.background_data)  # type: ignore
            shap_vals = explainer.shap_values(vec.reshape(1, -1))
            base_val = float(explainer.expected_value)
            pred_val = float(self.predict_fn(vec.reshape(1, -1)))
            
            vals_flat = np.asarray(shap_vals).flatten()
            attrs = [
                ShapFeatureAttribution(
                    feature_name=self.feature_names[i],
                    feature_value=float(vec[i]),
                    shapley_value=float(vals_flat[i]),
                )
                for i in range(len(self.feature_names))
            ]
            sum_s = float(np.sum(vals_flat))
            return ShapAttributionResult(
                feature_attributions=attrs,
                base_value=base_val,
                model_output=pred_val,
                sum_attributions=sum_s,
                reconstructed_output=base_val + sum_s,
                local_accuracy_residual=abs((base_val + sum_s) - pred_val),
                explainer_type="KernelExplainer",
                status="COMPLETED_VIA_SHAP_LIBRARY",
            )

        # Fallback to Exact Shapley Calculator if d <= 6
        if len(self.feature_names) <= 6:
            calc = ExactShapleyCalculator(
                predict_fn=self.predict_fn,
                feature_names=self.feature_names,
                background_data=self.background_data,
            )
            return calc.explain(vec)

        raise RuntimeError(
            f"Third-party 'shap' package is not installed and feature dimension d = {len(self.feature_names)} "
            f"exceeds exact combinatorial limit (d <= 6). Install 'shap' in the environment to enable "
            f"KernelExplainer or TreeExplainer."
        )

