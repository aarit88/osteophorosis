"""
OsteoGuard AI - Controlled Synthetic Explainability Fixtures.

Provides deterministic synthetic models and controlled inputs for mathematically
verifying Grad-CAM spatial localization and SHAP feature attributions.

Zero patient data, zero clinical scans, zero stochastic training.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Deterministic Synthetic CNN Fixture for Grad-CAM
# =====================================================================

class ControlledSpatialCNN(nn.Module):
    """
    Deterministic 2-class, 2-channel CNN designed for analytical Grad-CAM verification.
    Architecture:
      Conv2d(1, 2, kernel_size=3, padding=1, bias=False) -> ReLU -> AdaptiveAvgPool2d(1) -> Flatten -> Linear(2, 2, bias=False)

    Filter 0 is fixed as an edge/patch detector that activates on bright pixels.
    Filter 1 is fixed as an inverted detector.
    Linear weights explicitly route Channel 0 to Class 0, and Channel 1 to Class 1.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2, 2, bias=False)

        # Set deterministic, non-random weights
        with torch.no_grad():
            # Filter 0: Uniform positive kernel (averages local intensity)
            self.conv1.weight[0, 0, :, :] = torch.tensor([
                [0.1, 0.1, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1],
            ], dtype=torch.float32)

            # Filter 1: Inverted negative kernel
            self.conv1.weight[1, 0, :, :] = torch.tensor([
                [-0.1, -0.1, -0.1],
                [-0.1, -0.2, -0.1],
                [-0.1, -0.1, -0.1],
            ], dtype=torch.float32)

            # Linear layer routing:
            # Class 0 = +1.0 * Channel 0 + 0.0 * Channel 1
            # Class 1 =  0.0 * Channel 0 + 1.0 * Channel 1
            self.fc.weight.data = torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
            ], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.relu(self.conv1(x))
        pooled = self.pool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.fc(flattened)
        return logits


class SyntheticInputSuite:
    """
    Factory generating 5 controlled synthetic images for Grad-CAM testing:
    1. Localized bright region (Top-Left quadrant).
    2. Localized dark region (Center of bright field).
    3. Uniform flat image (Zero spatial gradients).
    4. Two separate synthetic regions (Top-Left and Bottom-Right).
    5. Shifted synthetic region (Feature moved from Top-Left to Bottom-Right).
    """

    @staticmethod
    def create_bright_spot(size: int = 32, spot_val: float = 1.0, bg_val: float = 0.0) -> torch.Tensor:
        """1. Localized bright synthetic square in Top-Left (rows 4..12, cols 4..12)."""
        img = np.full((1, 1, size, size), bg_val, dtype=np.float32)
        img[0, 0, 4:12, 4:12] = spot_val
        return torch.from_numpy(img)

    @staticmethod
    def create_dark_spot(size: int = 32, spot_val: float = 0.0, bg_val: float = 1.0) -> torch.Tensor:
        """2. Dark localized synthetic region in Center (rows 12..20, cols 12..20)."""
        img = np.full((1, 1, size, size), bg_val, dtype=np.float32)
        img[0, 0, 12:20, 12:20] = spot_val
        return torch.from_numpy(img)

    @staticmethod
    def create_uniform_image(size: int = 32, val: float = 0.5) -> torch.Tensor:
        """3. Uniform flat image with zero spatial variance."""
        img = np.full((1, 1, size, size), val, dtype=np.float32)
        return torch.from_numpy(img)

    @staticmethod
    def create_two_regions(size: int = 32) -> torch.Tensor:
        """4. Two separate regions: Top-Left (4..10) and Bottom-Right (22..28)."""
        img = np.zeros((1, 1, size, size), dtype=np.float32)
        img[0, 0, 4:10, 4:10] = 1.0
        img[0, 0, 22:28, 22:28] = 0.8
        return torch.from_numpy(img)

    @staticmethod
    def create_shifted_spot(size: int = 32) -> torch.Tensor:
        """5. Shifted synthetic square moved to Bottom-Right (rows 20..28, cols 20..28)."""
        img = np.zeros((1, 1, size, size), dtype=np.float32)
        img[0, 0, 20:28, 20:28] = 1.0
        return torch.from_numpy(img)

    @classmethod
    def get_all_inputs(cls, size: int = 32) -> Dict[str, torch.Tensor]:
        return {
            "bright_spot_top_left": cls.create_bright_spot(size),
            "dark_spot_center": cls.create_dark_spot(size),
            "uniform_flat": cls.create_uniform_image(size),
            "two_regions": cls.create_two_regions(size),
            "shifted_spot_bottom_right": cls.create_shifted_spot(size),
        }


# =====================================================================
# 2. Controlled Additive Tabular Fixture for SHAP Verification
# =====================================================================

SYNTHETIC_FEATURE_NAMES: List[str] = [
    "cortical_thickness_mm",
    "trabecular_attenuation_hu",
    "endosteal_scalloping_ratio",
    "calcium_mass_mg_cm3",
    "microfracture_density_idx",
]

# Analytical weights for linear model: f(x) = sum(w_i * x_i) + bias
ANALYTICAL_WEIGHTS = np.array([3.0, -2.0, 1.5, -0.5, 4.0], dtype=np.float64)
ANALYTICAL_BIAS = 10.0


def analytical_additive_model(x_batch: np.ndarray) -> np.ndarray:
    """
    Controlled additive linear model:
      f(x) = 3.0*x0 - 2.0*x1 + 1.5*x2 - 0.5*x3 + 4.0*x4 + 10.0
    """
    x = np.asarray(x_batch, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.dot(x, ANALYTICAL_WEIGHTS) + ANALYTICAL_BIAS


def compute_exact_analytical_shapley(
    x_instance: np.ndarray,
    background_data: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Derives exact theoretical Shapley values for an additive linear function:
      f(x) = sum_i (w_i * x_i) + b
    For an additive model, the exact Shapley contribution for feature i is independent
    of interactions and simplifies to:
      phi_i(x) = w_i * (x_i - E[x_i])
    where E[x_i] = mean of feature i across the background dataset.
    Base value:
      phi_0 = sum_i (w_i * E[x_i]) + b

    Returns:
      (phi_vector, phi_0)
    """
    x = np.asarray(x_instance, dtype=np.float64).flatten()
    bg = np.asarray(background_data, dtype=np.float64)
    mu_bg = np.mean(bg, axis=0)

    phi_vector = ANALYTICAL_WEIGHTS * (x - mu_bg)
    phi_0 = float(np.dot(ANALYTICAL_WEIGHTS, mu_bg) + ANALYTICAL_BIAS)

    return phi_vector, phi_0


def create_deterministic_background_dataset(n_samples: int = 20) -> np.ndarray:
    """
    Generates a deterministic 5-feature background dataset for SHAP testing.
    Uses fixed mathematical sequences to avoid stochastic variation.
    """
    bg = np.zeros((n_samples, 5), dtype=np.float64)
    for i in range(n_samples):
        # Feature 0: Cortical thickness [2.0 - 5.0 mm]
        bg[i, 0] = 2.0 + (i % 7) * 0.5
        # Feature 1: Trabecular attenuation [100 - 300 HU]
        bg[i, 1] = 100.0 + (i * 10.5) % 200.0
        # Feature 2: Endosteal scalloping [0.0 - 0.8]
        bg[i, 2] = 0.1 * (i % 9)
        # Feature 3: Calcium mass [500 - 1000 mg/cm3]
        bg[i, 3] = 500.0 + (i * 25.0) % 500.0
        # Feature 4: Microfracture density [0 - 10]
        bg[i, 4] = float(i % 11)
    return bg

