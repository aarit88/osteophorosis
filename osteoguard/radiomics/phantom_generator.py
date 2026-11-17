"""
OsteoGuard AI - Synthetic Digital Phantom Generator.

Produces completely synthetic, deterministic digital phantoms with analytically
derivable intensity distributions, spatial geometries, and masking configurations
for independent radiomics verification.

Zero patient data is used. Zero external datasets are downloaded.
All phantoms are mathematically defined and fully reproducible.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class SyntheticDigitalPhantomSuite:
    """
    Deterministic generation suite for synthetic radiomics verification phantoms.
    """

    @staticmethod
    def create_uniform_phantom(
        shape: Tuple[int, int] = (64, 64),
        intensity: float = 100.0,
        margin: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Creates a constant uniform intensity phantom with a central square mask.
        Analytical Properties:
            - Mean = intensity
            - Variance = 0.0
            - Std = 0.0
            - Min = intensity
            - Max = intensity
            - Range = 0.0
            - GLCM Contrast = 0.0
            - GLCM Homogeneity = 1.0
        """
        image = np.zeros(shape, dtype=np.float64)
        mask = np.zeros(shape, dtype=np.uint8)

        r_start, r_end = margin, shape[0] - margin
        c_start, c_end = margin, shape[1] - margin

        image[r_start:r_end, c_start:c_end] = intensity
        mask[r_start:r_end, c_start:c_end] = 1

        n_pixels = float((r_end - r_start) * (c_end - c_start))
        analytical_props = {
            "n_pixels": n_pixels,
            "mean": float(intensity),
            "variance": 0.0,
            "std": 0.0,
            "min": float(intensity),
            "max": float(intensity),
            "range": 0.0,
            "median": float(intensity),
            "glcm_contrast": 0.0,
            "glcm_homogeneity": 1.0,
            "glcm_asm": 1.0,
        }
        return image, mask, analytical_props

    @staticmethod
    def create_checkerboard_phantom(
        shape: Tuple[int, int] = (64, 64),
        tile_size: int = 8,
        low_val: float = 10.0,
        high_val: float = 90.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Creates a symmetric two-level checkerboard phantom.
        Analytical Properties:
            - Mean = (low_val + high_val) / 2.0
            - Variance = ((low_val - mean)^2 + (high_val - mean)^2) / 2.0
            - Min = low_val, Max = high_val, Range = high_val - low_val
        """
        rows, cols = shape
        image = np.zeros(shape, dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                tile_r = r // tile_size
                tile_c = c // tile_size
                if (tile_r + tile_c) % 2 == 0:
                    image[r, c] = low_val
                else:
                    image[r, c] = high_val

        mask = np.ones(shape, dtype=np.uint8)
        mean_val = (low_val + high_val) / 2.0
        var_val = ((low_val - mean_val) ** 2 + (high_val - mean_val) ** 2) / 2.0

        analytical_props = {
            "n_pixels": float(rows * cols),
            "mean": float(mean_val),
            "variance": float(var_val),
            "std": float(np.sqrt(var_val)),
            "min": float(low_val),
            "max": float(high_val),
            "range": float(high_val - low_val),
            "median": float(mean_val),
        }
        return image, mask, analytical_props

    @staticmethod
    def create_linear_gradient_phantom(
        shape: Tuple[int, int] = (64, 64),
        start_val: float = 10.0,
        end_val: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Creates a linear horizontal intensity ramp phantom.
        Analytical Properties:
            - Mean = (start_val + end_val) / 2.0
            - Monotonic gradient across columns
        """
        rows, cols = shape
        ramp = np.linspace(start_val, end_val, cols, dtype=np.float64)
        image = np.tile(ramp, (rows, 1))
        mask = np.ones(shape, dtype=np.uint8)

        mean_val = float(np.mean(image))
        var_val = float(np.var(image))

        analytical_props = {
            "n_pixels": float(rows * cols),
            "mean": mean_val,
            "variance": var_val,
            "std": float(np.sqrt(var_val)),
            "min": float(start_val),
            "max": float(end_val),
            "range": float(end_val - start_val),
            "median": float((start_val + end_val) / 2.0),
        }
        return image, mask, analytical_props

    @staticmethod
    def create_step_wedge_phantom(
        shape: Tuple[int, int] = (64, 64),
        levels: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Creates a discrete 4-level vertical step wedge phantom.
        """
        if levels is None:
            levels = [10.0, 30.0, 50.0, 70.0]

        rows, cols = shape
        n_levels = len(levels)
        step_width = cols // n_levels
        image = np.zeros(shape, dtype=np.float64)

        for i, val in enumerate(levels):
            c_start = i * step_width
            c_end = cols if i == n_levels - 1 else (i + 1) * step_width
            image[:, c_start:c_end] = val

        mask = np.ones(shape, dtype=np.uint8)
        analytical_props = {
            "n_pixels": float(rows * cols),
            "mean": float(np.mean(levels)),
            "min": float(min(levels)),
            "max": float(max(levels)),
            "range": float(max(levels) - min(levels)),
            "levels_count": float(n_levels),
        }
        return image, mask, analytical_props

    @staticmethod
    def create_geometric_annulus_phantom(
        shape: Tuple[int, int] = (64, 64),
        r_inner: int = 10,
        r_outer: int = 24,
        intensity: float = 75.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Creates a circular annulus (doughnut) phantom with curved ROI boundary.
        """
        image = np.zeros(shape, dtype=np.float64)
        mask = np.zeros(shape, dtype=np.uint8)

        cy, cx = shape[0] // 2, shape[1] // 2
        y, x = np.ogrid[:shape[0], :shape[1]]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2

        annulus_region = (dist_sq >= r_inner ** 2) & (dist_sq <= r_outer ** 2)
        image[annulus_region] = intensity
        mask[annulus_region] = 1

        n_pixels = float(np.sum(mask > 0))
        analytical_props = {
            "n_pixels": n_pixels,
            "mean": float(intensity),
            "variance": 0.0,
            "min": float(intensity),
            "max": float(intensity),
            "range": 0.0,
        }
        return image, mask, analytical_props

    @staticmethod
    def create_anisotropic_spacing_phantom(
        shape: Tuple[int, int] = (64, 64),
        spacing: Tuple[float, float] = (0.5, 1.5),
        intensity: float = 80.0,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Dict[str, float]]:
        """
        Creates a phantom paired with physical voxel spacing parameters (dy, dx) in mm.
        """
        image, mask, props = SyntheticDigitalPhantomSuite.create_uniform_phantom(
            shape=shape, intensity=intensity, margin=10
        )
        props["spacing_y"] = float(spacing[0])
        props["spacing_x"] = float(spacing[1])
        props["pixel_area_mm2"] = float(spacing[0] * spacing[1])
        return image, mask, spacing, props

