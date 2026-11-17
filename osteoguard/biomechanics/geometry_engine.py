"""
OsteoGuard AI - Cross-Sectional Biomechanical Geometry Engine.

Computes 2D geometric cross-sectional moments of area from cortical bone masks
using explicit physical coordinates and voxel/pixel spacing (mm).

Formulas:
    Cross-Sectional Area:
        A = sum(M(y, x) * dy * dx)  [mm^2]
    Centroid:
        x_bar = sum((x * dx) * M(y, x)) / sum(M(y, x))  [mm]
        y_bar = sum((y * dy) * M(y, x)) / sum(M(y, x))  [mm]
    Central Second Moments of Area:
        I_x = sum((y * dy - y_bar)^2 * M(y, x) * dy * dx)  [mm^4]
        I_y = sum((x * dx - x_bar)^2 * M(y, x) * dy * dx)  [mm^4]
        I_xy = sum((x * dx - x_bar) * (y * dy - y_bar) * M(y, x) * dy * dx)  [mm^4]
    Polar Second Moment of Area:
        J_p = I_x + I_y  [mm^4]
    Principal Moments:
        I_max, I_min = (I_x + I_y)/2 +/- sqrt(((I_x - I_y)/2)^2 + I_xy^2)  [mm^4]
        theta_p = 0.5 * atan2(-2 * I_xy, I_x - I_y)  [radians or degrees]

IMPORTANT PHYSICAL & CLINICAL DISTINCTIONS:
    1. J_p is the POLAR SECOND MOMENT OF AREA (units: mm^4), quantifying purely
       geometric resistance of a 2D cross-section to torsional shear.
    2. J_p is NOT mass moment of inertia (units: kg*m^2).
    3. 2D geometric moments are NOT equivalent to full 3D CT Rigidity Analysis (CTRA).
       CTRA integrates voxel-specific Young's moduli E(HU) to compute 3D axial,
       bending, and torsional rigidities across heterogeneous tissue densities.
    4. 2D geometric moment ratios do NOT directly predict clinical fracture probability.
"""

import math
from typing import Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field


class CrossSectionalGeometryResult(BaseModel):
    """
    Structured outcome of the 2D cross-sectional geometry evaluation.
    """
    # Physical spacing
    pixel_spacing_y_mm: float = Field(..., gt=0.0, description="Vertical pixel spacing (mm)")
    pixel_spacing_x_mm: float = Field(..., gt=0.0, description="Horizontal pixel spacing (mm)")
    
    # 0th Moment: Area
    cortical_area_mm2: float = Field(..., ge=0.0, description="Cortical cross-sectional area (mm^2)")
    pixel_count: int = Field(..., ge=0, description="Total segmented cortical pixels")
    
    # 1st Moments: Centroid
    centroid_x_mm: float = Field(..., description="Centroid x-coordinate in physical space (mm)")
    centroid_y_mm: float = Field(..., description="Centroid y-coordinate in physical space (mm)")
    
    # 2nd Moments of Area
    ix_mm4: float = Field(..., ge=0.0, description="Second moment of area about horizontal centroidal axis I_x (mm^4)")
    iy_mm4: float = Field(..., ge=0.0, description="Second moment of area about vertical centroidal axis I_y (mm^4)")
    ixy_mm4: float = Field(..., description="Product moment of area I_xy (mm^4)")
    
    # Polar Second Moment of Area
    polar_j_mm4: float = Field(..., ge=0.0, description="Polar second moment of area J_p = I_x + I_y (mm^4)")
    
    # Principal Moments
    i_max_mm4: float = Field(..., ge=0.0, description="Maximum principal second moment of area (mm^4)")
    i_min_mm4: float = Field(..., ge=0.0, description="Minimum principal second moment of area (mm^4)")
    principal_angle_deg: float = Field(..., description="Orientation of principal axis (degrees)")
    
    # Retention relative to intact reference (if provided)
    area_retention_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.5,
        description="Remaining cortical area fraction relative to intact baseline"
    )
    polar_j_retention_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.5,
        description="Remaining polar second moment fraction relative to intact baseline"
    )
    imin_retention_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.5,
        description="Remaining minimum principal bending moment fraction relative to intact baseline"
    )
    
    # Scientific caveats
    disclaimer: str = Field(
        default=(
            "RESEARCH ONLY: 2D geometric moment calculations quantify cross-sectional "
            "shape resistance (mm^4). They do not account for bone mineral density "
            "heterogeneity, 3D cortical geometry, material properties, or patient-specific "
            "loads, and do not constitute CT Rigidity Analysis (CTRA) or clinical fracture probability."
        )
    )


class CrossSectionalGeometryEngine:
    """
    Deterministic numerical moment calculator for 2D bone cross-sections.
    """

    @staticmethod
    def validate_mask(mask: np.ndarray) -> np.ndarray:
        """
        Validates that input mask is a 2D numpy array with binary values.
        """
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"Mask must be a numpy ndarray, got {type(mask)}")
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2-dimensional (H, W), got shape {mask.shape}")
        if mask.size == 0:
            raise ValueError("Mask array is empty.")
        
        # Check binary nature
        unique_vals = np.unique(mask)
        for u in unique_vals:
            if not np.isclose(u, 0.0) and not np.isclose(u, 1.0):
                raise ValueError(
                    f"Mask must contain only binary values {0, 1}. Found value: {u}"
                )
        
        binary_mask = (mask > 0.5).astype(np.float64)
        if np.sum(binary_mask) == 0:
            raise ValueError("Mask contains zero cortical bone pixels. Area is zero.")
            
        return binary_mask

    @classmethod
    def compute_geometry(
        cls,
        mask: np.ndarray,
        pixel_spacing_mm: Union[float, Tuple[float, float]] = 0.5,
        intact_baseline: Optional[CrossSectionalGeometryResult] = None,
    ) -> CrossSectionalGeometryResult:
        """
        Computes 2D geometric moments of area for the provided cortical cross-section mask.

        Args:
            mask: 2D binary array (H, W) where 1 indicates cortical bone and 0 background.
            pixel_spacing_mm: Spacing in mm. Either scalar (dy=dx) or tuple (dy, dx).
            intact_baseline: Optional precomputed intact geometry for computing retention ratios.

        Returns:
            CrossSectionalGeometryResult containing all calculated geometric moments.
        """
        binary_mask = cls.validate_mask(mask)
        h, w = binary_mask.shape

        if isinstance(pixel_spacing_mm, (int, float)):
            dy_mm = float(pixel_spacing_mm)
            dx_mm = float(pixel_spacing_mm)
        elif isinstance(pixel_spacing_mm, (tuple, list)) and len(pixel_spacing_mm) == 2:
            dy_mm = float(pixel_spacing_mm[0])
            dx_mm = float(pixel_spacing_mm[1])
        else:
            raise ValueError(
                f"pixel_spacing_mm must be a positive float or (dy, dx) tuple, got {pixel_spacing_mm}"
            )

        if dy_mm <= 0.0 or dx_mm <= 0.0:
            raise ValueError(f"Pixel spacing values must be strictly positive: dy={dy_mm}, dx={dx_mm}")

        # Pixel area
        dA = dy_mm * dx_mm
        pixel_count = int(np.sum(binary_mask))
        area_mm2 = float(pixel_count * dA)

        # Coordinate grids in mm
        # y is row index (vertical), x is column index (horizontal)
        y_indices, x_indices = np.indices((h, w), dtype=np.float64)
        y_coords_mm = y_indices * dy_mm
        x_coords_mm = x_indices * dx_mm

        # Centroid (1st central moment)
        total_active = np.sum(binary_mask)
        x_bar_mm = float(np.sum(x_coords_mm * binary_mask) / total_active)
        y_bar_mm = float(np.sum(y_coords_mm * binary_mask) / total_active)

        # Centered coordinates
        dx_centered_mm = x_coords_mm - x_bar_mm
        dy_centered_mm = y_coords_mm - y_bar_mm

        # Central second moments of area
        # I_x is moment about horizontal centroidal axis = integral(y^2 dA)
        ix_mm4 = float(np.sum((dy_centered_mm ** 2) * binary_mask) * dA)
        # I_y is moment about vertical centroidal axis = integral(x^2 dA)
        iy_mm4 = float(np.sum((dx_centered_mm ** 2) * binary_mask) * dA)
        # I_xy is product moment of area = integral(x * y dA)
        ixy_mm4 = float(np.sum(dx_centered_mm * dy_centered_mm * binary_mask) * dA)

        # Polar second moment of area: J_p = I_x + I_y
        polar_j_mm4 = ix_mm4 + iy_mm4

        # Principal second moments of area
        # (I_x + I_y)/2 +/- sqrt(((I_x - I_y)/2)^2 + I_xy^2)
        diff_half = (ix_mm4 - iy_mm4) / 2.0
        r = math.sqrt(diff_half ** 2 + ixy_mm4 ** 2)
        avg = (ix_mm4 + iy_mm4) / 2.0

        i_max_mm4 = float(avg + r)
        i_min_mm4 = float(max(0.0, avg - r))

        # Principal angle theta_p in degrees: 0.5 * atan2(-2 * I_xy, I_x - I_y)
        principal_angle_rad = 0.5 * math.atan2(-2.0 * ixy_mm4, ix_mm4 - iy_mm4)
        principal_angle_deg = float(math.degrees(principal_angle_rad))

        # Retention ratios
        area_retention = None
        polar_retention = None
        imin_retention = None

        if intact_baseline is not None:
            if intact_baseline.cortical_area_mm2 > 0:
                area_retention = float(min(1.5, max(0.0, area_mm2 / intact_baseline.cortical_area_mm2)))
            if intact_baseline.polar_j_mm4 > 0:
                polar_retention = float(min(1.5, max(0.0, polar_j_mm4 / intact_baseline.polar_j_mm4)))
            if intact_baseline.i_min_mm4 > 0:
                imin_retention = float(min(1.5, max(0.0, i_min_mm4 / intact_baseline.i_min_mm4)))

        return CrossSectionalGeometryResult(
            pixel_spacing_y_mm=dy_mm,
            pixel_spacing_x_mm=dx_mm,
            cortical_area_mm2=round(area_mm2, 4),
            pixel_count=pixel_count,
            centroid_x_mm=round(x_bar_mm, 4),
            centroid_y_mm=round(y_bar_mm, 4),
            ix_mm4=round(ix_mm4, 4),
            iy_mm4=round(iy_mm4, 4),
            ixy_mm4=round(ixy_mm4, 4),
            polar_j_mm4=round(polar_j_mm4, 4),
            i_max_mm4=round(i_max_mm4, 4),
            i_min_mm4=round(i_min_mm4, 4),
            principal_angle_deg=round(principal_angle_deg, 4),
            area_retention_ratio=round(area_retention, 4) if area_retention is not None else None,
            polar_j_retention_ratio=round(polar_retention, 4) if polar_retention is not None else None,
            imin_retention_ratio=round(imin_retention, 4) if imin_retention is not None else None,
        )

