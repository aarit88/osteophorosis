"""
OsteoGuard AI - Deterministic Synthetic Bone Cross-Section Phantom Suite.

Generates 8 reproducible digital bone cross-section phantoms for mathematical
and biomechanical geometry validation. Zero patient data is used.

Phantoms:
    1. Intact circular cortical ring (Analytical benchmark)
    2. Uniform cortical thinning (Concentric endosteal canal expansion)
    3. Symmetric cortical defect (Angular wedge defect)
    4. Eccentric cortical defect (Unilateral cortical saucerization)
    5. Lytic cavity (Intracortical circular defect)
    6. Multiple small cortical defects (Permeative / moth-eaten cortex)
    7. Non-circular elliptical cross-section (Anisotropic geometry: I_x != I_y)
    8. Anisotropic pixel-spacing phantom (Non-square voxel sampling validation)
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class BonePhantom:
    """Container for synthetic bone cross-section phantom."""
    name: str
    mask: np.ndarray
    spacing_y_mm: float
    spacing_x_mm: float
    description: str
    parameters: Dict[str, Any]
    analytical_area_mm2: Optional[float] = None
    analytical_ix_mm4: Optional[float] = None
    analytical_iy_mm4: Optional[float] = None
    analytical_polar_j_mm4: Optional[float] = None


class SyntheticBonePhantomSuite:
    """
    Factory generating 8 deterministic synthetic phantoms for cross-sectional biomechanics.
    """

    @staticmethod
    def create_intact_annulus(
        outer_radius_mm: float = 15.0,
        inner_radius_mm: float = 10.0,
        grid_size: int = 100,
        spacing_mm: float = 0.5,
    ) -> BonePhantom:
        """
        1. Intact circular cortical ring.
        Analytical Area: pi * (R^2 - r^2)
        Analytical I_x = I_y = (pi / 4) * (R^4 - r^4)
        Analytical J_p = (pi / 2) * (R^4 - r^4)
        """
        h = w = grid_size
        cy_mm = (h - 1) / 2.0 * spacing_mm
        cx_mm = (w - 1) / 2.0 * spacing_mm

        y_idx, x_idx = np.indices((h, w), dtype=np.float64)
        y_mm = y_idx * spacing_mm
        x_mm = x_idx * spacing_mm

        r_dist = np.sqrt((x_mm - cx_mm) ** 2 + (y_mm - cy_mm) ** 2)
        mask = ((r_dist >= inner_radius_mm) & (r_dist <= outer_radius_mm)).astype(np.float64)

        # Analytical values
        a_anal = math.pi * (outer_radius_mm ** 2 - inner_radius_mm ** 2)
        i_anal = (math.pi / 4.0) * (outer_radius_mm ** 4 - inner_radius_mm ** 4)
        j_anal = (math.pi / 2.0) * (outer_radius_mm ** 4 - inner_radius_mm ** 4)

        return BonePhantom(
            name="intact_circular_ring",
            mask=mask,
            spacing_y_mm=spacing_mm,
            spacing_x_mm=spacing_mm,
            description="Intact circular cortical annulus representing normal diaphysis",
            parameters={
                "outer_radius_mm": outer_radius_mm,
                "inner_radius_mm": inner_radius_mm,
                "grid_size": grid_size,
                "spacing_mm": spacing_mm,
            },
            analytical_area_mm2=round(a_anal, 4),
            analytical_ix_mm4=round(i_anal, 4),
            analytical_iy_mm4=round(i_anal, 4),
            analytical_polar_j_mm4=round(j_anal, 4),
        )

    @staticmethod
    def create_uniform_thinning(
        outer_radius_mm: float = 15.0,
        expanded_inner_radius_mm: float = 13.0,
        grid_size: int = 100,
        spacing_mm: float = 0.5,
    ) -> BonePhantom:
        """
        2. Uniform cortical thinning.
        Cortical thickness reduced from 5.0mm to 2.0mm.
        """
        h = w = grid_size
        cy_mm = (h - 1) / 2.0 * spacing_mm
        cx_mm = (w - 1) / 2.0 * spacing_mm

        y_idx, x_idx = np.indices((h, w), dtype=np.float64)
        y_mm = y_idx * spacing_mm
        x_mm = x_idx * spacing_mm

        r_dist = np.sqrt((x_mm - cx_mm) ** 2 + (y_mm - cy_mm) ** 2)
        mask = ((r_dist >= expanded_inner_radius_mm) & (r_dist <= outer_radius_mm)).astype(np.float64)

        a_anal = math.pi * (outer_radius_mm ** 2 - expanded_inner_radius_mm ** 2)
        i_anal = (math.pi / 4.0) * (outer_radius_mm ** 4 - expanded_inner_radius_mm ** 4)
        j_anal = (math.pi / 2.0) * (outer_radius_mm ** 4 - expanded_inner_radius_mm ** 4)

        return BonePhantom(
            name="uniform_cortical_thinning",
            mask=mask,
            spacing_y_mm=spacing_mm,
            spacing_x_mm=spacing_mm,
            description="Uniform endosteal thinning simulating metabolic osteopenia or diffuse erosion",
            parameters={
                "outer_radius_mm": outer_radius_mm,
                "expanded_inner_radius_mm": expanded_inner_radius_mm,
                "grid_size": grid_size,
                "spacing_mm": spacing_mm,
            },
            analytical_area_mm2=round(a_anal, 4),
            analytical_ix_mm4=round(i_anal, 4),
            analytical_iy_mm4=round(i_anal, 4),
            analytical_polar_j_mm4=round(j_anal, 4),
        )

    @staticmethod
    def create_symmetric_defect(
        outer_radius_mm: float = 15.0,
        inner_radius_mm: float = 10.0,
        wedge_angle_deg: float = 60.0,
        grid_size: int = 100,
        spacing_mm: float = 0.5,
    ) -> BonePhantom:
        """
        3. Symmetric cortical defect.
        Intact annulus with angular sector defect removed symmetrically across x-axis.
        """
        h = w = grid_size
        cy_mm = (h - 1) / 2.0 * spacing_mm
        cx_mm = (w - 1) / 2.0 * spacing_mm

        y_idx, x_idx = np.indices((h, w), dtype=np.float64)
        y_mm = y_idx * spacing_mm
        x_mm = x_idx * spacing_mm

        dx = x_mm - cx_mm
        dy = y_mm - cy_mm
        r_dist = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)  # [-pi, pi]

        half_wedge_rad = math.radians(wedge_angle_deg / 2.0)
        in_annulus = (r_dist >= inner_radius_mm) & (r_dist <= outer_radius_mm)
        in_wedge = (np.abs(angle) <= half_wedge_rad)

        mask = (in_annulus & (~in_wedge)).astype(np.float64)

        # Theoretical area: intact area * (1 - wedge_angle / 360)
        a_intact = math.pi * (outer_radius_mm ** 2 - inner_radius_mm ** 2)
        a_anal = a_intact * (1.0 - (wedge_angle_deg / 360.0))

        return BonePhantom(
            name="symmetric_cortical_defect",
            mask=mask,
            spacing_y_mm=spacing_mm,
            spacing_x_mm=spacing_mm,
            description="Symmetric angular notch defect simulating focal cortical erosion",
            parameters={
                "outer_radius_mm": outer_radius_mm,
                "inner_radius_mm": inner_radius_mm,
                "wedge_angle_deg": wedge_angle_deg,
                "grid_size": grid_size,
                "spacing_mm": spacing_mm,
            },
            analytical_area_mm2=round(a_anal, 4),
            analytical_ix_mm4=None,
            analytical_iy_mm4=None,
            analytical_polar_j_mm4=None,
        )

    @staticmethod
    def create_eccentric_defect(
        outer_radius_mm: float = 15.0,
        inner_radius_mm: float = 10.0,
        defect_center_x_mm: float = 12.5,
        defect_center_y_mm: float = 0.0,
        defect_radius_mm: float = 5.0,
        grid_size: int = 100,
        spacing_mm: float = 0.5,
    ) -> BonePhantom:
        """
        4. Eccentric cortical defect.
        Unilateral cortical saucerization/scalloping defect inducing centroid shift.
        """
        h = w = grid_size
        cy_mm = (h - 1) / 2.0 * spacing_mm
        cx_mm = (w - 1) / 2.0 * spacing_mm

        y_idx, x_idx = np.indices((h, w), dtype=np.float64)
        y_mm = y_idx * spacing_mm
        x_mm = x_idx * spacing_mm

        dx = x_mm - cx_mm
        dy = y_mm - cy_mm
        r_dist = np.sqrt(dx ** 2 + dy ** 2)

        in_annulus = (r_dist >= inner_radius_mm) & (r_dist <= outer_radius_mm)

        # Defect centered at (cx_mm + defect_center_x_mm, cy_mm + defect_center_y_mm)
        def_dx = x_mm - (cx_mm + defect_center_x_mm)
        def_dy = y_mm - (cy_mm + defect_center_y_mm)
        in_defect = (np.sqrt(def_dx ** 2 + def_dy ** 2) <= defect_radius_mm)

        mask = (in_annulus & (~in_defect)).astype(np.float64)

        return BonePhantom(
            name="eccentric_cortical_defect",
            mask=mask,
            spacing_y_mm=spacing_mm,
            spacing_x_mm=spacing_mm,
            description="Unilateral cortical defect inducing centroid translation and asymmetry",
            parameters={
                "outer_radius_mm": outer_radius_mm,
                "inner_radius_mm": inner_radius_mm,
                "defect_center_x_mm": defect_center_x_mm,
                "defect_center_y_mm": defect_center_y_mm,
                "defect_radius_mm": defect_radius_mm,
                "grid_size": grid_size,
                "spacing_mm": spacing_mm,
            },
            analytical_area_mm2=None,
            analytical_ix_mm4=None,
            analytical_iy_mm4=None,
            analytical_polar_j_mm4=None,
        )

    @staticmethod
    def create_lytic_cavity(
        outer_radius_mm: float = 15.0,
        inner_radius_mm: float = 10.0,
        cavity_center_x_mm: float = 12.0,
        cavity_center_y_mm: float = 0.0,
        cavity_radius_mm: float = 2.2,
        grid_size: int = 100,
        spacing_mm: float = 0.5,
    ) -> BonePhantom:
        """
        5. Lytic cavity.
        Subcortical circular osteolytic defect within cortex.
        """
        h = w = grid_size
        cy_mm = (h - 1) / 2.0 * spacing_mm
        cx_mm = (w - 1) / 2.0 * spacing_mm

        y_idx, x_idx = np.indices((h, w), dtype=np.float64)
        y_mm = y_idx * spacing_mm
        x_mm = x_idx * spacing_mm

        dx = x_mm - cx_mm
        dy = y_mm - cy_mm
        r_dist = np.sqrt(dx ** 2 + dy ** 2)

        in_annulus = (r_dist >= inner_radius_mm) & (r_dist <= outer_radius_mm)

        cav_dx = x_mm - (cx_mm + cavity_center_x_mm)
        cav_dy = y_mm - (cy_mm + cavity_center_y_mm)
        in_cavity = (np.sqrt(cav_dx ** 2 + cav_dy ** 2) <= cavity_radius_mm)

        mask = (in_annulus & (~in_cavity)).astype(np.float64)

        # Cavity area is strictly subtracted from intact annulus if cavity fits fully inside cortex
        a_intact = math.pi * (outer_radius_mm ** 2 - inner_radius_mm ** 2)
        a_cavity = math.pi * (cavity_radius_mm ** 2)
        a_anal = a_intact - a_cavity

        return BonePhantom(
            name="lytic_cavity",
            mask=mask,
            spacing_y_mm=spacing_mm,
            spacing_x_mm=spacing_mm,
            description="Intracortical circular osteolytic lesion simulating localized bone metastasis",
            parameters={
                "outer_radius_mm": outer_radius_mm,
                "inner_radius_mm": inner_radius_mm,
                "cavity_center_x_mm": cavity_center_x_mm,
                "cavity_center_y_mm": cavity_center_y_mm,
                "cavity_radius_mm": cavity_radius_mm,
                "grid_size": grid_size,
                "spacing_mm": spacing_mm,
            },
            analytical_area_mm2=round(a_anal, 4),
            analytical_ix_mm4=None,
            analytical_iy_mm4=None,
            analytical_polar_j_mm4=None,
        )

    @staticmethod
    def create_multiple_small_defects(
        outer_radius_mm: float = 15.0,
        inner_radius_mm: float = 10.0,
        num_defects: int = 8,
        defect_radius_mm: float = 1.0,
        grid_size: int = 100,
        spacing_mm: float = 0.5,
    ) -> BonePhantom:
        """
        6. Multiple small cortical defects.
        Permeative / moth-eaten osteolytic defects distributed symmetrically around mid-cortex.
        """
        h = w = grid_size
        cy_mm = (h - 1) / 2.0 * spacing_mm
        cx_mm = (w - 1) / 2.0 * spacing_mm

        y_idx, x_idx = np.indices((h, w), dtype=np.float64)
        y_mm = y_idx * spacing_mm
        x_mm = x_idx * spacing_mm

        dx = x_mm - cx_mm
        dy = y_mm - cy_mm
        r_dist = np.sqrt(dx ** 2 + dy ** 2)

        in_annulus = (r_dist >= inner_radius_mm) & (r_dist <= outer_radius_mm)

        mid_radius_mm = (outer_radius_mm + inner_radius_mm) / 2.0
        in_any_defect = np.zeros_like(in_annulus, dtype=bool)

        for i in range(num_defects):
            theta = 2.0 * math.pi * i / num_defects
            px = cx_mm + mid_radius_mm * math.cos(theta)
            py = cy_mm + mid_radius_mm * math.sin(theta)
            d = np.sqrt((x_mm - px) ** 2 + (y_mm - py) ** 2)
            in_any_defect |= (d <= defect_radius_mm)

        mask = (in_annulus & (~in_any_defect)).astype(np.float64)

        a_intact = math.pi * (outer_radius_mm ** 2 - inner_radius_mm ** 2)
        a_defects = num_defects * math.pi * (defect_radius_mm ** 2)
        a_anal = a_intact - a_defects

        return BonePhantom(
            name="multiple_small_defects",
            mask=mask,
            spacing_y_mm=spacing_mm,
            spacing_x_mm=spacing_mm,
            description="Multiple punctate defects simulating permeative/moth-eaten osteolysis",
            parameters={
                "outer_radius_mm": outer_radius_mm,
                "inner_radius_mm": inner_radius_mm,
                "num_defects": num_defects,
                "defect_radius_mm": defect_radius_mm,
                "grid_size": grid_size,
                "spacing_mm": spacing_mm,
            },
            analytical_area_mm2=round(a_anal, 4),
            analytical_ix_mm4=None,
            analytical_iy_mm4=None,
            analytical_polar_j_mm4=None,
        )

    @staticmethod
    def create_elliptical_annulus(
        outer_semi_a_mm: float = 18.0,
        outer_semi_b_mm: float = 12.0,
        thickness_mm: float = 3.5,
        grid_size: int = 120,
        spacing_mm: float = 0.5,
    ) -> BonePhantom:
        """
        7. Non-circular elliptical cross-section.
        Demonstrates I_x != I_y and principal axes alignment.
        Semi-axes:
            Outer: A_o = 18.0mm (horizontal), B_o = 12.0mm (vertical)
            Inner: A_i = 14.5mm, B_i = 8.5mm
        Analytical Area: pi * (A_o * B_o - A_i * B_i)
        Analytical I_x (bending about horizontal axis, integral y^2):
            (pi / 4) * (A_o * B_o^3 - A_i * B_i^3)
        Analytical I_y (bending about vertical axis, integral x^2):
            (pi / 4) * (A_o^3 * B_o - A_i^3 * B_i)
        """
        inner_semi_a_mm = outer_semi_a_mm - thickness_mm
        inner_semi_b_mm = outer_semi_b_mm - thickness_mm

        h = w = grid_size
        cy_mm = (h - 1) / 2.0 * spacing_mm
        cx_mm = (w - 1) / 2.0 * spacing_mm

        y_idx, x_idx = np.indices((h, w), dtype=np.float64)
        y_mm = y_idx * spacing_mm
        x_mm = x_idx * spacing_mm

        dx = x_mm - cx_mm
        dy = y_mm - cy_mm

        outer_ell = (dx / outer_semi_a_mm) ** 2 + (dy / outer_semi_b_mm) ** 2
        inner_ell = (dx / inner_semi_a_mm) ** 2 + (dy / inner_semi_b_mm) ** 2

        mask = ((outer_ell <= 1.0) & (inner_ell >= 1.0)).astype(np.float64)

        a_anal = math.pi * (outer_semi_a_mm * outer_semi_b_mm - inner_semi_a_mm * inner_semi_b_mm)
        ix_anal = (math.pi / 4.0) * (outer_semi_a_mm * (outer_semi_b_mm ** 3) - inner_semi_a_mm * (inner_semi_b_mm ** 3))
        iy_anal = (math.pi / 4.0) * ((outer_semi_a_mm ** 3) * outer_semi_b_mm - (inner_semi_a_mm ** 3) * inner_semi_b_mm)
        j_anal = ix_anal + iy_anal

        return BonePhantom(
            name="non_circular_elliptical_annulus",
            mask=mask,
            spacing_y_mm=spacing_mm,
            spacing_x_mm=spacing_mm,
            description="Elliptical diaphysis demonstrating anisotropic principal moments (I_x != I_y)",
            parameters={
                "outer_semi_a_mm": outer_semi_a_mm,
                "outer_semi_b_mm": outer_semi_b_mm,
                "thickness_mm": thickness_mm,
                "grid_size": grid_size,
                "spacing_mm": spacing_mm,
            },
            analytical_area_mm2=round(a_anal, 4),
            analytical_ix_mm4=round(ix_anal, 4),
            analytical_iy_mm4=round(iy_anal, 4),
            analytical_polar_j_mm4=round(j_anal, 4),
        )

    @staticmethod
    def create_anisotropic_spacing_annulus(
        outer_radius_mm: float = 15.0,
        inner_radius_mm: float = 10.0,
        grid_size_y: int = 120,
        grid_size_x: int = 60,
        spacing_y_mm: float = 0.4,
        spacing_x_mm: float = 0.8,
    ) -> BonePhantom:
        """
        8. Anisotropic pixel-spacing phantom.
        Validates that physical moment calculations remain isotropic when voxel grid is rectangular.
        """
        cy_mm = (grid_size_y - 1) / 2.0 * spacing_y_mm
        cx_mm = (grid_size_x - 1) / 2.0 * spacing_x_mm

        y_idx, x_idx = np.indices((grid_size_y, grid_size_x), dtype=np.float64)
        y_mm = y_idx * spacing_y_mm
        x_mm = x_idx * spacing_x_mm

        r_dist = np.sqrt((x_mm - cx_mm) ** 2 + (y_mm - cy_mm) ** 2)
        mask = ((r_dist >= inner_radius_mm) & (r_dist <= outer_radius_mm)).astype(np.float64)

        a_anal = math.pi * (outer_radius_mm ** 2 - inner_radius_mm ** 2)
        i_anal = (math.pi / 4.0) * (outer_radius_mm ** 4 - inner_radius_mm ** 4)
        j_anal = (math.pi / 2.0) * (outer_radius_mm ** 4 - inner_radius_mm ** 4)

        return BonePhantom(
            name="anisotropic_spacing_annulus",
            mask=mask,
            spacing_y_mm=spacing_y_mm,
            spacing_x_mm=spacing_x_mm,
            description="Circular annulus sampled on non-square anisotropic grid (dy=0.4mm, dx=0.8mm)",
            parameters={
                "outer_radius_mm": outer_radius_mm,
                "inner_radius_mm": inner_radius_mm,
                "grid_size_y": grid_size_y,
                "grid_size_x": grid_size_x,
                "spacing_y_mm": spacing_y_mm,
                "spacing_x_mm": spacing_x_mm,
            },
            analytical_area_mm2=round(a_anal, 4),
            analytical_ix_mm4=round(i_anal, 4),
            analytical_iy_mm4=round(i_anal, 4),
            analytical_polar_j_mm4=round(j_anal, 4),
        )

    @classmethod
    def get_all_phantoms(cls) -> Dict[str, BonePhantom]:
        """Returns all 8 synthetic bone cross-section phantoms."""
        return {
            "intact_circular_ring": cls.create_intact_annulus(),
            "uniform_cortical_thinning": cls.create_uniform_thinning(),
            "symmetric_cortical_defect": cls.create_symmetric_defect(),
            "eccentric_cortical_defect": cls.create_eccentric_defect(),
            "lytic_cavity": cls.create_lytic_cavity(),
            "multiple_small_defects": cls.create_multiple_small_defects(),
            "non_circular_elliptical_annulus": cls.create_elliptical_annulus(),
            "anisotropic_spacing_annulus": cls.create_anisotropic_spacing_annulus(),
        }

