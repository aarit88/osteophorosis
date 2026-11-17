"""
OsteoGuard AI - Continuous Lesion Geometry & Exploratory Biomechanical Proxy Engine.

Implements the continuous lesion-to-cortex ratio (LCR) calculation and the
exploratory continuous fracture risk proxy formulated during Stage 02:
  R_fracture = [ 1 / (1 + exp(-kappa * (LCR - tau))) ] * M_site * M_morph * 100%

IMPORTANT SCIENTIFIC DISTINCTION & RESEARCH STATUS:
    - THIS IS AN EXPLORATORY COMPUTATIONAL PROXY DEVELOPED IN STAGE 02.
    - IT IS NOT PART OF THE CLASSIC MIRELS (1989) CRITERIA.
    - IT HAS NOT UNDERGONE CLINICAL VALIDATION ON PATIENT OUTCOMES.
    - ALL FORMULAS AND PARAMETERS (kappa=12.5, tau=0.33) ARE MARKED AS:
      STATUS: EXPLORATORY / REQUIRING FACULTY ADVISOR AND CLINICAL REVIEW.
    - For the clinically validated discrete orthopedic criteria, see:
      `osteoguard.biomechanics.mirels.ConventionalMirelsScorer`.
"""

import math
from typing import Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field

from osteoguard.constants import (
    FRACTURE_ALERT_THRESHOLD_PCT,
    FRACTURE_KAPPA_SLOPE,
    FRACTURE_TAU_THRESHOLD,
    MORPHOLOGY_MULTIPLIERS,
    SITE_MULTIPLIERS,
)


def calculate_lesion_to_cortex_ratio(
    d_lesion_mm: float,
    d_bone_mm: float,
) -> float:
    """
    Computes the dimensionless Lesion-to-Cortex Ratio (LCR):
        LCR = D_lesion / D_bone

    Args:
        d_lesion_mm: Maximum transverse diameter of the osteolytic lesion (mm).
        d_bone_mm: Total transverse outer diameter of the cortical bone at lesion level (mm).

    Returns:
        float: Dimensionless LCR ratio clamped to [0.0, 1.0].

    Raises:
        ValueError: If diameters are negative, non-finite, or if d_bone_mm <= 0.
    """
    if not math.isfinite(d_lesion_mm) or not math.isfinite(d_bone_mm):
        raise ValueError(f"Diameters must be finite numbers: d_lesion={d_lesion_mm}, d_bone={d_bone_mm}")
    if d_lesion_mm < 0.0:
        raise ValueError(f"Lesion diameter cannot be negative: {d_lesion_mm}")
    if d_bone_mm <= 0.0:
        raise ValueError(f"Bone diameter must be strictly positive: {d_bone_mm}")

    raw_ratio = d_lesion_mm / d_bone_mm
    return float(min(1.0, max(0.0, raw_ratio)))


class FractureRiskEvaluation(BaseModel):
    """Container for exploratory continuous biomechanical proxy results."""
    lcr: float = Field(..., ge=0.0, le=1.0, description="Lesion-to-Cortex Ratio [0.0 - 1.0]")
    risk_percent: float = Field(..., ge=0.0, le=100.0, description="Exploratory fracture proxy probability [0.0 - 100.0%]")
    risk_level: str = Field(..., description="Stratified risk level: low | medium | high")
    urgent_alert: bool = Field(..., description="Requires immediate orthopedic consultation (>= 80%)")
    site_multiplier: float = Field(..., description="Applied anatomical multiplier M_site")
    morphology_multiplier: float = Field(..., description="Applied lesion morphology multiplier M_morph")
    polar_moment_retention_ratio: float = Field(..., description="Remaining torsional strength J_defect / J_intact")
    clinical_recommendation: str = Field(..., description="Guidance (strictly academic research proxy)")
    exploratory_status: str = Field(
        default="EXPLORATORY_PROXY_REQUIRING_CLINICAL_REVIEW",
        description="Formal designation indicating this equation is uncalibrated on patient outcomes"
    )


class BiomechanicalFractureEngine:
    """
    Computes structural stability degradation and continuous fracture risk
    derived from lesion dimensions and cortical anatomy.
    Marked as EXPLORATORY PROXY.
    """

    def __init__(
        self,
        kappa: float = FRACTURE_KAPPA_SLOPE,
        tau: float = FRACTURE_TAU_THRESHOLD,
        alert_threshold: float = FRACTURE_ALERT_THRESHOLD_PCT,
    ):
        if kappa <= 0.0:
            raise ValueError(f"Kappa slope parameter must be positive, got {kappa}")
        if tau < 0.0 or tau > 1.0:
            raise ValueError(f"Tau threshold must be in [0, 1], got {tau}")
        self.kappa = kappa
        self.tau = tau
        self.alert_threshold = alert_threshold

    def calculate_polar_moment_retention(
        self,
        lcr: float,
        r_outer_mm: float = 15.0,
        cortical_thickness_mm: float = 4.0,
    ) -> float:
        """
        Calculates polar moment of inertia retention ratio (J_defect / J_intact).
        For an intact cylindrical bone:
          J_intact = (pi / 2) * (r_outer^4 - r_inner^4)
        """
        if lcr < 0.0:
            raise ValueError(f"LCR cannot be negative: {lcr}")
        if r_outer_mm <= 0.0:
            raise ValueError(f"Outer radius must be positive: {r_outer_mm}")
        if cortical_thickness_mm <= 0.0 or cortical_thickness_mm >= r_outer_mm:
            raise ValueError(
                f"Cortical thickness must be positive and less than outer radius: {cortical_thickness_mm}"
            )

        clamped_lcr = min(1.0, max(0.0, lcr))
        r_inner_mm = max(1.0, r_outer_mm - cortical_thickness_mm)
        j_intact = (math.pi / 2.0) * (r_outer_mm ** 4 - r_inner_mm ** 4)

        # Defect arc angle theta_defect proportional to cortical lesion involvement
        theta_defect = min(2.0 * math.pi, max(0.0, clamped_lcr * math.pi))
        j_defect = j_intact * (1.0 - (theta_defect / (2.0 * math.pi))) - (2.0 / 3.0) * r_outer_mm * (cortical_thickness_mm ** 3)
        j_defect = max(0.0, j_defect)

        retention = j_defect / (j_intact + 1e-8)
        return float(min(1.0, max(0.0, retention)))

    def evaluate_risk(
        self,
        lcr: float,
        anatomical_site: str = "default",
        morphology: str = "default",
    ) -> FractureRiskEvaluation:
        """
        Evaluates the exploratory continuous fracture risk proxy equation:
          R_fracture = [1 / (1 + exp(-kappa * (LCR - tau)))] * M_site * M_morph * 100%
        """
        if not math.isfinite(lcr):
            raise ValueError(f"LCR must be a finite number, got {lcr}")
        if lcr < 0.0:
            raise ValueError(f"LCR cannot be negative, got {lcr}")

        clamped_lcr = float(min(1.0, max(0.0, lcr)))

        m_site = SITE_MULTIPLIERS.get(anatomical_site.lower(), SITE_MULTIPLIERS["default"])
        m_morph = MORPHOLOGY_MULTIPLIERS.get(morphology.lower(), MORPHOLOGY_MULTIPLIERS["default"])

        # Calibrated Sigmoid core
        exponent = -self.kappa * (clamped_lcr - self.tau)
        # Prevent numerical overflow
        clamped_exp = min(50.0, max(-50.0, exponent))
        base_sigmoid = 1.0 / (1.0 + math.exp(clamped_exp))

        raw_percent = base_sigmoid * m_site * m_morph * 100.0
        risk_percent = float(min(100.0, max(0.0, round(raw_percent, 2))))

        urgent_alert = risk_percent >= self.alert_threshold

        if risk_percent >= self.alert_threshold:
            risk_level = "high"
            recommendation = (
                "CRITICAL: Impending pathological fracture hazard (>=80%). "
                "Urgent orthopedic oncology consultation recommended for prophylactic fixation. "
                "(EXPLORATORY PROXY: Requires clinical review and correlation with discrete Mirels scoring)."
            )
        elif risk_percent >= 30.0:
            risk_level = "medium"
            recommendation = (
                "EXPLORATORY PROXY (MODERATE): Cortical thinning proxy in intermediate zone. "
                "Requires correlation with classical Mirels assessment and axial CT imaging."
            )
        else:
            risk_level = "low"
            recommendation = (
                "EXPLORATORY PROXY (LOW): Lesion involvement minimal on proxy equation. "
                "Standard surveillance protocol."
            )

        polar_retention = self.calculate_polar_moment_retention(clamped_lcr)

        return FractureRiskEvaluation(
            lcr=clamped_lcr,
            risk_percent=risk_percent,
            risk_level=risk_level,
            urgent_alert=urgent_alert,
            site_multiplier=m_site,
            morphology_multiplier=m_morph,
            polar_moment_retention_ratio=round(polar_retention, 4),
            clinical_recommendation=recommendation,
        )
