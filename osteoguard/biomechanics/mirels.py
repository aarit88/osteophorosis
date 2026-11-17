"""
OsteoGuard AI - Conventional Mirels Scoring Engine.

Implements the classic 1989 Mirels scoring system for evaluating impending
pathological fracture risk in long-bone metastatic lesions.

Reference:
    Mirels, H. (1989). "Metastatic disease in long bones. A proposed scoring system
    for diagnosing impending pathologic fractures." Clinical Orthopaedics and
    Related Research, (249), 256-264.

IMPORTANT RESEARCH BOUNDARY & CLINICAL DISCLAIMER:
    This software component implements the classical orthopedic scoring logic
    for academic research and engineering validation only. It is NOT an approved
    medical device, diagnostic instrument, or clinical decision tool.
    Mirels scoring was established and validated solely for metastatic carcinoma
    involving the long bones. It is NOT valid for metabolic bone diseases
    (e.g., Paget's disease of bone), genetic collagen disorders (e.g., Osteogenesis
    Imperfecta), primary bone tumors, or non-metastatic skeletal lesions.
"""

import math
from enum import IntEnum, Enum
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class MirelsSite(IntEnum):
    """Anatomical site score."""
    UPPER_EXTREMITY = 1
    LOWER_EXTREMITY = 2
    PERITROCHANTERIC = 3


class MirelsPain(IntEnum):
    """Pain intensity score."""
    MILD = 1
    MODERATE = 2
    FUNCTIONAL = 3


class MirelsRadiographicNature(IntEnum):
    """Radiographic lesion appearance score."""
    BLASTIC = 1
    MIXED = 2
    LYTIC = 3


class MirelsLesionSize(IntEnum):
    """Lesion size relative to bone diameter score."""
    LESS_THAN_ONE_THIRD = 1
    ONE_THIRD_TO_TWO_THIRDS = 2
    GREATER_THAN_TWO_THIRDS = 3


class MirelsRecommendationTier(str, Enum):
    LOW_RISK = "LOW_RISK"
    INTERMEDIATE_RISK = "INTERMEDIATE_RISK"
    HIGH_RISK = "HIGH_RISK"


# Canonical string mapping for robust string-based input handling
SITE_SYNONYMS: Dict[str, MirelsSite] = {
    "upper": MirelsSite.UPPER_EXTREMITY,
    "upper_extremity": MirelsSite.UPPER_EXTREMITY,
    "upper extremity": MirelsSite.UPPER_EXTREMITY,
    "arm": MirelsSite.UPPER_EXTREMITY,
    "humerus": MirelsSite.UPPER_EXTREMITY,
    "radius": MirelsSite.UPPER_EXTREMITY,
    "ulna": MirelsSite.UPPER_EXTREMITY,
    "1": MirelsSite.UPPER_EXTREMITY,
    
    "lower": MirelsSite.LOWER_EXTREMITY,
    "lower_extremity": MirelsSite.LOWER_EXTREMITY,
    "lower extremity": MirelsSite.LOWER_EXTREMITY,
    "leg": MirelsSite.LOWER_EXTREMITY,
    "femur_shaft": MirelsSite.LOWER_EXTREMITY,
    "femur diaphysis": MirelsSite.LOWER_EXTREMITY,
    "femur_diaphysis": MirelsSite.LOWER_EXTREMITY,
    "tibia": MirelsSite.LOWER_EXTREMITY,
    "fibula": MirelsSite.LOWER_EXTREMITY,
    "2": MirelsSite.LOWER_EXTREMITY,
    
    "peritrochanteric": MirelsSite.PERITROCHANTERIC,
    "peri-trochanteric": MirelsSite.PERITROCHANTERIC,
    "femur_peritrochanteric": MirelsSite.PERITROCHANTERIC,
    "femur_neck": MirelsSite.PERITROCHANTERIC,
    "femoral neck": MirelsSite.PERITROCHANTERIC,
    "trochanter": MirelsSite.PERITROCHANTERIC,
    "subtrochanteric": MirelsSite.PERITROCHANTERIC,
    "3": MirelsSite.PERITROCHANTERIC,
}

PAIN_SYNONYMS: Dict[str, MirelsPain] = {
    "mild": MirelsPain.MILD,
    "mild_pain": MirelsPain.MILD,
    "1": MirelsPain.MILD,
    
    "moderate": MirelsPain.MODERATE,
    "moderate_pain": MirelsPain.MODERATE,
    "2": MirelsPain.MODERATE,
    
    "functional": MirelsPain.FUNCTIONAL,
    "functional_pain": MirelsPain.FUNCTIONAL,
    "severe": MirelsPain.FUNCTIONAL,
    "weight_bearing": MirelsPain.FUNCTIONAL,
    "3": MirelsPain.FUNCTIONAL,
}

NATURE_SYNONYMS: Dict[str, MirelsRadiographicNature] = {
    "blastic": MirelsRadiographicNature.BLASTIC,
    "osteoblastic": MirelsRadiographicNature.BLASTIC,
    "sclerotic": MirelsRadiographicNature.BLASTIC,
    "1": MirelsRadiographicNature.BLASTIC,
    
    "mixed": MirelsRadiographicNature.MIXED,
    "mixed_lytic_blastic": MirelsRadiographicNature.MIXED,
    "2": MirelsRadiographicNature.MIXED,
    
    "lytic": MirelsRadiographicNature.LYTIC,
    "osteolytic": MirelsRadiographicNature.LYTIC,
    "3": MirelsRadiographicNature.LYTIC,
}

SIZE_SYNONYMS: Dict[str, MirelsLesionSize] = {
    "<1/3": MirelsLesionSize.LESS_THAN_ONE_THIRD,
    "< 1/3": MirelsLesionSize.LESS_THAN_ONE_THIRD,
    "less_than_one_third": MirelsLesionSize.LESS_THAN_ONE_THIRD,
    "small": MirelsLesionSize.LESS_THAN_ONE_THIRD,
    "1": MirelsLesionSize.LESS_THAN_ONE_THIRD,
    
    "1/3-2/3": MirelsLesionSize.ONE_THIRD_TO_TWO_THIRDS,
    "1/3 to 2/3": MirelsLesionSize.ONE_THIRD_TO_TWO_THIRDS,
    "one_third_to_two_thirds": MirelsLesionSize.ONE_THIRD_TO_TWO_THIRDS,
    "medium": MirelsLesionSize.ONE_THIRD_TO_TWO_THIRDS,
    "2": MirelsLesionSize.ONE_THIRD_TO_TWO_THIRDS,
    
    ">2/3": MirelsLesionSize.GREATER_THAN_TWO_THIRDS,
    "> 2/3": MirelsLesionSize.GREATER_THAN_TWO_THIRDS,
    "greater_than_two_thirds": MirelsLesionSize.GREATER_THAN_TWO_THIRDS,
    "large": MirelsLesionSize.GREATER_THAN_TWO_THIRDS,
    "3": MirelsLesionSize.GREATER_THAN_TWO_THIRDS,
}


def parse_site(val: Union[str, int, MirelsSite]) -> MirelsSite:
    if isinstance(val, MirelsSite):
        return val
    if isinstance(val, int) and val in (1, 2, 3):
        return MirelsSite(val)
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized in SITE_SYNONYMS:
            return SITE_SYNONYMS[normalized]
    raise ValueError(
        f"Invalid Mirels site input: {val!r}. Allowed categories: 'upper_extremity' (1), "
        f"'lower_extremity' (2), 'peritrochanteric' (3)."
    )


def parse_pain(val: Union[str, int, MirelsPain]) -> MirelsPain:
    if isinstance(val, MirelsPain):
        return val
    if isinstance(val, int) and val in (1, 2, 3):
        return MirelsPain(val)
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized in PAIN_SYNONYMS:
            return PAIN_SYNONYMS[normalized]
    raise ValueError(
        f"Invalid Mirels pain input: {val!r}. Allowed categories: 'mild' (1), "
        f"'moderate' (2), 'functional' (3)."
    )


def parse_nature(val: Union[str, int, MirelsRadiographicNature]) -> MirelsRadiographicNature:
    if isinstance(val, MirelsRadiographicNature):
        return val
    if isinstance(val, int) and val in (1, 2, 3):
        return MirelsRadiographicNature(val)
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized in NATURE_SYNONYMS:
            return NATURE_SYNONYMS[normalized]
    raise ValueError(
        f"Invalid Mirels radiographic nature input: {val!r}. Allowed categories: 'blastic' (1), "
        f"'mixed' (2), 'lytic' (3)."
    )


def parse_size(val: Union[str, int, MirelsLesionSize, float]) -> MirelsLesionSize:
    if isinstance(val, MirelsLesionSize):
        return val
    # Exact integer category [1, 2, 3]
    if isinstance(val, int) and not isinstance(val, bool):
        if val in (1, 2, 3):
            return MirelsLesionSize(val)
        raise ValueError(
            f"Invalid Mirels lesion size integer: {val}. Allowed score integers: 1, 2, 3."
        )
    # Continuous fraction [0.0, 1.0]
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError(f"Lesion size fraction must be a finite number: {val}")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"Lesion size ratio must be between 0.0 and 1.0, got: {val}")
        if val < 1.0 / 3.0:
            return MirelsLesionSize.LESS_THAN_ONE_THIRD
        elif val <= 2.0 / 3.0:
            return MirelsLesionSize.ONE_THIRD_TO_TWO_THIRDS
        else:
            return MirelsLesionSize.GREATER_THAN_TWO_THIRDS
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized in SIZE_SYNONYMS:
            return SIZE_SYNONYMS[normalized]
    raise ValueError(
        f"Invalid Mirels lesion size input: {val!r}. Allowed categories: '<1/3' (1), "
        f"'1/3-2/3' (2), '>2/3' (3), or continuous ratio float in [0.0, 1.0]."
    )


class MirelsScoreResult(BaseModel):
    """
    Structured outcome of the conventional Mirels scoring assessment.
    """
    site_score: int = Field(..., ge=1, le=3, description="Anatomical site score [1, 2, 3]")
    pain_score: int = Field(..., ge=1, le=3, description="Pain score [1, 2, 3]")
    nature_score: int = Field(..., ge=1, le=3, description="Radiographic nature score [1, 2, 3]")
    size_score: int = Field(..., ge=1, le=3, description="Lesion size score [1, 2, 3]")
    total_score: int = Field(..., ge=4, le=12, description="Total Mirels score [4 - 12]")
    risk_tier: MirelsRecommendationTier = Field(..., description="Risk tier based on classic criteria")
    fracture_risk_estimate_pct: float = Field(..., description="Historical cohort fracture rate from 1989 series")
    prophylactic_fixation_indicated: bool = Field(..., description="True if total score >= 9")
    clinical_interpretation: str = Field(..., description="Orthopedic literature guidance")
    target_pathology: str = Field(
        default="Metastatic carcinoma of long bones",
        description="Pathological indication under which Mirels is clinically validated"
    )
    disclaimer: str = Field(
        default=(
            "RESEARCH ONLY: Mirels scoring was developed specifically for metastatic "
            "lesions of long bones. Not valid for Paget's disease, Osteogenesis Imperfecta, "
            "or non-metastatic bone lesions. This tool does not provide medical diagnoses."
        )
    )


class ConventionalMirelsScorer:
    """
    Evaluator for the conventional 4-factor Mirels scoring protocol (1989).
    """

    # Classic 1989 reported fracture incidence per total score:
    # Score <= 7: ~4% or lower
    # Score 8: ~15%
    # Score 9: ~33%
    # Score 10+: ~50 - 100%
    HISTORICAL_RISK_MAP = {
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 4.0,
        8: 15.0,
        9: 33.3,
        10: 50.0,
        11: 66.7,
        12: 100.0,
    }

    @classmethod
    def calculate_score(
        cls,
        site: Union[str, int, MirelsSite],
        pain: Union[str, int, MirelsPain],
        nature: Union[str, int, MirelsRadiographicNature],
        size: Union[str, int, MirelsLesionSize, float],
        disease_context: str = "metastasis",
    ) -> MirelsScoreResult:
        """
        Calculates conventional Mirels score.

        Args:
            site: Anatomical location (1=Upper, 2=Lower, 3=Peritrochanteric)
            pain: Pain severity (1=Mild, 2=Moderate, 3=Functional)
            nature: Radiographic type (1=Blastic, 2=Mixed, 3=Lytic)
            size: Size relative to cortex (1=<1/3, 2=1/3-2/3, 3=>2/3)
            disease_context: Clinical context (default 'metastasis')

        Returns:
            MirelsScoreResult with scores, risk tier, and interpretation.

        Raises:
            ValueError: If any input factor is unrecognized or invalid.
        """
        norm_context = disease_context.strip().lower()
        if norm_context in ("paget", "oi", "osteogenesis_imperfecta", "healthy", "normal"):
            warning_msg = (
                f"NOTICE: Mirels scoring was developed exclusively for metastatic lesions in "
                f"long bones and is clinically unvalidated for {disease_context!r}. "
                f"Paget's disease involves hyper-remodeling and microdamage, OI involves Type I "
                f"collagen deficiency, and healthy controls exhibit no osteolytic lesion."
            )
        else:
            warning_msg = (
                "Valid for metastatic carcinoma of long bones. "
                "Consult orthopedic oncology for formal clinical judgment."
            )

        parsed_site = parse_site(site)
        parsed_pain = parse_pain(pain)
        parsed_nature = parse_nature(nature)
        parsed_size = parse_size(size)

        s_site = int(parsed_site)
        s_pain = int(parsed_pain)
        s_nature = int(parsed_nature)
        s_size = int(parsed_size)

        total = s_site + s_pain + s_nature + s_size

        if total < 4 or total > 12:
            raise ValueError(f"Calculated Mirels total {total} is outside valid [4, 12] range.")

        est_pct = cls.HISTORICAL_RISK_MAP.get(total, 50.0)

        if total <= 7:
            tier = MirelsRecommendationTier.LOW_RISK
            fixation = False
            interpretation = (
                f"Total Score {total}: Low risk of pathological fracture (~{est_pct:.0f}% in Mirels 1989 cohort). "
                f"Historically managed non-operatively with radiotherapy and clinical surveillance. "
                f"{warning_msg}"
            )
        elif total == 8:
            tier = MirelsRecommendationTier.INTERMEDIATE_RISK
            fixation = False  # Borderline, clinical judgment
            interpretation = (
                f"Total Score {total}: Borderline / Intermediate fracture risk (~{est_pct:.0f}% in Mirels 1989 cohort). "
                f"Requires careful clinical judgment, consideration of anatomical demands, and close surveillance. "
                f"{warning_msg}"
            )
        else:
            tier = MirelsRecommendationTier.HIGH_RISK
            fixation = True
            interpretation = (
                f"Total Score {total}: High risk of impending pathological fracture (~{est_pct:.0f}% in Mirels 1989 cohort). "
                f"Prophylactic internal fixation strongly indicated prior to radiation or systemic therapy. "
                f"{warning_msg}"
            )

        return MirelsScoreResult(
            site_score=s_site,
            pain_score=s_pain,
            nature_score=s_nature,
            size_score=s_size,
            total_score=total,
            risk_tier=tier,
            fracture_risk_estimate_pct=est_pct,
            prophylactic_fixation_indicated=fixation,
            clinical_interpretation=interpretation,
        )
