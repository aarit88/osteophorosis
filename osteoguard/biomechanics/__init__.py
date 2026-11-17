"""
OsteoGuard AI - Biomechanics and Fracture Risk Module.

Contains:
    1. ConventionalMirelsScorer: Classical discrete 4-factor Mirels (1989) scoring system.
    2. CrossSectionalGeometryEngine: 2D second moments of area and polar moment engine.
    3. SyntheticBonePhantomSuite: Deterministic 8-phantom synthetic test suite.
    4. BiomechanicalFractureEngine: Continuous LCR proxy engine (Stage 02 exploratory model).
"""

from osteoguard.biomechanics.mirels import (
    ConventionalMirelsScorer,
    MirelsScoreResult,
    MirelsSite,
    MirelsPain,
    MirelsRadiographicNature,
    MirelsLesionSize,
    MirelsRecommendationTier,
)
from osteoguard.biomechanics.geometry_engine import (
    CrossSectionalGeometryEngine,
    CrossSectionalGeometryResult,
)
from osteoguard.biomechanics.phantom_suite import (
    SyntheticBonePhantomSuite,
    BonePhantom,
)
from osteoguard.biomechanics.fracture_risk import (
    BiomechanicalFractureEngine,
    FractureRiskEvaluation,
    calculate_lesion_to_cortex_ratio,
)

__all__ = [
    # Conventional Mirels
    "ConventionalMirelsScorer",
    "MirelsScoreResult",
    "MirelsSite",
    "MirelsPain",
    "MirelsRadiographicNature",
    "MirelsLesionSize",
    "MirelsRecommendationTier",
    # Cross-Sectional Geometry Engine
    "CrossSectionalGeometryEngine",
    "CrossSectionalGeometryResult",
    # Synthetic Phantoms
    "SyntheticBonePhantomSuite",
    "BonePhantom",
    # Continuous proxy
    "BiomechanicalFractureEngine",
    "FractureRiskEvaluation",
    "calculate_lesion_to_cortex_ratio",
]
