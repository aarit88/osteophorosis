"""
OsteoGuard AI - Canonical Constants and Architectural Specifications.
Defines the strictly enforced 4-class taxonomy, IBSI feature definitions,
modality configurations, and biomechanical parameters.
"""

from typing import Dict, List, Tuple
from enum import Enum


# =====================================================================
# Strict Target Pathology Taxonomy (Rule 9: Never silently change)
# =====================================================================
class BonePathologyClass(str, Enum):
    METASTASIS = "metastasis"
    PAGET = "paget"
    OI = "oi"
    HEALTHY = "healthy"


TARGET_CLASSES: Tuple[str, ...] = (
    BonePathologyClass.METASTASIS.value,
    BonePathologyClass.PAGET.value,
    BonePathologyClass.OI.value,
    BonePathologyClass.HEALTHY.value,
)

CLASS_TO_IDX: Dict[str, int] = {
    BonePathologyClass.METASTASIS.value: 0,
    BonePathologyClass.PAGET.value: 1,
    BonePathologyClass.OI.value: 2,
    BonePathologyClass.HEALTHY.value: 3,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

CLASS_DISPLAY_NAMES: Dict[str, str] = {
    BonePathologyClass.METASTASIS.value: "Secondary Bone Metastasis",
    BonePathologyClass.PAGET.value: "Paget's Disease of Bone",
    BonePathologyClass.OI.value: "Osteogenesis Imperfecta",
    BonePathologyClass.HEALTHY.value: "Healthy Skeletal Control",
}

NUM_CLASSES: int = len(TARGET_CLASSES)


# =====================================================================
# Modality Strategy (Rule 8: Never mix without scientific strategy)
# =====================================================================
class ModalityType(str, Enum):
    XRAY = "xray"
    CT = "ct"
    BONE_SCAN = "bone_scan"


SUPPORTED_MODALITIES: Tuple[str, ...] = (
    ModalityType.XRAY.value,
    ModalityType.CT.value,
    ModalityType.BONE_SCAN.value,
)

# Standard CT Bone Window Settings (Paper: WL = +400 HU, WW = 1500 HU)
CT_BONE_WINDOW_LEVEL: float = 400.0
CT_BONE_WINDOW_WIDTH: float = 1500.0

# Radiograph CLAHE Parameters (Paper: 8x8 contextual tiles, clip limit beta = 2.0)
XRAY_CLAHE_CLIP_LIMIT: float = 2.0
XRAY_CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)


# =====================================================================
# Network Dimensionalities (Paper Section: End-to-End Computational Pipeline)
# =====================================================================
SEMANTIC_EMBEDDING_DIM: int = 512
RADIOMICS_FEATURE_DIM: int = 102
FUSION_VECTOR_DIM: int = SEMANTIC_EMBEDDING_DIM + RADIOMICS_FEATURE_DIM  # 614
MLP_HIDDEN_DIM_1: int = 256
MLP_HIDDEN_DIM_2: int = 64
MLP_DROPOUT_RATE: float = 0.4

# Multi-class Focal Loss Parameters
FOCAL_LOSS_GAMMA: float = 2.0
DEFAULT_CLASS_WEIGHTS: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


# =====================================================================
# Biomechanical Fracture Risk Parameters (Paper & KSCST Proposal)
# =====================================================================
FRACTURE_KAPPA_SLOPE: float = 12.5       # Calibrated sensitivity slope
FRACTURE_TAU_THRESHOLD: float = 0.33    # Critical 1/3 cortical threshold
FRACTURE_ALERT_THRESHOLD_PCT: float = 80.0 # Urgency consultation alert

# Anatomical site multipliers (M_site)
SITE_MULTIPLIERS: Dict[str, float] = {
    "femur_diaphysis": 1.30,
    "femur_neck": 1.35,
    "subtrochanteric": 1.35,
    "pelvis": 1.15,
    "spine_vertebra": 1.25,
    "tibia": 1.10,
    "humerus": 1.05,
    "default": 1.00,
}

# Lesion morphology multipliers (M_morph)
MORPHOLOGY_MULTIPLIERS: Dict[str, float] = {
    "osteolytic": 1.30,       # High cortical destruction
    "mixed": 1.15,            # Coexisting lysis and sclerosis
    "osteoblastic": 1.00,     # Dense sclerotic remodeling
    "permeative": 1.35,       # Aggressive moth-eaten invasion
    "default": 1.00,
}


# =====================================================================
# IBSI Radiomic Standard Codes and Counts (Exact 102 Features)
# =====================================================================
IBSI_CATEGORIES: Dict[str, Dict[str, any]] = {
    "first_order": {
        "ibsi_code": "IBSI:Q3RU",
        "count": 18,
        "description": "First-Order Intensity Statistics (histogram distribution)",
    },
    "glcm": {
        "ibsi_code": "IBSI:P88C",
        "count": 26,
        "description": "Gray Level Co-occurrence Matrix (second-order texture)",
    },
    "glrlm": {
        "ibsi_code": "IBSI:9SA3",
        "count": 16,
        "description": "Gray Level Run-Length Matrix (linear trabecular continuity)",
    },
    "glszm": {
        "ibsi_code": "IBSI:4NB3",
        "count": 16,
        "description": "Gray Level Size Zone Matrix (regional cluster size)",
    },
    "gldm_ngtdm": {
        "ibsi_code": "IBSI:26DV",
        "count": 26,
        "description": "Gray Level Dependence & Neighborhood Tone Difference Matrices",
    },
}

TOTAL_RADIOMIC_FEATURES: int = sum(cat["count"] for cat in IBSI_CATEGORIES.values())  # Exactly 102
assert TOTAL_RADIOMIC_FEATURES == RADIOMICS_FEATURE_DIM, "IBSI feature count mismatch"

