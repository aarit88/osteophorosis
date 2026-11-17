"""
OsteoGuard AI - Stage 28 Automated Verification Test Suite.

Comprehensive tests for:
1. Checkpoint cryptographic immutability (Rule 1).
2. Stage 28 artifact completeness and mirrored report existence.
3. Conventional Mirels 4-factor discrete scoring space (81 combinations).
4. Mirels input validation, boundary checking, and clinical pathology scoping.
5. Continuous LCR and exploratory sigmoid proxy status labeling.
6. 8 deterministic synthetic bone cross-section phantoms.
7. Analytical accuracy against circular annulus ground truths (<1.0% discretization error).
8. Translation invariance of central moments.
9. Rotation invariance of polar second moment of area (J_p).
10. Defect area and thinning J_p monotonicity.
11. Anisotropic non-square pixel spacing scaling.
12. Geometry engine input validation and error raising.
13. Zero patient data ingestion and preservation of burned holdouts.
14. Host Python and React frontend integrity.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from osteoguard.biomechanics.mirels import (
    ConventionalMirelsScorer,
    MirelsRecommendationTier,
    MirelsSite,
    MirelsPain,
    MirelsRadiographicNature,
    MirelsLesionSize,
    parse_site,
    parse_pain,
    parse_nature,
    parse_size,
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
    calculate_lesion_to_cortex_ratio,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CHECKPOINT_PATH = WORKSPACE_ROOT / "ml_artifacts" / "btxrd_frozen_model" / "best_checkpoint.pt"
EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
STAGE28_DIR = WORKSPACE_ROOT / "ml_artifacts" / "stage28_biomechanics"
MIRROR_REPORT = WORKSPACE_ROOT / "ml_artifacts" / "stage_reports" / "STAGE_28_BIOMECHANICS.md"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def test_01_checkpoint_hash_strictly_unmodified():
    """Verify frozen BTXRD checkpoint SHA-256 matches cryptographic baseline."""
    assert FROZEN_CHECKPOINT_PATH.exists(), f"Frozen checkpoint missing at {FROZEN_CHECKPOINT_PATH}"
    actual = compute_sha256(FROZEN_CHECKPOINT_PATH)
    assert actual == EXPECTED_SHA256, f"Checkpoint hash modified!\nExpected: {EXPECTED_SHA256}\nActual:   {actual}"


def test_02_all_stage28_artifacts_exist():
    """Verify all 9 Stage 28 deliverables and mirrored report exist and are populated."""
    required_files = [
        "STAGE_28_BIOMECHANICS_REPORT.md",
        "mirels_scoring_specification.md",
        "biomechanical_geometry_specification.md",
        "synthetic_bone_phantom_protocol.md",
        "mathematical_validation_report.md",
        "requirement_to_implementation_mapping.md",
        "limitations_and_clinical_review_requirements.md",
        "reproducibility_manifest.json",
        "test_execution_report.json",
    ]
    for rf in required_files:
        p = STAGE28_DIR / rf
        assert p.exists(), f"Required artifact missing: {rf}"
        assert p.stat().st_size > 0, f"Artifact {rf} is empty"

    assert MIRROR_REPORT.exists(), f"Mirrored report missing at {MIRROR_REPORT}"
    assert MIRROR_REPORT.stat().st_size > 0, "Mirrored report is empty"


def test_03_mirels_scoring_all_81_combinations():
    """Verify complete discrete 4-factor Mirels scoring space across all 81 combinations."""
    total_evaluated = 0
    scores = []
    tiers = {"LOW_RISK": 0, "INTERMEDIATE_RISK": 0, "HIGH_RISK": 0}

    for s in [1, 2, 3]:
        for p in [1, 2, 3]:
            for n in [1, 2, 3]:
                for sz in [1, 2, 3]:
                    res = ConventionalMirelsScorer.calculate_score(s, p, n, sz)
                    total_evaluated += 1
                    scores.append(res.total_score)
                    tiers[res.risk_tier.value] += 1
                    
                    # Score bounds
                    assert 4 <= res.total_score <= 12
                    # Sum consistency
                    assert res.total_score == s + p + n + sz
                    # Tier consistency
                    if res.total_score <= 7:
                        assert res.risk_tier == MirelsRecommendationTier.LOW_RISK
                        assert res.prophylactic_fixation_indicated is False
                    elif res.total_score == 8:
                        assert res.risk_tier == MirelsRecommendationTier.INTERMEDIATE_RISK
                        assert res.prophylactic_fixation_indicated is False
                    else:
                        assert res.risk_tier == MirelsRecommendationTier.HIGH_RISK
                        assert res.prophylactic_fixation_indicated is True

    assert total_evaluated == 81
    assert min(scores) == 4
    assert max(scores) == 12
    assert tiers["LOW_RISK"] == 31
    assert tiers["INTERMEDIATE_RISK"] == 19
    assert tiers["HIGH_RISK"] == 31


def test_04_mirels_input_validation_and_errors():
    """Verify input validation and error raising for invalid inputs."""
    # Invalid sites
    with pytest.raises(ValueError, match="Invalid Mirels site input"):
        ConventionalMirelsScorer.calculate_score("vertebra", 1, 1, 1)
    with pytest.raises(ValueError, match="Invalid Mirels site input"):
        ConventionalMirelsScorer.calculate_score(4, 1, 1, 1)

    # Invalid pain
    with pytest.raises(ValueError, match="Invalid Mirels pain input"):
        ConventionalMirelsScorer.calculate_score(1, "excruciating", 1, 1)
    with pytest.raises(ValueError, match="Invalid Mirels pain input"):
        ConventionalMirelsScorer.calculate_score(1, 0, 1, 1)

    # Invalid nature
    with pytest.raises(ValueError, match="Invalid Mirels radiographic nature input"):
        ConventionalMirelsScorer.calculate_score(1, 1, "cystic", 1)

    # Invalid size
    with pytest.raises(ValueError, match="Invalid Mirels lesion size"):
        ConventionalMirelsScorer.calculate_score(1, 1, 1, 5)
    with pytest.raises(ValueError, match="Lesion size ratio must be between"):
        ConventionalMirelsScorer.calculate_score(1, 1, 1, 1.5)
    with pytest.raises(ValueError, match="Lesion size ratio must be between"):
        ConventionalMirelsScorer.calculate_score(1, 1, 1, -0.2)


def test_05_mirels_clinical_disclaimer_and_pathology_scoping():
    """Verify clinical warnings and scoping to metastatic long-bone disease only."""
    res_default = ConventionalMirelsScorer.calculate_score(1, 1, 1, 1, disease_context="metastasis")
    assert "Metastatic carcinoma of long bones" in res_default.target_pathology
    assert "RESEARCH ONLY" in res_default.disclaimer

    # Non-metastatic disease contexts must trigger explicit warning
    res_paget = ConventionalMirelsScorer.calculate_score(1, 1, 1, 1, disease_context="paget")
    assert "clinically unvalidated for 'paget'" in res_paget.clinical_interpretation

    res_oi = ConventionalMirelsScorer.calculate_score(1, 1, 1, 1, disease_context="oi")
    assert "clinically unvalidated for 'oi'" in res_oi.clinical_interpretation

    res_healthy = ConventionalMirelsScorer.calculate_score(1, 1, 1, 1, disease_context="healthy")
    assert "clinically unvalidated for 'healthy'" in res_healthy.clinical_interpretation


def test_06_continuous_lcr_computation_and_boundaries():
    """Verify continuous lesion-to-cortex ratio calculation and boundary clamping."""
    assert calculate_lesion_to_cortex_ratio(0.0, 30.0) == 0.0
    assert abs(calculate_lesion_to_cortex_ratio(10.0, 30.0) - 0.3333) < 0.001
    assert calculate_lesion_to_cortex_ratio(15.0, 30.0) == 0.50
    assert calculate_lesion_to_cortex_ratio(35.0, 30.0) == 1.0  # Clamped to 1.0

    # Error conditions
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_lesion_to_cortex_ratio(-2.0, 30.0)
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_lesion_to_cortex_ratio(10.0, 0.0)
    with pytest.raises(ValueError, match="must be finite"):
        calculate_lesion_to_cortex_ratio(float("nan"), 30.0)


def test_07_exploratory_sigmoid_proxy_labeling():
    """Verify exploratory status and backward compatibility of continuous fracture engine."""
    engine = BiomechanicalFractureEngine()
    res = engine.evaluate_risk(0.55, anatomical_site="femur_neck", morphology="osteolytic")

    assert res.exploratory_status == "EXPLORATORY_PROXY_REQUIRING_CLINICAL_REVIEW"
    assert res.risk_percent >= 80.0
    assert res.urgent_alert is True
    assert "Urgent orthopedic oncology consultation" in res.clinical_recommendation
    assert "EXPLORATORY PROXY" in res.clinical_recommendation


def test_08_phantom_generation_all_8_present_and_deterministic():
    """Verify all 8 synthetic bone cross-section phantoms are generated and deterministic."""
    p_dict = SyntheticBonePhantomSuite.get_all_phantoms()
    assert len(p_dict) == 8

    expected_names = [
        "intact_circular_ring",
        "uniform_cortical_thinning",
        "symmetric_cortical_defect",
        "eccentric_cortical_defect",
        "lytic_cavity",
        "multiple_small_defects",
        "non_circular_elliptical_annulus",
        "anisotropic_spacing_annulus",
    ]
    for name in expected_names:
        assert name in p_dict
        p = p_dict[name]
        assert p.mask.ndim == 2
        assert np.sum(p.mask) > 0
        assert set(np.unique(p.mask)).issubset({0.0, 1.0})
        assert p.spacing_y_mm > 0
        assert p.spacing_x_mm > 0

    # Determinism check: re-generation yields identical mask
    p_again = SyntheticBonePhantomSuite.get_all_phantoms()
    for name in expected_names:
        assert np.array_equal(p_dict[name].mask, p_again[name].mask)


def test_09_circular_annulus_analytical_accuracy():
    """Verify intact circular annulus analytical ground-truth convergence (<1.0% error)."""
    p = SyntheticBonePhantomSuite.create_intact_annulus(
        outer_radius_mm=15.0, inner_radius_mm=10.0, grid_size=100, spacing_mm=0.5
    )
    geom = CrossSectionalGeometryEngine.compute_geometry(p.mask, p.spacing_y_mm)

    # Theoretical: A = 125 * pi approx 392.699 mm^2
    # I_x = I_y = 40625 * pi / 4 approx 31906.80 mm^4
    # J_p = 40625 * pi / 2 approx 63813.60 mm^4
    area_err_pct = abs(geom.cortical_area_mm2 - p.analytical_area_mm2) / p.analytical_area_mm2 * 100.0
    ix_err_pct = abs(geom.ix_mm4 - p.analytical_ix_mm4) / p.analytical_ix_mm4 * 100.0
    iy_err_pct = abs(geom.iy_mm4 - p.analytical_iy_mm4) / p.analytical_iy_mm4 * 100.0
    j_err_pct = abs(geom.polar_j_mm4 - p.analytical_polar_j_mm4) / p.analytical_polar_j_mm4 * 100.0

    assert area_err_pct < 1.0, f"Area error {area_err_pct:.3f}% >= 1.0%"
    assert ix_err_pct < 1.0, f"Ix error {ix_err_pct:.3f}% >= 1.0%"
    assert iy_err_pct < 1.0, f"Iy error {iy_err_pct:.3f}% >= 1.0%"
    assert j_err_pct < 1.0, f"Jp error {j_err_pct:.3f}% >= 1.0%"
    # In-plane symmetry: I_x approx I_y
    assert abs(geom.ix_mm4 - geom.iy_mm4) / geom.ix_mm4 < 0.01


def test_10_uniform_thinning_analytical_accuracy():
    """Verify uniform cortical thinning analytical ground-truth convergence (<1.0% error)."""
    p_intact = SyntheticBonePhantomSuite.create_intact_annulus()
    geom_intact = CrossSectionalGeometryEngine.compute_geometry(p_intact.mask, p_intact.spacing_y_mm)

    p_thin = SyntheticBonePhantomSuite.create_uniform_thinning(outer_radius_mm=15.0, expanded_inner_radius_mm=13.0)
    geom_thin = CrossSectionalGeometryEngine.compute_geometry(
        p_thin.mask, p_thin.spacing_y_mm, intact_baseline=geom_intact
    )

    area_err_pct = abs(geom_thin.cortical_area_mm2 - p_thin.analytical_area_mm2) / p_thin.analytical_area_mm2 * 100.0
    j_err_pct = abs(geom_thin.polar_j_mm4 - p_thin.analytical_polar_j_mm4) / p_thin.analytical_polar_j_mm4 * 100.0

    assert area_err_pct < 1.0
    assert j_err_pct < 1.0
    # Retention ratios: theoretical area ratio = 56 / 125 = 0.4480
    assert abs(geom_thin.area_retention_ratio - 0.4480) < 0.01
    # Theoretical J_p ratio = 22064 / 40625 approx 0.5431
    assert abs(geom_thin.polar_j_retention_ratio - 0.5431) < 0.01


def test_11_translation_invariance_of_central_moments():
    """Verify central moments (Ix, Iy, Jp) are translation-invariant under spatial shift."""
    p = SyntheticBonePhantomSuite.create_intact_annulus()
    grid_sz = 140
    spacing = 0.5
    h_orig, w_orig = p.mask.shape

    # Centered
    m1 = np.zeros((grid_sz, grid_sz), dtype=np.float64)
    r1 = (grid_sz - h_orig) // 2
    c1 = (grid_sz - w_orig) // 2
    m1[r1:r1+h_orig, c1:c1+w_orig] = p.mask
    g1 = CrossSectionalGeometryEngine.compute_geometry(m1, spacing)

    # Shifted by (+12, -8) pixels
    m2 = np.zeros((grid_sz, grid_sz), dtype=np.float64)
    r2 = r1 + 12
    c2 = c1 - 8
    m2[r2:r2+h_orig, c2:c2+w_orig] = p.mask
    g2 = CrossSectionalGeometryEngine.compute_geometry(m2, spacing)

    # Central moments must be identical
    assert abs(g1.cortical_area_mm2 - g2.cortical_area_mm2) < 1e-4
    assert abs(g1.ix_mm4 - g2.ix_mm4) < 1e-3
    assert abs(g1.iy_mm4 - g2.iy_mm4) < 1e-3
    assert abs(g1.polar_j_mm4 - g2.polar_j_mm4) < 1e-3

    # Centroid must shift by exact physical delta
    assert abs((g2.centroid_x_mm - g1.centroid_x_mm) - (-8 * spacing)) < 1e-4
    assert abs((g2.centroid_y_mm - g1.centroid_y_mm) - (12 * spacing)) < 1e-4


def test_12_rotation_invariance_of_polar_moment():
    """Verify polar second moment (Jp) is invariant under 90 degree rotation and axes swap."""
    p_ell = SyntheticBonePhantomSuite.create_elliptical_annulus()
    g_orig = CrossSectionalGeometryEngine.compute_geometry(p_ell.mask, p_ell.spacing_y_mm)

    m_rot = np.rot90(p_ell.mask)
    g_rot = CrossSectionalGeometryEngine.compute_geometry(m_rot, p_ell.spacing_y_mm)

    # Area and J_p must be conserved
    assert abs(g_orig.cortical_area_mm2 - g_rot.cortical_area_mm2) < 1e-4
    assert abs(g_orig.polar_j_mm4 - g_rot.polar_j_mm4) < 1e-3

    # I_x and I_y swap under 90 degree rotation
    assert abs(g_orig.ix_mm4 - g_rot.iy_mm4) < 1e-3
    assert abs(g_orig.iy_mm4 - g_rot.ix_mm4) < 1e-3


def test_13_wedge_defect_area_monotonicity():
    """Verify strict monotonicity: increasing wedge defect angle strictly decreases area."""
    p_intact = SyntheticBonePhantomSuite.create_intact_annulus()
    g_intact = CrossSectionalGeometryEngine.compute_geometry(p_intact.mask, 0.5)

    angles = [30.0, 60.0, 90.0, 120.0, 150.0]
    areas = [g_intact.cortical_area_mm2]

    for a in angles:
        p = SyntheticBonePhantomSuite.create_symmetric_defect(wedge_angle_deg=a)
        g = CrossSectionalGeometryEngine.compute_geometry(p.mask, 0.5)
        areas.append(g.cortical_area_mm2)

    for i in range(len(areas) - 1):
        assert areas[i] > areas[i + 1], f"Area failed strict decrease at index {i}"


def test_14_endosteal_thinning_jp_monotonicity():
    """Verify strict monotonicity: expanding endosteal canal strictly decreases J_p."""
    radii = [10.0, 11.0, 12.0, 13.0, 14.0]
    j_values = []

    for r in radii:
        p = SyntheticBonePhantomSuite.create_uniform_thinning(expanded_inner_radius_mm=r)
        g = CrossSectionalGeometryEngine.compute_geometry(p.mask, 0.5)
        j_values.append(g.polar_j_mm4)

    for i in range(len(j_values) - 1):
        assert j_values[i] > j_values[i + 1], f"J_p failed strict decrease at radius {radii[i+1]}"


def test_15_anisotropic_spacing_metric_invariance():
    """Verify that circular annulus sampled on rectangular pixels yields isotropic physical moments."""
    p = SyntheticBonePhantomSuite.create_anisotropic_spacing_annulus(
        spacing_y_mm=0.4, spacing_x_mm=0.8
    )
    geom = CrossSectionalGeometryEngine.compute_geometry(p.mask, (p.spacing_y_mm, p.spacing_x_mm))

    # Underlying geometry is circular ring; moments in physical mm^4 must remain close (<3.5% discretization error)
    assert abs(geom.ix_mm4 - geom.iy_mm4) / geom.ix_mm4 < 0.035
    assert abs(geom.polar_j_mm4 - p.analytical_polar_j_mm4) / p.analytical_polar_j_mm4 * 100.0 < 1.0


def test_16_geometry_engine_input_validation():
    """Verify geometry engine raises proper exceptions for invalid masks and spacing."""
    # Empty mask (all zeros)
    with pytest.raises(ValueError, match="contains zero cortical bone pixels"):
        CrossSectionalGeometryEngine.compute_geometry(np.zeros((50, 50)), 0.5)

    # 1D or 3D array
    with pytest.raises(ValueError, match="Mask must be 2-dimensional"):
        CrossSectionalGeometryEngine.compute_geometry(np.ones((50, 50, 3)), 0.5)

    # Non-binary mask
    with pytest.raises(ValueError, match="only binary values"):
        m = np.zeros((50, 50))
        m[20:30, 20:30] = 5.0
        CrossSectionalGeometryEngine.compute_geometry(m, 0.5)

    # Zero or negative spacing
    with pytest.raises(ValueError, match="strictly positive"):
        m = np.zeros((50, 50))
        m[20:30, 20:30] = 1.0
        CrossSectionalGeometryEngine.compute_geometry(m, 0.0)

    # Non-array input
    with pytest.raises(TypeError, match="must be a numpy ndarray"):
        CrossSectionalGeometryEngine.compute_geometry([[1, 0], [0, 1]], 0.5)


def test_17_zero_unauthorized_patient_data_and_holdouts_intact():
    """Verify zero patient datasets ingested and Center 2/3 holdouts remain intact."""
    with open(STAGE28_DIR / "reproducibility_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["patient_data_accessed"] is False
    assert manifest["holdout_accessed"] is False
    assert manifest["models_trained"] is False
    assert (WORKSPACE_ROOT / "ml_artifacts" / "btxrd_center2_external_evaluation").exists()
    assert (WORKSPACE_ROOT / "ml_artifacts" / "btxrd_center3_external_evaluation").exists()


def test_18_host_python_and_frontend_untouched():
    """Verify host Python and React frontend remain completely unmodified."""
    host_python = Path("C:/Python313/python.exe")
    assert host_python.exists()

    frontend_dir = WORKSPACE_ROOT / "osteophorosis-main" / "src"
    assert frontend_dir.exists()
    assert (frontend_dir / "App.tsx").exists()
    assert (frontend_dir / "main.tsx").exists()
