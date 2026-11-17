"""
Unit tests for Biomechanical Impending Pathological Fracture Risk Engine.
Validates the continuous Mirels-aligned equation and polar moment degradation.
"""

import pytest
from osteoguard.biomechanics.fracture_risk import BiomechanicalFractureEngine


def test_fracture_risk_baseline():
    """Verify fracture risk at baseline (zero cortical lesion)."""
    engine = BiomechanicalFractureEngine(kappa=12.5, tau=0.33, alert_threshold=80.0)
    res = engine.evaluate_risk(lcr=0.0, anatomical_site="default", morphology="default")

    assert res.lcr == 0.0
    assert res.risk_percent < 5.0
    assert res.risk_level == "low"
    assert res.urgent_alert is False
    assert res.polar_moment_retention_ratio > 0.90


def test_fracture_risk_critical_threshold():
    """Verify behavior at the critical 1/3 cortical threshold (LCR = tau = 0.33)."""
    engine = BiomechanicalFractureEngine(kappa=12.5, tau=0.33)
    res = engine.evaluate_risk(lcr=0.33, anatomical_site="default", morphology="default")

    # At LCR = tau, sigmoid(0) = 0.50 -> 50%
    assert abs(res.risk_percent - 50.0) < 1.0
    assert res.risk_level == "medium"


def test_fracture_risk_high_urgency_trigger():
    """Verify high-risk alert (>= 80%) on extensive osteolytic lesion in femur neck."""
    engine = BiomechanicalFractureEngine(kappa=12.5, tau=0.33, alert_threshold=80.0)
    res = engine.evaluate_risk(
        lcr=0.55,
        anatomical_site="femur_neck",       # M_site = 1.35
        morphology="osteolytic",            # M_morph = 1.30
    )

    assert res.risk_percent >= 80.0
    assert res.risk_level == "high"
    assert res.urgent_alert is True
    assert "Urgent orthopedic oncology consultation" in res.clinical_recommendation


def test_polar_moment_monotonic_degradation():
    """Verify polar moment of inertia retention decreases monotonically with LCR."""
    engine = BiomechanicalFractureEngine()
    retentions = [engine.calculate_polar_moment_retention(lcr=x / 10.0) for x in range(11)]

    for i in range(len(retentions) - 1):
        assert retentions[i] >= retentions[i + 1], "Polar moment retention must decrease as lesion grows"

