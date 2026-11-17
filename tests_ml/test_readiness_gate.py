"""
Unit tests for the Zero-Dataset Readiness Verification Gate.
Enforces that model training is HARD-BLOCKED when real clinical datasets are absent,
protecting against synthetic data pollution (Rule 5) and metric fabrication (Rule 7).
"""

from pathlib import Path
import pytest
from PIL import Image

from osteoguard.data.readiness import (
    DatasetReadinessGate,
    DatasetReadinessError,
    ReadinessCheckResult,
)


def test_readiness_gate_nonexistent_directory(tmp_path):
    """Verify hard block when dataset directory does not exist."""
    gate = DatasetReadinessGate()
    fake_path = tmp_path / "does_not_exist"
    result = gate.check_readiness(fake_path)

    assert result.is_ready is False
    assert result.status == "BLOCKED"
    assert "does not exist" in result.reason

    with pytest.raises(DatasetReadinessError) as exc_info:
        gate.enforce_readiness(fake_path)
    assert "BLOCKED" in str(exc_info.value)


def test_readiness_gate_empty_directory(tmp_path):
    """Verify hard block when dataset directory contains zero images."""
    empty_dir = tmp_path / "empty_cohort"
    empty_dir.mkdir()

    gate = DatasetReadinessGate()
    result = gate.check_readiness(empty_dir)

    assert result.is_ready is False
    assert result.status == "BLOCKED"
    assert result.real_sample_count == 0

    with pytest.raises(DatasetReadinessError):
        gate.enforce_readiness(empty_dir)


def test_readiness_gate_missing_target_classes(tmp_path):
    """Verify hard block when one or more required target classes are missing."""
    cohort_dir = tmp_path / "partial_cohort"
    cohort_dir.mkdir()
    # Create only metastasis folder
    met_dir = cohort_dir / "metastasis"
    met_dir.mkdir()

    img = Image.new("L", (64, 64), color=128)
    img.save(met_dir / "PT01_01.png")

    gate = DatasetReadinessGate(min_patients_per_class=1)
    result = gate.check_readiness(cohort_dir)

    assert result.is_ready is False
    assert result.status == "BLOCKED"
    assert "Missing required target classes" in result.reason
    assert "paget" in result.missing_classes
    assert "oi" in result.missing_classes
    assert "healthy" in result.missing_classes


def test_readiness_gate_cleared_on_verified_cohort(tmp_path):
    """Verify clearance granted when all 4 classes meet patient thresholds."""
    cohort_dir = tmp_path / "verified_cohort"
    cohort_dir.mkdir()
    classes = ["metastasis", "paget", "oi", "healthy"]

    # Populate 3 distinct patients per class with valid images
    for cls in classes:
        cls_dir = cohort_dir / cls
        cls_dir.mkdir()
        for p_idx in range(3):
            img = Image.new("L", (64, 64), color=100 + p_idx * 20)
            img.save(cls_dir / f"PT_{cls}_{p_idx:02d}_01.png")

    gate = DatasetReadinessGate(min_patients_per_class=2)
    result = gate.check_readiness(cohort_dir)

    assert result.is_ready is True
    assert result.status == "READY"
    assert result.real_sample_count == 12  # 4 classes * 3 samples
    assert len(result.missing_classes) == 0

    # enforce_readiness should return without raising exception
    enforced = gate.enforce_readiness(cohort_dir)
    assert enforced.is_ready is True

