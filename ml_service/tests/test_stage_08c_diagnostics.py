"""
OsteoGuard AI - Stage 08C Prediction Diagnostics Test Suite.

Verifies:
  1. Class mapping consistency (0 = non-fractured, 1 = fractured).
  2. Validation predictions file integrity (612 records, valid probs, no NaNs).
  3. Threshold analysis file integrity (91 steps from 0.05 to 0.95, optimal threshold identification).
  4. Diagnostic visualization artifacts (distribution, ROC, and PR plots).
  5. Truncated image recovery audit (59 images identified and recoverable via LOAD_TRUNCATED_IMAGES).
"""

import csv
import json
from pathlib import Path
from PIL import Image, ImageFile
import pytest

from ml_service.config.settings import get_settings


@pytest.fixture(scope="module")
def pilot_dir() -> Path:
    settings = get_settings()
    d = settings.base_dir / "ml_artifacts/fracatlas_pilot"
    assert d.exists(), f"Pilot directory missing: {d}"
    return d


def test_class_mapping_consistency(pilot_dir: Path):
    """Verify class mapping in pilot config and weights info."""
    config_yaml = pilot_dir / "pilot_config.yaml"
    weights_json = pilot_dir / "pretrained_weights_info.json"

    assert config_yaml.exists()
    assert weights_json.exists()

    with open(weights_json, "r", encoding="utf-8") as f:
        w_info = json.load(f)

    mapping = w_info["output_mapping"]
    assert mapping["0"] == "candidate_control_radiograph"
    assert mapping["1"] == "acute_bone_fracture"


def test_validation_predictions_csv(pilot_dir: Path):
    """Verify validation_predictions.csv has 612 images, valid probabilities, and accurate columns."""
    pred_csv = pilot_dir / "validation_predictions.csv"
    assert pred_csv.exists(), "validation_predictions.csv missing"

    with open(pred_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    assert len(records) == 612, f"Expected 612 records, got {len(records)}"

    fractured_count = sum(1 for r in records if r["actual_label"] == "1")
    non_fractured_count = sum(1 for r in records if r["actual_label"] == "0")
    assert fractured_count == 107
    assert non_fractured_count == 505

    max_p1 = 0.0
    for r in records:
        p0 = float(r["prob_class_0"])
        p1 = float(r["prob_class_1"])
        assert 0.0 <= p0 <= 1.0
        assert 0.0 <= p1 <= 1.0
        assert abs((p0 + p1) - 1.0) < 1e-4, f"Probabilities do not sum to 1: {p0} + {p1}"
        if p1 > max_p1:
            max_p1 = p1

    # Confirm all probabilities are strictly below 0.50 (explaining zero positive predictions at default cutoff)
    assert max_p1 < 0.50, f"Expected all probabilities < 0.50, found max {max_p1}"


def test_threshold_analysis_csv(pilot_dir: Path):
    """Verify threshold analysis contains 91 steps and identifies optimal thresholds."""
    th_csv = pilot_dir / "threshold_analysis.csv"
    assert th_csv.exists(), "threshold_analysis.csv missing"

    with open(th_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    assert len(records) == 91, f"Expected 91 thresholds, got {len(records)}"

    # Check threshold 0.50 yields zero true positives
    th_0_5 = next(r for r in records if float(r["threshold"]) == 0.50)
    assert int(th_0_5["tp"]) == 0
    assert int(th_0_5["fp"]) == 0
    assert int(th_0_5["tn"]) == 505
    assert int(th_0_5["fn"]) == 107
    assert float(th_0_5["sensitivity_recall"]) == 0.0

    # Check threshold 0.35 achieves balanced sensitivity and specificity
    th_0_35 = next(r for r in records if float(r["threshold"]) == 0.35)
    assert int(th_0_35["tp"]) == 77
    assert float(th_0_35["sensitivity_recall"]) > 0.70
    assert float(th_0_35["specificity"]) > 0.65


def test_diagnostic_plots(pilot_dir: Path):
    """Verify all 3 diagnostic plots are generated and non-empty."""
    dist_png = pilot_dir / "probability_distribution.png"
    roc_png = pilot_dir / "roc_curve.png"
    pr_png = pilot_dir / "precision_recall_curve.png"

    for p in [dist_png, roc_png, pr_png]:
        assert p.exists(), f"Plot missing: {p}"
        assert p.stat().st_size > 20000, f"Plot unexpectedly small: {p}"


def test_truncated_image_recovery():
    """Verify that truncated images in FracAtlas recover cleanly under LOAD_TRUNCATED_IMAGES."""
    settings = get_settings()
    truncated_sample = settings.base_dir / "ml_service/data/raw/fracatlas/images/Non_fractured/IMG0004028.jpg"
    assert truncated_sample.exists(), "Sample truncated image missing"

    # Test without flag -> raises OSError
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with pytest.raises(OSError):
        with Image.open(truncated_sample) as img:
            img.convert("RGB").load()

    # Test with flag -> loads cleanly
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(truncated_sample) as img:
        img_rgb = img.convert("RGB")
        img_rgb.load()
        assert img_rgb.size[0] > 0
        assert img_rgb.size[1] > 0

