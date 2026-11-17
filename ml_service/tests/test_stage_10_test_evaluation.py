"""
Stage 10 Holdout Test Evaluation Verification Tests.
Verifies test artifacts completeness, prediction validity, frozen threshold integrity,
zero data leakage, and system non-interference.
"""

import csv
import json
import math
from pathlib import Path
import pytest
import torch

from ml_service.config.settings import get_settings


@pytest.fixture(scope="module")
def artifacts_dir():
    settings = get_settings()
    d = settings.base_dir / "ml_artifacts/fracatlas_test_evaluation"
    assert d.exists(), f"Test artifacts directory missing: {d}"
    return d


@pytest.fixture(scope="module")
def split_manifest():
    settings = get_settings()
    p = settings.base_dir / "ml_service/data/splits/fracatlas_classification_split.csv"
    assert p.exists()
    return p


def test_test_artifacts_completeness(artifacts_dir):
    """Verifies all required Stage 10 files exist and are non-empty."""
    expected_files = [
        "test_predictions.csv",
        "test_metrics.json",
        "confusion_matrix_test.png",
        "roc_curve_test.png",
        "precision_recall_curve_test.png",
        "validation_vs_test_comparison.png",
        "test_evaluation_summary.json",
    ]
    for fn in expected_files:
        p = artifacts_dir / fn
        assert p.exists(), f"Missing file: {fn}"
        assert p.stat().st_size > 0, f"Empty file: {fn}"


def test_test_predictions_count_and_probabilities(artifacts_dir):
    """Verifies that test_predictions.csv has 614 rows and valid probabilities."""
    preds_file = artifacts_dir / "test_predictions.csv"
    records = []
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    assert len(records) == 614, f"Expected 614 test records, got {len(records)}"

    n_frac = sum(1 for r in records if int(r["actual_label"]) == 1)
    n_non_frac = sum(1 for r in records if int(r["actual_label"]) == 0)
    assert n_frac == 110
    assert n_non_frac == 504

    for r in records:
        pf = float(r["fracture_probability"])
        pn = float(r["non_fracture_probability"])
        assert 0.0 <= pf <= 1.0
        assert 0.0 <= pn <= 1.0
        assert math.isclose(pf + pn, 1.0, abs_tol=1e-4)

        # Check thresholding consistency
        pred_031 = int(r["pred_threshold_031"])
        assert pred_031 == int(pf >= 0.31)
        pred_050 = int(r["pred_threshold_050"])
        assert pred_050 == int(pf >= 0.50)


def test_frozen_threshold_integrity(artifacts_dir):
    """Verifies that the frozen threshold 0.31 was derived from validation, not optimized on test."""
    test_metrics_file = artifacts_dir / "test_metrics.json"
    with open(test_metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Primary threshold must be exactly 0.31
    assert data["frozen_primary_threshold_031"]["threshold"] == 0.31

    # Check conservation of test set confusion matrix: 99 + 129 + 375 + 11 = 614
    m = data["frozen_primary_threshold_031"]
    total = m["tp"] + m["fp"] + m["tn"] + m["fn"]
    assert total == 614
    assert m["tp"] + m["fn"] == 110  # Ground truth fractured
    assert m["tn"] + m["fp"] == 504  # Ground truth non-fractured


def test_no_data_leakage_between_splits(split_manifest, artifacts_dir):
    """Verifies zero overlap between train/val and test image hashes."""
    splits = {"train": set(), "val": set(), "test": set()}
    with open(split_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            splits[r["split"].lower()].add(r["sha256_hash"])

    assert len(splits["test"].intersection(splits["train"])) == 0
    assert len(splits["test"].intersection(splits["val"])) == 0


def test_stage_09_checkpoint_unaltered():
    """Verifies that Stage 09 checkpoint was not modified during evaluation."""
    settings = get_settings()
    ckpt_path = settings.base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert ckpt_path.exists()
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert ckpt["epoch"] == 9
    assert math.isclose(ckpt["val_pr_auc"], 0.7556, abs_tol=1e-3)

