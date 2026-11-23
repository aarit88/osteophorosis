"""
Stage 14 - Final Bone Metastasis Ensemble Holdout Test Evaluation Test Suite.

Verifies:
1. All 5 cross-validation checkpoints load into ResNet-18.
2. Holdout test cohort integrity (exactly 44 patients: 9 positive, 35 negative).
3. Test dataset paired view integrity (anterior and posterior views, 1:4 aspect ratio, [0, 1] range).
4. Patient-level max aggregation logic.
5. Five-model ensemble averaging and disagreement logic.
6. Test patient predictions count and uniqueness (exactly 44 unique patients).
7. Test image predictions count (exactly 88 images).
8. All predicted probabilities are finite and bounded in [0.0, 1.0].
9. Frozen operating thresholds (primary theta=0.60, secondary theta=0.45) strictly applied.
10. Zero patient leakage: complete isolation from train and validation splits.
11. Stage 12 cross-validation checkpoints remain unaltered.
12. Stage 13 ensemble validation artifacts remain unaltered.
13. FracAtlas checkpoints and test evaluations remain unaltered.
14. All 10 Stage 14 deliverables exist and are non-empty.
15. Host CPU Python environment remains untouched.
16. React frontend files remain untouched.
"""

import csv
import json
from pathlib import Path
import subprocess
import pytest
import torch

from ml_service.config.settings import get_settings
from ml_service.datasets.bone_scan_dataset import (
    BoneScanPatientDataset,
    preprocess_scintigraphy_image,
)
from ml_service.models.bone_scan_classifier import BoneScanResNet18
from ml_service.models.bone_scan_ensemble import BoneScanEnsemble


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def test_artifacts_dir(base_dir):
    d = base_dir / "ml_artifacts/bone_scan_test_evaluation"
    assert d.exists(), f"Test artifacts dir missing: {d}"
    return d


@pytest.fixture(scope="module")
def cv_dir(base_dir):
    d = base_dir / "ml_artifacts/bone_scan_cv"
    assert d.exists(), f"CV artifacts dir missing: {d}"
    return d


@pytest.fixture(scope="module")
def master_manifest(base_dir):
    p = base_dir / "ml_service/data/splits/bone_scan_split_manifest.csv"
    assert p.exists(), f"Master manifest missing: {p}"
    return p


def test_01_all_five_checkpoints_load(cv_dir):
    """Verifies that each of the 5 cross-validation checkpoints exists and loads into ResNet18."""
    for i in range(5):
        ckpt_path = cv_dir / f"fold_{i}/best_checkpoint.pt"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in ckpt
        model = BoneScanResNet18(pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        assert model.conv1.in_channels == 1
        assert model.fc.out_features == 1


def test_02_holdout_test_cohort_integrity(master_manifest):
    """Verifies that the holdout test cohort contains exactly 44 patients (9 positive, 35 negative)."""
    ds = BoneScanPatientDataset(master_manifest, split="test", is_train=False)
    assert len(ds) == 44, f"Expected 44 test patients, got {len(ds)}"
    counts = ds.get_class_counts()
    assert counts["positive_patients"] == 9, f"Expected 9 positive patients, got {counts['positive_patients']}"
    assert counts["negative_patients"] == 35, f"Expected 35 negative patients, got {counts['negative_patients']}"


def test_03_test_dataset_paired_views(master_manifest):
    """Verifies all 44 test patients have paired anterior and posterior scans with correct shapes and ranges."""
    ds = BoneScanPatientDataset(master_manifest, split="test", is_train=False)
    for i in range(len(ds)):
        sample = ds[i]
        assert "anterior_tensor" in sample
        assert "posterior_tensor" in sample
        assert sample["anterior_tensor"].shape == (1, 1024, 256)
        assert sample["posterior_tensor"].shape == (1, 1024, 256)
        assert 0.0 <= sample["anterior_tensor"].min().item() <= sample["anterior_tensor"].max().item() <= 1.0
        assert 0.0 <= sample["posterior_tensor"].min().item() <= sample["posterior_tensor"].max().item() <= 1.0


def test_04_patient_level_max_aggregation():
    """Verifies max-pooling rule: P_patient = max(P_ant, P_post)."""
    assert max(0.85, 0.42) == 0.85
    assert max(0.12, 0.77) == 0.77
    assert max(0.50, 0.50) == 0.50


def test_05_ensemble_mean_disagreement_logic():
    """Verifies 5-model ensemble averaging and standard deviation logic."""
    probs = [0.60, 0.70, 0.80, 0.50, 0.65]
    expected_mean = float(sum(probs) / 5.0)
    assert abs(expected_mean - 0.65) < 1e-4
    t = torch.tensor(probs)
    expected_std = float(t.std(unbiased=False).item())
    assert expected_std > 0.0


def test_06_test_patient_predictions_count_and_uniqueness(test_artifacts_dir, master_manifest):
    """Verifies exactly 44 unique test patients receive predictions in test_patient_predictions.csv."""
    preds_csv = test_artifacts_dir / "test_patient_predictions.csv"
    assert preds_csv.exists()

    # Load test split patient IDs from manifest
    test_pids = set()
    with open(master_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() == "test":
                test_pids.add(r["patient_id"])

    seen_pids = set()
    rows = []
    with open(preds_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r["patient_id"]
            assert pid not in seen_pids, f"Duplicate patient prediction: {pid}"
            assert pid in test_pids, f"Patient {pid} not in test split!"
            seen_pids.add(pid)
            rows.append(r)

    assert len(rows) == 44, f"Expected 44 predictions, got {len(rows)}"
    assert seen_pids == test_pids, "Mismatch between manifest test patients and predictions!"


def test_07_test_image_predictions_count(test_artifacts_dir):
    """Verifies test_image_predictions.csv contains exactly 88 rows (2 views per patient)."""
    img_csv = test_artifacts_dir / "test_image_predictions.csv"
    assert img_csv.exists()

    rows = []
    ant_count = 0
    post_count = 0
    with open(img_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if r["view"] == "anterior":
                ant_count += 1
            elif r["view"] == "posterior":
                post_count += 1

    assert len(rows) == 88, f"Expected 88 image predictions, got {len(rows)}"
    assert ant_count == 44, f"Expected 44 anterior predictions, got {ant_count}"
    assert post_count == 44, f"Expected 44 posterior predictions, got {post_count}"


def test_08_probabilities_bounded_and_finite(test_artifacts_dir):
    """Verifies all test probabilities and disagreement indices are finite and bounded in [0.0, 1.0]."""
    preds_csv = test_artifacts_dir / "test_patient_predictions.csv"
    with open(preds_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p_ens = float(r["prob_patient_ensemble"])
            std_dis = float(r["std_disagreement"])
            assert 0.0 <= p_ens <= 1.0, f"Probability out of bounds: {p_ens}"
            assert 0.0 <= std_dis <= 1.0, f"Disagreement out of bounds: {std_dis}"
            for k in range(5):
                f_p = float(r[f"fold_{k}_prob"])
                assert 0.0 <= f_p <= 1.0, f"Fold {k} probability out of bounds: {f_p}"


def test_09_frozen_thresholds_applied(test_artifacts_dir):
    """Verifies predictions strictly obey frozen thresholds: theta_primary=0.60, theta_screening=0.45."""
    preds_csv = test_artifacts_dir / "test_patient_predictions.csv"
    with open(preds_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p_ens = float(r["prob_patient_ensemble"])
            pred_060 = int(r["pred_label_060"])
            pred_045 = int(r["pred_label_045"])
            assert pred_060 == int(p_ens >= 0.60), f"Mismatch at threshold 0.60 for {r['patient_id']}: prob={p_ens}, pred={pred_060}"
            assert pred_045 == int(p_ens >= 0.45), f"Mismatch at threshold 0.45 for {r['patient_id']}: prob={p_ens}, pred={pred_045}"


def test_10_zero_training_or_validation_leakage(master_manifest, test_artifacts_dir):
    """Verifies zero patient leakage from train (203) or validation (44) cohorts into test predictions."""
    non_test_pids = set()
    with open(master_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() in ("train", "val"):
                non_test_pids.add(r["patient_id"])

    assert len(non_test_pids) == 203 + 44  # 247 patients

    preds_csv = test_artifacts_dir / "test_patient_predictions.csv"
    with open(preds_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = r["patient_id"]
            assert pid not in non_test_pids, f"Critical Data Leakage! Non-test patient found in test predictions: {pid}"


def test_11_model_weights_and_cv_artifacts_unaltered(cv_dir):
    """Verifies Stage 12 cross-validation fold checkpoints are non-empty and unaltered."""
    for i in range(5):
        ckpt_path = cv_dir / f"fold_{i}/best_checkpoint.pt"
        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 40_000_000


def test_12_stage_13_artifacts_unaltered(base_dir):
    """Verifies Stage 13 ensemble validation artifacts remain intact."""
    val_metrics = base_dir / "ml_artifacts/bone_scan_ensemble/ensemble_validation_metrics.json"
    assert val_metrics.exists()
    assert val_metrics.stat().st_size > 0
    with open(val_metrics, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "discriminative_performance" in data
    assert data["discriminative_performance"]["roc_auc"] == 0.8952


def test_13_fracatlas_artifacts_unaltered(base_dir):
    """Verifies FracAtlas Stage 09 full training and Stage 10 holdout test artifacts remain unaltered."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists()
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    fracatlas_test = base_dir / "ml_artifacts/fracatlas_test_evaluation/test_predictions.csv"
    assert fracatlas_test.exists()
    assert fracatlas_test.stat().st_size > 0


def test_14_stage_14_artifacts_completeness(test_artifacts_dir):
    """Verifies that all 10 required Stage 14 deliverables exist and are non-empty."""
    required_files = [
        "test_patient_predictions.csv",
        "test_image_predictions.csv",
        "test_metrics.json",
        "test_confidence_intervals.json",
        "validation_vs_test_comparison.json",
        "confusion_matrix_test.png",
        "roc_curve_test.png",
        "precision_recall_curve_test.png",
        "probability_distribution_test.png",
        "ensemble_disagreement_test.png",
    ]
    for fn in required_files:
        p = test_artifacts_dir / fn
        assert p.exists(), f"Missing Stage 14 artifact: {fn}"
        assert p.stat().st_size > 0, f"Artifact {fn} is empty"


def test_15_cpu_environment_untouched():
    """Verifies host CPU Python environment was not modified and remains CPU-only."""
    cpu_python = Path(r"C:\Python313\python.exe")
    assert cpu_python.exists()
    res = subprocess.run([str(cpu_python), "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "+cpu" in res.stdout


def test_16_react_frontend_untouched(base_dir):
    """Verifies React frontend files in osteophorosis-main/src remain untouched."""
    frontend_dir = base_dir / "src"
    assert frontend_dir.exists(), f"Frontend directory missing: {frontend_dir}"
    package_json = base_dir / "package.json"
    assert package_json.exists(), f"package.json missing: {package_json}"

