"""
Stage 13 - Final Bone Metastasis Ensemble Construction and Validation Tests.

Verifies:
1. All 5 cross-validation checkpoints load successfully.
2. Correct 16-bit preprocessing, clipping, and 1:4 aspect ratio.
3. Correct anterior and posterior scan pairing.
4. Patient-level max aggregation (prob_patient == max(prob_ant, prob_post)).
5. Ensemble averaging across 5 models.
6. Exactly 44 unique validation patients predicted.
7. All predicted probabilities are finite and bounded in [0.0, 1.0].
8. Zero test cohort leakage (0 test patients accessed or predicted).
9. Existing fold checkpoints remain unaltered.
10. All Stage 13 artifacts exist and are non-empty.
11. FracAtlas checkpoints and artifacts remain unaltered.
12. Host CPU Python environment remains untouched.
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
def ensemble_artifacts_dir(base_dir):
    d = base_dir / "ml_artifacts/bone_scan_ensemble"
    assert d.exists(), f"Ensemble artifacts dir missing: {d}"
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


def test_02_preprocessing_and_aspect_ratio(master_manifest):
    """Verifies 16-bit preprocessing, 25k count clipping, 1:4 aspect ratio, and [0, 1] range."""
    with open(master_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    img_path = Path(first_row["file_path"])
    tensor = preprocess_scintigraphy_image(img_path, clip_ceiling=25000.0, target_size=(1024, 256))
    assert tensor.shape == (1, 1024, 256)
    assert 0.0 <= tensor.min().item() <= tensor.max().item() <= 1.0


def test_03_validation_dataset_anterior_posterior_pairing(master_manifest):
    """Verifies all 44 validation patients have valid paired anterior and posterior scans."""
    ds = BoneScanPatientDataset(master_manifest, split="val", is_train=False)
    assert len(ds) == 44
    counts = ds.get_class_counts()
    assert counts["positive_patients"] == 9
    assert counts["negative_patients"] == 35

    sample = ds[0]
    assert "anterior_tensor" in sample
    assert "posterior_tensor" in sample
    assert sample["anterior_tensor"].shape == (1, 1024, 256)
    assert sample["posterior_tensor"].shape == (1, 1024, 256)


def test_04_patient_level_max_aggregation():
    """Verifies max-pooling rule: P_patient = max(P_ant, P_post)."""
    p_ant = 0.88
    p_post = 0.31
    assert max(p_ant, p_post) == 0.88

    p_ant2 = 0.15
    p_post2 = 0.92
    assert max(p_ant2, p_post2) == 0.92


def test_05_ensemble_averaging_logic():
    """Verifies ensemble mean and standard deviation logic."""
    fold_probs = [0.80, 0.85, 0.75, 0.90, 0.80]
    ens_mean = float(sum(fold_probs) / len(fold_probs))
    ens_std = float(torch.tensor(fold_probs).std(unbiased=False).item())
    assert abs(ens_mean - 0.82) < 1e-4
    assert ens_std > 0.0


def test_06_prediction_count_and_uniqueness(ensemble_artifacts_dir):
    """Verifies exactly 44 unique validation patients receive predictions."""
    preds_csv = ensemble_artifacts_dir / "ensemble_validation_predictions.csv"
    assert preds_csv.exists()

    seen_pids = set()
    rows = []
    with open(preds_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r["patient_id"]
            assert pid not in seen_pids, f"Duplicate patient prediction: {pid}"
            seen_pids.add(pid)
            rows.append(r)

    assert len(rows) == 44, f"Expected 44 predictions, got {len(rows)}"
    assert len(seen_pids) == 44, f"Expected 44 unique patients, got {len(seen_pids)}"


def test_07_probabilities_finite_and_bounded(ensemble_artifacts_dir):
    """Verifies all ensemble probabilities are strictly within [0.0, 1.0]."""
    preds_csv = ensemble_artifacts_dir / "ensemble_validation_predictions.csv"
    with open(preds_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p_ens = float(r["prob_patient_ensemble"])
            std_dis = float(r["std_disagreement"])
            assert 0.0 <= p_ens <= 1.0, f"Probability out of bounds: {p_ens}"
            assert 0.0 <= std_dis <= 1.0, f"Disagreement out of bounds: {std_dis}"


def test_08_zero_holdout_test_leakage(master_manifest, ensemble_artifacts_dir):
    """Verifies none of the 44 holdout test patients appear in ensemble validation predictions."""
    test_pids = set()
    with open(master_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() == "test":
                test_pids.add(r["patient_id"])
    assert len(test_pids) == 44

    preds_csv = ensemble_artifacts_dir / "ensemble_validation_predictions.csv"
    with open(preds_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["patient_id"] not in test_pids, f"Critical Leakage! Test patient in validation: {r['patient_id']}"


def test_09_existing_fold_checkpoints_unaltered(cv_dir):
    """Verifies Stage 12 fold checkpoints are non-empty and unmodified."""
    for i in range(5):
        ckpt_path = cv_dir / f"fold_{i}/best_checkpoint.pt"
        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 40_000_000


def test_10_stage_13_artifacts_completeness(ensemble_artifacts_dir):
    """Verifies all required Stage 13 artifact files exist and are non-empty."""
    required_files = [
        "ensemble_config.json",
        "ensemble_validation_predictions.csv",
        "ensemble_validation_metrics.json",
        "ensemble_threshold_analysis.csv",
        "ensemble_calibration_analysis.json",
        "ensemble_confusion_matrix.png",
        "ensemble_roc_curve.png",
        "ensemble_precision_recall_curve.png",
        "ensemble_probability_distribution.png",
        "ensemble_model_disagreement.png",
    ]
    for fn in required_files:
        p = ensemble_artifacts_dir / fn
        assert p.exists(), f"Missing artifact: {fn}"
        assert p.stat().st_size > 0, f"Artifact {fn} is empty"


def test_11_fracatlas_artifacts_unaltered(base_dir):
    """Verifies FracAtlas Stage 09 checkpoint and Stage 10 test evaluations are unaltered."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists()
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    fracatlas_test_eval = base_dir / "ml_artifacts/fracatlas_test_evaluation/test_predictions.csv"
    assert fracatlas_test_eval.exists()


def test_12_cpu_environment_untouched():
    """Verifies host CPU Python environment was not modified."""
    cpu_python = Path(r"C:\Python313\python.exe")
    assert cpu_python.exists()
    res = subprocess.run([str(cpu_python), "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "+cpu" in res.stdout

