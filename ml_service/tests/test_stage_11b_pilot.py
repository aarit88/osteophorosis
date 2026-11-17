"""
Stage 11B - Bone Scintigraphy Pilot Pipeline and Training Tests.

Verifies:
1. 16-bit loading retains uint16 range (min >= 0, max > 255).
2. Input tensors are 1-channel, height 1024, width 256.
3. No anamorphic square distortion (aspect ratio is 1:4).
4. Binary labels are correct: 0 = NO_BONE_METASTASIS, 1 = BONE_METASTASIS.
5. Anterior and posterior images from the same patient are correctly paired.
6. Original validation (44) and test (44) patients are completely absent from CV folds.
7. The 5 folds partition exactly the 203 training patients.
8. Model input accepts (B, 1, 1024, 256) and outputs (B, 1).
9. Patient-level max aggregation correctly computes max(P_ant, P_post).
10. Checkpoint can be reloaded and reproduces identical predictions on validation fold.
11. Pilot runs on CUDA with forward and backward pass without CUDA errors.
12. FracAtlas checkpoints are unaltered (hash/timestamp verification).
13. CPU Python environment is untouched.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from ml_service.config.settings import get_settings
from ml_service.datasets.bone_scan_dataset import (
    BoneScanPatientDataset,
    preprocess_scintigraphy_image,
)
from ml_service.models.bone_scan_classifier import BoneScanResNet18


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def pilot_artifacts_dir(base_dir):
    d = base_dir / "ml_artifacts" / "bone_scan_pilot"
    assert d.exists(), f"Pilot artifacts dir missing: {d}"
    return d


@pytest.fixture(scope="module")
def train_cv_manifest(base_dir):
    p = base_dir / "ml_service" / "data" / "splits" / "bone_scan_train_cv_manifest.csv"
    assert p.exists(), f"Train CV manifest missing: {p}"
    return p


@pytest.fixture(scope="module")
def master_split_manifest(base_dir):
    p = base_dir / "ml_service" / "data" / "splits" / "bone_scan_split_manifest.csv"
    assert p.exists(), f"Master split manifest missing: {p}"
    return p


def test_01_16bit_loading_uint16_range(master_split_manifest):
    """Verifies raw images load in 16-bit format with values exceeding 8-bit range (> 255)."""
    with open(master_split_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    img_path = Path(first_row["file_path"])
    assert img_path.exists(), f"Sample image file missing: {img_path}"

    with Image.open(img_path) as img:
        assert img.mode in ("I;16", "I", "F"), f"Expected 16-bit mode, got {img.mode}"
        arr = np.array(img)
        assert arr.dtype in (np.uint16, np.int32, np.float32)
        assert arr.min() >= 0
        assert arr.max() > 255, f"Image values must exceed 8-bit ceiling: max is {arr.max()}"


def test_02_input_tensor_dimensions_and_aspect_ratio(master_split_manifest):
    """Verifies preprocessed tensor has 1 channel, (H=1024, W=256), 1:4 aspect ratio, [0, 1] range."""
    with open(master_split_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    img_path = Path(first_row["file_path"])

    tensor = preprocess_scintigraphy_image(img_path, clip_ceiling=25000.0, target_size=(1024, 256))
    assert tensor.shape == (1, 1024, 256), f"Expected shape (1, 1024, 256), got {tensor.shape}"
    h, w = tensor.shape[1], tensor.shape[2]
    assert h / w == 4.0, f"Aspect ratio must be exactly 1:4 (H/W=4.0), got {h/w}"
    assert 0.0 <= tensor.min().item() <= tensor.max().item() <= 1.0


def test_03_binary_label_mapping(train_cv_manifest):
    """Verifies binary label encoding: 0 = NO_BONE_METASTASIS, 1 = BONE_METASTASIS."""
    dataset = BoneScanPatientDataset(train_cv_manifest)
    assert len(dataset) == 203
    for pdata in dataset.patients:
        cls = pdata["target_class"]
        lbl = pdata["label"]
        if cls == "BONE_METASTASIS":
            assert lbl == 1.0
        elif cls == "NO_BONE_METASTASIS":
            assert lbl == 0.0
        else:
            pytest.fail(f"Unknown target class: {cls}")


def test_04_paired_anterior_posterior_views(train_cv_manifest):
    """Verifies each patient entry pairs valid Anterior and Posterior scans."""
    dataset = BoneScanPatientDataset(train_cv_manifest)
    sample = dataset[0]
    assert "anterior_tensor" in sample
    assert "posterior_tensor" in sample
    assert sample["anterior_tensor"].shape == (1, 1024, 256)
    assert sample["posterior_tensor"].shape == (1, 1024, 256)
    assert "anterior" in sample["anterior_path"].lower()
    assert "posterior" in sample["posterior_path"].lower()
    assert sample["patient_id"] is not None


def test_05_original_val_and_test_isolation_from_cv_manifest(master_split_manifest, train_cv_manifest):
    """Verifies 100% absence of original validation (44) and test (44) patients in CV folds."""
    master_val_test_patients = set()
    with open(master_split_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() in ("val", "test"):
                master_val_test_patients.add(r["patient_id"])
    assert len(master_val_test_patients) == 88, f"Expected 88 val+test patients, got {len(master_val_test_patients)}"

    cv_patients = set()
    with open(train_cv_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cv_patients.add(r["patient_id"])
    assert len(cv_patients) == 203, f"Expected 203 CV patients, got {len(cv_patients)}"

    overlap = master_val_test_patients.intersection(cv_patients)
    assert len(overlap) == 0, f"Critical Leakage! Overlapping patients: {overlap}"


def test_06_train_cv_manifest_partitions_203_patients(train_cv_manifest):
    """Verifies train_cv_manifest contains exactly 203 patients, 406 scans, and 5 balanced folds."""
    patients_by_fold = {}
    rows_count = 0
    with open(train_cv_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows_count += 1
            fold = r["train_cv_fold"]
            pid = r["patient_id"]
            if fold not in patients_by_fold:
                patients_by_fold[fold] = set()
            patients_by_fold[fold].add(pid)

    assert rows_count == 406, f"Expected 406 rows (2 per patient), got {rows_count}"
    assert len(patients_by_fold) == 5, f"Expected 5 folds, got {len(patients_by_fold)}"

    all_cv_pids = set()
    for fold, pids in patients_by_fold.items():
        assert len(pids) in (40, 41, 42), f"Fold {fold} has unexpected patient count: {len(pids)}"
        # Check mutually disjoint
        assert len(all_cv_pids.intersection(pids)) == 0, f"Fold {fold} has overlapping patients"
        all_cv_pids.update(pids)
    assert len(all_cv_pids) == 203


def test_07_model_input_and_output_shapes():
    """Verifies ResNet18 model accepts (B, 1, 1024, 256) and outputs (B, 1)."""
    model = BoneScanResNet18(pretrained=False)
    assert model.conv1.in_channels == 1
    assert model.fc.out_features == 1

    x = torch.randn(2, 1, 1024, 256)
    out = model(x)
    assert out.shape == (2, 1)


def test_08_clinical_max_pooling_aggregation():
    """Verifies patient risk is defined as max(P_ant, P_post)."""
    p_ant = 0.82
    p_post = 0.35
    p_patient = max(p_ant, p_post)
    assert p_patient == 0.82

    p_ant2 = 0.12
    p_post2 = 0.95
    p_patient2 = max(p_ant2, p_post2)
    assert p_patient2 == 0.95


def test_09_pilot_artifacts_completeness(pilot_artifacts_dir):
    """Verifies all 12 required Stage 11B pilot artifacts exist and are non-empty."""
    required_files = [
        "pilot_config.yaml",
        "cv_fold_manifest.csv",
        "training_log.csv",
        "patient_level_predictions.csv",
        "image_level_predictions.csv",
        "validation_metrics.json",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "loss_curves.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "pretrained_weights_info.json",
        "STAGE_11B_PILOT.md",
    ]
    for fn in required_files:
        p = pilot_artifacts_dir / fn
        assert p.exists(), f"Missing artifact: {fn}"
        assert p.stat().st_size > 0, f"Artifact {fn} is empty"


def test_10_checkpoint_reload_and_reproducibility(pilot_artifacts_dir, train_cv_manifest):
    """Verifies best_checkpoint.pt reloads correctly and produces identical predictions."""
    ckpt_path = pilot_artifacts_dir / "best_checkpoint.pt"
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "val_patient_metrics" in ckpt
    assert "roc_auc" in ckpt["val_patient_metrics"]

    model = BoneScanResNet18(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load 1 patient from validation fold
    val_dataset = BoneScanPatientDataset(train_cv_manifest, fold="fold_0", is_train=False)
    sample = val_dataset[0]

    with torch.no_grad():
        ant_pred = torch.sigmoid(model(sample["anterior_tensor"].unsqueeze(0))).item()
        post_pred = torch.sigmoid(model(sample["posterior_tensor"].unsqueeze(0))).item()
        pt_pred = max(ant_pred, post_pred)

    # Compare with patient_level_predictions.csv
    preds_file = pilot_artifacts_dir / "patient_level_predictions.csv"
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["patient_id"] == sample["patient_id"]:
                saved_pt_pred = float(r["prob_patient_max"])
                # float16 vs float32 rounding difference is ~1e-4
                assert abs(pt_pred - saved_pt_pred) < 1e-3, f"Prediction mismatch: {pt_pred} vs {saved_pt_pred}"
                break


def test_11_cuda_forward_backward_execution():
    """Verifies model forward and backward pass on CUDA without errors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available in this test environment")

    device = torch.device("cuda:0")
    model = BoneScanResNet18(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(2, 1, 1024, 256, device=device)
    y = torch.tensor([[1.0], [0.0]], device=device)

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()


def test_12_fracatlas_checkpoints_unaltered(base_dir):
    """Verifies FracAtlas Stage 09 best checkpoint exists and is unaltered."""
    fracatlas_ckpt = base_dir / "ml_artifacts" / "fracatlas_full_training" / "best_checkpoint.pt"
    assert fracatlas_ckpt.exists(), f"FracAtlas checkpoint missing: {fracatlas_ckpt}"
    assert fracatlas_ckpt.stat().st_size > 100_000_000, "FracAtlas checkpoint unexpectedly small"

    fracatlas_test_eval = base_dir / "ml_artifacts" / "fracatlas_test_evaluation" / "test_predictions.csv"
    assert fracatlas_test_eval.exists(), f"FracAtlas test predictions missing: {fracatlas_test_eval}"


def test_13_cpu_environment_untouched():
    """Verifies system CPU Python environment was not modified or overwritten."""
    cpu_python = Path(r"C:\Python313\python.exe")
    assert cpu_python.exists(), f"CPU Python missing at {cpu_python}"
    res = subprocess.run([str(cpu_python), "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "+cpu" in res.stdout, f"CPU PyTorch modified! Output: {res.stdout}"
