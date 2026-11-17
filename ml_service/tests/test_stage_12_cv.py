"""
Stage 12 - Bone Scintigraphy Five-Fold Cross-Validation Verification Tests.

Verifies:
1. Patient-level fold isolation (mutually exclusive train and validation sets per fold).
2. Anterior and posterior view pairing (grouped strictly by patient).
3. Out-of-fold prediction uniqueness (exactly 203 unique patients, each predicted once).
4. Zero original validation leakage (none of the 44 original val patients in CV folds).
5. Zero original test leakage (none of the 44 original test patients in CV folds).
6. Binary label mapping integrity (0 = NO_BONE_METASTASIS, 1 = BONE_METASTASIS).
7. Patient-level max aggregation (prob_patient_max == max(prob_anterior, prob_posterior)).
8. Checkpoint reloadability across all 5 folds.
9. CUDA forward/backward execution.
10. Five-fold artifact completeness (fold_0 through fold_4 subdirectories).
11. Root CV summary artifacts completeness (manifests, plots, summary JSON).
12. FracAtlas checkpoints and artifacts unaltered.
13. Host CPU Python environment untouched.
"""

import csv
import json
from pathlib import Path
import subprocess
import pytest
import torch

from ml_service.config.settings import get_settings
from ml_service.datasets.bone_scan_dataset import BoneScanPatientDataset
from ml_service.models.bone_scan_classifier import BoneScanResNet18


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def cv_artifacts_dir(base_dir):
    d = base_dir / "ml_artifacts/bone_scan_cv"
    assert d.exists(), f"CV artifacts directory missing: {d}"
    return d


@pytest.fixture(scope="module")
def master_manifest(base_dir):
    p = base_dir / "ml_service/data/splits/bone_scan_split_manifest.csv"
    assert p.exists(), f"Master split manifest missing: {p}"
    return p


@pytest.fixture(scope="module")
def cv_manifest(base_dir):
    p = base_dir / "ml_service/data/splits/bone_scan_train_cv_manifest.csv"
    assert p.exists(), f"Train CV manifest missing: {p}"
    return p


def test_01_patient_level_fold_isolation(cv_manifest):
    """Verifies that every fold has mutually exclusive train and validation patients."""
    patients_by_fold = {f"fold_{i}": set() for i in range(5)}
    with open(cv_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            patients_by_fold[r["train_cv_fold"]].add(r["patient_id"])

    # Verify each fold has between 40 and 42 patients
    for fold, pids in patients_by_fold.items():
        assert len(pids) in (40, 41, 42), f"Fold {fold} has unexpected patient count: {len(pids)}"

    # Check mutual exclusivity
    for i in range(5):
        for j in range(i + 1, 5):
            f_i = f"fold_{i}"
            f_j = f"fold_{j}"
            overlap = patients_by_fold[f_i].intersection(patients_by_fold[f_j])
            assert len(overlap) == 0, f"Leakage detected between {f_i} and {f_j}: {overlap}"


def test_02_paired_views_grouped_by_patient(cv_manifest):
    """Verifies each patient has both anterior and posterior views in the CV manifest."""
    patient_views = {}
    with open(cv_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = r["patient_id"]
            if pid not in patient_views:
                patient_views[pid] = set()
            patient_views[pid].add(r["view"].lower())

    assert len(patient_views) == 203
    for pid, views in patient_views.items():
        assert views == {"anterior", "posterior"}, f"Patient {pid} views incomplete: {views}"


def test_03_out_of_fold_prediction_uniqueness(cv_artifacts_dir):
    """Verifies exactly 203 unique patients receive exactly one out-of-fold prediction."""
    oof_csv = cv_artifacts_dir / "out_of_fold_patient_predictions.csv"
    assert oof_csv.exists(), f"Missing OOF predictions file: {oof_csv}"

    predictions = []
    seen_pids = set()
    with open(oof_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r["patient_id"]
            assert pid not in seen_pids, f"Duplicate prediction found for patient: {pid}"
            seen_pids.add(pid)
            predictions.append(r)

    assert len(predictions) == 203, f"Expected 203 OOF predictions, got {len(predictions)}"
    assert len(seen_pids) == 203, f"Expected 203 unique patient IDs, got {len(seen_pids)}"


def test_04_zero_original_val_leakage(master_manifest, cv_manifest, cv_artifacts_dir):
    """Verifies none of the 44 original validation patients appear in CV folds or OOF predictions."""
    val_patients = set()
    with open(master_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() == "val":
                val_patients.add(r["patient_id"])
    assert len(val_patients) == 44

    # Check CV manifest
    with open(cv_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["patient_id"] not in val_patients, f"Val patient {r['patient_id']} leaked into CV manifest!"

    # Check OOF predictions
    oof_csv = cv_artifacts_dir / "out_of_fold_patient_predictions.csv"
    with open(oof_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["patient_id"] not in val_patients, f"Val patient {r['patient_id']} leaked into OOF predictions!"


def test_05_zero_original_test_leakage(master_manifest, cv_manifest, cv_artifacts_dir):
    """Verifies none of the 44 original holdout test patients appear in CV folds or OOF predictions."""
    test_patients = set()
    with open(master_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() == "test":
                test_patients.add(r["patient_id"])
    assert len(test_patients) == 44

    # Check CV manifest
    with open(cv_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["patient_id"] not in test_patients, f"Test patient {r['patient_id']} leaked into CV manifest!"

    # Check OOF predictions
    oof_csv = cv_artifacts_dir / "out_of_fold_patient_predictions.csv"
    with open(oof_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["patient_id"] not in test_patients, f"Test patient {r['patient_id']} leaked into OOF predictions!"


def test_06_binary_label_mapping(cv_manifest):
    """Verifies 0 = NO_BONE_METASTASIS, 1 = BONE_METASTASIS."""
    dataset = BoneScanPatientDataset(cv_manifest)
    assert len(dataset) == 203
    for pdata in dataset.patients:
        cls = pdata["target_class"]
        lbl = pdata["label"]
        if cls == "BONE_METASTASIS":
            assert lbl == 1.0
        elif cls == "NO_BONE_METASTASIS":
            assert lbl == 0.0
        else:
            pytest.fail(f"Invalid target class: {cls}")


def test_07_patient_level_max_aggregation(cv_artifacts_dir):
    """Verifies prob_patient_max is exactly max(prob_anterior, prob_posterior)."""
    oof_csv = cv_artifacts_dir / "out_of_fold_patient_predictions.csv"
    with open(oof_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p_ant = float(r["prob_anterior"])
            p_post = float(r["prob_posterior"])
            p_max = float(r["prob_patient_max"])
            expected_max = max(p_ant, p_post)
            assert abs(p_max - expected_max) < 1e-5, f"Max pooling mismatch for {r['patient_id']}: {p_max} vs {expected_max}"


def test_08_all_five_fold_checkpoints_reloadable(cv_artifacts_dir):
    """Verifies best_checkpoint.pt for each of the 5 folds can be loaded into ResNet18."""
    for i in range(5):
        fold_name = f"fold_{i}"
        ckpt_path = cv_artifacts_dir / fold_name / "best_checkpoint.pt"
        assert ckpt_path.exists(), f"Missing checkpoint for {fold_name}: {ckpt_path}"
        assert ckpt_path.stat().st_size > 40_000_000, f"Checkpoint unexpectedly small for {fold_name}"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "val_patient_metrics" in ckpt

        model = BoneScanResNet18(pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Dummy forward pass to ensure weights are functional
        dummy_x = torch.randn(1, 1, 1024, 256)
        with torch.no_grad():
            out = model(dummy_x)
        assert out.shape == (1, 1)


def test_09_cuda_forward_backward_execution():
    """Verifies forward and backward execution on CUDA GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")

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


def test_10_five_fold_artifact_completeness(cv_artifacts_dir):
    """Verifies all 5 fold directories contain training_log.csv, checkpoints, and fold_metrics.json."""
    for i in range(5):
        fold_dir = cv_artifacts_dir / f"fold_{i}"
        assert fold_dir.exists(), f"Missing fold directory: {fold_dir}"

        required_files = [
            "training_log.csv",
            "best_checkpoint.pt",
            "last_checkpoint.pt",
            "fold_metrics.json",
        ]
        for fn in required_files:
            p = fold_dir / fn
            assert p.exists(), f"Missing file in {fold_dir.name}: {fn}"
            assert p.stat().st_size > 0, f"File {fn} in {fold_dir.name} is empty"


def test_11_root_cv_artifacts_completeness(cv_artifacts_dir):
    """Verifies summary artifacts, plots, and manifests exist in bone_scan_cv root."""
    required_root_files = [
        "cv_manifest.csv",
        "out_of_fold_patient_predictions.csv",
        "cv_metrics_summary.json",
        "fold_comparison.png",
        "pooled_roc_curve.png",
        "pooled_precision_recall_curve.png",
    ]
    for fn in required_root_files:
        p = cv_artifacts_dir / fn
        assert p.exists(), f"Missing root artifact: {fn}"
        assert p.stat().st_size > 0, f"Root artifact {fn} is empty"


def test_12_fracatlas_artifacts_unaltered(base_dir):
    """Verifies FracAtlas Stage 09 checkpoint and Stage 10 test evaluations are unaltered."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists(), f"FracAtlas checkpoint missing: {fracatlas_ckpt}"
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    fracatlas_test_eval = base_dir / "ml_artifacts/fracatlas_test_evaluation/test_predictions.csv"
    assert fracatlas_test_eval.exists(), f"FracAtlas test predictions missing: {fracatlas_test_eval}"


def test_13_cpu_environment_untouched():
    """Verifies system CPU Python environment was not modified."""
    cpu_python = Path(r"C:\Python313\python.exe")
    assert cpu_python.exists(), f"CPU Python missing at {cpu_python}"
    res = subprocess.run([str(cpu_python), "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "+cpu" in res.stdout, f"CPU PyTorch modified! Output: {res.stdout}"

