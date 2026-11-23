"""
Stage 18 - BTXRD Final Model Freeze and External Evaluation Readiness Audit Tests.

Verifies:
1. Original Stage 17 best checkpoint unchanged and valid SHA-256.
2. Frozen copy in ml_artifacts/btxrd_frozen_model/ has identical SHA-256 hash.
3. Completeness of all 6 frozen model artifacts.
4. Authoritative three-class mapping consistency.
5. Model architecture integrity, parameter counts, and reload without error.
6. Frozen inference engine predict_image output contract and probability sum.
7. Validation metrics exact reproduction audit (Macro-F1: 0.8103, Brier: 0.3168).
8. Multi-center boundaries: Center 2 (549) and Center 3 (259) sealed holdouts.
9. Epidemiological documentation of Center 3 low normal prevalence (27 normal).
10. Conflict quarantine enforcement (all 4 duplicate conflicts excluded).
11. Calibration status verified as uncalibrated raw softmax.
12. Preservation of FracAtlas checkpoints and reports.
13. Preservation of Bone Scintigraphy checkpoints and test metrics.
14. Preservation of Stage 16 pilot and Stage 17 full training artifacts.
15. Preservation of host CPU Python environment and React frontend.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import pytest
import torch

from ml_service.config.settings import get_settings
from ml_service.datasets.btxrd_dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from ml_service.inference.btxrd_inference import (
    EXPECTED_CHECKPOINT_SHA256,
    FROZEN_MODEL_VERSION,
    BTXRDFrozenClassifier,
    audit_saved_validation_predictions,
    calculate_file_sha256,
)
from ml_service.models.btxrd_classifier import BTXRDResNet34


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def original_checkpoint(base_dir):
    p = base_dir / "ml_artifacts/btxrd_full_training/best_checkpoint.pt"
    assert p.exists(), f"Original checkpoint missing: {p}"
    return p


@pytest.fixture(scope="module")
def frozen_model_dir(base_dir):
    d = base_dir / "ml_artifacts/btxrd_frozen_model"
    assert d.exists(), f"Frozen model directory missing: {d}"
    return d


@pytest.fixture(scope="module")
def btxrd_manifest(base_dir):
    p = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    assert p.exists(), f"BTXRD split manifest missing: {p}"
    return p


def test_01_original_checkpoint_exists_and_hash(original_checkpoint):
    """Verifies original Stage 17 checkpoint exists, is non-empty, and has expected SHA-256."""
    assert original_checkpoint.stat().st_size > 200_000_000
    actual_hash = calculate_file_sha256(original_checkpoint)
    assert actual_hash == EXPECTED_CHECKPOINT_SHA256, f"Hash mismatch: {actual_hash}"


def test_02_frozen_copy_hash_match(frozen_model_dir, original_checkpoint):
    """Verifies copied checkpoint in frozen directory matches original SHA-256 bit-for-bit."""
    frozen_ckpt = frozen_model_dir / "best_checkpoint.pt"
    assert frozen_ckpt.exists()
    frozen_hash = calculate_file_sha256(frozen_ckpt)
    original_hash = calculate_file_sha256(original_checkpoint)
    assert frozen_hash == original_hash
    assert frozen_hash == EXPECTED_CHECKPOINT_SHA256

    hash_txt = frozen_model_dir / "checkpoint_sha256.txt"
    assert hash_txt.exists()
    assert hash_txt.read_text(encoding="utf-8").strip() == EXPECTED_CHECKPOINT_SHA256


def test_03_frozen_model_directory_manifest_completeness(frozen_model_dir):
    """Verifies all 6 required files exist in ml_artifacts/btxrd_frozen_model/."""
    required_files = [
        "best_checkpoint.pt",
        "model_config.json",
        "preprocessing_config.json",
        "class_mapping.json",
        "checkpoint_sha256.txt",
        "model_freeze_manifest.json",
    ]
    for fname in required_files:
        fpath = frozen_model_dir / fname
        assert fpath.exists(), f"Required frozen model file missing: {fname}"
        assert fpath.stat().st_size > 0, f"File is empty: {fname}"


def test_04_class_mapping_consistency(frozen_model_dir):
    """Verifies authoritative 3-class mapping in config and code."""
    mapping_file = frozen_model_dir / "class_mapping.json"
    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["class_to_idx"]["NORMAL_BONE_RADIOGRAPH"] == 0
    assert data["class_to_idx"]["BENIGN_BONE_TUMOR"] == 1
    assert data["class_to_idx"]["PRIMARY_MALIGNANT_TUMOR"] == 2
    assert data["num_classes"] == 3


def test_05_model_architecture_and_reload(frozen_model_dir):
    """Verifies clean model instantiation and weight reload from frozen directory."""
    frozen_ckpt = frozen_model_dir / "best_checkpoint.pt"
    ckpt = torch.load(frozen_ckpt, map_location="cpu")
    assert ckpt["epoch"] == 18

    model = BTXRDResNet34(num_classes=3, pretrained=False, dropout_p=0.4, freeze_early=True)
    model.unfreeze_layer3()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = model.get_model_metadata()
    assert meta["trainable_parameters"] == 19_938_307
    assert meta["frozen_parameters"] == 1_347_904
    assert meta["total_parameters"] == 21_286_211

    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 3)
    assert not torch.isnan(out).any()


def test_06_frozen_classifier_predict_single_image(frozen_model_dir, tmp_path):
    """Verifies BTXRDFrozenClassifier prediction API contract on single image."""
    test_img = Image.new("RGB", (600, 400), color=(128, 128, 128))
    img_path = tmp_path / "sample_xray.png"
    test_img.save(img_path)

    classifier = BTXRDFrozenClassifier(device=torch.device("cpu"), verify_sha256=True)
    res = classifier.predict_image(img_path)

    required_keys = [
        "normal_probability",
        "benign_probability",
        "malignant_probability",
        "predicted_class_idx",
        "predicted_class_name",
        "model_version",
        "checkpoint_sha256",
        "calibration_status",
    ]
    for k in required_keys:
        assert k in res, f"Missing key: {k}"

    assert res["model_version"] == FROZEN_MODEL_VERSION
    assert res["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert res["calibration_status"] == "uncalibrated_raw_softmax"
    assert res["predicted_class_idx"] in [0, 1, 2]
    assert res["predicted_class_name"] in ["NORMAL_BONE_RADIOGRAPH", "BENIGN_BONE_TUMOR", "PRIMARY_MALIGNANT_TUMOR"]

    prob_sum = res["normal_probability"] + res["benign_probability"] + res["malignant_probability"]
    assert abs(prob_sum - 1.0) < 1e-4


def test_07_validation_performance_reproduction_audit(base_dir):
    """Verifies exact reproduction of all Stage 17 validation metrics."""
    res = audit_saved_validation_predictions()
    assert res["sample_count"] == 588
    assert res["macro_f1"] == 0.8103
    assert res["macro_pr_auc"] == 0.87
    assert res["balanced_accuracy"] == 0.8290
    assert res["overall_accuracy"] == 0.8180
    assert res["macro_roc_auc_ovr"] == 0.9114
    assert res["brier_score"] == 0.3168
    assert res["confusion_matrix"] == [[290, 26, 3], [65, 154, 9], [1, 3, 37]]

    # Malignant metrics
    mal_metrics = res["per_class_metrics"]["PRIMARY_MALIGNANT_TUMOR"]
    assert mal_metrics["support"] == 41
    assert mal_metrics["sensitivity_recall"] == 0.9024
    assert mal_metrics["specificity"] == 0.9781
    assert mal_metrics["precision_ppv"] == 0.7551
    assert mal_metrics["f1_score"] == 0.8222


def test_08_center_2_and_center_3_metadata_and_boundaries(btxrd_manifest, base_dir):
    """Verifies multi-center metadata without running evaluation on external holdouts."""
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    c2_rows = [r for r in rows if r["center"].lower() == "center_2"]
    c3_rows = [r for r in rows if r["center"].lower() == "center_3"]

    assert len(c2_rows) == 549, f"Center 2 count expected 549, got {len(c2_rows)}"
    assert len(c3_rows) == 259, f"Center 3 count expected 259, got {len(c3_rows)}"

    # Ensure no external evaluation results exist yet
    c2_eval_dir = base_dir / "ml_artifacts/btxrd_center2_evaluation"
    c3_eval_dir = base_dir / "ml_artifacts/btxrd_center3_evaluation"
    assert not c2_eval_dir.exists(), "Center 2 evaluation directory should not exist in Stage 18!"
    assert not c3_eval_dir.exists(), "Center 3 evaluation directory should not exist in Stage 18!"


def test_09_center_3_low_normal_prevalence_documentation(btxrd_manifest, frozen_model_dir):
    """Verifies Center 3 has exactly 27 normal radiographs and is documented in manifest."""
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    c3_normals = [r for r in rows if r["center"].lower() == "center_3" and r["target_class"] == "NORMAL_BONE_RADIOGRAPH"]
    assert len(c3_normals) == 27

    manifest_file = frozen_model_dir / "model_freeze_manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        m_data = json.load(f)

    c3_target = m_data["external_evaluation_targets"]["center_3"]
    assert c3_target["class_distribution"]["NORMAL_BONE_RADIOGRAPH"] == 27
    assert "epidemiological_notes" in c3_target


def test_10_conflict_quarantine_enforcement(btxrd_manifest):
    """Verifies all 4 conflicting duplicate images remain quarantined with conflict_flag=TRUE."""
    expected_conflicts = {"IMG000591.jpeg", "IMG003456.jpeg", "IMG000707.jpeg", "IMG003346.jpeg"}
    actual_conflicts = set()

    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("conflict_flag", "").strip().lower() == "true":
                actual_conflicts.add(row["image_id"])

    assert actual_conflicts == expected_conflicts


def test_11_calibration_policy_audit(frozen_model_dir):
    """Verifies that probability calibration policy is documented as raw uncalibrated softmax."""
    manifest_file = frozen_model_dir / "model_freeze_manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        m_data = json.load(f)

    cal_status = m_data["inference_specification"]["probability_calibration_status"]
    assert "uncalibrated" in cal_status.lower()
    assert "0.3168" in cal_status


def test_12_fracatlas_artifacts_preservation(base_dir):
    """Verifies FracAtlas Stage 09 checkpoint and Stage 10 report remain intact."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists()
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    fracatlas_rep = base_dir / "ml_artifacts/stage_reports/STAGE_10_FRACATLAS_HOLDOUT_EVALUATION.md"
    assert fracatlas_rep.exists()


def test_13_bone_scintigraphy_artifacts_preservation(base_dir):
    """Verifies Bone Scintigraphy fold 0 checkpoint and test metrics remain intact."""
    scint_fold0 = base_dir / "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"
    assert scint_fold0.exists()

    scint_test_eval = base_dir / "ml_artifacts/bone_scan_test_evaluation/test_metrics.json"
    assert scint_test_eval.exists()


def test_14_stage_16_and_17_artifacts_preservation(base_dir):
    """Verifies Stage 16 pilot checkpoint and all Stage 17 deliverables remain untouched."""
    pilot_ckpt = base_dir / "ml_artifacts/btxrd_pilot/best_checkpoint.pt"
    assert pilot_ckpt.exists()

    stage17_dir = base_dir / "ml_artifacts/btxrd_full_training"
    assert (stage17_dir / "best_checkpoint.pt").exists()
    assert (stage17_dir / "training_log.csv").exists()
    assert (stage17_dir / "validation_metrics.json").exists()
    assert (stage17_dir / "validation_predictions.csv").exists()


def test_15_cpu_python_and_frontend_preservation(base_dir):
    """Verifies host CPU Python and React frontend are completely intact."""
    cpu_py = Path(r"C:\Python313\python.exe")
    assert cpu_py.exists()
    cmd = [str(cpu_py), "-c", "import torch; print(torch.__version__)"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "+cpu" in res.stdout

    frontend_file = base_dir / "src/App.tsx"
    assert frontend_file.exists()

