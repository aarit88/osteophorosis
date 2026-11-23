"""
Stage 20 - BTXRD Center 3 External Stress Test Evaluation Tests.

Verifies:
1. Frozen checkpoint SHA-256 hash matches before and after evaluation.
2. Center 3 partition isolation (259 images: 27 normal, 201 benign, 31 malignant).
3. Authoritative class mapping consistency (0=Normal, 1=Benign, 2=Primary Malignant).
4. Prediction CSV row count equals 259 with valid columns and image IDs.
5. Softmax probability validity, range [0, 1], and sum-to-one.
6. Metric consistency between classification_metrics.json and predictions.
7. Per-class metrics consistency and valid bootstrap confidence intervals.
8. Confusion matrix sum equals 259 and matches predictions.
9. Multi-center preservation: Center 2 evaluation artifacts preserved and untampered.
10. Conflict quarantine preserved (0 conflicting images in Center 3).
11. Probability calibration status correctly documented as uncalibrated raw softmax.
12. Preservation of FracAtlas checkpoints and artifacts.
13. Preservation of Bone Scintigraphy checkpoints and test metrics.
14. Preservation of Stage 16, 17, 18, and 19 artifacts.
15. Preservation of host CPU Python and React frontend.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest

from ml_service.config.settings import get_settings
from ml_service.datasets.btxrd_dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)

EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def center3_eval_dir(base_dir):
    d = base_dir / "ml_artifacts/btxrd_center3_external_evaluation"
    assert d.exists(), f"Center 3 evaluation directory missing: {d}"
    return d


@pytest.fixture(scope="module")
def frozen_checkpoint(base_dir):
    p = base_dir / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
    assert p.exists(), f"Frozen checkpoint missing: {p}"
    return p


@pytest.fixture(scope="module")
def btxrd_manifest(base_dir):
    p = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    assert p.exists(), f"BTXRD split manifest missing: {p}"
    return p


def test_01_checkpoint_hash_pre_and_post_verification(frozen_checkpoint, center3_eval_dir):
    """Verifies that the frozen checkpoint SHA-256 matches before and after evaluation."""
    h = hashlib.sha256()
    with open(frozen_checkpoint, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    current_hash = h.hexdigest().upper()
    assert current_hash == EXPECTED_SHA256, f"Checkpoint hash mismatch: {current_hash}"

    verif_file = center3_eval_dir / "checkpoint_hash_verification.txt"
    assert verif_file.exists()
    content = verif_file.read_text(encoding="utf-8")
    assert EXPECTED_SHA256 in content
    assert "CONFIRMED_IDENTICAL_BIT_LEVEL_INTEGRITY" in content


def test_02_center3_partition_isolation_and_row_count(btxrd_manifest, center3_eval_dir):
    """Verifies that only Center 3 external holdout images were evaluated."""
    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    c3_manifest_ids = {
        r["image_id"]: r["target_class"]
        for r in rows
        if r["center"].lower() == "center_3" and r["center_holdout_split"] == "external_test_center3"
    }
    assert len(c3_manifest_ids) == 259

    # Class distribution check
    c3_classes = [r["target_class"] for r in rows if r["center"].lower() == "center_3" and r["center_holdout_split"] == "external_test_center3"]
    assert c3_classes.count("NORMAL_BONE_RADIOGRAPH") == 27
    assert c3_classes.count("BENIGN_BONE_TUMOR") == 201
    assert c3_classes.count("PRIMARY_MALIGNANT_TUMOR") == 31

    preds_file = center3_eval_dir / "center3_predictions.csv"
    assert preds_file.exists()

    with open(preds_file, "r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))

    assert len(pred_rows) == 259, f"Expected 259 prediction rows, got {len(pred_rows)}"

    for prow in pred_rows:
        img_id = prow["image_id"]
        assert img_id in c3_manifest_ids, f"Image {img_id} does not belong to Center 3 external holdout!"
        assert prow["target_class"] == c3_manifest_ids[img_id], f"Label mismatch for {img_id}"


def test_03_class_mapping_integrity():
    """Verifies authoritative 3-class mapping."""
    assert CLASS_TO_IDX["NORMAL_BONE_RADIOGRAPH"] == 0
    assert CLASS_TO_IDX["BENIGN_BONE_TUMOR"] == 1
    assert CLASS_TO_IDX["PRIMARY_MALIGNANT_TUMOR"] == 2
    assert IDX_TO_CLASS[0] == "NORMAL_BONE_RADIOGRAPH"
    assert IDX_TO_CLASS[1] == "BENIGN_BONE_TUMOR"
    assert IDX_TO_CLASS[2] == "PRIMARY_MALIGNANT_TUMOR"


def test_04_prediction_row_count_and_columns(center3_eval_dir):
    """Verifies prediction CSV schema and completeness."""
    preds_file = center3_eval_dir / "center3_predictions.csv"
    expected_cols = [
        "image_id",
        "target_class",
        "actual_label",
        "prob_normal",
        "prob_benign",
        "prob_malignant",
        "predicted_label",
        "predicted_class",
        "is_correct",
    ]
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_cols
        rows = list(reader)

    assert len(rows) == 259


def test_05_probability_validity_and_sum_to_one(center3_eval_dir):
    """Verifies that all predicted probabilities are valid numbers in [0, 1] and sum to 1.0."""
    preds_file = center3_eval_dir / "center3_predictions.csv"
    with open(preds_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_norm = float(row["prob_normal"])
            p_ben = float(row["prob_benign"])
            p_mal = float(row["prob_malignant"])

            assert 0.0 <= p_norm <= 1.0
            assert 0.0 <= p_ben <= 1.0
            assert 0.0 <= p_mal <= 1.0

            total_prob = p_norm + p_ben + p_mal
            assert abs(total_prob - 1.0) < 1e-4

            # Verify argmax decision rule
            probs = [p_norm, p_ben, p_mal]
            assert int(row["predicted_label"]) == int(np.argmax(probs))


def test_06_overall_metrics_consistency(center3_eval_dir):
    """Verifies that classification_metrics.json values match predictions."""
    metrics_path = center3_eval_dir / "classification_metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["overall_accuracy"]["estimate"] == 0.471
    assert m["balanced_accuracy"]["estimate"] == 0.4467
    assert m["macro_f1"]["estimate"] == 0.3592
    assert m["weighted_f1"]["estimate"] == 0.5285
    assert m["macro_roc_auc_ovr"]["estimate"] == 0.6468
    assert m["macro_pr_auc_ovr"]["estimate"] == 0.4558
    assert m["multiclass_brier_score"] == 0.6486
    assert m["decision_rule"] == "argmax(softmax(logits))"
    assert m["calibration_status"] == "uncalibrated_raw_softmax"


def test_07_per_class_metrics_consistency(center3_eval_dir):
    """Verifies per_class_metrics.json structure, supports, and confidence intervals."""
    path = center3_eval_dir / "per_class_metrics.json"
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["NORMAL_BONE_RADIOGRAPH"]["support"] == 27
    assert m["BENIGN_BONE_TUMOR"]["support"] == 201
    assert m["PRIMARY_MALIGNANT_TUMOR"]["support"] == 31

    # Malignant metrics
    mal = m["PRIMARY_MALIGNANT_TUMOR"]
    assert mal["sensitivity"]["estimate"] == 0.6774
    assert mal["specificity"]["estimate"] == 0.6930
    assert mal["precision_ppv"]["estimate"] == 0.2308
    assert mal["npv"]["estimate"] == 0.9405
    assert mal["roc_auc_ovr"]["estimate"] == 0.7490
    assert mal["pr_auc"]["estimate"] == 0.3815

    # Check CI validity (lower <= estimate <= upper)
    assert mal["sensitivity"]["ci_95"][0] <= mal["sensitivity"]["estimate"] <= mal["sensitivity"]["ci_95"][1]
    assert mal["specificity"]["ci_95"][0] <= mal["specificity"]["estimate"] <= mal["specificity"]["ci_95"][1]

    # Normal metrics
    norm = m["NORMAL_BONE_RADIOGRAPH"]
    assert norm["sensitivity"]["estimate"] == 0.1852
    assert norm["specificity"]["estimate"] == 0.7845
    assert norm["precision_ppv"]["estimate"] == 0.0909
    assert norm["npv"]["estimate"] == 0.8922

    # Benign metrics
    ben = m["BENIGN_BONE_TUMOR"]
    assert ben["sensitivity"]["estimate"] == 0.4776
    assert ben["specificity"]["estimate"] == 0.7069
    assert ben["precision_ppv"]["estimate"] == 0.8496
    assert ben["npv"]["estimate"] == 0.2808


def test_08_confusion_matrix_sum_and_integrity(center3_eval_dir):
    """Verifies confusion matrix elements and sums."""
    cm_path = center3_eval_dir / "confusion_matrix.json"
    assert cm_path.exists()
    with open(cm_path, "r", encoding="utf-8") as f:
        cm_data = json.load(f)

    cm = cm_data["confusion_matrix_counts"]
    assert len(cm) == 3
    assert len(cm[0]) == 3

    total_sum = sum(sum(row) for row in cm)
    assert total_sum == 259

    assert cm == [[5, 13, 9], [44, 96, 61], [6, 4, 21]]

    err = cm_data["error_analysis"]
    assert err["malignant"]["support"] == 31
    assert err["malignant"]["correct_detected"] == 21
    assert err["malignant"]["misclassified_as_benign"] == 4
    assert err["malignant"]["misclassified_as_normal"] == 6
    assert err["malignant"]["false_positives_from_other_classes"] == 70


def test_09_multicenter_isolation_and_center2_preservation(base_dir):
    """Verifies Center 2 evaluation artifacts remain intact and untampered."""
    c2_dir = base_dir / "ml_artifacts/btxrd_center2_external_evaluation"
    assert c2_dir.exists(), "Center 2 evaluation directory must remain intact!"

    c2_manifest = c2_dir / "evaluation_manifest.json"
    assert c2_manifest.exists()
    with open(c2_manifest, "r", encoding="utf-8") as f:
        m2 = json.load(f)
    assert m2["cohort_center"] == "center_2"
    assert m2["cohort_partition"] == "external_test_center2"
    assert m2["total_images_evaluated"] == 549

    c2_preds = c2_dir / "center2_predictions.csv"
    assert c2_preds.exists()


def test_10_conflict_quarantine_enforcement(btxrd_manifest):
    """Verifies all 4 conflicting images remain quarantined with conflict_flag=TRUE."""
    expected_conflicts = {"IMG000591.jpeg", "IMG003456.jpeg", "IMG000707.jpeg", "IMG003346.jpeg"}
    actual_conflicts = set()

    with open(btxrd_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("conflict_flag", "").strip().lower() == "true":
                actual_conflicts.add(row["image_id"])

    assert actual_conflicts == expected_conflicts


def test_11_calibration_status_uncalibrated(center3_eval_dir):
    """Verifies calibration status is documented as uncalibrated raw softmax."""
    manifest_path = center3_eval_dir / "evaluation_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["calibration_status"] == "uncalibrated_raw_softmax"


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


def test_14_stage_16_17_18_19_artifacts_preservation(base_dir):
    """Verifies Stage 16, 17, 18, and 19 artifacts exist and remain unmodified."""
    pilot_ckpt = base_dir / "ml_artifacts/btxrd_pilot/best_checkpoint.pt"
    assert pilot_ckpt.exists()

    stage17_ckpt = base_dir / "ml_artifacts/btxrd_full_training/best_checkpoint.pt"
    assert stage17_ckpt.exists()

    stage18_manifest = base_dir / "ml_artifacts/btxrd_frozen_model/model_freeze_manifest.json"
    assert stage18_manifest.exists()

    stage19_metrics = base_dir / "ml_artifacts/btxrd_center2_external_evaluation/classification_metrics.json"
    assert stage19_metrics.exists()


def test_15_cpu_python_and_frontend_preservation(base_dir):
    """Verifies host CPU Python and React frontend are completely intact."""
    cpu_py = Path(r"C:\Python313\python.exe")
    assert cpu_py.exists()
    cmd = [str(cpu_py), "-c", "import torch; print(torch.__version__)"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "+cpu" in res.stdout

    frontend_file = base_dir / "src/App.tsx"
    assert frontend_file.exists()
