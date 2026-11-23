"""
Stage 21 - BTXRD Multi-Center Generalization Audit and Model Disposition Tests.

Verifies:
1. Frozen checkpoint SHA-256 hash remains invariant.
2. All 10 Stage 21 audit artifacts and plots exist in ml_artifacts/btxrd_multicenter_audit.
3. Stage 17-20 reports and prediction files are preserved intact.
4. Center 2 and Center 3 remain strictly external evaluation cohorts.
5. multicenter_performance_comparison.csv contains accurate data for all 3 centers.
6. generalization_gap_analysis.json contains verified metrics and drops.
7. cross_center_error_analysis.json matches verified confusion matrices.
8. model_readiness_decision.md contains formal decisions and 3-class vs 4-class project distinction.
9. statistical_limitations.md documents image-level bootstrap, uncalibrated softmax, and prevalence dependence.
10. remediation_roadmap.md documents the 7 technical pillars and permanent holdout sealing.
11. Preservation of FracAtlas checkpoints and artifacts.
12. Preservation of Bone Scintigraphy checkpoints and test metrics.
13. Preservation of host CPU Python and React frontend.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

from ml_service.config.settings import get_settings

EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"


@pytest.fixture(scope="module")
def base_dir():
    return get_settings().base_dir


@pytest.fixture(scope="module")
def audit_dir(base_dir):
    d = base_dir / "ml_artifacts/btxrd_multicenter_audit"
    assert d.exists(), f"Stage 21 audit directory missing: {d}"
    return d


@pytest.fixture(scope="module")
def frozen_checkpoint(base_dir):
    p = base_dir / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
    assert p.exists(), f"Frozen checkpoint missing: {p}"
    return p


def test_01_checkpoint_hash_unmodified(frozen_checkpoint):
    """Verifies that the frozen checkpoint SHA-256 remains completely unaltered."""
    h = hashlib.sha256()
    with open(frozen_checkpoint, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    current_hash = h.hexdigest().upper()
    assert current_hash == EXPECTED_SHA256, f"Checkpoint hash mismatch: {current_hash}"


def test_02_all_stage21_artifacts_exist(audit_dir):
    """Verifies that all 10 required Stage 21 files and plots exist with non-zero size."""
    expected_files = [
        "multicenter_performance_comparison.csv",
        "generalization_gap_analysis.json",
        "cross_center_error_analysis.json",
        "statistical_limitations.md",
        "model_readiness_decision.md",
        "remediation_roadmap.md",
        "STAGE_21_MULTICENTER_GENERALIZATION_AUDIT.md",
        "multicenter_metric_comparison.png",
        "classwise_sensitivity_specificity.png",
        "multicenter_confusion_matrices.png",
    ]
    for fn in expected_files:
        fp = audit_dir / fn
        assert fp.exists(), f"Missing Stage 21 artifact: {fn}"
        assert fp.stat().st_size > 0, f"Empty Stage 21 artifact: {fn}"


def test_03_stage_17_to_20_reports_and_predictions_preserved(base_dir):
    """Verifies that earlier stage reports and predictions are preserved."""
    # Stage 17
    assert (base_dir / "ml_artifacts/stage_reports/STAGE_17_BTXRD_FULL_TRAINING.md").exists()
    c1_preds = base_dir / "ml_artifacts/btxrd_full_training/validation_predictions.csv"
    assert c1_preds.exists()
    with open(c1_preds) as f:
        assert len(list(csv.DictReader(f))) == 588

    # Stage 18
    assert (base_dir / "ml_artifacts/stage_reports/STAGE_18_BTXRD_MODEL_FREEZE.md").exists()
    assert (base_dir / "ml_artifacts/btxrd_frozen_model/model_freeze_manifest.json").exists()

    # Stage 19
    assert (base_dir / "ml_artifacts/stage_reports/STAGE_19_CENTER2_EXTERNAL_EVALUATION.md").exists()
    c2_preds = base_dir / "ml_artifacts/btxrd_center2_external_evaluation/center2_predictions.csv"
    assert c2_preds.exists()
    with open(c2_preds) as f:
        assert len(list(csv.DictReader(f))) == 549

    # Stage 20
    assert (base_dir / "ml_artifacts/stage_reports/STAGE_20_CENTER3_EXTERNAL_EVALUATION.md").exists()
    c3_preds = base_dir / "ml_artifacts/btxrd_center3_external_evaluation/center3_predictions.csv"
    assert c3_preds.exists()
    with open(c3_preds) as f:
        assert len(list(csv.DictReader(f))) == 259


def test_04_center2_and_center3_isolation(base_dir):
    """Verifies that Center 2 and Center 3 remain external holdouts."""
    manifest_p = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    assert manifest_p.exists()
    with open(manifest_p) as f:
        rows = list(csv.DictReader(f))

    c2_rows = [r for r in rows if r["center"].lower() == "center_2"]
    assert len(c2_rows) == 549
    assert all(r["center_holdout_split"] == "external_test_center2" for r in c2_rows)

    c3_rows = [r for r in rows if r["center"].lower() == "center_3"]
    assert len(c3_rows) == 259
    assert all(r["center_holdout_split"] == "external_test_center3" for r in c3_rows)


def test_05_multicenter_performance_csv_integrity(audit_dir):
    """Verifies data integrity of multicenter_performance_comparison.csv."""
    csv_p = audit_dir / "multicenter_performance_comparison.csv"
    with open(csv_p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3

    cohorts = [r["cohort"] for r in rows]
    assert cohorts == ["Center 1", "Center 2", "Center 3"]

    c1 = rows[0]
    assert int(c1["n_samples"]) == 588
    assert float(c1["macro_f1"]) == 0.8103
    assert float(c1["balanced_accuracy"]) == 0.8290

    c2 = rows[1]
    assert int(c2["n_samples"]) == 549
    assert float(c2["macro_f1"]) == 0.5069
    assert float(c2["balanced_accuracy"]) == 0.5374

    c3 = rows[2]
    assert int(c3["n_samples"]) == 259
    assert float(c3["macro_f1"]) == 0.3592
    assert float(c3["balanced_accuracy"]) == 0.4467


def test_06_generalization_gap_json_integrity(audit_dir):
    """Verifies generalization_gap_analysis.json values."""
    p = audit_dir / "generalization_gap_analysis.json"
    with open(p) as f:
        data = json.load(f)

    c1_c2 = data["center1_to_center2_gap"]
    assert c1_c2["macro_f1"]["center1"] == 0.8103
    assert c1_c2["macro_f1"]["center2"] == 0.5069
    assert c1_c2["macro_f1"]["absolute_drop"] == -0.3034

    c1_c3 = data["center1_to_center3_gap"]
    assert c1_c3["macro_f1"]["center1"] == 0.8103
    assert c1_c3["macro_f1"]["center3"] == 0.3592
    assert c1_c3["macro_f1"]["absolute_drop"] == -0.4511
    assert c1_c3["normal_sensitivity_collapse"]["center3"] == 0.1852


def test_07_cross_center_error_analysis_json_integrity(audit_dir):
    """Verifies cross_center_error_analysis.json values and confusion matrices."""
    p = audit_dir / "cross_center_error_analysis.json"
    with open(p) as f:
        data = json.load(f)

    assert data["center1_errors"]["confusion_matrix"] == [[290, 26, 3], [65, 154, 9], [1, 3, 37]]
    assert data["center2_errors"]["confusion_matrix"] == [[119, 107, 33], [53, 120, 41], [10, 21, 45]]
    assert data["center3_errors"]["confusion_matrix"] == [[5, 13, 9], [44, 96, 61], [6, 4, 21]]

    # Malignant false negatives
    assert data["center2_errors"]["malignant"]["false_negatives"] == 31
    assert data["center3_errors"]["malignant"]["false_negatives"] == 10


def test_08_model_disposition_decisions(audit_dir):
    """Verifies model_readiness_decision.md contains formal decisions and project scope distinction."""
    p = audit_dir / "model_readiness_decision.md"
    content = p.read_text(encoding="utf-8")

    assert "CONDITIONAL PASS" in content
    assert "FAIL" in content
    assert "STRICTLY PROHIBITED" in content
    assert "UNSUITABLE IN CURRENT FORM" in content

    # Project scope distinction
    assert "Secondary Bone Metastasis" in content
    assert "Paget's Disease" in content
    assert "Osteogenesis Imperfecta" in content


def test_09_statistical_limitations_integrity(audit_dir):
    """Verifies statistical_limitations.md documents key methodological boundaries."""
    p = audit_dir / "statistical_limitations.md"
    content = p.read_text(encoding="utf-8").lower()

    assert "1,000 bootstrap iterations" in content
    assert "patient accession" in content or "patient identifier" in content or "patient id" in content
    assert "raw, uncalibrated softmax" in content
    assert "prevalence dependence" in content
    assert "high npv must not be presented as proof of safe malignancy rule-out" in content


def test_10_remediation_roadmap_integrity(audit_dir):
    """Verifies remediation_roadmap.md documents technical pillars and holdout sealing."""
    p = audit_dir / "remediation_roadmap.md"
    content = p.read_text(encoding="utf-8")

    assert "Permanent Sealing of Center 2 and Center 3" in content
    assert "Patient-Level Identifiers" in content
    assert "Acquisition Protocol Harmonization" in content
    assert "Multi-Reader Review" in content or "Consensus" in content
    assert "Split Conformal Prediction" in content or "Conformal" in content


def test_11_fracatlas_artifacts_preservation(base_dir):
    """Verifies FracAtlas Stage 09 checkpoint and Stage 10 report remain intact."""
    fracatlas_ckpt = base_dir / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
    assert fracatlas_ckpt.exists()
    assert fracatlas_ckpt.stat().st_size > 100_000_000

    fracatlas_rep = base_dir / "ml_artifacts/stage_reports/STAGE_10_FRACATLAS_HOLDOUT_EVALUATION.md"
    assert fracatlas_rep.exists()


def test_12_bone_scintigraphy_artifacts_preservation(base_dir):
    """Verifies Bone Scintigraphy fold 0 checkpoint and test metrics remain intact."""
    scint_fold0 = base_dir / "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"
    assert scint_fold0.exists()

    scint_test_eval = base_dir / "ml_artifacts/bone_scan_test_evaluation/test_metrics.json"
    assert scint_test_eval.exists()


def test_13_cpu_python_and_frontend_preservation(base_dir):
    """Verifies host CPU Python and React frontend are completely intact."""
    cpu_py = Path(r"C:\Python313\python.exe")
    assert cpu_py.exists()
    cmd = [str(cpu_py), "-c", "import torch; print(torch.__version__)"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "+cpu" in res.stdout

    frontend_file = base_dir / "src/App.tsx"
    assert frontend_file.exists()

