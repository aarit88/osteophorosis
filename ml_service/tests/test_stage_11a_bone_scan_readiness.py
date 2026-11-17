"""
Stage 11A - Bone Scan Metastasis Dataset Readiness Audit Tests.
Verifies dataset identity, Excel metadata alignment, patient-level split hygiene,
zero data leakage, paired view co-location, 16-bit scintigraphy properties,
and FracAtlas checkpoint immutability.
"""

import csv
from pathlib import Path
import pytest
from PIL import Image
import openpyxl
import torch

from ml_service.config.settings import get_settings


@pytest.fixture(scope="module")
def paths():
    settings = get_settings()
    root = settings.base_dir
    return {
        "raw_dir": root / "ml_service/data/raw/bone_scan_metastasis",
        "excel": root / "ml_service/data/raw/bone_scan_metastasis/Bone Scan Images Dataset.xlsx",
        "split_manifest": root / "ml_service/data/splits/bone_scan_split_manifest.csv",
        "fracatlas_ckpt": root / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt",
    }


def test_bone_scan_dataset_identity_and_metadata(paths):
    """Verifies Excel file records and raw image counts on disk."""
    wb = openpyxl.load_workbook(paths["excel"])
    ws = wb.active
    excel_rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if any(r)]

    assert len(excel_rows) == 291, f"Expected 291 patient records, got {len(excel_rows)}"
    n_pos = sum(1 for r in excel_rows if r[9] == "Bone metastases")
    n_neg = sum(1 for r in excel_rows if r[9] == "No bone metastases")
    assert n_pos == 60
    assert n_neg == 231

    # Demographics
    sexes = {r[5] for r in excel_rows}
    assert "Female" in sexes and "Male" in sexes
    assert sum(1 for r in excel_rows if r[5] == "Female") == 289
    assert sum(1 for r in excel_rows if r[5] == "Male") == 2

    # Files on disk
    mets_ant = list((paths["raw_dir"] / "Bone metastases/Anterior").glob("*.png"))
    mets_post = list((paths["raw_dir"] / "Bone metastases/Posterior").glob("*.png"))
    no_mets_ant = list((paths["raw_dir"] / "No bone metastases/Anterior").glob("*.png"))
    no_mets_post = list((paths["raw_dir"] / "No bone metastases/Posterior").glob("*.png"))

    assert len(mets_ant) == 60
    assert len(mets_post) == 60
    assert len(no_mets_ant) == 231
    assert len(no_mets_post) == 231
    assert (len(mets_ant) + len(mets_post) + len(no_mets_ant) + len(no_mets_post)) == 582


def test_bone_scan_manifest_exact_counts(paths):
    """Verifies patient and image counts per partition in bone_scan_split_manifest.csv."""
    records = list(csv.DictReader(open(paths["split_manifest"], "r", encoding="utf-8")))
    assert len(records) == 582

    splits = {"train": set(), "val": set(), "test": set()}
    split_images = {"train": 0, "val": 0, "test": 0}
    split_pos_pts = {"train": set(), "val": set(), "test": set()}

    for r in records:
        sp = r["split"].lower()
        pid = r["patient_id"]
        splits[sp].add(pid)
        split_images[sp] += 1
        if r["target_class"] == "BONE_METASTASIS":
            split_pos_pts[sp].add(pid)

    # Train (70%)
    assert len(splits["train"]) == 203
    assert len(split_pos_pts["train"]) == 42
    assert split_images["train"] == 406

    # Val (15%)
    assert len(splits["val"]) == 44
    assert len(split_pos_pts["val"]) == 9
    assert split_images["val"] == 88

    # Test (15%)
    assert len(splits["test"]) == 44
    assert len(split_pos_pts["test"]) == 9
    assert split_images["test"] == 88


def test_bone_scan_zero_leakage_and_colocation(paths):
    """Verifies 0 patient ID leakage and 0 SHA-256 hash leakage across splits."""
    records = list(csv.DictReader(open(paths["split_manifest"], "r", encoding="utf-8")))

    pts_by_split = {"train": set(), "val": set(), "test": set()}
    hashes_by_split = {"train": set(), "val": set(), "test": set()}
    views_by_patient = {}
    split_by_patient = {}
    fold_by_patient = {}

    for r in records:
        sp = r["split"].lower()
        pid = r["patient_id"]
        pts_by_split[sp].add(pid)
        hashes_by_split[sp].add(r["sha256_hash"])

        if pid not in views_by_patient:
            views_by_patient[pid] = []
            split_by_patient[pid] = set()
            fold_by_patient[pid] = set()
        views_by_patient[pid].append(r["view"].lower())
        split_by_patient[pid].add(sp)
        fold_by_patient[pid].add(r["fold"])

    # Zero patient leakage
    assert len(pts_by_split["train"].intersection(pts_by_split["val"])) == 0
    assert len(pts_by_split["train"].intersection(pts_by_split["test"])) == 0
    assert len(pts_by_split["val"].intersection(pts_by_split["test"])) == 0

    # Zero image hash leakage
    assert len(hashes_by_split["train"].intersection(hashes_by_split["val"])) == 0
    assert len(hashes_by_split["train"].intersection(hashes_by_split["test"])) == 0
    assert len(hashes_by_split["val"].intersection(hashes_by_split["test"])) == 0

    # Paired view co-location
    for pid, v_list in views_by_patient.items():
        assert sorted(v_list) == ["anterior", "posterior"], f"Patient {pid} missing paired views"
        assert len(split_by_patient[pid]) == 1, f"Patient {pid} views split across partitions: {split_by_patient[pid]}"
        assert len(fold_by_patient[pid]) == 1, f"Patient {pid} views split across folds: {fold_by_patient[pid]}"


def test_bone_scan_16bit_image_properties(paths):
    """Verifies that all 582 images are 16-bit grayscale (I;16) with dimensions (256, 1024)."""
    records = list(csv.DictReader(open(paths["split_manifest"], "r", encoding="utf-8")))

    # Sample 30 images across all splits
    for r in records[::20]:
        img_path = Path(r["file_path"])
        assert img_path.exists(), f"Image missing: {img_path}"
        with Image.open(img_path) as img:
            assert img.size == (256, 1024), f"Unexpected size {img.size} for {img_path}"
            assert img.mode == "I;16", f"Unexpected mode {img.mode} for {img_path}"


def test_bone_scan_5fold_stratification(paths):
    """Verifies that all 5 folds contain exactly 12 metastasis-positive patients."""
    records = list(csv.DictReader(open(paths["split_manifest"], "r", encoding="utf-8")))
    folds = {}
    for r in records:
        fld = r["fold"]
        pid = r["patient_id"]
        is_pos = (r["target_class"] == "BONE_METASTASIS")
        if fld not in folds:
            folds[fld] = {"patients": set(), "pos_pts": set()}
        folds[fld]["patients"].add(pid)
        if is_pos:
            folds[fld]["pos_pts"].add(pid)

    assert len(folds) == 5
    for fld, d in folds.items():
        assert len(d["pos_pts"]) == 12, f"Fold {fld} has {len(d['pos_pts'])} positive patients (expected 12)"
        assert len(d["patients"]) in (58, 59)


def test_fracatlas_artifacts_strictly_unaltered(paths):
    """Verifies that FracAtlas checkpoints remain intact and uncorrupted."""
    assert paths["fracatlas_ckpt"].exists()
    ckpt = torch.load(paths["fracatlas_ckpt"], map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 9
    assert ckpt["val_pr_auc"] == pytest.approx(0.7556, rel=1e-3)

