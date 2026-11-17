"""
Stage 06.2 Split Audit and Data Leakage Prevention Tests.

Verifies:
  1. Bone Scan:
     - 291 patients, 582 images.
     - Strict patient-level stratification (70% train, 15% val, 15% test).
     - Co-location of paired Anterior and Posterior views in the same split and fold.
     - 0 patient overlap across splits.
     - 0 image content (SHA-256) overlap across splits.
     - Negative class strictly NO_BONE_METASTASIS (never HEALTHY_CONTROL).
     - 5-fold CV assignment balance.
  2. FracAtlas Classification:
     - Exactly 4,083 rows matching authoritative dataset.csv.
     - 719 fractured, 3,364 non-fractured.
     - 0 duplicate hash leakage across train, val, test.
     - Authoritative resolution of duplicate filename cases (IMG0003375, IMG0003376) to Fractured/.
     - All referenced file paths exist.
  3. FracAtlas Segmentation:
     - Exactly 719 rows matching official 574/82/63 split.
     - All 719 map to COCO masks.
     - 0 overlap across train, val, test.
  4. BTXRD:
     - Exactly 3,746 rows.
     - Duplicate-grouped stratified split (0 duplicate hash leakage across train, val, test).
     - Conflicted duplicate pairs flagged with conflict_flag == 'TRUE'.
     - Multi-center distribution and external holdout split (Center 2 and Center 3 isolated).
     - Target classes strictly PRIMARY_MALIGNANT_TUMOR, BENIGN_BONE_TUMOR, NORMAL_BONE_RADIOGRAPH.
"""

from collections import Counter, defaultdict
import csv
from pathlib import Path
import pytest

from ml_service.config.settings import get_settings


@pytest.fixture(scope="module")
def splits_dir() -> Path:
    settings = get_settings()
    return settings.base_dir / "ml_service/data/splits"


# ==============================================================================
# 1. BONE SCAN METASTASIS SPLIT AUDIT
# ==============================================================================

def test_bone_scan_split_integrity(splits_dir: Path):
    manifest_path = splits_dir / "bone_scan_split_manifest.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Total rows: 291 patients * 2 views = 582 images
    assert len(rows) == 582, f"Expected 582 rows, got {len(rows)}"

    patient_views = defaultdict(list)
    patient_splits = {}
    patient_folds = {}
    patient_classes = {}
    split_hashes = defaultdict(set)

    for r in rows:
        pid = r["patient_id"]
        view = r["view"]
        split = r["split"]
        fold = r["fold"]
        cls = r["target_class"]
        sha = r["sha256_hash"]
        fpath = Path(r["file_path"])

        # File exists
        assert fpath.exists(), f"Image file missing: {fpath}"

        # Class validation: NO_BONE_METASTASIS or BONE_METASTASIS (NEVER HEALTHY_CONTROL)
        assert cls in ["BONE_METASTASIS", "NO_BONE_METASTASIS"], f"Invalid class: {cls}"
        assert cls != "HEALTHY_CONTROL", "Negative bone scan must NOT be labeled HEALTHY_CONTROL"

        patient_views[pid].append(view)
        split_hashes[split].add(sha)

        if pid not in patient_splits:
            patient_splits[pid] = split
            patient_folds[pid] = fold
            patient_classes[pid] = cls
        else:
            # Enforce paired views co-located in identical split and fold
            assert patient_splits[pid] == split, f"Patient {pid} views in different splits!"
            assert patient_folds[pid] == fold, f"Patient {pid} views in different folds!"
            assert patient_classes[pid] == cls, f"Patient {pid} has conflicting classes!"

    # Exactly 291 unique patients
    assert len(patient_views) == 291, f"Expected 291 patients, got {len(patient_views)}"

    # Every patient has exactly Anterior and Posterior
    for pid, views in patient_views.items():
        assert sorted(views) == ["Anterior", "Posterior"], f"Patient {pid} views: {views}"

    # Verify patient count per split: 70% Train (203), 15% Val (44), 15% Test (44)
    split_patient_counts = Counter(patient_splits.values())
    assert split_patient_counts["train"] == 203
    assert split_patient_counts["val"] == 44
    assert split_patient_counts["test"] == 44

    # Verify image count per split: 406 train, 88 val, 88 test
    split_img_counts = Counter(r["split"] for r in rows)
    assert split_img_counts["train"] == 406
    assert split_img_counts["val"] == 88
    assert split_img_counts["test"] == 88

    # Verify zero patient overlap across splits
    train_pts = {pid for pid, sp in patient_splits.items() if sp == "train"}
    val_pts = {pid for pid, sp in patient_splits.items() if sp == "val"}
    test_pts = {pid for pid, sp in patient_splits.items() if sp == "test"}

    assert len(train_pts & val_pts) == 0, f"Patient leakage between train and val: {train_pts & val_pts}"
    assert len(train_pts & test_pts) == 0, f"Patient leakage between train and test: {train_pts & test_pts}"
    assert len(val_pts & test_pts) == 0, f"Patient leakage between val and test: {val_pts & test_pts}"

    # Verify zero image hash overlap across splits
    assert len(split_hashes["train"] & split_hashes["val"]) == 0, "Hash leakage between train and val"
    assert len(split_hashes["train"] & split_hashes["test"]) == 0, "Hash leakage between train and test"
    assert len(split_hashes["val"] & split_hashes["test"]) == 0, "Hash leakage between val and test"

    # Verify 5-fold cross-validation balance
    fold_counts = Counter(patient_folds.values())
    for f_idx in range(5):
        f_name = f"fold_{f_idx}"
        # Total patients per fold should be 58 or 59 (291 / 5 = 58.2)
        assert fold_counts[f_name] in [58, 59], f"Fold {f_name} count: {fold_counts[f_name]}"

    # Verify metastasis balance across folds (12 positive patients in each of the 5 folds)
    fold_mets_counts = Counter(patient_folds[pid] for pid, cls in patient_classes.items() if cls == "BONE_METASTASIS")
    for f_idx in range(5):
        f_name = f"fold_{f_idx}"
        assert fold_mets_counts[f_name] == 12, f"Fold {f_name} mets count: {fold_mets_counts[f_name]}"


# ==============================================================================
# 2. FRACATLAS CLASSIFICATION SPLIT AUDIT
# ==============================================================================

def test_fracatlas_classification_split(splits_dir: Path):
    manifest_path = splits_dir / "fracatlas_classification_split.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Must contain exactly 4,083 records matching dataset.csv
    assert len(rows) == 4083, f"Expected 4083 rows, got {len(rows)}"

    # Class balance: 719 fractured, 3364 non-fractured
    frac_count = sum(1 for r in rows if r["fractured"] == "1")
    nonfrac_count = sum(1 for r in rows if r["fractured"] == "0")
    assert frac_count == 719, f"Expected 719 fractured, got {frac_count}"
    assert nonfrac_count == 3364, f"Expected 3364 non-fractured, got {nonfrac_count}"

    # Split sizes: ~70% train (2,857), ~15% val (612), ~15% test (614)
    split_counts = Counter(r["split"] for r in rows)
    assert split_counts["train"] == 2857, f"Train count: {split_counts['train']}"
    assert split_counts["val"] == 612, f"Val count: {split_counts['val']}"
    assert split_counts["test"] == 614, f"Test count: {split_counts['test']}"

    # Check file existence & duplicate hash leakage
    split_hashes = defaultdict(set)
    for r in rows:
        fpath = Path(r["file_path"])
        assert fpath.exists(), f"Missing image file: {fpath}"
        split_hashes[r["split"]].add(r["sha256_hash"])

    # Zero duplicate content leakage across splits
    assert len(split_hashes["train"] & split_hashes["val"]) == 0, "Duplicate hash leakage between train and val"
    assert len(split_hashes["train"] & split_hashes["test"]) == 0, "Duplicate hash leakage between train and test"
    assert len(split_hashes["val"] & split_hashes["test"]) == 0, "Duplicate hash leakage between val and test"

    # Verify duplicate filename case resolution (IMG0003375 and IMG0003376)
    row_map = {r["image_id"]: r for r in rows}
    assert "IMG0003375.jpg" in row_map
    assert "IMG0003376.jpg" in row_map

    rec_3375 = row_map["IMG0003375.jpg"]
    rec_3376 = row_map["IMG0003376.jpg"]
    assert rec_3375["fractured"] == "1"
    assert rec_3376["fractured"] == "1"
    assert "Fractured" in rec_3375["file_path"]
    assert "Fractured" in rec_3376["file_path"]


# ==============================================================================
# 3. FRACATLAS SEGMENTATION SPLIT AUDIT
# ==============================================================================

def test_fracatlas_segmentation_split(splits_dir: Path):
    manifest_path = splits_dir / "fracatlas_segmentation_split.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Exactly 719 fractured images (574 train + 82 val + 63 test)
    assert len(rows) == 719, f"Expected 719 rows, got {len(rows)}"

    split_counts = Counter(r["split"] for r in rows)
    assert split_counts["train"] == 574, f"Train count: {split_counts['train']}"
    assert split_counts["val"] == 82, f"Val count: {split_counts['val']}"
    assert split_counts["test"] == 63, f"Test count: {split_counts['test']}"

    # Verify all files exist and have polygon annotations
    for r in rows:
        fpath = Path(r["file_path"])
        assert fpath.exists(), f"Image missing: {fpath}"
        mask_path = Path(r["mask_json_path"])
        assert mask_path.exists(), f"Mask JSON missing: {mask_path}"
        assert int(r["num_polygons"]) >= 1, f"Image {r['image_id']} has 0 polygons in segmentation split"

    # Zero overlap of images across splits
    img_splits = defaultdict(set)
    for r in rows:
        img_splits[r["split"]].add(r["image_id"])

    assert len(img_splits["train"] & img_splits["val"]) == 0
    assert len(img_splits["train"] & img_splits["test"]) == 0
    assert len(img_splits["val"] & img_splits["test"]) == 0


# ==============================================================================
# 4. BTXRD SPLIT AUDIT
# ==============================================================================

def test_btxrd_split_integrity(splits_dir: Path):
    manifest_path = splits_dir / "btxrd_split_manifest.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Exactly 3,746 images
    assert len(rows) == 3746, f"Expected 3746 rows, got {len(rows)}"

    # Check target classes (NEVER secondary metastasis)
    valid_classes = {"PRIMARY_MALIGNANT_TUMOR", "BENIGN_BONE_TUMOR", "NORMAL_BONE_RADIOGRAPH"}
    for r in rows:
        assert r["target_class"] in valid_classes, f"Illegal target class: {r['target_class']}"
        assert "SECONDARY" not in r["target_class"], "Primary bone tumors must NEVER map to secondary metastasis"

    # Multi-center distribution check: Center 1 (2,938), Center 2 (549), Center 3 (259)
    center_counts = Counter(r["center"] for r in rows)
    assert center_counts["center_1"] == 2938
    assert center_counts["center_2"] == 549
    assert center_counts["center_3"] == 259

    # Center holdout split isolation check:
    # Center 2 must strictly be external_test_center2
    # Center 3 must strictly be external_test_center3
    # Center 1 must only be train or val
    for r in rows:
        c = r["center"]
        ch_split = r["center_holdout_split"]
        if c == "center_1":
            assert ch_split in ["train", "val"], f"Center 1 has invalid holdout split: {ch_split}"
        elif c == "center_2":
            assert ch_split == "external_test_center2", f"Center 2 has invalid holdout split: {ch_split}"
        elif c == "center_3":
            assert ch_split == "external_test_center3", f"Center 3 has invalid holdout split: {ch_split}"

    # Duplicate content grouping check:
    # Every sha256_hash that appears multiple times MUST have identical stratified_split
    sha_to_strat_splits = defaultdict(set)
    for r in rows:
        sha_to_strat_splits[r["sha256_hash"]].add(r["stratified_split"])

    for sha, splits in sha_to_strat_splits.items():
        assert len(splits) == 1, f"SHA {sha[:12]} leaked across stratified splits: {splits}"

    # Verify conflict flag tracking
    # There must be flagged conflict rows
    conflict_rows = [r for r in rows if r["conflict_flag"] == "TRUE"]
    assert len(conflict_rows) > 0, "Expected conflicting duplicate rows to be flagged"

    # Exactly 4 rows (2 pairs) have conflicting labels
    assert len(conflict_rows) == 4, f"Expected 4 conflict rows (2 pairs), got {len(conflict_rows)}"
    conflict_ids = {r["image_id"] for r in conflict_rows}
    assert "IMG000591.jpeg" in conflict_ids
    assert "IMG003456.jpeg" in conflict_ids
    assert "IMG000707.jpeg" in conflict_ids
    assert "IMG003346.jpeg" in conflict_ids

    # All files exist
    for r in rows:
        assert Path(r["file_path"]).exists(), f"Image file missing: {r['file_path']}"
