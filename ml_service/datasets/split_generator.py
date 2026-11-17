"""
OsteoGuard AI - Stage 06.2 Dataset Partitioning & Data Leakage Prevention Engine.
Generates:
  1. bone_scan_split_manifest.csv (Patient-level stratified split + 5-fold CV)
  2. fracatlas_classification_split.csv (Full 4,083 classification split, duplicate-resolved)
  3. fracatlas_segmentation_split.csv (Official 574/82/63 fracture mask split)
  4. btxrd_split_manifest.csv (Duplicate-grouped stratified + Center holdout split)

Strictly enforces:
  - Zero patient leakage in bone scan.
  - Grouping of paired Anterior and Posterior views.
  - Deduplication and conflict tracking.
  - Strict negative mapping rules (NO_BONE_METASTASIS != HEALTHY_CONTROL; Sarcomas != Metastases).
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import openpyxl

from ml_service.config.settings import get_settings
from ml_service.datasets.registry_verifier import compute_file_sha256
from ml_service.utils.logging import logger


SEED = 42


# ==============================================================================
# 1. BONE SCAN PARTITIONING (Patient-Level Grouping & 5-Fold Cross-Validation)
# ==============================================================================

def generate_bone_scan_splits(
    base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Partitions the Bone Scan Metastasis Dataset (291 patients, 582 images) strictly
    at the patient level:
      - 70% Train (203 patients, 406 images)
      - 15% Val (44 patients, 88 images)
      - 15% Test (44 patients, 88 images)
      - Supplementary 5-fold patient cross-validation (folds 0..4).
    Both Anterior and Posterior views for any patient are strictly co-located.
    """
    settings = get_settings()
    root = base_dir or settings.base_dir
    bs_dir = root / "ml_service/data/raw/bone_scan_metastasis"
    out_dir = output_dir or (root / "ml_service/data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = bs_dir / "Bone Scan Images Dataset.xlsx"
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]

    patients_mets = []
    patients_nomets = []

    for r in rows:
        if r[0] is None:
            continue
        pid = int(r[0])
        diag = str(r[9]).strip()
        ant_name = str(r[1]).strip()
        post_name = str(r[2]).strip()
        if not ant_name.lower().endswith(".png"):
            ant_name += ".png"
        if not post_name.lower().endswith(".png"):
            post_name += ".png"

        folder = "Bone metastases" if diag == "Bone metastases" else "No bone metastases"
        ant_file = bs_dir / folder / "Anterior" / ant_name
        post_file = bs_dir / folder / "Posterior" / post_name

        patient_record = {
            "patient_id": f"PT_BS_{pid:03d}",
            "numeric_id": pid,
            "diag": "BONE_METASTASIS" if diag == "Bone metastases" else "NO_BONE_METASTASIS",
            "ant_file": ant_file,
            "post_file": post_file,
            "ant_name": ant_name,
            "post_name": post_name,
        }

        if diag == "Bone metastases":
            patients_mets.append(patient_record)
        else:
            patients_nomets.append(patient_record)

    # Deterministic patient-level shuffling
    rng = random.Random(seed)
    patients_mets_sorted = sorted(patients_mets, key=lambda p: p["numeric_id"])
    patients_nomets_sorted = sorted(patients_nomets, key=lambda p: p["numeric_id"])
    rng.shuffle(patients_mets_sorted)
    rng.shuffle(patients_nomets_sorted)

    # 70% Train, 15% Val, 15% Test
    # Mets (60): 42 train, 9 val, 9 test
    mets_train = patients_mets_sorted[:42]
    mets_val = patients_mets_sorted[42:51]
    mets_test = patients_mets_sorted[51:60]

    # NoMets (231): 161 train, 35 val, 35 test
    nomets_train = patients_nomets_sorted[:161]
    nomets_val = patients_nomets_sorted[161:196]
    nomets_test = patients_nomets_sorted[196:231]

    patient_to_split = {}
    for p in mets_train + nomets_train:
        patient_to_split[p["patient_id"]] = "train"
    for p in mets_val + nomets_val:
        patient_to_split[p["patient_id"]] = "val"
    for p in mets_test + nomets_test:
        patient_to_split[p["patient_id"]] = "test"

    # Supplementary 5-fold cross-validation assignment (all 291 patients)
    # 60 mets -> 12 per fold; 231 nomets -> 46 per fold (47 in fold 4)
    patient_to_fold = {}
    for idx, p in enumerate(patients_mets_sorted):
        fold_id = f"fold_{idx % 5}"
        patient_to_fold[p["patient_id"]] = fold_id
    for idx, p in enumerate(patients_nomets_sorted):
        fold_id = f"fold_{idx % 5}"
        patient_to_fold[p["patient_id"]] = fold_id

    # Build individual image records (582 rows)
    all_records = []
    all_patients = patients_mets_sorted + patients_nomets_sorted

    for p in sorted(all_patients, key=lambda x: x["numeric_id"]):
        pid = p["patient_id"]
        split = patient_to_split[pid]
        fold = patient_to_fold[pid]
        diag = p["diag"]

        # Anterior
        sha_ant = compute_file_sha256(p["ant_file"])
        all_records.append({
            "patient_id": pid,
            "image_id": p["ant_name"],
            "file_path": str(p["ant_file"].as_posix()),
            "view": "Anterior",
            "target_class": diag,
            "split": split,
            "fold": fold,
            "source_id": "IICS_UNA_BONE_SCAN_METASTASIS",
            "sha256_hash": sha_ant,
        })

        # Posterior
        sha_post = compute_file_sha256(p["post_file"])
        all_records.append({
            "patient_id": pid,
            "image_id": p["post_name"],
            "file_path": str(p["post_file"].as_posix()),
            "view": "Posterior",
            "target_class": diag,
            "split": split,
            "fold": fold,
            "source_id": "IICS_UNA_BONE_SCAN_METASTASIS",
            "sha256_hash": sha_post,
        })

    # Export manifest
    out_csv = out_dir / "bone_scan_split_manifest.csv"
    fieldnames = [
        "patient_id", "image_id", "file_path", "view", "target_class",
        "split", "fold", "source_id", "sha256_hash"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    logger.info(f"Generated {out_csv} with {len(all_records)} images across 291 patients.")
    return all_records, out_csv


# ==============================================================================
# 2. FRACATLAS PARTITIONING (Classification Split & Segmentation Split)
# ==============================================================================

def generate_fracatlas_splits(
    base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path, Path]:
    """
    1. Generates full 4,083 classification split (70% train, 15% val, 15% test).
       Authoritatively resolves duplicate filename cases (IMG0003375, IMG0003376) to Fractured/.
    2. Generates official 574/82/63 fracture segmentation split mapping to COCO masks.
    """
    settings = get_settings()
    root = base_dir or settings.base_dir
    frac_dir = root / "ml_service/data/raw/fracatlas"
    out_dir = output_dir or (root / "ml_service/data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = frac_dir / "dataset.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        dataset_records = list(csv.DictReader(f))

    images_dir = frac_dir / "images"

    # Part A: Classification split (4,083 records)
    fractured_records = []
    non_fractured_records = []

    for r in dataset_records:
        img_id = r["image_id"]
        is_fractured = r.get("fractured", "0") == "1"

        # Authoritative file path resolution:
        # If fractured == 1: point strictly to Fractured/
        # If fractured == 0: point strictly to Non_fractured/
        if is_fractured:
            img_file = images_dir / "Fractured" / img_id
        else:
            img_file = images_dir / "Non_fractured" / img_id

        if not img_file.exists():
            # Fallback
            alt = images_dir / ("Non_fractured" if is_fractured else "Fractured") / img_id
            if alt.exists():
                img_file = alt

        # Determine anatomy
        anatomy = "skeleton"
        for bc in ["hand", "leg", "hip", "shoulder", "mixed"]:
            if r.get(bc) == "1":
                anatomy = bc
                break

        # Determine projection
        proj = "AP"
        for vc in ["frontal", "lateral", "oblique"]:
            if r.get(vc) == "1":
                proj = vc.capitalize()
                break

        sha256 = compute_file_sha256(img_file)

        item = {
            "image_id": img_id,
            "file_path": str(img_file.as_posix()),
            "target_class": "acute_bone_fracture" if is_fractured else "candidate_control_radiograph",
            "fractured": "1" if is_fractured else "0",
            "anatomic_region": anatomy,
            "view_projection": proj,
            "sha256_hash": sha256,
        }

        if is_fractured:
            fractured_records.append(item)
        else:
            non_fractured_records.append(item)

    # Deterministic duplicate-grouped stratified split
    # Group records by sha256_hash so duplicate radiographs never cross splits
    sha_to_items = defaultdict(list)
    for it in fractured_records + non_fractured_records:
        sha_to_items[it["sha256_hash"]].append(it)

    frac_hashes = []
    nonfrac_hashes = []
    for sha, items in sha_to_items.items():
        if items[0]["fractured"] == "1":
            frac_hashes.append(sha)
        else:
            nonfrac_hashes.append(sha)

    rng = random.Random(seed)
    frac_hashes.sort()
    nonfrac_hashes.sort()
    rng.shuffle(frac_hashes)
    rng.shuffle(nonfrac_hashes)

    # 70% Train, 15% Val, 15% Test over unique content hashes
    n_f_tr = int(round(len(frac_hashes) * 0.70))
    n_f_val = int(round(len(frac_hashes) * 0.15))
    f_tr_h = set(frac_hashes[:n_f_tr])
    f_val_h = set(frac_hashes[n_f_tr:n_f_tr + n_f_val])
    f_te_h = set(frac_hashes[n_f_tr + n_f_val:])

    n_nf_tr = int(round(len(nonfrac_hashes) * 0.70))
    n_nf_val = int(round(len(nonfrac_hashes) * 0.15))
    nf_tr_h = set(nonfrac_hashes[:n_nf_tr])
    nf_val_h = set(nonfrac_hashes[n_nf_tr:n_nf_tr + n_nf_val])
    nf_te_h = set(nonfrac_hashes[n_nf_tr + n_nf_val:])

    train_hashes = f_tr_h | nf_tr_h
    val_hashes = f_val_h | nf_val_h

    classification_split_records = []
    for it in fractured_records + non_fractured_records:
        h = it["sha256_hash"]
        d = dict(it)
        if h in train_hashes:
            d["split"] = "train"
        elif h in val_hashes:
            d["split"] = "val"
        else:
            d["split"] = "test"
        classification_split_records.append(d)

    # Sort deterministically
    classification_split_records = sorted(classification_split_records, key=lambda x: x["image_id"])

    # Export classification split CSV
    class_csv = out_dir / "fracatlas_classification_split.csv"
    class_fields = [
        "image_id", "file_path", "target_class", "fractured",
        "anatomic_region", "view_projection", "split", "sha256_hash"
    ]
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=class_fields)
        writer.writeheader()
        for r in classification_split_records:
            writer.writerow(r)

    # Part B: Segmentation split (574 train / 82 val / 63 test)
    split_dir = frac_dir / "Utilities/Fracture Split"
    train_split_imgs = [r[0] for r in csv.reader(open(split_dir / "train.csv"))][1:]
    val_split_imgs = [r[0] for r in csv.reader(open(split_dir / "valid.csv"))][1:]
    test_split_imgs = [r[0] for r in csv.reader(open(split_dir / "test.csv"))][1:]

    coco_path = frac_dir / "Annotations/COCO JSON/COCO_fracture_masks.json"
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    coco_img_map = {img["file_name"]: img["id"] for img in coco["images"]}
    coco_anno_count = Counter(anno["image_id"] for anno in coco["annotations"])

    segmentation_records = []

    def make_seg_record(img_id: str, split_name: str) -> Dict[str, Any]:
        img_file = images_dir / "Fractured" / img_id
        if not img_file.exists():
            img_file = images_dir / "Non_fractured" / img_id
        sha = compute_file_sha256(img_file)
        c_id = coco_img_map.get(img_id)
        num_poly = coco_anno_count.get(c_id, 0)
        return {
            "image_id": img_id,
            "file_path": str(img_file.as_posix()),
            "mask_json_path": str(coco_path.as_posix()),
            "num_polygons": num_poly,
            "split": split_name,
            "sha256_hash": sha,
        }

    for im in train_split_imgs:
        segmentation_records.append(make_seg_record(im, "train"))
    for im in val_split_imgs:
        segmentation_records.append(make_seg_record(im, "val"))
    for im in test_split_imgs:
        segmentation_records.append(make_seg_record(im, "test"))

    segmentation_records = sorted(segmentation_records, key=lambda x: x["image_id"])

    seg_csv = out_dir / "fracatlas_segmentation_split.csv"
    seg_fields = [
        "image_id", "file_path", "mask_json_path", "num_polygons",
        "split", "sha256_hash"
    ]
    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seg_fields)
        writer.writeheader()
        for r in segmentation_records:
            writer.writerow(r)

    logger.info(f"Generated FracAtlas classification split ({len(classification_split_records)} rows) and segmentation split ({len(segmentation_records)} rows).")
    return classification_split_records, segmentation_records, class_csv, seg_csv


# ==============================================================================
# 3. BTXRD PARTITIONING (Duplicate-Grouped Stratified & Center Holdout)
# ==============================================================================

def generate_btxrd_splits(
    base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Partitions BTXRD (3,746 images):
      1. Groups by exact duplicate content (SHA-256) to prevent content leakage.
      2. Detects and flags 2 duplicate content pairs with conflicting labels in original data.
      3. Provides two split protocols:
         - stratified_split: 70% Train, 15% Val, 15% Test.
         - center_holdout_split: Center 1 (train/val), Center 2 (external_test_center2), Center 3 (external_test_center3).
    """
    settings = get_settings()
    root = base_dir or settings.base_dir
    btx_dir = root / "ml_service/data/raw/btxrd"
    out_dir = output_dir or (root / "ml_service/data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = btx_dir / "dataset.xlsx"
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))[1:]

    images_dir = btx_dir / "images"

    # Step 1: Read all rows and compute SHA-256
    raw_items = []
    sha_to_items = defaultdict(list)

    for r in rows:
        img_id = str(r[0]).strip()
        center = int(r[1])
        tumor = r[19] == 1 or r[19] == "1"
        benign = r[20] == 1 or r[20] == "1"
        malignant = r[21] == 1 or r[21] == "1"

        if tumor:
            if malignant:
                cat = "PRIMARY_MALIGNANT_TUMOR"
                short = "malignant"
            else:
                cat = "BENIGN_BONE_TUMOR"
                short = "benign"
        else:
            cat = "NORMAL_BONE_RADIOGRAPH"
            short = "normal"

        img_file = images_dir / img_id
        if not img_file.exists():
            for ext in [".jpeg", ".jpg", ".png"]:
                cand = images_dir / f"{Path(img_id).stem}{ext}"
                if cand.exists():
                    img_file = cand
                    break

        sha = compute_file_sha256(img_file)

        item = {
            "image_id": img_id,
            "file_path": str(img_file.as_posix()),
            "target_class": cat,
            "center": center,
            "tumor": "1" if tumor else "0",
            "benign": "1" if benign else "0",
            "malignant": "1" if malignant else "0",
            "sha256_hash": sha,
            "category_code": short,
        }
        raw_items.append(item)
        sha_to_items[sha].append(item)

    # Step 2: Flag conflicts
    # Identify duplicate groups with discordant category_code
    conflicted_hashes = set()
    for sha, items in sha_to_items.items():
        if len(items) > 1:
            cats = {it["category_code"] for it in items}
            if len(cats) > 1:
                conflicted_hashes.add(sha)

    # Step 3: Duplicate-Group Aware Stratified Partitioning (70/15/15)
    # Group records by sha256_hash so identical images are never split
    unique_hashes = sorted(list(sha_to_items.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_hashes)

    # Partition unique hashes stratified by primary category
    hash_to_stratified_split = {}
    cat_to_hashes = defaultdict(list)
    for h in unique_hashes:
        primary_cat = sha_to_items[h][0]["category_code"]
        cat_to_hashes[primary_cat].append(h)

    for cat, h_list in cat_to_hashes.items():
        n = len(h_list)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        n_test = n - (n_train + n_val)

        train_h = h_list[:n_train]
        val_h = h_list[n_train:n_train + n_val]
        test_h = h_list[n_train + n_val:]

        for h in train_h:
            hash_to_stratified_split[h] = "train"
        for h in val_h:
            hash_to_stratified_split[h] = "val"
        for h in test_h:
            hash_to_stratified_split[h] = "test"

    # Step 4: Center-Holdout Partitioning
    # Center 1 (2,938 images) -> train (80%) / val (20%)
    # Center 2 (549 images) -> external_test_center2
    # Center 3 (259 images) -> external_test_center3
    c1_hashes = [h for h in unique_hashes if sha_to_items[h][0]["center"] == 1]
    rng.shuffle(c1_hashes)
    n_c1_train = int(round(len(c1_hashes) * 0.80))
    c1_train_hashes = set(c1_hashes[:n_c1_train])
    c1_val_hashes = set(c1_hashes[n_c1_train:])

    # Step 5: Build final manifest records
    final_records = []
    for item in raw_items:
        h = item["sha256_hash"]
        center = item["center"]
        is_conflict = h in conflicted_hashes

        strat_split = hash_to_stratified_split[h]

        if center == 1:
            center_split = "train" if h in c1_train_hashes else "val"
        elif center == 2:
            center_split = "external_test_center2"
        else:
            center_split = "external_test_center3"

        record = {
            "image_id": item["image_id"],
            "file_path": item["file_path"],
            "target_class": item["target_class"],
            "center": f"center_{center}",
            "tumor": item["tumor"],
            "benign": item["benign"],
            "malignant": item["malignant"],
            "stratified_split": strat_split,
            "center_holdout_split": center_split,
            "sha256_hash": h,
            "conflict_flag": "TRUE" if is_conflict else "FALSE",
        }
        final_records.append(record)

    final_records = sorted(final_records, key=lambda x: x["image_id"])

    out_csv = out_dir / "btxrd_split_manifest.csv"
    fields = [
        "image_id", "file_path", "target_class", "center", "tumor",
        "benign", "malignant", "stratified_split", "center_holdout_split",
        "sha256_hash", "conflict_flag"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in final_records:
            writer.writerow(r)

    logger.info(f"Generated BTXRD split manifest ({len(final_records)} rows, {len(conflicted_hashes)} conflict groups).")
    return final_records, out_csv


def generate_all_splits():
    print("Generating Stage 06.2 dataset split manifests...")
    bs_records, bs_path = generate_bone_scan_splits()
    print(f"  Bone Scan Split Manifest: {bs_path} ({len(bs_records)} rows)")

    fa_class, fa_seg, fa_c_path, fa_s_path = generate_fracatlas_splits()
    print(f"  FracAtlas Classification Split: {fa_c_path} ({len(fa_class)} rows)")
    print(f"  FracAtlas Segmentation Split: {fa_s_path} ({len(fa_seg)} rows)")

    btx_records, btx_path = generate_btxrd_splits()
    print(f"  BTXRD Split Manifest: {btx_path} ({len(btx_records)} rows)")

    print("All Stage 06.2 split manifests generated successfully.")


if __name__ == "__main__":
    generate_all_splits()
