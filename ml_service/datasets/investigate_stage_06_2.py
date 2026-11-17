"""
Deep investigation script for Stage 06.2.
Audits:
  1. FracAtlas duplicates, missing files, and COCO mask mapping.
  2. Bone scan exact patient counts, views, and fold distribution.
  3. BTXRD center-wise class distribution, duplicate hashes, and label conflict check.
"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import openpyxl


def audit_fracatlas():
    print("=" * 80)
    print("AUDITING FRACATLAS FOR PARTITIONING")
    frac_dir = Path("ml_service/data/raw/fracatlas")
    csv_path = frac_dir / "dataset.csv"

    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"Total dataset.csv records: {len(records)}")

    # Check images existence
    images_dir = frac_dir / "images"
    missing_files = []
    found_files = []
    
    for r in records:
        img_id = r["image_id"]
        is_fractured = r.get("fractured") == "1"
        folder = "Fractured" if is_fractured else "Non_fractured"
        p = images_dir / folder / img_id
        if not p.exists():
            # Check alternative folder
            alt = images_dir / ("Non_fractured" if is_fractured else "Fractured") / img_id
            if alt.exists():
                found_files.append((r, alt, "ALT_FOLDER"))
            else:
                missing_files.append((r, p))
        else:
            found_files.append((r, p, "NORMAL"))

    print(f"Found files: {len(found_files)}, Missing files: {len(missing_files)}")
    alt_files = [f for f in found_files if f[2] == "ALT_FOLDER"]
    print(f"Files in alternate folder: {len(alt_files)}")
    for r, p, _ in alt_files:
        print(f"  {r['image_id']}: label fractured={r['fractured']} found in {p.parent.name}")

    # Inspect official segmentation split
    split_dir = frac_dir / "Utilities/Fracture Split"
    train_split = [r[0] for r in csv.reader(open(split_dir / "train.csv"))][1:]
    val_split = [r[0] for r in csv.reader(open(split_dir / "valid.csv"))][1:]
    test_split = [r[0] for r in csv.reader(open(split_dir / "test.csv"))][1:]

    print(f"Official fracture split: train={len(train_split)}, val={len(val_split)}, test={len(test_split)}, sum={len(train_split)+len(val_split)+len(test_split)}")
    
    # Check COCO masks
    coco_path = frac_dir / "Annotations/COCO JSON/COCO_fracture_masks.json"
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    coco_img_map = {img["file_name"]: img["id"] for img in coco["images"]}
    coco_anno_count = Counter(anno["image_id"] for anno in coco["annotations"])
    print(f"COCO images indexed: {len(coco_img_map)}")
    print(f"COCO annotations count: {len(coco['annotations'])}")
    
    # Check that all official split images exist in COCO
    all_split_imgs = set(train_split + val_split + test_split)
    unmatched = all_split_imgs - set(coco_img_map.keys())
    print(f"Split images missing from COCO: {len(unmatched)}")


def audit_bone_scan():
    print("\n" + "=" * 80)
    print("AUDITING BONE SCAN METASTASIS DATASET FOR PARTITIONING")
    bs_dir = Path("ml_service/data/raw/bone_scan_metastasis")
    excel_path = bs_dir / "Bone Scan Images Dataset.xlsx"

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    patients = [r for r in rows if r[0] is not None]

    print(f"Total patient rows: {len(patients)}")
    diag_counts = Counter(r[9] for r in patients)
    print(f"Diagnosis distribution: {diag_counts}")

    # Verify both views exist on disk
    missing = []
    for r in patients:
        pid = int(r[0])
        ant = f"{r[1]}.png" if not str(r[1]).endswith(".png") else str(r[1])
        post = f"{r[2]}.png" if not str(r[2]).endswith(".png") else str(r[2])
        folder = "Bone metastases" if r[9] == "Bone metastases" else "No bone metastases"

        ant_p = bs_dir / folder / "Anterior" / ant
        post_p = bs_dir / folder / "Posterior" / post

        if not ant_p.exists():
            missing.append(str(ant_p))
        if not post_p.exists():
            missing.append(str(post_p))

    print(f"Missing images: {len(missing)}")


def audit_btxrd():
    print("\n" + "=" * 80)
    print("AUDITING BTXRD FOR PARTITIONING")
    btx_dir = Path("ml_service/data/raw/btxrd")
    excel_path = btx_dir / "dataset.xlsx"

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))[1:]

    # Center-wise category distribution
    # header: ('image_id', 'center', ..., 'tumor', 'benign', 'malignant', ...)
    center_dist = defaultdict(lambda: Counter())

    import hashlib
    img_dir = btx_dir / "images"
    sha_map = {}
    sha_to_rows = defaultdict(list)

    for r in rows:
        img_id = str(r[0]).strip()
        center = r[1]
        tumor = r[19] == 1 or r[19] == "1"
        benign = r[20] == 1 or r[20] == "1"
        malignant = r[21] == 1 or r[21] == "1"

        if tumor:
            cat = "Malignant" if malignant else "Benign"
        else:
            cat = "Normal"

        center_dist[center][cat] += 1
        center_dist[center]["Total"] += 1

        img_p = img_dir / img_id
        if img_p.exists():
            h = hashlib.sha256(open(img_p, "rb").read()).hexdigest()
            sha_map[img_id] = h
            sha_to_rows[h].append((img_id, center, cat, r))

    print("Center-wise Category Distribution:")
    for c in sorted(center_dist.keys()):
        print(f"  Center {c}: Total={center_dist[c]['Total']}, Normal={center_dist[c]['Normal']}, Benign={center_dist[c]['Benign']}, Malignant={center_dist[c]['Malignant']}")

    # Check the 21 duplicate hashes for conflicting labels
    dup_hashes = {h: items for h, items in sha_to_rows.items() if len(items) > 1}
    print(f"\nDuplicate hashes count: {len(dup_hashes)}")
    
    conflict_count = 0
    cross_center_dups = 0
    for h, items in dup_hashes.items():
        cats = {it[2] for it in items}
        centers = {it[1] for it in items}
        filenames = [it[0] for it in items]
        if len(cats) > 1:
            conflict_count += 1
            print(f"  CONFLICT DETECTED in hash {h[:10]}: categories={cats}, files={filenames}")
        if len(centers) > 1:
            cross_center_dups += 1
            print(f"  Cross-center duplicate in hash {h[:10]}: centers={centers}, files={filenames}")

    print(f"Label conflicts across duplicate hashes: {conflict_count}")
    print(f"Cross-center duplicate hashes: {cross_center_dups}")


if __name__ == "__main__":
    audit_fracatlas()
    audit_bone_scan()
    audit_btxrd()

