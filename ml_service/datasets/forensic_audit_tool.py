"""
OsteoGuard AI - Comprehensive Forensic Audit Tool for Stage 06.1.
Performs in-depth analysis on:
  1. FracAtlas.zip
  2. Bone Scan Metastasis Dataset.zip
  3. BTXRD.zip
"""

from collections import Counter, defaultdict
import csv
from io import BytesIO, StringIO
from pathlib import Path
import zipfile
import openpyxl


def audit_fracatlas(zf: zipfile.ZipFile):
    print("\n" + "=" * 80)
    print("TASK 2 FORENSIC AUDIT: FRACATLAS")

    # 1. Inspect dataset.csv
    csv_members = [m for m in zf.namelist() if Path(m).name.lower() == "dataset.csv"]
    print(f"Target classification CSV: {csv_members}")
    
    dataset_records = []
    if csv_members:
        content = zf.read(csv_members[0]).decode("utf-8", errors="ignore")
        reader = list(csv.DictReader(StringIO(content)))
        dataset_records = reader
        print(f"Total rows in dataset.csv: {len(reader)}")
        if reader:
            print(f"Columns: {list(reader[0].keys())}")
            # Identify label column (fractured)
            fracture_counts = Counter(r.get("fractured", r.get("fracture", "")) for r in reader)
            print(f"Fracture distribution (fractured column): {fracture_counts}")

            # Check patient identifier or image_id
            id_counts = Counter(r.get("image_id", "") for r in reader)
            print(f"Unique image_ids in dataset.csv: {len(id_counts)}")

            # Check anatomical / bone representation
            bones = [r.get("bone", "") for r in reader if r.get("bone")]
            if bones:
                print(f"Bones distribution (sample 5): {Counter(bones).most_common(5)}")

    # 2. Check fractured and non_fractured image counts inside images directory
    all_imgs = [m for m in zf.namelist() if Path(m).suffix.lower() in [".jpg", ".jpeg", ".png"]]
    print(f"Total image files in FracAtlas: {len(all_imgs)}")

    fractured_imgs = [m for m in all_imgs if "fractured" in Path(m).parts and "non_fractured" not in Path(m).parts]
    non_fractured_imgs = [m for m in all_imgs if "non_fractured" in Path(m).parts]
    print(f"Images in 'Fractured' folders: {len(fractured_imgs)}")
    print(f"Images in 'Non_fractured' folders: {len(non_fractured_imgs)}")

    # 3. Check duplicate filenames
    filenames = [Path(m).name for m in all_imgs]
    fn_counts = Counter(filenames)
    dups = {k: v for k, v in fn_counts.items() if v > 1}
    print(f"Duplicate image filenames across archive: {len(dups)}")
    if dups:
        print(f"Sample duplicate filenames: {list(dups.items())[:5]}")

    # 4. Check COCO, VGG, VOC annotations
    coco_files = [m for m in zf.namelist() if "coco" in m.lower() and m.endswith(".json")]
    vgg_files = [m for m in zf.namelist() if "vgg" in m.lower() or "via" in m.lower() or ("annotation" in m.lower() and m.endswith(".json") and "coco" not in m.lower())]
    voc_files = [m for m in zf.namelist() if m.endswith(".xml")]
    txt_yolo_files = [m for m in zf.namelist() if "yolo" in m.lower() or (m.endswith(".txt") and "annotation" in m.lower())]

    print(f"COCO annotation files: {coco_files}")
    print(f"VGG / other JSON annotation files: {vgg_files}")
    print(f"PASCAL VOC XML annotation files: {len(voc_files)}")
    print(f"YOLO TXT annotation files: {len(txt_yolo_files)}")

    # 5. Check pre-existing splits
    split_files = [m for m in zf.namelist() if any(k in m.lower() for k in ["train", "valid", "test", "split"]) and (m.endswith(".csv") or m.endswith(".txt"))]
    print(f"Pre-existing train/val/test split files: {split_files}")
    for sf in split_files:
        try:
            sample_content = zf.read(sf).decode("utf-8", errors="ignore").strip().splitlines()
            print(f"  {sf}: {len(sample_content)} lines (first 3: {sample_content[:3]})")
        except Exception as e:
            print(f"  {sf}: error reading: {e}")


def audit_bone_scan(zf: zipfile.ZipFile):
    print("\n" + "=" * 80)
    print("TASK 3 FORENSIC AUDIT: BONE METASTASIS DATASET")

    # 1. Read Excel workbook
    xlsx_members = [m for m in zf.namelist() if m.endswith(".xlsx") and not Path(m).name.startswith("~$")]
    print(f"Excel workbook: {xlsx_members}")

    patients_meta = {}
    if xlsx_members:
        data = BytesIO(zf.read(xlsx_members[0]))
        wb = openpyxl.load_workbook(data, data_only=True)
        sheet = wb.active
        rows = list(sheet.iter_rows(values_only=True))
        header = rows[0]
        data_rows = rows[1:]

        valid_patient_rows = []
        for r in data_rows:
            # Check if row has valid patient entry
            patient_id = r[0]
            if patient_id is not None and str(patient_id).strip() != "":
                valid_patient_rows.append(r)

        print(f"Total non-empty patient rows in Excel: {len(valid_patient_rows)}")

        # Extract patient metadata
        # Header: ('PATIENT', 'IMAGE ANTERIOR VIEW', 'IMAGE POSTERIOR VIEW', 'FORMAT', 'ACQUISITION DATE', 'SEX', 'AGE', 'WEIGHT', 'HEIGHT', 'DIAGNOSIS CLASSIFICATION')
        diag_counts = Counter()
        images_expected = set()

        for r in valid_patient_rows:
            pid = str(r[0]).strip()
            ant_img = str(r[1]).strip() if r[1] else ""
            post_img = str(r[2]).strip() if r[2] else ""
            fmt = str(r[3]).strip() if r[3] else "PNG"
            diag = str(r[9]).strip() if len(r) > 9 and r[9] else "UNKNOWN"

            diag_counts[diag] += 1
            if ant_img:
                images_expected.add(ant_img if ant_img.lower().endswith(".png") else f"{ant_img}.png")
            if post_img:
                images_expected.add(post_img if post_img.lower().endswith(".png") else f"{post_img}.png")

            patients_meta[pid] = {
                "patient_id": pid,
                "anterior_image": ant_img,
                "posterior_image": post_img,
                "diagnosis": diag,
                "sex": r[5],
                "age": r[6],
                "weight": r[7],
                "height": r[8],
            }

        print(f"Unique patients parsed: {len(patients_meta)}")
        print(f"Diagnosis distribution: {diag_counts}")
        print(f"Total expected image filenames from Excel: {len(images_expected)}")

    # 2. Inspect physical images in archive
    png_members = [m for m in zf.namelist() if Path(m).suffix.lower() == ".png"]
    png_basenames = {Path(m).name for m in png_members}
    print(f"Total PNG images in archive: {len(png_members)}")
    print(f"Unique PNG basenames: {len(png_basenames)}")

    # Check match between expected and actual
    if images_expected:
        missing = images_expected - png_basenames
        extra = png_basenames - images_expected
        print(f"Missing images referenced in Excel: {len(missing)}")
        if missing:
            print(f"  Sample missing: {list(missing)[:5]}")
        print(f"Extra images in archive not in Excel: {len(extra)}")
        if extra:
            print(f"  Sample extra: {list(extra)[:5]}")


def audit_btxrd(zf: zipfile.ZipFile):
    print("\n" + "=" * 80)
    print("TASK 4 FORENSIC AUDIT: BTXRD")

    # 1. Read dataset.xlsx (skip ~$ lock files)
    xlsx_members = [m for m in zf.namelist() if m.endswith(".xlsx") and not Path(m).name.startswith("~$")]
    print(f"Valid Excel workbooks: {xlsx_members}")

    if xlsx_members:
        data = BytesIO(zf.read(xlsx_members[0]))
        wb = openpyxl.load_workbook(data, data_only=True)
        sheet = wb["Sheet1"]
        rows = list(sheet.iter_rows(values_only=True))
        header = rows[0]
        data_rows = rows[1:]
        print(f"Total rows in Sheet1: {len(data_rows)}")
        print(f"Columns ({len(header)}): {header}")

        # Check pathology columns
        # 'tumor', 'benign', 'malignant', 'osteochondroma', 'multiple osteochondromas', 'simple bone cyst', 'giant cell tumor', 'osteofibroma', 'synovial osteochondroma', 'other bt', 'osteosarcoma', 'other mt'
        path_cols = ['tumor', 'benign', 'malignant', 'osteochondroma', 'multiple osteochondromas', 'simple bone cyst', 'giant cell tumor', 'osteofibroma', 'synovial osteochondroma', 'other bt', 'osteosarcoma', 'other mt']
        col_indices = {name: idx for idx, name in enumerate(header) if name in path_cols}

        print("Pathology counts in BTXRD Excel:")
        for name, idx in col_indices.items():
            pos_count = sum(1 for r in data_rows if r[idx] == 1 or r[idx] == "1")
            print(f"  {name}: {pos_count}")

    # 2. Inspect image files
    img_members = [m for m in zf.namelist() if Path(m).suffix.lower() in [".jpg", ".jpeg", ".png"]]
    print(f"Total image files in BTXRD: {len(img_members)}")

    # 3. Inspect JSON annotations
    json_members = [m for m in zf.namelist() if m.endswith(".json")]
    print(f"Total JSON annotation files: {len(json_members)}")
    if json_members:
        # Sample one JSON
        sample_json = json_members[0]
        try:
            content = zf.read(sample_json).decode("utf-8")
            print(f"Sample JSON ({Path(sample_json).name}) length: {len(content)} chars (first 150 chars: {content[:150]})")
        except Exception as e:
            print(f"Error reading sample JSON: {e}")


def main():
    archives_dir = Path("ml_service/data/raw/_archives")
    for arc in sorted(archives_dir.glob("*.zip")):
        with zipfile.ZipFile(arc, "r") as zf:
            if "fracatlas" in arc.name.lower():
                audit_fracatlas(zf)
            elif "bone scan" in arc.name.lower():
                audit_bone_scan(zf)
            elif "btxrd" in arc.name.lower():
                audit_btxrd(zf)


if __name__ == "__main__":
    main()
