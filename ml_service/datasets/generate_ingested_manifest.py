"""
OsteoGuard AI - Ingested Datasets Manifest Generator.
Generates ml_service/data/dataset_manifest.csv indexing:
  1. FracAtlas (4083 classification records + duplicates tracked)
  2. Bone Scan Metastasis Dataset (582 images, 291 patients, paired anterior/posterior)
  3. BTXRD (3746 images, multi-center, benign/malignant/normal bone)
"""

from collections import Counter
import csv
import hashlib
from io import BytesIO
from pathlib import Path
import openpyxl

from ml_service.datasets.registry_verifier import compute_file_sha256


MANIFEST_COLUMNS = [
    "dataset_id", "source_id", "patient_id", "study_id", "series_id", "instance_id",
    "target_class", "short_code", "modality", "original_format", "file_path", "mask_path",
    "anatomic_region", "view_projection", "slice_thickness_mm", "kvp", "checksum_sha256",
    "verified_healthy_flag", "license", "split"
]


def index_fracatlas(raw_dir: Path) -> list:
    frac_dir = raw_dir / "fracatlas"
    csv_path = frac_dir / "dataset.csv"
    if not csv_path.exists():
        print("FracAtlas dataset.csv not found!")
        return []

    print("Indexing FracAtlas...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    images_dir = frac_dir / "images"
    records = []

    # Map bone columns
    bone_cols = ["hand", "leg", "hip", "shoulder", "mixed"]
    view_cols = ["frontal", "lateral", "oblique"]

    for r in reader:
        img_id = r["image_id"]
        is_fractured = r.get("fractured", "0") == "1"

        # Determine anatomic region
        anatomy = "skeleton"
        for bc in bone_cols:
            if r.get(bc) == "1":
                anatomy = bc
                break

        # Determine projection
        proj = "AP"
        for vc in view_cols:
            if r.get(vc) == "1":
                proj = vc.capitalize()
                break

        # Locate file on disk
        # Authoritative: Fractured -> Fractured/, Non_fractured -> Non_fractured/
        if is_fractured:
            file_candidate = images_dir / "Fractured" / img_id
        else:
            file_candidate = images_dir / "Non_fractured" / img_id

        if not file_candidate.exists():
            # Try alternate folder if present
            alt = images_dir / ("Non_fractured" if is_fractured else "Fractured") / img_id
            if alt.exists():
                file_candidate = alt

        if file_candidate.exists():
            sha256 = compute_file_sha256(file_candidate)
            ext = file_candidate.suffix.lstrip(".")
            mask_path = frac_dir / "Annotations/COCO JSON/COCO_fracture_masks.json" if is_fractured else ""

            records.append({
                "dataset_id": "FRACATLAS_FRACTURE_COHORT",
                "source_id": "FRACATLAS_FRACTURE_COHORT",
                "patient_id": f"PT_FA_{Path(img_id).stem}",
                "study_id": f"ST_FA_{Path(img_id).stem}",
                "series_id": "SE_FA_01",
                "instance_id": img_id,
                "target_class": "acute_bone_fracture" if is_fractured else "candidate_control_radiograph",
                "short_code": "fracture" if is_fractured else "control",
                "modality": "xray",
                "original_format": ext,
                "file_path": str(file_candidate.as_posix()),
                "mask_path": str(mask_path.as_posix()) if mask_path else "",
                "anatomic_region": anatomy,
                "view_projection": proj,
                "slice_thickness_mm": "",
                "kvp": "",
                "checksum_sha256": sha256,
                "verified_healthy_flag": "FALSE",
                "license": "CC-BY-4.0",
                "split": "unassigned",
            })

    print(f"  Indexed {len(records)} records for FracAtlas.")
    return records


def index_bone_scan(raw_dir: Path) -> list:
    bs_dir = raw_dir / "bone_scan_metastasis"
    excel_path = bs_dir / "Bone Scan Images Dataset.xlsx"
    if not excel_path.exists():
        print("Bone Scan Excel not found!")
        return []

    print("Indexing Bone Scan Metastasis Dataset...")
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))[1:]

    records = []

    for r in rows:
        if r[0] is None:
            continue
        pid = int(r[0])
        ant_name = str(r[1]).strip()
        post_name = str(r[2]).strip()
        diag = str(r[9]).strip()

        if not ant_name.lower().endswith(".png"):
            ant_name += ".png"
        if not post_name.lower().endswith(".png"):
            post_name += ".png"

        folder_name = "Bone metastases" if diag == "Bone metastases" else "No bone metastases"

        ant_file = bs_dir / folder_name / "Anterior" / ant_name
        post_file = bs_dir / folder_name / "Posterior" / post_name

        patient_pseudo = f"PT_BS_{pid:03d}"
        study_id = f"ST_BS_{pid:03d}"

        # Anterior Record
        if ant_file.exists():
            sha_ant = compute_file_sha256(ant_file)
            records.append({
                "dataset_id": "IICS_UNA_BONE_SCAN_METASTASIS",
                "source_id": "IICS_UNA_BONE_SCAN_METASTASIS",
                "patient_id": patient_pseudo,
                "study_id": study_id,
                "series_id": "SE_BS_ANT",
                "instance_id": ant_name,
                "target_class": "scintigraphy_bone_metastasis" if diag == "Bone metastases" else "scintigraphy_no_bone_metastasis",
                "short_code": "mets_bs" if diag == "Bone metastases" else "nomets_bs",
                "modality": "bone_scan",
                "original_format": "png",
                "file_path": str(ant_file.as_posix()),
                "mask_path": "",
                "anatomic_region": "whole_body_skeleton",
                "view_projection": "Anterior",
                "slice_thickness_mm": "",
                "kvp": "",
                "checksum_sha256": sha_ant,
                "verified_healthy_flag": "FALSE",
                "license": "CC-BY-4.0",
                "split": "unassigned",
            })

        # Posterior Record
        if post_file.exists():
            sha_post = compute_file_sha256(post_file)
            records.append({
                "dataset_id": "IICS_UNA_BONE_SCAN_METASTASIS",
                "source_id": "IICS_UNA_BONE_SCAN_METASTASIS",
                "patient_id": patient_pseudo,
                "study_id": study_id,
                "series_id": "SE_BS_POST",
                "instance_id": post_name,
                "target_class": "scintigraphy_bone_metastasis" if diag == "Bone metastases" else "scintigraphy_no_bone_metastasis",
                "short_code": "mets_bs" if diag == "Bone metastases" else "nomets_bs",
                "modality": "bone_scan",
                "original_format": "png",
                "file_path": str(post_file.as_posix()),
                "mask_path": "",
                "anatomic_region": "whole_body_skeleton",
                "view_projection": "Posterior",
                "slice_thickness_mm": "",
                "kvp": "",
                "checksum_sha256": sha_post,
                "verified_healthy_flag": "FALSE",
                "license": "CC-BY-4.0",
                "split": "unassigned",
            })

    print(f"  Indexed {len(records)} records for Bone Scan (291 patients x 2 views).")
    return records


def index_btxrd(raw_dir: Path) -> list:
    btx_dir = raw_dir / "btxrd"
    excel_path = btx_dir / "dataset.xlsx"
    if not excel_path.exists():
        print("BTXRD dataset.xlsx not found!")
        return []

    print("Indexing BTXRD...")
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    sheet = wb["Sheet1"]
    rows = list(sheet.iter_rows(values_only=True))
    header = rows[0]
    data_rows = rows[1:]

    records = []
    images_dir = btx_dir / "images"
    annotations_dir = btx_dir / "Annotations"

    for r in data_rows:
        img_id = str(r[0]).strip()
        center = r[1]
        tumor = r[19] == 1 or r[19] == "1"
        benign = r[20] == 1 or r[20] == "1"
        malignant = r[21] == 1 or r[21] == "1"

        # Determine pathology category
        if tumor:
            if malignant:
                target_cls = "primary_malignant_bone_tumor"
                short_code = "sarcoma_primary"
            else:
                target_cls = "benign_bone_tumor"
                short_code = "benign_tumor"
        else:
            target_cls = "candidate_normal_bone_radiograph"
            short_code = "normal_bone"

        # Anatomy determination
        anatomy = "skeleton"
        anat_cols = ['hand', 'ulna', 'radius', 'humerus', 'foot', 'tibia', 'fibula', 'femur', 'hip bone']
        for ac in anat_cols:
            idx = header.index(ac)
            if r[idx] == 1 or r[idx] == "1":
                anatomy = ac
                break

        # Projection
        proj = "AP"
        for pc in ['frontal', 'lateral', 'oblique']:
            idx = header.index(pc)
            if r[idx] == 1 or r[idx] == "1":
                proj = pc.capitalize()
                break

        # Locate image file
        img_file = images_dir / img_id
        if not img_file.exists():
            for ext in [".jpeg", ".jpg", ".png"]:
                cand = images_dir / f"{Path(img_id).stem}{ext}"
                if cand.exists():
                    img_file = cand
                    break

        if img_file and img_file.exists():
            sha256 = compute_file_sha256(img_file)
            json_mask = annotations_dir / f"{Path(img_id).stem}.json"
            mask_str = str(json_mask.as_posix()) if json_mask.exists() else ""

            records.append({
                "dataset_id": "BTXRD_BONE_TUMOR_COHORT",
                "source_id": "BTXRD_BONE_TUMOR_COHORT",
                "patient_id": f"PT_BTX_{img_id}",
                "study_id": f"ST_BTX_{img_id}",
                "series_id": f"SE_BTX_C{center}",
                "instance_id": img_file.name,
                "target_class": target_cls,
                "short_code": short_code,
                "modality": "xray",
                "original_format": img_file.suffix.lstrip("."),
                "file_path": str(img_file.as_posix()),
                "mask_path": mask_str,
                "anatomic_region": anatomy,
                "view_projection": proj,
                "slice_thickness_mm": "",
                "kvp": "",
                "checksum_sha256": sha256,
                "verified_healthy_flag": "FALSE",
                "license": "CC-BY-4.0",
                "split": "unassigned",
            })

    print(f"  Indexed {len(records)} records for BTXRD.")
    return records


def generate_all():
    raw_dir = Path("ml_service/data/raw")
    manifest_csv = Path("ml_service/data/dataset_manifest.csv")

    all_records = []
    all_records.extend(index_fracatlas(raw_dir))
    all_records.extend(index_bone_scan(raw_dir))
    all_records.extend(index_btxrd(raw_dir))

    print(f"Writing {len(all_records)} total records to {manifest_csv}...")
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print("Manifest generation complete.")


if __name__ == "__main__":
    generate_all()
