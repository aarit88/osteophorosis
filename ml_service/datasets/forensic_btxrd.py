"""
Forensic inspection of BTXRD dataset.
"""
from collections import Counter
from io import BytesIO
import hashlib
from pathlib import Path
import zipfile
import openpyxl


def audit():
    arc_name = Path("ml_service/data/raw/_archives/BTXRD.zip")
    with zipfile.ZipFile(arc_name) as zf:
        excel_name = "BTXRD/dataset.xlsx"
        data = BytesIO(zf.read(excel_name))
        wb = openpyxl.load_workbook(data, data_only=True)
        sheet = wb["Sheet1"]
        rows = list(sheet.iter_rows(values_only=True))
        header = rows[0]
        data_rows = rows[1:]

        print(f"Total rows in BTXRD dataset.xlsx: {len(data_rows)}")
        print("Header columns:", header)

        # Check image files in archive
        img_members = [m for m in zf.namelist() if Path(m).suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"Total image files in BTXRD: {len(img_members)}")

        # Check duplicate image filenames
        basenames = [Path(m).name for m in img_members]
        fn_counts = Counter(basenames)
        fn_dups = {k: v for k, v in fn_counts.items() if v > 1}
        print(f"Duplicate filenames: {len(fn_dups)}")

        # Check SHA-256 duplicates across image files
        sha_map = {}
        for m in img_members:
            sha_map[m] = hashlib.sha256(zf.read(m)).hexdigest()

        sha_counts = Counter(sha_map.values())
        content_dups = {h: c for h, c in sha_counts.items() if c > 1}
        print(f"Duplicate image content hashes: {len(content_dups)}")
        if content_dups:
            for h, c in list(content_dups.items())[:5]:
                matched = [Path(m).name for m, s in sha_map.items() if s == h]
                print(f"  Hash {h[:10]}... shared by {c} files: {matched}")

        # Check non-tumor / normal rows
        tumor_idx = header.index("tumor")
        benign_idx = header.index("benign")
        mal_idx = header.index("malignant")

        tumor_counts = Counter(r[tumor_idx] for r in data_rows)
        print("Tumor column distribution:", tumor_counts)

        # Centers / Hospitals
        center_idx = header.index("center")
        centers = Counter(r[center_idx] for r in data_rows)
        print("Clinical centers / acquisition sites:", centers)

        # Age and Gender
        gender_idx = header.index("gender")
        genders = Counter(r[gender_idx] for r in data_rows)
        print("Gender distribution:", genders)

        # JSON annotations check
        json_members = [m for m in zf.namelist() if m.endswith(".json")]
        print(f"Total JSON annotation files: {len(json_members)}")


if __name__ == "__main__":
    audit()

