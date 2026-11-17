"""
Forensic audit of Bone Scan Dataset.
"""
from collections import Counter
from io import BytesIO
import hashlib
from pathlib import Path
import zipfile
import openpyxl


def audit():
    arc_name = Path("ml_service/data/raw/_archives/Bone Scan Images Dataset for Study of Bone Metastases in Adult Breast Cancer Patients at IICS-UNA Paraguay.zip")
    with zipfile.ZipFile(arc_name) as zf:
        excel_name = "Bone Scan Images Dataset for Study of Bone Metastases in Adult Breast Cancer Patients at IICS-UNA Paraguay/Bone Scan Images Dataset.xlsx"
        data = BytesIO(zf.read(excel_name))
        wb = openpyxl.load_workbook(data, data_only=True)
        sheet = wb.active
        rows = list(sheet.iter_rows(values_only=True))[1:]

        patients = []
        sha_map = {}

        for r in rows:
            if r[0] is None:
                continue
            pid, ant, post, fmt, acq_date, sex, age, wt, ht, diag = r[:10]
            ant_fn = str(ant).strip() if ant else ""
            post_fn = str(post).strip() if post else ""
            if not ant_fn.lower().endswith(".png"):
                ant_fn += ".png"
            if not post_fn.lower().endswith(".png"):
                post_fn += ".png"

            patients.append({
                "pid": str(pid).strip(),
                "ant": ant_fn,
                "post": post_fn,
                "diag": str(diag).strip(),
                "sex": str(sex).strip() if sex else "",
                "age": age,
                "weight": wt,
                "height": ht,
            })

        print("Total parsed patients:", len(patients))
        diag_counts = Counter(p["diag"] for p in patients)
        print("Diagnosis counts:", diag_counts)

        # Sex counts
        sex_counts = Counter(p["sex"] for p in patients)
        print("Sex counts:", sex_counts)

        prefix = "Bone Scan Images Dataset for Study of Bone Metastases in Adult Breast Cancer Patients at IICS-UNA Paraguay"
        namelist = set(zf.namelist())
        missing_files = []

        for p in patients:
            ant_path = f"{prefix}/{p['ant']}"
            post_path = f"{prefix}/{p['post']}"

            if ant_path not in namelist:
                missing_files.append(ant_path)
            else:
                sha_map[ant_path] = hashlib.sha256(zf.read(ant_path)).hexdigest()

            if post_path not in namelist:
                missing_files.append(post_path)
            else:
                sha_map[post_path] = hashlib.sha256(zf.read(post_path)).hexdigest()

        print("Missing image files referenced in Excel:", len(missing_files))
        print("Total verified image files:", len(sha_map))

        # Check content duplicates
        sha_counts = Counter(sha_map.values())
        content_dups = {sha: count for sha, count in sha_counts.items() if count > 1}
        print("Duplicate image content hashes count:", len(content_dups))
        if content_dups:
            for sha, count in content_dups.items():
                matched = [Path(f).name for f, h in sha_map.items() if h == sha]
                print(f"  Hash {sha[:12]}... shared by ({count} files): {matched}")


if __name__ == "__main__":
    audit()

