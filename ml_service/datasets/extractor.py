"""
OsteoGuard AI - Safe Dataset Archive Extractor.
Extracts verified raw datasets with path-traversal (Zip Slip) protection.
Target directories:
  1. ml_service/data/raw/fracatlas/
  2. ml_service/data/raw/bone_scan_metastasis/
  3. ml_service/data/raw/btxrd/
"""

import os
from pathlib import Path
import sys
import zipfile


def safe_extract_archive(archive_path: Path, target_dir: Path, strip_root_folder: bool = True):
    """
    Extracts all members of archive_path into target_dir with strict Zip Slip protection.
    If strip_root_folder is True and all members share a single top-level folder,
    that folder prefix is stripped so target_dir contains the dataset contents directly.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_target = target_dir.resolve()
    print(f"Extracting {archive_path.name} -> {target_dir}...")

    with zipfile.ZipFile(archive_path, "r") as zf:
        infolist = zf.infolist()
        print(f"  Archive contains {len(infolist)} members.")

        # Determine common root prefix if applicable
        common_root = None
        if strip_root_folder:
            top_parts = set()
            for info in infolist:
                parts = Path(info.filename).parts
                if parts:
                    top_parts.add(parts[0])
            if len(top_parts) == 1:
                common_root = list(top_parts)[0]
                print(f"  Detected single root folder '{common_root}' - stripping for clean extraction.")

        extracted_count = 0
        for info in infolist:
            filename = info.filename

            # Zip Slip security check: Disallow absolute paths and directory traversal
            if ".." in filename or filename.startswith(("/", "\\")):
                raise ValueError(f"Security Alert: Path traversal attempt detected in archive member: {filename}")

            if common_root and filename.startswith(common_root):
                rel_path = Path(filename).relative_to(common_root)
            else:
                rel_path = Path(filename)

            if str(rel_path) in [".", ""]:
                continue

            destination = (resolved_target / rel_path).resolve()

            # Enforce destination is strictly inside target directory
            try:
                destination.relative_to(resolved_target)
            except ValueError:
                raise ValueError(f"Security Alert: Member {filename} attempts escape outside {resolved_target}")

            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(destination, "wb") as dst:
                    # Stream chunks
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                extracted_count += 1

        print(f"  Successfully extracted {extracted_count} files into {target_dir}.\n", flush=True)


def extract_all():
    archives_dir = Path("ml_service/data/raw/_archives")
    raw_dir = Path("ml_service/data/raw")

    mapping = [
        ("FracAtlas.zip", raw_dir / "fracatlas"),
        ("Bone Scan Images Dataset for Study of Bone Metastases in Adult Breast Cancer Patients at IICS-UNA Paraguay.zip", raw_dir / "bone_scan_metastasis"),
        ("BTXRD.zip", raw_dir / "btxrd"),
    ]

    for arc_filename, target_folder in mapping:
        arc_path = archives_dir / arc_filename
        if not arc_path.exists():
            print(f"ERROR: Archive not found: {arc_path}", file=sys.stderr)
            continue
        safe_extract_archive(arc_path, target_folder, strip_root_folder=True)

    print("All archives extracted safely with path-traversal protection.")


if __name__ == "__main__":
    extract_all()

