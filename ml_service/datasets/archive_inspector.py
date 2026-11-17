"""
OsteoGuard AI - Dataset Archive Forensic Inspector.
Inspects zip archives in ml_service/data/raw/_archives/:
  1. Archive integrity check (CRC-32 via zipfile.testzip()).
  2. Size, compression ratio, file count, and member hierarchy.
  3. Image file format enumeration.
  4. Metadata inspection (CSV, Excel, JSON).
"""

from collections import Counter
from io import BytesIO
from pathlib import Path
import zipfile
import openpyxl


def inspect_archives(archives_dir: Path):
    archives = list(archives_dir.glob("*.zip"))
    print(f"Found {len(archives)} archive(s) in {archives_dir}")

    for arc in sorted(archives, key=lambda p: p.name):
        print("=" * 80)
        print(f"ARCHIVE: {arc.name}")
        compressed_size = arc.stat().st_size
        print(f"File size: {compressed_size:,} bytes ({compressed_size / (1024 * 1024):.2f} MB)")

        try:
            with zipfile.ZipFile(arc, "r") as zf:
                print("Testing archive CRC-32 integrity...")
                corrupt_file = zf.testzip()
                if corrupt_file:
                    print(f"ERROR: Corrupt member found: {corrupt_file}")
                else:
                    print("Integrity: PASS (All CRC-32 checksums valid)")

                infolist = zf.infolist()
                total_uncompressed = sum(info.file_size for info in infolist)
                print(f"Total members: {len(infolist)}")
                print(f"Uncompressed size: {total_uncompressed:,} bytes ({total_uncompressed / (1024 * 1024):.2f} MB)")

                # Extension counter
                ext_counter = Counter(Path(info.filename).suffix.lower() for info in infolist if not info.is_dir())
                print("Extension breakdown:")
                for ext, count in ext_counter.most_common(15):
                    print(f"  {ext or '[no extension]'}: {count}")

                # Top-level paths
                top_levels = set()
                for info in infolist:
                    parts = Path(info.filename).parts
                    if parts:
                        top_levels.add(parts[0])
                print(f"Top-level directories/files: {sorted(list(top_levels))}")

                # List metadata files
                metadata_files = [info.filename for info in infolist if Path(info.filename).suffix.lower() in [".csv", ".xlsx", ".xls", ".json", ".txt", ".xml", ".yaml", ".yml"]]
                print(f"Metadata files ({len(metadata_files)}):")
                for mf in sorted(metadata_files)[:30]:
                    print(f"  - {mf}")

        except Exception as e:
            print(f"EXCEPTION inspecting {arc.name}: {e}")


if __name__ == "__main__":
    archives_path = Path("ml_service/data/raw/_archives")
    inspect_archives(archives_path)

