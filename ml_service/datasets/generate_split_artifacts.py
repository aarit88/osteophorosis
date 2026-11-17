"""
OsteoGuard AI - Dataset Audit and Split Artifact Generator.
Generates:
  1. dataset_quality_report.json
  2. class_distribution.csv
  3. duplicate_audit.csv
  4. split_manifest.csv
"""

from pathlib import Path
import shutil
import sys

from ml_service.datasets.auditor import DatasetAuditor
from ml_service.datasets.partitioner import PatientLevelPartitioner


def generate_artifacts():
    auditor = DatasetAuditor()
    audit_res = auditor.audit_dataset_directory()
    print(f"Audit status: {audit_res['status']}")
    print(f"Total samples: {audit_res['total_samples']}")
    print(f"Blockers: {audit_res['blockers']}")

    splits_dir = Path("ml_service/data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    exported = auditor.export_audit_artifacts(audit_res, splits_dir)

    partitioner = PatientLevelPartitioner()
    partitioned, manifest_path = partitioner.partition_from_audit(
        audit_res,
        output_path=splits_dir / "split_manifest.csv"
    )
    print(f"Split manifest rows: {len(partitioned)}")

    # Mirror to ml_artifacts/reports
    reports_dir = Path("ml_artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        "dataset_quality_report.json",
        "class_distribution.csv",
        "duplicate_audit.csv",
        "split_manifest.csv",
    ]
    for fname in artifacts:
        src = splits_dir / fname
        dst = reports_dir / fname
        shutil.copyfile(src, dst)
        print(f"Generated: {src} -> {dst}")

    print("All 4 artifacts generated successfully.")


if __name__ == "__main__":
    generate_artifacts()

