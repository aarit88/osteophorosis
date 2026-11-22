#!/usr/bin/env python3
"""
OsteoGuard AI — Stage 33/34 Recursive Dataset Integrity, Metadata Completeness,
Duplicate Patient/Image, and Clinical Label Ontology Validator.

Recursively scans `ml_service/data/raw/` (and optional manifest), validates physical file
integrity (format, byte size, dimensions, SHA-256 hash), checks metadata & Stage 34
diagnostic provenance completeness, detects duplicate patient IDs / conflicting diagnoses
and duplicate/near-duplicate images, and outputs a structured audit summary.

Usage:
  python3 ml_service/scripts/validate_dataset.py
  python3 ml_service/scripts/validate_dataset.py --manifest ml_service/data/manifests/track_d_manifest.json
  python3 ml_service/scripts/validate_dataset.py --strict --json-out ml_service/data/reports/dataset_validation_cli.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_service.datasets.clinical_ontology import ONTOLOGY_VERSION
from ml_service.datasets.track_d_intake import (
    discover_track_d_records,
    evaluate_track_d_readiness,
    generate_leakage_safe_splits,
    inspect_image_file,
    validate_track_d_dataset,
)


def scan_raw_directory_files(raw_dir: Path, root_dir: Path) -> Dict[str, Any]:
    """
    Recursively scans every non-hidden file under `ml_service/data/raw/` and verifies
    file size, binary header format, pixel dimensions, SHA-256 digest, and duplicate hashes.
    """
    files_inspected: List[Dict[str, Any]] = []
    sha_to_paths: Dict[str, List[str]] = defaultdict(list)
    patient_id_to_paths: Dict[str, List[str]] = defaultdict(list)
    invalid_files: List[Dict[str, Any]] = []

    if raw_dir.exists():
        for file_path in sorted(raw_dir.rglob("*")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            rel = file_path.relative_to(root_dir).as_posix()
            size_bytes = file_path.stat().st_size
            info = inspect_image_file(file_path)
            stem = file_path.stem
            inferred_pid = stem.rsplit("_", 1)[0] if stem.startswith(("PT_", "SUBJ_")) and "_" in stem[3:] else ""
            if inferred_pid:
                patient_id_to_paths[inferred_pid].append(rel)
            if info.get("sha256"):
                sha_to_paths[str(info["sha256"])].append(rel)

            entry = {
                "path": rel,
                "size_bytes": size_bytes,
                "valid": bool(info["valid"]),
                "error": info.get("error"),
                "image_format": info.get("image_format"),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "bit_depth": info.get("bit_depth", 0),
                "sha256": info.get("sha256", ""),
                "inferred_patient_id": inferred_pid or None,
            }
            files_inspected.append(entry)
            if not info["valid"]:
                invalid_files.append(entry)

    duplicate_image_hashes = {sha: paths for sha, paths in sha_to_paths.items() if len(paths) > 1}
    repeated_patient_ids = {pid: paths for pid, paths in patient_id_to_paths.items() if len(paths) > 1}

    return {
        "raw_dir": str(raw_dir),
        "raw_files_found": len(files_inspected),
        "valid_raw_files": len(files_inspected) - len(invalid_files),
        "invalid_raw_files": len(invalid_files),
        "duplicate_image_hashes_count": len(duplicate_image_hashes),
        "duplicate_image_hashes": duplicate_image_hashes,
        "repeated_patient_ids_in_raw": repeated_patient_ids,
        "files": files_inspected,
    }


def run_dataset_validation_cli(
    root_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    raw_dir = root_dir / "ml_service" / "data" / "raw"
    raw_scan = scan_raw_directory_files(raw_dir=raw_dir, root_dir=root_dir)

    records, discovery_mode = discover_track_d_records(root_dir=root_dir, manifest_path=manifest_path)
    validation = validate_track_d_dataset(root_dir=root_dir, records=records)
    splits = generate_leakage_safe_splits(validation["passed_records"], seed=42)
    readiness = evaluate_track_d_readiness(validation, splits)

    patient_occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in validation["records"]:
        pid = str(rec.get("patient_id") or "").strip()
        if pid:
            patient_occurrences[pid].append({
                "sample_id": rec.get("sample_id"),
                "diagnostic_class": rec.get("diagnostic_class"),
                "imaging_modality": rec.get("imaging_modality"),
                "image_path": rec.get("image_path"),
            })
    duplicate_patient_ids = {
        pid: items for pid, items in patient_occurrences.items() if len(items) > 1
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ontology_version": ONTOLOGY_VERSION,
        "discovery_mode": discovery_mode,
        "raw_directory_scan": raw_scan,
        "metadata_and_provenance": {
            "total_records_inspected": validation["total_inspected"],
            "passed_count": validation["passed_count"],
            "quarantined_count": validation["quarantined_count"],
            "missing_metadata_fields_count": validation["missing_metadata_fields_count"],
            "label_provenance_completeness_pct": validation["label_provenance_completeness_pct"],
            "healthy_control_eligibility_failures": validation["healthy_control_eligibility_failures"],
        },
        "duplicates_and_conflicts": {
            "exact_image_duplicates": validation["exact_duplicate_count"],
            "near_perceptual_duplicates": validation["near_duplicate_count"],
            "conflicting_patient_diagnoses": validation["conflicting_patient_count"],
            "conflicting_diagnoses_records": validation["conflicting_diagnoses_count"],
            "duplicate_patient_ids_count": len(duplicate_patient_ids),
            "duplicate_patient_ids": duplicate_patient_ids,
        },
        "adjudication_summary": validation["adjudication_counts"],
        "accepted_by_class": validation["accepted_by_class"],
        "readiness_status": readiness["readiness_status"],
        "ready_for_training": readiness["ready_for_training"],
        "gates": readiness["gates"],
        "blockers": readiness["blockers"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OsteoGuard AI — Recursive Raw Dataset Integrity, Metadata & Ontology Validator"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=REPO_ROOT,
        help="Workspace root directory (defaults to repository root)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit Track D JSON or CSV manifest path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path to save the validation JSON report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if quarantined samples, corrupt files, or blocked gates exist",
    )
    args = parser.parse_args(argv)

    summary = run_dataset_validation_cli(
        root_dir=args.root_dir.resolve(),
        manifest_path=args.manifest,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if args.strict and (
        summary["raw_directory_scan"]["invalid_raw_files"] > 0
        or summary["metadata_and_provenance"]["quarantined_count"] > 0
        or not summary["ready_for_training"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
