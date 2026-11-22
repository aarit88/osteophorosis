#!/usr/bin/env python3
"""
OsteoGuard AI — Stage 33 Track D Dataset Intake, Validation, Split & Readiness CLI.

Usage:
  python3 ml_service/scripts/run_track_d_intake.py --all
  python3 ml_service/scripts/run_track_d_intake.py --manifest ml_service/data/manifests/track_d_manifest.json --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_service.datasets.track_d_intake import run_track_d_intake_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OsteoGuard AI Stage 33 — Track D Clinical Dataset Intake & Readiness CLI"
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
        help="Optional path to Track D JSON or CSV manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for patient-level splits (default: 42)",
    )
    parser.add_argument(
        "--holdout-institutions",
        nargs="*",
        default=None,
        help="Optional institution_id list isolated into external_test",
    )
    parser.add_argument(
        "--copy-quarantined",
        action="store_true",
        help="Copy rejected files into ml_service/data/quarantine/{modality}/{class}/ without deleting originals",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run validation, leakage-safe splitting, and readiness reporting end-to-end",
    )
    args = parser.parse_args(argv)

    result = run_track_d_intake_pipeline(
        root_dir=args.root_dir.resolve(),
        manifest_path=args.manifest,
        seed=args.seed,
        holdout_institutions=args.holdout_institutions,
        copy_quarantined_files=args.copy_quarantined,
        write_outputs=True,
    )
    readiness = result["readiness"]
    print(json.dumps({
        "readiness_status": readiness["readiness_status"],
        "ready_for_training": readiness["ready_for_training"],
        "total_inspected_samples": readiness["total_inspected_samples"],
        "valid_samples": readiness["valid_samples"],
        "quarantined_samples": readiness["quarantined_samples"],
        "unique_patients": readiness["unique_patients"],
        "institution_count": readiness["institution_count"],
        "class_counts": readiness["class_counts"],
        "modality_counts": readiness["modality_counts"],
        "split_distribution": readiness["split_distribution"],
        "gates": readiness["gates"],
        "blockers": readiness["blockers"],
        "outputs": result["outputs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
