"""
OsteoGuard AI - Training and Readiness CLI Entrypoint.
Usage:
  python -m osteoguard.training.train_cli --check-readiness-only --dataset-dir <path>
  python -m osteoguard.training.train_cli --dataset-dir <path>
"""

import argparse
from pathlib import Path
import sys

from osteoguard.data.readiness import DatasetReadinessGate, DatasetReadinessError
from osteoguard.training.trainer import OsteoGuardTrainer


def main():
    parser = argparse.ArgumentParser(description="OsteoGuard AI ML Training & Verification CLI")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/clinical_cohort",
        help="Path to clinical dataset root containing 4 class folders",
    )
    parser.add_argument(
        "--check-readiness-only",
        action="store_true",
        help="Perform pre-flight verification without training",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet101",
        choices=["resnet101", "swin_t"],
        help="Deep semantic backbone to deploy",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_dir)

    print("=" * 70)
    print("   OSTEO GUARD AI - MEDICAL ML PIPELINE EXECUTION GATE")
    print("=" * 70)
    print(f"Target Path: {dataset_path.resolve()}")
    print(f"Semantic Backbone: {args.backbone}")

    gate = DatasetReadinessGate()
    result = gate.check_readiness(dataset_path)

    print(f"\n[Readiness Verdict]: {result.status}")
    print(f"[Reason]: {result.reason}")
    print(f"[Real Samples Count]: {result.real_sample_count}")
    print(f"[Real Patients Count]: {result.real_patient_count}")

    if not result.is_ready:
        print("\n[BLOCKED] Training cannot proceed until authentic clinical data is provided.")
        print("Required Actions:")
        for action in result.action_required:
            print(f"  * {action}")
        if args.check_readiness_only:
            sys.exit(0)
        else:
            sys.exit(1)

    print("\n[PASSED] Dataset verified. Training clearance GRANTED.")
    if args.check_readiness_only:
        sys.exit(0)

    trainer = OsteoGuardTrainer()
    trainer.train(dataset_path)


if __name__ == "__main__":
    main()

