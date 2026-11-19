"""
OsteoGuard AI - Training and Readiness CLI Entrypoint.
Usage:
  python -m osteoguard.training.train_cli --check-readiness-only --dataset-dir ml_service/data/raw
  python -m osteoguard.training.train_cli --dataset-dir ml_service/data/raw --epochs 60 --backbone resnet101
"""

import argparse
from pathlib import Path
import sys

from osteoguard.config import ModelConfig, TrainingConfig
from osteoguard.data.readiness import DatasetReadinessGate
from osteoguard.training.trainer import OsteoGuardTrainer


def main():
    parser = argparse.ArgumentParser(description="OsteoGuard AI ML Training & Verification CLI")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="ml_service/data/raw",
        help="Path to clinical dataset root containing 4 class folders (metastasis/, paget/, oi/, healthy/)",
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
    parser.add_argument(
        "--modality",
        type=str,
        default="xray",
        choices=["xray", "ct", "bone_scan"],
        help="Imaging modality preprocessing pathway",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument(
        "--min-patients",
        type=int,
        default=10,
        help="Minimum distinct patients required per class",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained backbone initialization",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_dir)

    print("=" * 70)
    print("   OSTEOGUARD AI - 4-CLASS CLINICAL ML PIPELINE EXECUTION GATE")
    print("=" * 70)
    print(f"Target Path       : {dataset_path.resolve()}")
    print(f"Semantic Backbone : {args.backbone}")
    print(f"Modality Pathway  : {args.modality}")

    gate = DatasetReadinessGate(min_patients_per_class=args.min_patients)
    result = gate.check_readiness(dataset_path)

    print(f"\n[Readiness Verdict]   : {result.status}")
    print(f"[Reason]              : {result.reason}")
    print(f"[Real Samples Count]  : {result.real_sample_count}")
    print(f"[Real Patients Count] : {result.real_patient_count}")

    if not result.is_ready:
        print("\n[BLOCKED] Training cannot proceed until authentic 4-class clinical data is provided.")
        print("Required Actions:")
        for action in result.action_required:
            print(f"  * {action}")
        if args.check_readiness_only:
            sys.exit(0)
        sys.exit(1)

    print("\n[PASSED] Dataset verified. Training clearance GRANTED.")
    if args.check_readiness_only:
        sys.exit(0)

    model_cfg = ModelConfig(
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    trainer = OsteoGuardTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        min_patients_per_class=args.min_patients,
    )
    summary = trainer.train(dataset_path, modality=args.modality)
    print(
        f"[VERIFIED] Model trained. Test Macro-F1={summary['test_metrics']['macro_f1']:.4f}, "
        f"Test Accuracy={summary['test_metrics']['accuracy']:.4f}, "
        f"Temperature={summary['temperature']:.4f}"
    )


if __name__ == "__main__":
    main()
