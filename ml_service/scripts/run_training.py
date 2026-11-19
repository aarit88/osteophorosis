"""
OsteoGuard AI - 4-Class Patient-Stratified K-Fold Training Script.

Loads verified patient cohorts from `ml_service/data/raw/{metastasis,paget,oi,healthy}/`,
enforces DatasetReadinessGate (zero synthetic or empty-cohort training), executes
patient-level stratified K-fold cross-validation via the `osteoguard` library,
fits post-hoc Temperature Scaling and Mahalanobis OOD bounds per fold, and saves
production-ready `.pth` checkpoints to `ml_artifacts/checkpoints/` alongside
structured logs and pooled out-of-fold evaluation reports in `ml_artifacts/logs/`.

Usage:
  python3 ml_service/scripts/run_training.py --check-readiness-only
  python3 ml_service/scripts/run_training.py --dataset-dir ml_service/data/raw --folds 5 --epochs 60 --backbone resnet101
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
import logging
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TARGET_CLASS_DIRS: Tuple[str, ...] = ("metastasis", "paget", "oi", "healthy")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".dicom"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OsteoGuard AI - Stratified K-Fold 4-Class Clinical Training Runner"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="ml_service/data/raw",
        help="Root directory containing metastasis/, paget/, oi/, healthy/ subdirectories",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="ml_artifacts/checkpoints",
        help="Output directory for production-ready .pth checkpoints",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="ml_artifacts/logs",
        help="Output directory for structured JSON logs, epoch CSVs, and CV summary reports",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of patient-level stratified cross-validation folds (>= 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum training epochs per fold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for PyTorch DataLoaders",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW base learning rate",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet101",
        choices=["resnet101", "swin_t"],
        help="Deep semantic backbone architecture",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="xray",
        choices=["xray", "ct", "bone_scan"],
        help="Imaging modality preprocessing pathway",
    )
    parser.add_argument(
        "--min-patients",
        type=int,
        default=10,
        help="Minimum distinct patients required per target class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker processes",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained backbone weights",
    )
    parser.add_argument(
        "--check-readiness-only",
        action="store_true",
        help="Validate environment dependencies and raw clinical dataset readiness without training",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit JSON-formatted console logs instead of human-readable text",
    )
    return parser.parse_args(argv)


def _create_console_logger(json_format: bool = False) -> logging.Logger:
    logger = logging.getLogger("osteoguard.kfold")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if json_format:
            class _StdlibJsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    return json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    })
            handler.setFormatter(_StdlibJsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        logger.addHandler(handler)
    return logger


def _scan_raw_class_counts(dataset_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {cls: 0 for cls in TARGET_CLASS_DIRS}
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return counts
    for cls in TARGET_CLASS_DIRS:
        cls_dir = dataset_dir / cls
        if cls_dir.exists() and cls_dir.is_dir():
            counts[cls] = sum(
                1
                for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
    return counts


def _check_ml_dependencies() -> List[str]:
    missing: List[str] = []
    for mod in ("torch", "torchvision", "numpy", "PIL", "pydantic", "scipy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def build_patient_stratified_kfold_splits(
    samples: Sequence[Any],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Any], List[Any], set, set]]:
    """
    Groups samples by patient_id, stratifies patients across target pathology classes,
    and assigns every patient deterministically to exactly one validation fold.

    Returns a list of (train_samples, val_samples, train_patients, val_patients)
    with mathematically guaranteed zero patient leakage across splits.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for cross-validation, got {n_folds}")

    rng = random.Random(seed)
    patient_to_samples: Dict[str, List[Any]] = defaultdict(list)
    patient_class_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        patient_to_samples[sample.patient_id].append(sample)
        patient_class_votes[sample.patient_id][sample.label.lower()] += 1

    class_to_patients: Dict[str, List[str]] = defaultdict(list)
    for patient_id, votes in patient_class_votes.items():
        primary_class = max(sorted(votes.keys()), key=lambda c: votes[c])
        class_to_patients[primary_class].append(patient_id)

    min_patients_in_any_class = min((len(p) for p in class_to_patients.values()), default=0)
    if min_patients_in_any_class < n_folds:
        raise ValueError(
            f"Cannot execute {n_folds}-fold stratified CV: smallest class has only "
            f"{min_patients_in_any_class} distinct patient(s). Reduce --folds or expand cohort."
        )

    fold_val_patients: List[set] = [set() for _ in range(n_folds)]
    for cls in sorted(class_to_patients.keys()):
        patients = sorted(class_to_patients[cls])
        rng.shuffle(patients)
        for idx, patient_id in enumerate(patients):
            fold_val_patients[idx % n_folds].add(patient_id)

    all_patients = set(patient_to_samples.keys())
    assigned_union = set().union(*fold_val_patients)
    assert assigned_union == all_patients, "Stratified K-fold partition missed patients!"

    splits: List[Tuple[List[Any], List[Any], set, set]] = []
    for fold_idx in range(n_folds):
        val_pts = set(fold_val_patients[fold_idx])
        train_pts = all_patients - val_pts
        assert train_pts.isdisjoint(val_pts), f"Patient leakage detected in fold {fold_idx}!"

        train_samples = [s for p in sorted(train_pts) for s in patient_to_samples[p]]
        val_samples = [s for p in sorted(val_pts) for s in patient_to_samples[p]]
        splits.append((train_samples, val_samples, train_pts, val_pts))

    return splits


def run_kfold_training(args: argparse.Namespace, logger: logging.Logger) -> int:
    dataset_dir = Path(args.dataset_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    logs_dir = Path(args.logs_dir)

    logger.info("OsteoGuard AI - 4-Class Stratified K-Fold Training Gate")
    logger.info("Dataset path: %s", dataset_dir.resolve())

    raw_counts = _scan_raw_class_counts(dataset_dir)
    total_raw_files = sum(raw_counts.values())
    logger.info("Raw disk scan summary: %s (total=%d)", json.dumps(raw_counts), total_raw_files)

    missing_deps = _check_ml_dependencies()
    if missing_deps:
        logger.error(
            "Missing required Python ML dependencies (%s). "
            "Install `pip install -r ml_service/requirements.txt` on your GPU/Python host before running training.",
            ", ".join(missing_deps),
        )
        if total_raw_files == 0:
            logger.error(
                "Clinical data gate also BLOCKED: 0 verified patient scans detected under %s/{%s}.",
                dataset_dir,
                ",".join(TARGET_CLASS_DIRS),
            )
        return 1

    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from ml_service.utils.logging import setup_logger
    from osteoguard.config import ModelConfig, TrainingConfig
    from osteoguard.constants import IDX_TO_CLASS, NUM_CLASSES, TARGET_CLASSES
    from osteoguard.data.dataset import OsteoGuardDataset
    from osteoguard.data.readiness import DatasetReadinessError, DatasetReadinessGate
    from osteoguard.data.validator import DatasetValidator
    from osteoguard.models.loss import MulticlassFocalLoss
    from osteoguard.training.trainer import (
        OsteoGuardTrainer,
        _compute_classification_metrics,
        _set_deterministic_seed,
        _sha256_file,
    )
    from osteoguard.uncertainty.calibration import TemperatureScaler
    from osteoguard.uncertainty.ood_detector import MahalanobisFeatureDetector

    # Enforce strict clinical readiness gate (Rules 4, 5, 6, 7) before creating any training logs/artifacts
    gate = DatasetReadinessGate(min_patients_per_class=args.min_patients)
    readiness = gate.check_readiness(dataset_dir)
    logger.info(
        "Readiness Verdict: %s | Samples=%d | Patients=%d | Reason=%s",
        readiness.status,
        readiness.real_sample_count,
        readiness.real_patient_count,
        readiness.reason,
    )

    if not readiness.is_ready:
        logger.error("Training BLOCKED by clinical readiness gate.")
        for act in readiness.action_required:
            logger.error("Action required: %s", act)
        return 0 if args.check_readiness_only else 1

    if args.check_readiness_only:
        logger.info("Readiness check PASSED (--check-readiness-only specified; exiting cleanly).")
        return 0

    # Prepare artifact and integrated file-logging directories ONLY after dataset clearance
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "osteoguard_4class_kfold_train.log"
    run_logger = setup_logger(
        name="osteoguard.kfold.runner",
        level="INFO",
        json_format=args.json_logs,
        log_file=str(log_file_path),
    )

    _set_deterministic_seed(args.seed)
    run_logger.info(
        f"Starting {args.folds}-fold patient-stratified CV | backbone={args.backbone} | "
        f"modality={args.modality} | epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr}"
    )

    validator = DatasetValidator()
    samples = validator.collect_validated_samples(dataset_dir, modality=args.modality)
    if not samples:
        raise DatasetReadinessError("DatasetValidator returned 0 valid samples after readiness check.")

    splits = build_patient_stratified_kfold_splits(samples, n_folds=args.folds, seed=args.seed)

    model_cfg = ModelConfig(
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained,
    )
    train_cfg = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
    )

    shared_rad_cache: Dict[str, np.ndarray] = {}
    fold_results: List[Dict[str, Any]] = []
    oof_logits_list: List[np.ndarray] = []
    oof_probs_list: List[np.ndarray] = []
    oof_targets_list: List[np.ndarray] = []
    oof_meta_list: List[Dict[str, str]] = []

    best_fold_info: Optional[Dict[str, Any]] = None

    for fold_idx, (train_samples, val_samples, train_pts, val_pts) in enumerate(splits):
        run_logger.info(
            f"[Fold {fold_idx + 1}/{args.folds}] Train={len(train_samples)} samples ({len(train_pts)} pts) | "
            f"Val={len(val_samples)} samples ({len(val_pts)} pts) | zero patient leakage verified"
        )

        trainer = OsteoGuardTrainer(
            model_config=model_cfg,
            train_config=train_cfg,
            min_patients_per_class=args.min_patients,
        )

        train_ds = OsteoGuardDataset(
            train_samples,
            image_size=model_cfg.image_size,
            radiomics_cache=shared_rad_cache,
        )
        val_ds = OsteoGuardDataset(
            val_samples,
            image_size=model_cfg.image_size,
            radiomics_cache=shared_rad_cache,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=min(train_cfg.batch_size, max(len(train_ds), 1)),
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=min(train_cfg.batch_size, max(len(val_ds), 1)),
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = trainer.build_model()
        class_weights = trainer._compute_class_weights(train_samples)
        criterion = MulticlassFocalLoss(
            gamma=train_cfg.focal_loss_gamma,
            alpha=class_weights,
        ).to(trainer.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(train_cfg.epochs, 1),
            eta_min=train_cfg.learning_rate * 0.05,
        )

        best_val_f1 = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        best_state_dict: Optional[Dict[str, Any]] = None

        fold_csv_path = logs_dir / f"osteoguard_4class_fold_{fold_idx}_epochs.csv"
        with open(fold_csv_path, "w", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_roc_auc", "lr"])

            for epoch in range(1, train_cfg.epochs + 1):
                train_loss = trainer.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, val_logits, val_targets, _, _ = trainer._collect_predictions(
                    model, val_loader, criterion
                )
                scheduler.step()

                val_probs = TemperatureScaler(1.0).scale_probabilities(val_logits)
                val_metrics_epoch = _compute_classification_metrics(val_targets, val_probs)
                val_f1 = float(val_metrics_epoch["macro_f1"])
                val_auc = float(val_metrics_epoch["macro_roc_auc"])
                current_lr = float(optimizer.param_groups[0]["lr"])

                writer.writerow([
                    epoch,
                    round(train_loss, 6),
                    round(val_loss, 6),
                    round(val_acc, 4),
                    round(val_f1, 4),
                    round(val_auc, 4),
                    round(current_lr, 8),
                ])

                run_logger.info(
                    f"[Fold {fold_idx}] Epoch {epoch:03d}/{train_cfg.epochs:03d} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} | val_auc={val_auc:.4f}"
                )

                improved = (val_f1 > best_val_f1 + 1e-4) or (
                    abs(val_f1 - best_val_f1) <= 1e-4 and val_loss < best_val_loss
                )
                if improved:
                    best_val_f1 = val_f1
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= train_cfg.early_stopping_patience:
                        run_logger.info(
                            f"[Fold {fold_idx}] Early stopping at epoch {epoch} (best_epoch={best_epoch})."
                        )
                        break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Post-hoc temperature scaling calibration on fold validation logits
        _, _, val_logits, val_targets, _, val_meta = trainer._collect_predictions(model, val_loader, criterion)
        scaler = TemperatureScaler(initial_temperature=1.0)
        fitted_temp = scaler.fit(val_logits, val_targets, lr=0.05, max_iter=200) if len(val_targets) > 0 else 1.0
        cal_val_probs = scaler.scale_probabilities(val_logits)
        fold_val_metrics = _compute_classification_metrics(val_targets, cal_val_probs)

        # Fit Mahalanobis OOD detector strictly on this fold's training fusion embeddings
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=min(train_cfg.batch_size, max(len(train_ds), 1)),
            shuffle=False,
            num_workers=args.num_workers,
        )
        _, _, _, train_targets, train_fusion, _ = trainer._collect_predictions(model, train_eval_loader, criterion)
        ood_payload: Optional[Dict[str, Any]] = None
        try:
            detector = MahalanobisFeatureDetector(
                num_classes=NUM_CLASSES,
                feature_dim=model_cfg.fusion_dim,
                regularization=1e-2,
                threshold=25.0,
            )
            detector.fit(train_fusion, train_targets)
            train_dists = [detector.compute_distance(vec) for vec in train_fusion]
            calibrated_thresh = float(np.percentile(train_dists, 98.0)) if train_dists else 25.0
            ood_payload = {
                "centroids": detector.centroids.tolist() if detector.centroids is not None else None,
                "inv_cov": detector.inv_cov.tolist() if detector.inv_cov is not None else None,
                "threshold": round(max(calibrated_thresh, 5.0), 4),
            }
        except Exception as ood_err:
            run_logger.warning(f"[Fold {fold_idx}] Mahalanobis OOD fit skipped: {ood_err}")

        # Persist production-ready .pth checkpoint for this fold
        trained_at = datetime.now(timezone.utc).isoformat()
        fold_ckpt_path = checkpoints_dir / f"osteoguard_4class_fold_{fold_idx}.pth"
        checkpoint_bundle = {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg.model_dump(),
            "target_classes": list(TARGET_CLASSES),
            "fold": fold_idx,
            "n_folds": args.folds,
            "best_epoch": best_epoch,
            "temperature": float(fitted_temp),
            "ood_mahalanobis": ood_payload,
            "validation_metrics": fold_val_metrics,
            "class_weights": class_weights,
            "trained_at": trained_at,
        }
        torch.save(checkpoint_bundle, fold_ckpt_path)
        fold_sha256 = _sha256_file(fold_ckpt_path)

        run_logger.info(
            f"[Fold {fold_idx}] Saved production .pth checkpoint: {fold_ckpt_path.name} "
            f"(SHA-256={fold_sha256[:16]}..., T={fitted_temp:.4f}, "
            f"Macro-F1={fold_val_metrics['macro_f1']:.4f}, ECE={fold_val_metrics['ece']:.4f})"
        )

        fold_entry = {
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "temperature": float(fitted_temp),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_patients": len(train_pts),
            "val_patients": len(val_pts),
            "checkpoint_filename": fold_ckpt_path.name,
            "checkpoint_sha256": fold_sha256,
            "metrics": fold_val_metrics,
        }
        fold_results.append(fold_entry)

        oof_logits_list.append(val_logits)
        oof_probs_list.append(cal_val_probs)
        oof_targets_list.append(val_targets)
        oof_meta_list.extend(val_meta)

        score_key = (float(fold_val_metrics["macro_f1"]), -float(fold_val_metrics["nll"]))
        if best_fold_info is None or score_key > best_fold_info["score_key"]:
            best_fold_info = {
                "score_key": score_key,
                "fold_idx": fold_idx,
                "path": fold_ckpt_path,
                "metrics": fold_val_metrics,
                "temperature": float(fitted_temp),
                "best_epoch": best_epoch,
                "trained_at": trained_at,
            }

    assert best_fold_info is not None

    # Pooled Out-Of-Fold (OOF) Evaluation across all verified patient scans
    oof_probs = np.concatenate(oof_probs_list, axis=0)
    oof_targets = np.concatenate(oof_targets_list, axis=0)
    oof_metrics = _compute_classification_metrics(oof_targets, oof_probs)

    fold_f1s = [f["metrics"]["macro_f1"] for f in fold_results]
    fold_accs = [f["metrics"]["accuracy"] for f in fold_results]
    fold_aucs = [f["metrics"]["macro_roc_auc"] for f in fold_results]
    fold_eces = [f["metrics"]["ece"] for f in fold_results]

    aggregate_cv = {
        "n_folds": args.folds,
        "mean_accuracy": round(float(np.mean(fold_accs)), 4),
        "std_accuracy": round(float(np.std(fold_accs)), 4),
        "mean_macro_f1": round(float(np.mean(fold_f1s)), 4),
        "std_macro_f1": round(float(np.std(fold_f1s)), 4),
        "mean_macro_roc_auc": round(float(np.mean(fold_aucs)), 4),
        "std_macro_roc_auc": round(float(np.std(fold_aucs)), 4),
        "mean_ece": round(float(np.mean(fold_eces)), 4),
        "best_fold": best_fold_info["fold_idx"],
    }

    # Promote highest-performing fold checkpoint to production canonical `.pth` filename
    production_ckpt_path = checkpoints_dir / "osteoguard_4class_best.pth"
    shutil.copy2(best_fold_info["path"], production_ckpt_path)
    production_sha256 = _sha256_file(production_ckpt_path)

    sample_predictions: List[Dict[str, Any]] = []
    for idx in range(min(24, len(oof_meta_list))):
        pred_idx = int(np.argmax(oof_probs[idx]))
        gt_idx = int(oof_targets[idx])
        sample_predictions.append({
            "sample_id": oof_meta_list[idx]["sample_id"],
            "cohort": f"OOF CV ({oof_meta_list[idx]['patient_id']})",
            "metadata": f"{oof_meta_list[idx]['modality'].upper()} • Stratified {args.folds}-Fold OOF",
            "ground_truth": IDX_TO_CLASS[gt_idx],
            "predicted_label": IDX_TO_CLASS[pred_idx],
            "probability": round(float(oof_probs[idx][pred_idx]), 4),
            "is_correct": pred_idx == gt_idx,
        })

    cv_report = {
        "status": "TRAINED_VERIFIED",
        "model_id": "osteo-trackD-fourclass-unified",
        "trained_at": best_fold_info["trained_at"],
        "backbone": model_cfg.backbone_name,
        "modality": args.modality,
        "best_epoch": best_fold_info["best_epoch"],
        "best_fold": best_fold_info["fold_idx"],
        "temperature": best_fold_info["temperature"],
        "checkpoint_filename": production_ckpt_path.name,
        "checkpoint_sha256": production_sha256,
        "dataset_summary": {
            "total_samples": readiness.real_sample_count,
            "total_patients": readiness.real_patient_count,
            "n_folds": args.folds,
        },
        "cv_summary": aggregate_cv,
        "validation_metrics": best_fold_info["metrics"],
        "test_metrics": oof_metrics,
        "folds": fold_results,
        "sample_predictions": sample_predictions,
    }

    summary_path = checkpoints_dir / "four_class_training_summary.json"
    summary_path.write_text(json.dumps(cv_report, indent=2), encoding="utf-8")

    cv_log_summary_path = logs_dir / "osteoguard_4class_cv_summary.json"
    cv_log_summary_path.write_text(json.dumps(cv_report, indent=2), encoding="utf-8")

    run_logger.info(
        f"K-Fold Cross-Validation Complete! Production .pth saved: {production_ckpt_path} "
        f"(SHA-256={production_sha256}) | OOF Accuracy={oof_metrics['accuracy']:.4f} | "
        f"OOF Macro-F1={oof_metrics['macro_f1']:.4f} | Mean Fold Macro-F1={aggregate_cv['mean_macro_f1']:.4f}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = _create_console_logger(json_format=args.json_logs)
    try:
        return run_kfold_training(args, logger)
    except Exception as exc:
        logger.error("Training runner aborted: %s", exc, exc_info=False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
