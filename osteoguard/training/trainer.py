"""
OsteoGuard AI - Complete 4-Class Clinical Training, Calibration & Artifact Engine.
Implements:
  - Hard DatasetReadinessGate enforcement (zero synthetic/empty cohort training)
  - Patient-level stratified Train/Val/Test partitioning (zero patient leakage)
  - Class-weighted Multiclass Focal Loss (gamma = 2.0) + AdamW + CosineAnnealingLR
  - Early stopping on validation Macro-F1 / NLL
  - Post-hoc Temperature Scaling calibration & CalibrationDiagnostics (NLL, Brier, ECE)
  - Mahalanobis feature-space OOD fitting on fusion manifold
  - Cryptographic SHA-256 checkpoint, log, epoch CSV, and training summary JSON export
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from osteoguard.config import ModelConfig, TrainingConfig
from osteoguard.constants import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    TARGET_CLASSES,
)
from osteoguard.data.dataset import OsteoGuardDataset
from osteoguard.data.partitioning import PatientLevelPartitioner, SampleRecord
from osteoguard.data.readiness import DatasetReadinessGate, DatasetReadinessError
from osteoguard.data.sanitizer import PatientPrivacySanitizer
from osteoguard.data.validator import DatasetValidator
from osteoguard.models.loss import MulticlassFocalLoss
from osteoguard.models.osteoguard_net import OsteoGuardNet
from osteoguard.uncertainty.calibration import CalibrationDiagnostics, TemperatureScaler
from osteoguard.uncertainty.ood_detector import MahalanobisFeatureDetector


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Any]:
    """Computes accuracy, macro-F1, balanced accuracy, per-class F1, and OVR ROC-AUC without fragile assumptions."""
    preds = np.argmax(probs, axis=1)
    n = max(len(y_true), 1)
    accuracy = float(np.mean(preds == y_true))

    per_class: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []
    recall_list: List[float] = []

    for c_idx, cls_name in IDX_TO_CLASS.items():
        tp = int(np.sum((preds == c_idx) & (y_true == c_idx)))
        fp = int(np.sum((preds == c_idx) & (y_true != c_idx)))
        fn = int(np.sum((preds != c_idx) & (y_true == c_idx)))
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        support = int(np.sum(y_true == c_idx))
        per_class[cls_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            f1_list.append(f1)
            recall_list.append(rec)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    balanced_accuracy = float(np.mean(recall_list)) if recall_list else 0.0

    # Optional OVR ROC-AUC via pure NumPy trapezoidal rank statistic
    roc_aucs: List[float] = []
    for c_idx in range(NUM_CLASSES):
        y_bin = (y_true == c_idx).astype(np.int32)
        n_pos = int(np.sum(y_bin == 1))
        n_neg = int(np.sum(y_bin == 0))
        if n_pos > 0 and n_neg > 0:
            scores = probs[:, c_idx]
            order = np.argsort(-scores)
            y_sorted = y_bin[order]
            tps = np.cumsum(y_sorted) / float(n_pos)
            fps = np.cumsum(1 - y_sorted) / float(n_neg)
            tps = np.concatenate([[0.0], tps])
            fps = np.concatenate([[0.0], fps])
            auc_fn = getattr(np, "trapezoid", None) or np.trapz
            roc_aucs.append(float(auc_fn(tps, fps)))

    macro_roc_auc = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    cal_report = CalibrationDiagnostics.evaluate(y_true, probs)

    return {
        "n_samples": int(n),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_roc_auc": round(macro_roc_auc, 4),
        "nll": cal_report.negative_log_likelihood,
        "brier_score": cal_report.brier_score,
        "ece": cal_report.expected_calibration_error,
        "mce": cal_report.max_calibration_error,
        "per_class": per_class,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


class OsteoGuardTrainer:
    """
    Orchestrates verified training workflows for OsteoGuard AI.
    Enforces pre-flight readiness checks to prevent phantom or synthetic training.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        train_config: Optional[TrainingConfig] = None,
        min_patients_per_class: int = 10,
    ):
        self.model_config = model_config or ModelConfig()
        self.train_config = train_config or TrainingConfig()
        self.readiness_gate = DatasetReadinessGate(min_patients_per_class=min_patients_per_class)
        self.validator = DatasetValidator()
        self.sanitizer = PatientPrivacySanitizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_preflight_readiness_check(self, dataset_dir: Path):
        """
        Hard gate: Will raise DatasetReadinessError if real clinical data is absent.
        """
        return self.readiness_gate.enforce_readiness(dataset_dir)

    def build_model(self) -> OsteoGuardNet:
        """Instantiates the unified OsteoGuard dual-stream network."""
        model = OsteoGuardNet(
            backbone_name=self.model_config.backbone_name,
            pretrained_backbone=self.model_config.pretrained_backbone,
            in_channels=1,
            semantic_dim=self.model_config.semantic_dim,
            radiomics_dim=self.model_config.radiomics_dim,
            num_classes=self.model_config.num_classes,
            enable_unet_masking=self.model_config.enable_unet_masking,
        )
        return model.to(self.device)

    def train_epoch(
        self,
        model: OsteoGuardNet,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: MulticlassFocalLoss,
    ) -> float:
        """Executes one training epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, radiomics, targets, _ in dataloader:
            images = images.to(self.device)
            radiomics = radiomics.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(images, radiomics)
            loss = criterion(outputs["logits"], targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.train_config.gradient_clip_norm
            )
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        model: OsteoGuardNet,
        dataloader: DataLoader,
        criterion: MulticlassFocalLoss,
    ) -> Tuple[float, float]:
        """Evaluates loss and accuracy on validation/test loader."""
        loss, acc, _, _, _, _ = self._collect_predictions(model, dataloader, criterion)
        return loss, acc

    def _collect_predictions(
        self,
        model: OsteoGuardNet,
        dataloader: DataLoader,
        criterion: MulticlassFocalLoss,
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, str]]]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_logits: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_fusion: List[np.ndarray] = []
        all_meta: List[Dict[str, str]] = []

        with torch.no_grad():
            for images, radiomics, targets, meta in dataloader:
                images = images.to(self.device)
                radiomics = radiomics.to(self.device)
                targets = targets.to(self.device)

                outputs = model(images, radiomics)
                loss = criterion(outputs["logits"], targets)

                total_loss += float(loss.item())
                logits_np = outputs["logits"].detach().cpu().numpy()
                fusion_np = outputs["v_fusion"].detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()

                preds = np.argmax(logits_np, axis=1)
                correct += int(np.sum(preds == targets_np))
                total_samples += len(targets_np)

                all_logits.append(logits_np)
                all_targets.append(targets_np)
                all_fusion.append(fusion_np)

                batch_len = len(targets_np)
                for i in range(batch_len):
                    all_meta.append(
                        {
                            "sample_id": str(meta["sample_id"][i]),
                            "patient_id": self.sanitizer.generate_pseudonym(str(meta["patient_id"][i])),
                            "modality": str(meta["modality"][i]),
                            "class_name": str(meta["class_name"][i]),
                        }
                    )

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = float(correct) / max(total_samples, 1)
        logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, NUM_CLASSES))
        targets_arr = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0,), dtype=np.int64)
        fusion_arr = np.concatenate(all_fusion, axis=0) if all_fusion else np.empty((0, self.model_config.fusion_dim))
        return avg_loss, accuracy, logits_arr, targets_arr, fusion_arr, all_meta

    @staticmethod
    def _compute_class_weights(train_samples: List[SampleRecord]) -> List[float]:
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for s in train_samples:
            counts[CLASS_TO_IDX.get(s.label.lower(), 3)] += 1.0
        counts = np.maximum(counts, 1.0)
        inv = np.sum(counts) / (NUM_CLASSES * counts)
        normalized = inv / np.mean(inv)
        return [float(round(w, 4)) for w in normalized]

    def train(
        self,
        dataset_dir: Path,
        modality: str = "xray",
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        """
        End-to-end patient-stratified training, post-hoc temperature calibration,
        OOD manifold fitting, holdout evaluation, and cryptographically hashed checkpointing.
        """
        _set_deterministic_seed(self.train_config.seed)

        # 1. Enforce strict clinical readiness gate
        print(f"[OsteoGuard Trainer] Executing pre-flight readiness gate for: {dataset_dir}...")
        readiness = self.run_preflight_readiness_check(dataset_dir)

        # 2. Setup output directories
        self.train_config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.train_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.train_config.logs_dir.mkdir(parents=True, exist_ok=True)

        # 3. Collect validated samples & run patient-level stratified splitting (zero leakage)
        samples = self.validator.collect_validated_samples(dataset_dir, modality=modality)
        if not samples:
            raise DatasetReadinessError("No readable image samples returned after validation scan.")

        partitioner = PatientLevelPartitioner(
            train_ratio=self.train_config.train_split,
            val_ratio=self.train_config.val_split,
            test_ratio=self.train_config.test_split,
            seed=self.train_config.seed,
        )
        split = partitioner.partition(samples)
        if len(split.train_samples) == 0 or len(split.val_samples) == 0:
            raise DatasetReadinessError(
                "Patient-level partitioning produced an empty train or validation split. "
                "Ensure sufficient distinct patients per pathology class."
            )

        print(
            f"[OsteoGuard Trainer] Patient-level splits verified (zero leakage): "
            f"Train={len(split.train_samples)} ({len(split.train_patients)} pts) | "
            f"Val={len(split.val_samples)} ({len(split.val_patients)} pts) | "
            f"Test={len(split.test_samples)} ({len(split.test_patients)} pts)"
        )

        # 4. Build PyTorch datasets & DataLoaders
        shared_rad_cache: Dict[str, np.ndarray] = {}
        train_ds = OsteoGuardDataset(
            split.train_samples,
            image_size=self.model_config.image_size,
            radiomics_cache=shared_rad_cache,
        )
        val_ds = OsteoGuardDataset(
            split.val_samples,
            image_size=self.model_config.image_size,
            radiomics_cache=shared_rad_cache,
        )
        test_ds = OsteoGuardDataset(
            split.test_samples if split.test_samples else split.val_samples,
            image_size=self.model_config.image_size,
            radiomics_cache=shared_rad_cache,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=min(self.train_config.batch_size, max(len(train_ds), 1)),
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=min(self.train_config.batch_size, max(len(val_ds), 1)),
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=min(self.train_config.batch_size, max(len(test_ds), 1)),
            shuffle=False,
            num_workers=num_workers,
        )

        # 5. Model, class-weighted Focal Loss, Optimizer & Scheduler
        print(f"[OsteoGuard Trainer] Initializing OsteoGuardNet ({self.model_config.backbone_name}) on {self.device}")
        model = self.build_model()
        class_weights = self._compute_class_weights(split.train_samples)
        criterion = MulticlassFocalLoss(
            gamma=self.train_config.focal_loss_gamma,
            alpha=class_weights,
        ).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.train_config.epochs, 1),
            eta_min=self.train_config.learning_rate * 0.05,
        )

        checkpoint_path = self.train_config.checkpoints_dir / "osteoguard_4class_best.pt"
        log_path = self.train_config.logs_dir / "osteoguard_4class_train.log"
        csv_path = self.train_config.logs_dir / "osteoguard_4class_epochs.csv"
        summary_path = self.train_config.checkpoints_dir / "four_class_training_summary.json"

        best_val_macro_f1 = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        best_state_dict: Optional[Dict[str, Any]] = None

        log_lines: List[str] = [
            f"[{datetime.now(timezone.utc).isoformat()}] OsteoGuard 4-Class Clinical Training Started",
            f"Dataset: {Path(dataset_dir).resolve()} | Modality: {modality} | Backbone: {self.model_config.backbone_name}",
            f"Samples: total={readiness.real_sample_count}, patients={readiness.real_patient_count}, class_weights={class_weights}",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_roc_auc", "lr"])

            for epoch in range(1, self.train_config.epochs + 1):
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, val_logits, val_targets, _, _ = self._collect_predictions(
                    model, val_loader, criterion
                )
                scheduler.step()

                # Softmax for epoch metrics
                val_probs = TemperatureScaler(1.0).scale_probabilities(val_logits)
                val_epoch_metrics = _compute_classification_metrics(val_targets, val_probs)
                val_f1 = float(val_epoch_metrics["macro_f1"])
                val_auc = float(val_epoch_metrics["macro_roc_auc"])
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

                line = (
                    f"Epoch {epoch:03d}/{self.train_config.epochs:03d} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} | val_auc={val_auc:.4f}"
                )
                print(f"[OsteoGuard Trainer] {line}")
                log_lines.append(line)

                improved = (val_f1 > best_val_macro_f1 + 1e-4) or (
                    abs(val_f1 - best_val_macro_f1) <= 1e-4 and val_loss < best_val_loss
                )
                if improved:
                    best_val_macro_f1 = val_f1
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.train_config.early_stopping_patience:
                        stop_msg = f"Early stopping triggered at epoch {epoch} (best epoch={best_epoch})."
                        print(f"[OsteoGuard Trainer] {stop_msg}")
                        log_lines.append(stop_msg)
                        break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # 6. Post-hoc Temperature Scaling Calibration on Validation Split
        _, _, val_logits, val_targets, _, _ = self._collect_predictions(model, val_loader, criterion)
        scaler = TemperatureScaler(initial_temperature=1.0)
        if len(val_targets) > 0:
            fitted_temp = scaler.fit(val_logits, val_targets, lr=0.05, max_iter=200)
        else:
            fitted_temp = 1.0

        calibrated_val_probs = scaler.scale_probabilities(val_logits)
        val_metrics = _compute_classification_metrics(val_targets, calibrated_val_probs)

        # 7. Fit Mahalanobis OOD Detector on training fusion embeddings (zero patient leakage)
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=min(self.train_config.batch_size, max(len(train_ds), 1)),
            shuffle=False,
            num_workers=num_workers,
        )
        _, _, _, train_targets, train_fusion, _ = self._collect_predictions(model, train_eval_loader, criterion)
        ood_payload: Optional[Dict[str, Any]] = None
        try:
            detector = MahalanobisFeatureDetector(
                num_classes=NUM_CLASSES,
                feature_dim=self.model_config.fusion_dim,
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
            log_lines.append(f"Mahalanobis OOD fit skipped: {ood_err}")

        # 8. Evaluate Calibrated Model on Held-Out Test Split
        _, _, test_logits, test_targets, _, test_meta = self._collect_predictions(model, test_loader, criterion)
        calibrated_test_probs = scaler.scale_probabilities(test_logits)
        test_metrics = _compute_classification_metrics(test_targets, calibrated_test_probs)

        sample_predictions: List[Dict[str, Any]] = []
        for idx in range(min(24, len(test_meta))):
            pred_idx = int(np.argmax(calibrated_test_probs[idx]))
            gt_idx = int(test_targets[idx])
            sample_predictions.append({
                "sample_id": test_meta[idx]["sample_id"],
                "cohort": f"Clinical Holdout ({test_meta[idx]['patient_id']})",
                "metadata": f"{test_meta[idx]['modality'].upper()} • Calibrated T={fitted_temp:.2f}",
                "ground_truth": IDX_TO_CLASS[gt_idx],
                "predicted_label": IDX_TO_CLASS[pred_idx],
                "probability": round(float(calibrated_test_probs[idx][pred_idx]), 4),
                "is_correct": pred_idx == gt_idx,
            })

        # 9. Persist Verified Checkpoint + SHA-256 + Logs + Summary JSON
        checkpoint_bundle = {
            "model_state_dict": model.state_dict(),
            "model_config": self.model_config.model_dump(),
            "target_classes": list(TARGET_CLASSES),
            "temperature": float(fitted_temp),
            "ood_mahalanobis": ood_payload,
            "best_epoch": best_epoch,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint_bundle, checkpoint_path)
        checkpoint_sha256 = _sha256_file(checkpoint_path)

        log_lines.extend([
            f"Fitted Temperature Scalar T = {fitted_temp:.4f}",
            f"Validation Metrics (Calibrated): acc={val_metrics['accuracy']}, macro_f1={val_metrics['macro_f1']}, ece={val_metrics['ece']}, roc_auc={val_metrics['macro_roc_auc']}",
            f"Holdout Test Metrics (Calibrated): acc={test_metrics['accuracy']}, macro_f1={test_metrics['macro_f1']}, ece={test_metrics['ece']}, roc_auc={test_metrics['macro_roc_auc']}",
            f"Saved Verified Checkpoint: {checkpoint_path} (SHA-256: {checkpoint_sha256})",
        ])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        summary_payload = {
            "status": "TRAINED_VERIFIED",
            "model_id": "osteo-trackD-fourclass-unified",
            "trained_at": checkpoint_bundle["trained_at"],
            "backbone": self.model_config.backbone_name,
            "modality": modality,
            "best_epoch": best_epoch,
            "temperature": float(fitted_temp),
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_summary": {
                "total_samples": readiness.real_sample_count,
                "total_patients": readiness.real_patient_count,
                "train_samples": len(split.train_samples),
                "val_samples": len(split.val_samples),
                "test_samples": len(split.test_samples),
                "train_patients": len(split.train_patients),
                "val_patients": len(split.val_patients),
                "test_patients": len(split.test_patients),
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "sample_predictions": sample_predictions,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        print(f"[OsteoGuard Trainer] Training complete. Checkpoint SHA-256: {checkpoint_sha256}")
        return summary_payload
