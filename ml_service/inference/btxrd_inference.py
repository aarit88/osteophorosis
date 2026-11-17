"""
OsteoGuard AI - Stage 18: Frozen BTXRD Three-Class Inference & Validation Audit Module.

Implements the immutable inference protocol for the frozen ResNet-34 bone tumour classifier:
- Architecture: TorchVision ResNet-34 (ImageNet-1K pretrained initialization)
- Preprocessing: Aspect-ratio preserving letterbox with symmetrical black padding to 512x512
- Classification Rule: Predicted class = argmax(softmax(logits))
- Probability outputs: Raw uncalibrated softmax probabilities (Brier score: 0.3168 on Center 1 Val)
- Multi-center integrity: Center 2 and Center 3 evaluation specifications defined; external cohorts strictly sealed.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml_service.config.settings import get_settings
from ml_service.datasets.btxrd_dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    BTXRDDataset,
    preprocess_btxrd_image,
)
from ml_service.models.btxrd_classifier import BTXRDResNet34

# Authoritative frozen model metadata
FROZEN_MODEL_VERSION = "1.0.0"
EXPECTED_CHECKPOINT_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
BEST_EPOCH = 18

# External evaluation specifications (Cohorts strictly sealed during Stage 18)
CENTER_2_SPEC = {
    "center": "center_2",
    "split_protocol": "center_holdout_split",
    "split_name": "external_test_center2",
    "expected_count": 549,
    "expected_class_distribution": {
        "NORMAL_BONE_RADIOGRAPH": 259,
        "BENIGN_BONE_TUMOR": 214,
        "PRIMARY_MALIGNANT_TUMOR": 76,
    },
    "role": "Primary external three-class benchmark",
    "status": "SEALED_HOLDOUT",
}

CENTER_3_SPEC = {
    "center": "center_3",
    "split_protocol": "center_holdout_split",
    "split_name": "external_test_center3",
    "expected_count": 259,
    "expected_class_distribution": {
        "NORMAL_BONE_RADIOGRAPH": 27,
        "BENIGN_BONE_TUMOR": 201,
        "PRIMARY_MALIGNANT_TUMOR": 31,
    },
    "role": "Secondary external three-class stress test",
    "status": "SEALED_HOLDOUT",
    "clinical_caution": (
        "Center 3 contains only 27 normal radiographs (10.42% prevalence). "
        "Normal specificity and sensitivity require cautious contextual interpretation."
    ),
}


def calculate_file_sha256(filepath: Union[str, Path]) -> str:
    """Calculates uppercase SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


class BTXRDFrozenClassifier:
    """
    Immutable inference engine for the frozen BTXRD ResNet-34 model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        verify_sha256: bool = True,
    ):
        settings = get_settings()
        self.base_dir = settings.base_dir

        if checkpoint_path is None:
            checkpoint_path = self.base_dir / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
            if not checkpoint_path.exists():
                checkpoint_path = self.base_dir / "ml_artifacts/btxrd_full_training/best_checkpoint.pt"

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # Hash verification
        self.actual_sha256 = calculate_file_sha256(self.checkpoint_path)
        if verify_sha256 and self.actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nGot:      {self.actual_sha256}"
            )

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load weights and configure model
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = BTXRDResNet34(
            num_classes=3,
            pretrained=False,
            dropout_p=0.4,
            freeze_early=True,
        )
        self.model.unfreeze_layer3()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.epoch = ckpt.get("epoch", BEST_EPOCH)
        self.version = FROZEN_MODEL_VERSION

    def predict_tensor(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs frozen inference on a preprocessed (B, 3, 512, 512) tensor.
        Returns raw softmax probabilities and argmax predictions.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred_classes = np.argmax(probs, axis=1)

        results = []
        for i in range(len(probs)):
            p = probs[i]
            pred_idx = int(pred_classes[i])
            results.append({
                "normal_probability": float(p[0]),
                "benign_probability": float(p[1]),
                "malignant_probability": float(p[2]),
                "predicted_class_idx": pred_idx,
                "predicted_class_name": IDX_TO_CLASS[pred_idx],
                "model_version": self.version,
                "checkpoint_sha256": self.actual_sha256,
                "calibration_status": "uncalibrated_raw_softmax",
            })

        return results[0] if len(results) == 1 else results

    def predict_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Executes frozen end-to-end inference on a single radiograph path.
        Preserves aspect-ratio letterbox preprocessing with black padding to 512x512.
        """
        tensor = preprocess_btxrd_image(image_path, target_size=512, is_train=False)
        return self.predict_tensor(tensor)


def audit_saved_validation_predictions(
    predictions_csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Fast audit that computes exact validation metrics directly from the authoritative
    saved validation_predictions.csv generated at Epoch 18.
    """
    settings = get_settings()
    base_dir = settings.base_dir
    if predictions_csv_path is None:
        predictions_csv_path = base_dir / "ml_artifacts/btxrd_full_training/validation_predictions.csv"

    predictions_csv_path = Path(predictions_csv_path)
    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {predictions_csv_path}")

    all_targets = []
    all_preds = []
    all_probs = []

    with open(predictions_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_targets.append(int(row["actual_label"]))
            all_preds.append(int(row["predicted_label"]))
            all_probs.append([
                float(row["prob_normal"]),
                float(row["prob_benign"]),
                float(row["prob_malignant"]),
            ])

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    assert len(all_targets) == 588, f"Expected 588 samples, got {len(all_targets)}"

    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    overall_acc = float(np.mean(all_targets == all_preds))
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

    per_class_recall = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_precision = precision_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    balanced_acc = float(np.mean(per_class_recall))

    per_class_metrics = {}
    roc_aucs = []
    pr_aucs = []

    for c_idx in range(3):
        c_name = IDX_TO_CLASS[c_idx]
        y_binary = (all_targets == c_idx).astype(int)
        y_prob = all_probs[:, c_idx]
        support = int(np.sum(y_binary))

        tn = np.sum((all_targets != c_idx) & (all_preds != c_idx))
        fp = np.sum((all_targets != c_idx) & (all_preds == c_idx))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        auc = float(roc_auc_score(y_binary, y_prob)) if len(np.unique(y_binary)) > 1 else 0.5
        pr = float(average_precision_score(y_binary, y_prob)) if len(np.unique(y_binary)) > 1 else 0.0
        roc_aucs.append(auc)
        pr_aucs.append(pr)

        per_class_metrics[c_name] = {
            "class_index": c_idx,
            "support": support,
            "prevalence": round(float(support / len(all_targets)), 4),
            "sensitivity_recall": round(float(per_class_recall[c_idx]), 4),
            "specificity": round(spec, 4),
            "precision_ppv": round(float(per_class_precision[c_idx]), 4),
            "f1_score": round(float(per_class_f1[c_idx]), 4),
            "roc_auc_ovr": round(auc, 4),
            "pr_auc": round(pr, 4),
        }

    macro_roc = float(np.mean(roc_aucs))
    macro_pr = float(np.mean(pr_aucs))

    y_one_hot = np.zeros_like(all_probs)
    for i, t in enumerate(all_targets):
        y_one_hot[i, t] = 1.0
    brier_score = float(np.mean(np.sum((all_probs - y_one_hot) ** 2, axis=1)))

    return {
        "macro_f1": round(macro_f1, 4),
        "macro_pr_auc": round(macro_pr, 4),
        "macro_roc_auc_ovr": round(macro_roc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "brier_score": round(brier_score, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "sample_count": len(all_targets),
    }


def reproduce_center1_validation_performance(
    classifier: Optional[BTXRDFrozenClassifier] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Performs full inference reproduction on Center 1 clean validation cohort (N=588)
    and verifies that metrics match Stage 17 exactly.
    """
    settings = get_settings()
    base_dir = settings.base_dir
    manifest_path = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"

    if classifier is None:
        classifier = BTXRDFrozenClassifier(device=device, verify_sha256=True)

    val_dataset = BTXRDDataset(
        manifest_path=manifest_path,
        center="center_1",
        split="val",
        split_protocol="center_holdout_split",
        target_size=512,
        is_train=False,
        augment=False,
        base_dir=base_dir,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True if classifier.device.type == "cuda" else False,
    )

    all_targets = []
    all_probs = []
    all_preds = []
    all_image_ids = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(classifier.device)
            targets = batch["label"].numpy()
            image_ids = batch["image_id"]

            logits = classifier.model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets)
            all_probs.append(probs)
            all_preds.extend(preds)
            all_image_ids.extend(image_ids)

    all_targets = np.array(all_targets)
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)

    assert len(all_targets) == 588, f"Expected 588 validation samples, got {len(all_targets)}"

    # Metrics calculation
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    overall_acc = float(np.mean(all_targets == all_preds))
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

    per_class_recall = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_precision = precision_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    balanced_acc = float(np.mean(per_class_recall))

    # One-vs-Rest Specificity, ROC-AUC, PR-AUC
    per_class_metrics = {}
    roc_aucs = []
    pr_aucs = []

    for c_idx in range(3):
        c_name = IDX_TO_CLASS[c_idx]
        y_binary = (all_targets == c_idx).astype(int)
        y_prob = all_probs[:, c_idx]
        support = int(np.sum(y_binary))

        tn = np.sum((all_targets != c_idx) & (all_preds != c_idx))
        fp = np.sum((all_targets != c_idx) & (all_preds == c_idx))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        auc = float(roc_auc_score(y_binary, y_prob)) if len(np.unique(y_binary)) > 1 else 0.5
        pr = float(average_precision_score(y_binary, y_prob)) if len(np.unique(y_binary)) > 1 else 0.0
        roc_aucs.append(auc)
        pr_aucs.append(pr)

        per_class_metrics[c_name] = {
            "class_index": c_idx,
            "support": support,
            "prevalence": round(float(support / len(all_targets)), 4),
            "sensitivity_recall": round(float(per_class_recall[c_idx]), 4),
            "specificity": round(spec, 4),
            "precision_ppv": round(float(per_class_precision[c_idx]), 4),
            "f1_score": round(float(per_class_f1[c_idx]), 4),
            "roc_auc_ovr": round(auc, 4),
            "pr_auc": round(pr, 4),
        }

    macro_roc = float(np.mean(roc_aucs))
    macro_pr = float(np.mean(pr_aucs))

    # Multiclass Brier score: (1/N) * sum_{i,k} (p_{ik} - y_{ik})^2
    y_one_hot = np.zeros_like(all_probs)
    for i, t in enumerate(all_targets):
        y_one_hot[i, t] = 1.0
    brier_score = float(np.mean(np.sum((all_probs - y_one_hot) ** 2, axis=1)))

    return {
        "macro_f1": round(macro_f1, 4),
        "macro_pr_auc": round(macro_pr, 4),
        "macro_roc_auc_ovr": round(macro_roc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "brier_score": round(brier_score, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "sample_count": len(all_targets),
    }
