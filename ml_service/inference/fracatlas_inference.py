"""
OsteoGuard AI - Track A: FracAtlas ResNet-101 Binary Fracture Preliminary Inference Module.

Provides standalone, CPU/CUDA-compatible single-image preliminary research inference
for the Stage 09 ResNet-101 fracture classifier without downloading ImageNet weights or
touching the quarantined Stage 10 holdout test partition (N=614).

Authoritative Stage 09 / Stage 10 Specifications:
  - Architecture: TorchVision ResNet-101 with Linear(2048, 2) classification head
  - Preprocessing: RGB conversion -> Resize((224, 224)) -> ToTensor -> ImageNet normalize
  - Classes: 0 = candidate_control_radiograph (non-fractured), 1 = acute_bone_fracture
  - Frozen Screening Operating Thresholds (Stage 09 Validation):
      * 0.31: High-sensitivity screening (Test N=614: 90.0% sens, 74.4% spec, 97.15% NPV)
      * 0.38: Best Youden's J balanced point (Val sens 78.5%, spec 83.96%)
      * 0.47: Best validation F1 point (Val F1 0.6792)
      * 0.50: Default softmax boundary
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from ml_service.config.settings import get_settings

TRACK_A_VERSION = "1.0.0"
TRACK_A_DEFAULT_REL_PATH = Path("ml_artifacts/fracatlas_full_training/best_checkpoint.pt")
TRACK_A_CLASSES = {
    0: "candidate_control_radiograph",
    1: "acute_bone_fracture",
}
TRACK_A_OPERATING_THRESHOLDS = {
    "high_sensitivity_screening": 0.31,
    "best_youden_j": 0.38,
    "best_f1": 0.47,
    "default_0_50": 0.50,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def calculate_file_sha256(filepath: Union[str, Path]) -> str:
    """Calculates uppercase SHA-256 hash of a checkpoint file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def preprocess_fracatlas_image(
    image_path: Union[str, Path],
    img_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Applies deterministic Stage 09/10 validation/test preprocessing to a single radiograph."""
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"Radiograph image not found: {p}")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    with Image.open(p) as img:
        img_rgb = img.convert("RGB")
    return transform(img_rgb)


class FracAtlasFractureClassifier:
    """
    Standalone preliminary inference wrapper for Stage 09 FracAtlas ResNet-101.
    Offline-safe: initializes ResNet-101 with weights=None and loads model_state_dict.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        default_threshold: float = 0.31,
    ):
        settings = get_settings()
        self.base_dir = settings.base_dir

        if checkpoint_path is None:
            checkpoint_path = self.base_dir / TRACK_A_DEFAULT_REL_PATH

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Track A FracAtlas checkpoint not found at: {self.checkpoint_path}"
            )

        if not (0.0 < default_threshold < 1.0):
            raise ValueError(f"default_threshold must be in (0, 1), got {default_threshold}")
        self.default_threshold = float(default_threshold)

        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.actual_sha256 = calculate_file_sha256(self.checkpoint_path)
        self.actual_size_bytes = self.checkpoint_path.stat().st_size

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        model = models.resnet101(weights=None)
        model.fc = nn.Linear(2048, 2)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.epoch = ckpt.get("epoch", 9) if isinstance(ckpt, dict) else None
        self.val_pr_auc = ckpt.get("val_pr_auc") if isinstance(ckpt, dict) else None
        self.val_roc_auc = ckpt.get("val_roc_auc") if isinstance(ckpt, dict) else None

    def predict_tensor(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs deterministic inference on a preprocessed (1, 3, 224, 224) tensor."""
        thresh = self.default_threshold if threshold is None else float(threshold)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        p_non = float(probs[0, 0])
        p_frac = float(probs[0, 1])
        pred_idx = 1 if p_frac >= thresh else 0

        operating_calls = {
            name: {
                "threshold": th_val,
                "predicted_class_idx": 1 if p_frac >= th_val else 0,
                "predicted_class_name": TRACK_A_CLASSES[1 if p_frac >= th_val else 0],
            }
            for name, th_val in TRACK_A_OPERATING_THRESHOLDS.items()
        }

        return {
            "track": "Track A — FracAtlas ResNet-101 Binary Fracture Screening",
            "model_version": TRACK_A_VERSION,
            "checkpoint_epoch": self.epoch,
            "checkpoint_sha256": self.actual_sha256,
            "checkpoint_size_bytes": self.actual_size_bytes,
            "prob_non_fractured": p_non,
            "prob_fractured": p_frac,
            "active_threshold": thresh,
            "predicted_class_idx": pred_idx,
            "predicted_class_name": TRACK_A_CLASSES[pred_idx],
            "operating_point_calls": operating_calls,
            "research_notice": (
                "PRELIMINARY SINGLE-TRACK RESEARCH OUTPUT ONLY — Binary fracture screening, "
                "not a 4-class differential diagnosis and not cleared for clinical care."
            ),
        }

    def predict_image(
        self,
        image_path: Union[str, Path],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes end-to-end preliminary inference on a single radiograph file."""
        tensor = preprocess_fracatlas_image(image_path)
        return self.predict_tensor(tensor, threshold=threshold)
