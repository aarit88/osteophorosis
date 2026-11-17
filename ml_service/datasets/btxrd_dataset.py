"""
OsteoGuard AI - BTXRD Dataset Module.

Provides:
- Aspect-ratio-preserving letterbox preprocessing with black padding to 512x512.
- 3-channel broadcast for single-channel grayscale and RGB plain radiographs.
- ImageNet normalization.
- Center 1 partition loader with conflict quarantine enforcement.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

CLASS_TO_IDX = {
    "NORMAL_BONE_RADIOGRAPH": 0,
    "BENIGN_BONE_TUMOR": 1,
    "PRIMARY_MALIGNANT_TUMOR": 2,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_btxrd_image(
    image_path: Union[str, Path],
    target_size: int = 512,
    is_train: bool = False,
    augment: bool = False,
) -> torch.Tensor:
    """
    Loads an X-ray image (grayscale or RGB), letterboxes to target_size x target_size
    preserving aspect ratio with symmetrical black padding, converts to 3-channel tensor,
    and applies standard ImageNet normalization.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"BTXRD image missing: {p}")

    with Image.open(p) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size

    # Compute aspect-ratio-preserving scale factor
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Resize longest dimension with bilinear interpolation
    resized_img = img_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create black canvas and paste symmetrically
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized_img, (pad_x, pad_y))

    # Optional mild horizontal flip for training
    if is_train and augment and torch.rand(1).item() > 0.5:
        canvas = TF.hflip(canvas)

    # Convert to float32 tensor in range [0.0, 1.0]
    tensor = TF.to_tensor(canvas)

    # ImageNet normalization
    tensor_norm = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return tensor_norm


class BTXRDDataset(Dataset):
    """
    PyTorch Dataset for BTXRD three-class primary bone tumour classification.
    Strictly filters by hospital center, partition split, and conflict quarantine.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        center: str = "center_1",
        split: str = "train",
        split_protocol: str = "center_holdout_split",
        target_size: int = 512,
        is_train: bool = False,
        augment: bool = False,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.center = center.lower()
        self.split = split.lower()
        self.split_protocol = split_protocol
        self.target_size = target_size
        self.is_train = is_train
        self.augment = augment
        self.base_dir = Path(base_dir) if base_dir else self.manifest_path.resolve().parent.parent.parent.parent

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"BTXRD manifest missing: {self.manifest_path}")

        self.records: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Conflict quarantine: strictly skip any flagged conflicts
                if row.get("conflict_flag", "").strip().lower() == "true":
                    continue

                # 2. Filter by hospital center
                if row.get("center", "").strip().lower() != self.center:
                    continue

                # 3. Filter by split in the designated protocol column
                row_split = row.get(self.split_protocol, "").strip().lower()
                if row_split != self.split:
                    continue

                tgt_cls = row.get("target_class", "").strip()
                if tgt_cls not in CLASS_TO_IDX:
                    continue

                self.records.append({
                    "image_id": row["image_id"],
                    "file_path": row["file_path"],
                    "target_class": tgt_cls,
                    "label": CLASS_TO_IDX[tgt_cls],
                    "center": row["center"],
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        full_path = self.base_dir / item["file_path"]

        img_tensor = preprocess_btxrd_image(
            image_path=full_path,
            target_size=self.target_size,
            is_train=self.is_train,
            augment=self.augment,
        )

        return {
            "image": img_tensor,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "image_id": item["image_id"],
            "target_class": item["target_class"],
            "file_path": str(full_path),
        }

    def get_class_counts(self) -> Dict[str, int]:
        counts = {cls: 0 for cls in CLASS_TO_IDX}
        for item in self.records:
            counts[item["target_class"]] += 1
        return counts

    def get_class_frequencies_tensor(self) -> torch.Tensor:
        counts = self.get_class_counts()
        return torch.tensor([counts[IDX_TO_CLASS[i]] for i in range(3)], dtype=torch.float32)

