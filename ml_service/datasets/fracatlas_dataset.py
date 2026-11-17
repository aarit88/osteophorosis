"""
OsteoGuard AI - FracAtlas Dataset and DataLoader Module.

Provides PyTorch Dataset and DataLoader for binary fracture classification:
  - Class 0: Non-fractured (candidate_control_radiograph)
  - Class 1: Fractured (acute_bone_fracture)

Features:
  - RGB 3-channel conversion for plain radiographs.
  - Consistent 224x224 spatial standardization.
  - Data augmentations applied strictly to training partition.
  - Deterministic evaluation preprocessing for validation and testing.
  - Windows-safe multiprocessing (num_workers=0 by default).
"""

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from ml_service.config.settings import get_settings


def get_fracatlas_transforms(
    split: str = "train",
    img_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    Augmentation is applied ONLY to 'train'.
    'val' and 'test' preprocessing is strictly deterministic.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


class FracAtlasClassificationDataset(Dataset):
    """
    PyTorch Dataset for FracAtlas binary fracture classification.
    Reads records from fracatlas_classification_split.csv.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (224, 224),
        return_metadata: bool = False,
    ):
        settings = get_settings()
        root = settings.base_dir
        if manifest_path is None:
            self.manifest_path = root / "ml_service/data/splits/fracatlas_classification_split.csv"
        else:
            self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"FracAtlas split manifest not found: {self.manifest_path}")

        self.split = split.lower()
        self.img_size = img_size
        self.return_metadata = return_metadata

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_fracatlas_transforms(split=self.split, img_size=img_size)

        # Load and filter records
        self.records: List[Dict[str, str]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.split == "all" or row["split"].lower() == self.split:
                    self.records.append(row)

        if len(self.records) == 0:
            raise ValueError(f"No records found for split '{self.split}' in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]]:
        row = self.records[idx]
        file_path = Path(row["file_path"])

        if not file_path.exists():
            raise FileNotFoundError(f"FracAtlas image file not found: {file_path}")

        # Load image & convert to RGB (standard 3 channels)
        with Image.open(file_path) as img:
            image_rgb = img.convert("RGB")

        # Apply transformation pipeline
        image_tensor = self.transform(image_rgb)

        # Label: 0 = non-fractured, 1 = fractured
        label_int = int(row["fractured"])
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        if self.return_metadata:
            meta = {
                "image_id": row["image_id"],
                "anatomic_region": row.get("anatomic_region", "unknown"),
                "view_projection": row.get("view_projection", "unknown"),
                "sha256_hash": row.get("sha256_hash", ""),
                "split": row.get("split", self.split),
            }
            return image_tensor, label_tensor, meta

        return image_tensor, label_tensor

    def get_class_counts(self) -> Dict[int, int]:
        """Returns distribution of classes in this partition."""
        counts = {0: 0, 1: 0}
        for r in self.records:
            lbl = int(r["fractured"])
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts


def create_fracatlas_dataloaders(
    manifest_path: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: Tuple[int, int] = (224, 224),
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Factory function creating train, val, and test DataLoaders.
    Default num_workers=0 to ensure stable multiprocessing on Windows.
    """
    splits = ["train", "val", "test"]
    loaders = {}

    for sp in splits:
        ds = FracAtlasClassificationDataset(
            manifest_path=manifest_path,
            split=sp,
            img_size=img_size,
        )
        is_train = (sp == "train")
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        loaders[sp] = loader

    return loaders

