"""
OsteoGuard AI - Bone Scan Scintigraphy Patient-Aware Dataset Module.

Features:
  - 16-bit unsigned integer grayscale loading (mode I;16).
  - Native spatial aspect-ratio preservation: (W=256, H=1024).
  - Robust intensity windowing: clipped at 25,000 counts, scaled to [0, 1].
  - Patient-aware paired view interface: Anterior and Posterior scans are grouped.
  - Zero patient leakage across cross-validation folds.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


def preprocess_scintigraphy_image(
    file_path: Union[str, Path],
    clip_ceiling: float = 25000.0,
    target_size: Tuple[int, int] = (1024, 256),  # (H, W)
) -> torch.Tensor:
    """
    Loads 16-bit bone scintigraphy PNG, clips intensity, normalizes to [0, 1],
    and returns a float32 tensor of shape (1, H, W).
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Bone scan image file missing: {p}")

    with Image.open(p) as img:
        arr = np.array(img, dtype=np.float32)

    # Intensity clipping at 25,000 counts (preserves bone uptake, discards bladder saturation)
    arr_clipped = np.clip(arr, 0.0, clip_ceiling)

    # Min-max scale to [0.0, 1.0]
    arr_norm = arr_clipped / clip_ceiling

    # Convert to torch tensor of shape (1, H, W)
    tensor = torch.from_numpy(arr_norm).unsqueeze(0)

    # Verify or adjust dimensions (native is already 1024 x 256)
    if tensor.shape[1:] != target_size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return tensor


class BoneScanPatientDataset(Dataset):
    """
    Patient-level dataset grouping paired Anterior and Posterior whole-body scans.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: Optional[str] = None,
        fold: Optional[str] = None,
        exclude_fold: Optional[str] = None,
        is_train: bool = True,
        augment: bool = False,
        clip_ceiling: float = 25000.0,
        target_size: Tuple[int, int] = (1024, 256),
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.fold = fold
        self.exclude_fold = exclude_fold
        self.is_train = is_train
        self.augment = augment and is_train
        self.clip_ceiling = clip_ceiling
        self.target_size = target_size

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing: {self.manifest_path}")

        records = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # Filter by split if specified
                if split is not None:
                    r_split = r.get("split", "")
                    if r_split.lower() != split.lower():
                        continue
                f_val = r.get("train_cv_fold", r.get("fold", ""))
                # Filter by fold if specified
                if fold is not None and f_val.lower() != fold.lower():
                    continue
                # Exclude fold if specified
                if exclude_fold is not None and f_val.lower() == exclude_fold.lower():
                    continue
                records.append(r)

        # Group records by patient
        patients_dict = {}
        for r in records:
            pid = r["patient_id"]
            if pid not in patients_dict:
                patients_dict[pid] = {
                    "patient_id": pid,
                    "target_class": r["target_class"],
                    "label": 1.0 if r["target_class"] == "BONE_METASTASIS" else 0.0,
                    "anterior_record": None,
                    "posterior_record": None,
                }
            v = r["view"].lower()
            if v == "anterior":
                patients_dict[pid]["anterior_record"] = r
            elif v == "posterior":
                patients_dict[pid]["posterior_record"] = r

        # Verify all patients have both views
        self.patients = []
        for pid, pdata in patients_dict.items():
            if pdata["anterior_record"] is None or pdata["posterior_record"] is None:
                raise ValueError(f"Patient {pid} missing paired anterior/posterior view")
            self.patients.append(pdata)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pdata = self.patients[idx]
        ant_file = pdata["anterior_record"]["file_path"]
        post_file = pdata["posterior_record"]["file_path"]

        ant_tensor = preprocess_scintigraphy_image(ant_file, self.clip_ceiling, self.target_size)
        post_tensor = preprocess_scintigraphy_image(post_file, self.clip_ceiling, self.target_size)

        # Conservative data augmentation during training
        if self.augment:
            # Subtle random rotation (-5 to +5 deg)
            angle = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            ant_tensor = TF.rotate(ant_tensor, angle)
            post_tensor = TF.rotate(post_tensor, angle)

            # Subtle vertical translation (+/- 3%)
            trans_y = int(torch.empty(1).uniform_(-0.03, 0.03).item() * self.target_size[0])
            ant_tensor = TF.affine(ant_tensor, angle=0, translate=[0, trans_y], scale=1.0, shear=0)
            post_tensor = TF.affine(post_tensor, angle=0, translate=[0, trans_y], scale=1.0, shear=0)

        label_tensor = torch.tensor([pdata["label"]], dtype=torch.float32)

        return {
            "patient_id": pdata["patient_id"],
            "anterior_tensor": ant_tensor,
            "posterior_tensor": post_tensor,
            "label": label_tensor,
            "target_class": pdata["target_class"],
            "anterior_path": ant_file,
            "posterior_path": post_file,
        }

    def get_class_counts(self) -> Dict[str, int]:
        n_pos = sum(1 for p in self.patients if p["label"] == 1.0)
        n_neg = len(self.patients) - n_pos
        return {"positive_patients": n_pos, "negative_patients": n_neg, "total_patients": len(self.patients)}

