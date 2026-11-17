"""
OsteoGuard AI - PyTorch Dataset Ingestion Class.
Loads medical images, extracts or passes pre-computed IBSI radiomics vectors,
applies modality calibrations, and yields multimodal tensor tuples.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from osteoguard.constants import (
    CLASS_TO_IDX,
    RADIOMICS_FEATURE_DIM,
    TARGET_CLASSES,
)
from osteoguard.data.modality import ModalityStrategyEngine
from osteoguard.data.partitioning import SampleRecord
from osteoguard.radiomics.engine import IBSIRadiomicsEngine


class OsteoGuardDataset(Dataset):
    """
    Multimodal PyTorch Dataset yielding:
      - image: (1, H, W) normalized float tensor
      - radiomics: (102,) float tensor of IBSI descriptors
      - label: scalar int class index in {0, 1, 2, 3}
      - metadata: dict with sample_id, patient_id, modality
    """

    def __init__(
        self,
        samples: List[SampleRecord],
        image_size: Tuple[int, int] = (512, 512),
        radiomics_cache: Optional[Dict[str, np.ndarray]] = None,
        transform: Optional[Callable] = None,
        compute_radiomics: bool = True,
    ):
        self.samples = samples
        self.image_size = image_size
        self.radiomics_cache = radiomics_cache if radiomics_cache is not None else {}
        self.transform = transform
        self.compute_radiomics = compute_radiomics
        self.radiomics_engine = IBSIRadiomicsEngine() if compute_radiomics else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, str]]:
        record = self.samples[idx]
        file_path = Path(record.file_path)

        # 1. Load image as single-channel grayscale
        try:
            with Image.open(file_path) as pil_img:
                gray_img = pil_img.convert("L")
                resized_img = gray_img.resize(self.image_size, Image.Resampling.BILINEAR)
                raw_np = np.array(resized_img, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read image at {file_path}: {e}")

        # 2. Modality-specific radiometric calibration
        calibrated_np = ModalityStrategyEngine.preprocess_image(
            raw_np,
            modality=record.modality,
        )

        # 3. Radiomic extraction (or cached retrieval)
        if record.sample_id in self.radiomics_cache:
            v_rad_np = self.radiomics_cache[record.sample_id]
        elif self.compute_radiomics and self.radiomics_engine is not None:
            v_rad_np = self.radiomics_engine.extract_features(calibrated_np)
            self.radiomics_cache[record.sample_id] = v_rad_np
        else:
            v_rad_np = np.zeros(RADIOMICS_FEATURE_DIM, dtype=np.float32)

        # 4. Form PyTorch tensors
        # Shape (1, H, W)
        image_tensor = torch.from_numpy(calibrated_np).unsqueeze(0).float()
        radiomics_tensor = torch.from_numpy(v_rad_np).float()

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # 5. Label conversion
        cls_idx = CLASS_TO_IDX.get(record.label.lower(), 3)

        meta = {
            "sample_id": record.sample_id,
            "patient_id": record.patient_id,
            "modality": record.modality,
            "class_name": record.label,
        }

        return image_tensor, radiomics_tensor, cls_idx, meta

