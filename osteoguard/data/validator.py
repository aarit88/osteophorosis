"""
OsteoGuard AI - Dataset Integrity and Validation Engine.
Performs pre-flight validation on medical image files, label structures,
and directory hierarchies prior to any data ingestion.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from osteoguard.constants import (
    CLASS_TO_IDX,
    SUPPORTED_MODALITIES,
    TARGET_CLASSES,
)
from osteoguard.data.partitioning import SampleRecord


class DatasetValidationReport(BaseModel):
    is_valid: bool = Field(..., description="Overall dataset health verdict")
    total_samples: int = Field(default=0)
    total_patients: int = Field(default=0)
    samples_per_class: Dict[str, int] = Field(default_factory=dict)
    patients_per_class: Dict[str, int] = Field(default_factory=dict)
    corrupt_files: List[str] = Field(default_factory=list)
    unauthorized_classes: List[str] = Field(default_factory=list)
    validation_messages: List[str] = Field(default_factory=list)


class DatasetValidator:
    """
    Validates on-disk dataset folders and manifests against OsteoGuard AI protocols.
    """

    def __init__(self, allowed_classes: Tuple[str, ...] = TARGET_CLASSES):
        self.allowed_classes = set(allowed_classes)

    def validate_image_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Verifies if an image file can be read and contains valid pixels."""
        if not file_path.exists():
            return False, "File does not exist"
        try:
            with Image.open(file_path) as img:
                img.verify()
            # Reopen to test load and check dimensions
            with Image.open(file_path) as img:
                w, h = img.size
                if w < 32 or h < 32:
                    return False, f"Resolution too small: {w}x{h}"
            return True, None
        except Exception as e:
            return False, f"Image corruption error: {str(e)}"

    def audit_dataset_directory(self, data_root: Path) -> DatasetValidationReport:
        """
        Audits a dataset directory organized as:
          data_root/
            metastasis/
            paget/
            oi/
            healthy/
        """
        root = Path(data_root)
        if not root.exists() or not root.is_dir():
            return DatasetValidationReport(
                is_valid=False,
                validation_messages=[f"Directory does not exist: {root}"],
            )

        samples_per_class: Dict[str, int] = {c: 0 for c in self.allowed_classes}
        patients_per_class: Dict[str, Set[str]] = {c: set() for c in self.allowed_classes}
        corrupt_files: List[str] = []
        unauthorized_classes: List[str] = []
        messages: List[str] = []

        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if not subdirs:
            return DatasetValidationReport(
                is_valid=False,
                validation_messages=[f"No class subdirectories found in: {root}"],
            )

        for subdir in subdirs:
            class_name = subdir.name.lower()
            if class_name not in self.allowed_classes:
                unauthorized_classes.append(subdir.name)
                messages.append(f"Unauthorized diagnostic class detected: '{subdir.name}'")
                continue

            for file_path in subdir.glob("*.*"):
                if file_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"):
                    continue

                valid, err = self.validate_image_file(file_path)
                if not valid:
                    corrupt_files.append(str(file_path))
                    messages.append(f"Corrupt image: {file_path.name} - {err}")
                    continue

                # Parse patient ID: everything before the last underscore is patient ID (e.g. PATIENT01_slice02 -> PATIENT01)
                stem = file_path.stem
                if "_" in stem:
                    patient_id = stem.rsplit("_", 1)[0]
                else:
                    patient_id = stem

                samples_per_class[class_name] += 1
                patients_per_class[class_name].add(patient_id)

        total_samples = sum(samples_per_class.values())
        total_patients = sum(len(p) for p in patients_per_class.values())
        pat_counts = {c: len(p) for c, p in patients_per_class.items()}

        is_valid = (
            total_samples > 0
            and len(corrupt_files) == 0
            and len(unauthorized_classes) == 0
            and all(count > 0 for count in samples_per_class.values())
        )

        if total_samples == 0:
            messages.append("Dataset directory contains 0 valid samples.")
        elif any(count == 0 for count in samples_per_class.values()):
            missing = [c for c, count in samples_per_class.items() if count == 0]
            messages.append(f"Missing samples for required target classes: {missing}")

        return DatasetValidationReport(
            is_valid=is_valid,
            total_samples=total_samples,
            total_patients=total_patients,
            samples_per_class=samples_per_class,
            patients_per_class=pat_counts,
            corrupt_files=corrupt_files,
            unauthorized_classes=unauthorized_classes,
            validation_messages=messages,
        )
