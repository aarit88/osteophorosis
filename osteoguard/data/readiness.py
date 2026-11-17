"""
OsteoGuard AI - Dataset Readiness Verification Gate.
Strictly enforces:
  - Rule 4: Never invent dataset labels or patient records.
  - Rule 5: Never use generated synthetic images as real training data.
  - Rule 6: Never claim that a model has been trained unless actual training logs exist.
  - Rule 7: Never fabricate evaluation metrics.
Hard-blocks model training until verified real clinical datasets are supplied.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from osteoguard.constants import TARGET_CLASSES
from osteoguard.data.validator import DatasetValidator, DatasetValidationReport


class DatasetReadinessError(Exception):
    """Raised when training is attempted without verified real labelled clinical datasets."""
    pass


class ReadinessCheckResult(BaseModel):
    is_ready: bool = Field(..., description="Whether dataset is verified ready for real training")
    status: str = Field(..., description="'READY' or 'BLOCKED'")
    reason: str = Field(..., description="Diagnostic justification for status")
    dataset_path: str = Field(...)
    real_sample_count: int = Field(default=0)
    real_patient_count: int = Field(default=0)
    classes_detected: List[str] = Field(default_factory=list)
    missing_classes: List[str] = Field(default_factory=list)
    action_required: List[str] = Field(default_factory=list)


class DatasetReadinessGate:
    """
    Pre-training enforcement gate.
    Scans the proposed dataset repository, verifies patient stratification,
    checks for synthetic / mock artifact pollution, and halts training if zero real data exists.
    """

    def __init__(self, min_patients_per_class: int = 10):
        self.min_patients_per_class = min_patients_per_class
        self.validator = DatasetValidator()

    def check_readiness(self, dataset_dir: Path) -> ReadinessCheckResult:
        """
        Audits dataset directory and determines training clearance.
        """
        path = Path(dataset_dir)

        if not path.exists():
            return ReadinessCheckResult(
                is_ready=False,
                status="BLOCKED",
                reason=f"Dataset path does not exist: {path}",
                dataset_path=str(path),
                action_required=[
                    "Acquire verified institutional or open-access clinical imaging cohorts.",
                    "Organize into 4 class directories: metastasis/, paget/, oi/, healthy/.",
                    "Run readiness check prior to initiating training."
                ],
            )

        report = self.validator.audit_dataset_directory(path)

        if report.total_samples == 0:
            return ReadinessCheckResult(
                is_ready=False,
                status="BLOCKED",
                reason="ZERO real medical image samples detected on disk.",
                dataset_path=str(path),
                action_required=[
                    "Download authentic clinical DICOM/PNG radiographs for target bone disorders.",
                    "Strict Rule 5: Never generate synthetic images to use as real training data.",
                    "Strict Rule 4: Never invent patient records or labels.",
                ],
            )

        # Check all 4 target classes
        detected = [c for c, count in report.samples_per_class.items() if count > 0]
        missing = [c for c in TARGET_CLASSES if c not in detected]

        if missing:
            return ReadinessCheckResult(
                is_ready=False,
                status="BLOCKED",
                reason=f"Missing required target classes in clinical cohort: {missing}",
                dataset_path=str(path),
                real_sample_count=report.total_samples,
                real_patient_count=report.total_patients,
                classes_detected=detected,
                missing_classes=missing,
                action_required=[
                    f"Acquire validated imaging samples for missing classes: {missing}.",
                ],
            )

        # Check patient count thresholds to prevent training on single-patient leaks
        insufficient_patients = [
            c for c, count in report.patients_per_class.items()
            if count < self.min_patients_per_class
        ]
        if insufficient_patients:
            return ReadinessCheckResult(
                is_ready=False,
                status="BLOCKED",
                reason=(
                    f"Insufficient distinct patients in classes: {insufficient_patients}. "
                    f"Minimum required per class is {self.min_patients_per_class} to guarantee patient-level generalization."
                ),
                dataset_path=str(path),
                real_sample_count=report.total_samples,
                real_patient_count=report.total_patients,
                classes_detected=detected,
                action_required=[
                    "Expand clinical patient recruitment to meet minimum patient diversity threshold.",
                ],
            )

        return ReadinessCheckResult(
            is_ready=True,
            status="READY",
            reason="Verified real clinical dataset passed all validation and patient-stratification criteria.",
            dataset_path=str(path),
            real_sample_count=report.total_samples,
            real_patient_count=report.total_patients,
            classes_detected=detected,
            missing_classes=[],
            action_required=[],
        )

    def enforce_readiness(self, dataset_dir: Path) -> ReadinessCheckResult:
        """
        Hard enforcement method. Raises DatasetReadinessError if not ready.
        """
        result = self.check_readiness(dataset_dir)
        if not result.is_ready:
            raise DatasetReadinessError(
                f"[OsteoGuard Readiness Gate - BLOCKED]: {result.reason}\n"
                f"Actions required:\n" + "\n".join(f"  - {act}" for act in result.action_required)
            )
        return result

