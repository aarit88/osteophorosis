"""
OsteoGuard AI - ML Readiness Evaluation Service.
Audits the physical dataset store and checkpoint repository, computing the exact
finite state of the model lifecycle and enforcing strict medical AI governance:
  - Rule 4: Never invent dataset labels or patient records.
  - Rule 5: Never use generated synthetic images as real training data.
  - Rule 6: Never claim that a model has been trained unless actual training logs exist.
  - Rule 7: Never fabricate evaluation metrics or fake predictions.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import torch

from ml_service.app.schemas.model_state import ModelState
from ml_service.app.schemas.ml_status import (
    DatasetStatusResponse,
    MLHealthResponse,
    MLStatusResponse,
    ModelArchitectureInfo,
    ModelStatusResponse,
)
from ml_service.config.settings import MLSettings, get_settings
from ml_service.utils.exceptions import ModelNotReadyException
from ml_service.utils.logging import logger
from osteoguard.constants import TARGET_CLASSES
from osteoguard.data.validator import DatasetValidator


class MLReadinessService:
    """
    Evaluates dataset and checkpoint state against strict clinical validation gates.
    """

    def __init__(self, settings: Optional[MLSettings] = None):
        self.settings = settings or get_settings()
        self.validator = DatasetValidator()
        self._start_time = time.time()
        self._is_training = False  # Track live training execution in subsequent stages

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 2)

    def get_health_status(self) -> MLHealthResponse:
        """Returns runtime health and hardware telemetry."""
        device_str = "cuda" if torch.cuda.is_available() and self.settings.device == "cuda" else "cpu"
        return MLHealthResponse(
            status="ok",
            service=self.settings.service_name,
            version=self.settings.service_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            environment=self.settings.env,
            device=device_str,
            uptime_seconds=self.uptime_seconds,
        )

    def get_dataset_status(self) -> DatasetStatusResponse:
        """
        Audits physical storage for real clinical imaging cohorts.
        Strictly reports zero samples when no clinical data has been downloaded.
        """
        data_dir = self.settings.get_resolved_raw_dir()
        if not data_dir.exists():
            # Try raw parent data directory
            data_dir = self.settings.get_resolved_data_dir()

        if not data_dir.exists():
            return DatasetStatusResponse(
                status="EMPTY",
                dataset_path=str(data_dir),
                sample_count=0,
                patient_count=0,
                classes_present={c: 0 for c in TARGET_CLASSES},
                missing_classes=list(TARGET_CLASSES),
                is_ready=False,
                validation_passed=False,
                blockers=[
                    f"Dataset directory does not exist on disk: {data_dir}",
                    "Zero clinical datasets downloaded. Training clearance is hard-blocked.",
                ],
                action_required=[
                    "Acquire verified institutional or open-access DICOM/PNG radiographs.",
                    "Structure directories by target class: metastasis/, paget/, oi/, healthy/.",
                    "Ensure patient identifiers are preserved in filenames to enable zero-leakage splits.",
                ],
            )

        report = self.validator.audit_dataset_directory(data_dir)

        if report.total_samples == 0:
            return DatasetStatusResponse(
                status="EMPTY",
                dataset_path=str(data_dir),
                sample_count=0,
                patient_count=0,
                classes_present={c: 0 for c in TARGET_CLASSES},
                missing_classes=list(TARGET_CLASSES),
                is_ready=False,
                validation_passed=False,
                blockers=[
                    "ZERO real medical image samples detected on disk.",
                    "Strict Rule 4 & 5: No synthetic images or fake labels may be substituted.",
                    "Training cannot proceed without authentic medical images.",
                ],
                action_required=[
                    "Download authentic clinical imaging datasets.",
                    "Follow modality standardization guidelines (CT: WL +400/WW 1500; X-ray: CLAHE).",
                ],
            )

        # Classes detection
        detected = [c for c, count in report.samples_per_class.items() if count > 0]
        missing = [c for c in TARGET_CLASSES if c not in detected]

        blockers = []
        if missing:
            blockers.append(f"Missing clinical cohorts for required classes: {missing}")

        for cls, count in report.patients_per_class.items():
            if count < self.settings.min_patients_per_class:
                blockers.append(
                    f"Class '{cls}' has only {count} patients (minimum required: {self.settings.min_patients_per_class})"
                )

        if report.corrupt_files:
            blockers.append(f"Corrupt image files detected: {len(report.corrupt_files)} files cannot be read.")

        if report.unauthorized_classes:
            blockers.append(f"Unauthorized directory classes detected: {report.unauthorized_classes}")

        for msg in report.validation_messages:
            if "synthetic" in msg.lower() or "mock" in msg.lower():
                blockers.append(f"Synthetic data rule violation: {msg}")

        is_ready = len(blockers) == 0 and report.is_valid

        return DatasetStatusResponse(
            status="READY" if is_ready else ("INVALID" if report.total_samples > 0 else "EMPTY"),
            dataset_path=str(data_dir),
            sample_count=report.total_samples,
            patient_count=report.total_patients,
            classes_present=report.samples_per_class,
            missing_classes=missing,
            is_ready=is_ready,
            validation_passed=report.is_valid,
            blockers=blockers,
            action_required=["Proceed to training stage." if is_ready else "Resolve dataset blockers above."],
        )

    def get_active_checkpoint(self) -> Optional[Path]:
        """Checks if a verified model checkpoint exists on disk."""
        if self.settings.active_checkpoint_path:
            chk = Path(self.settings.active_checkpoint_path)
            if chk.is_file():
                return chk

        chk_dir = self.settings.get_resolved_checkpoints_dir()
        if chk_dir.exists():
            checkpoints = list(chk_dir.glob("*.pt")) + list(chk_dir.glob("*.pth"))
            if checkpoints:
                # Return newest
                return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        return None

    def determine_model_state(self) -> Tuple[ModelState, List[str]]:
        """
        Computes model state based on actual filesystem state.
        Never reports TRAINED or READY_FOR_INFERENCE unless genuine checkpoints exist.
        """
        blockers: List[str] = []

        # 1. Check configuration
        try:
            _ = self.settings.get_resolved_data_dir()
            _ = self.settings.get_resolved_checkpoints_dir()
        except Exception as e:
            return ModelState.NOT_CONFIGURED, [f"Configuration error: {str(e)}"]

        # 2. Check if training is in progress
        if self._is_training:
            return ModelState.TRAINING, []

        # 3. Check for trained weights checkpoint
        checkpoint = self.get_active_checkpoint()
        dataset_status = self.get_dataset_status()

        if checkpoint is None:
            # No weights on disk
            if dataset_status.sample_count == 0:
                blockers.append("No clinical dataset downloaded. State is AWAITING_DATA.")
                blockers.append("No trained model checkpoints exist in ml_artifacts/checkpoints/.")
                return ModelState.AWAITING_DATA, blockers
            elif not dataset_status.is_ready:
                blockers.extend(dataset_status.blockers)
                return ModelState.DATA_VALIDATION_REQUIRED, blockers
            else:
                return ModelState.READY_FOR_TRAINING, ["Dataset verified. Ready for training initialization."]

        # Checkpoint exists: verify it
        try:
            # Check if training logs exist
            logs_dir = self.settings.resolve_path(self.settings.logs_dir)
            has_logs = logs_dir.exists() and any(logs_dir.glob("*.log"))

            if not has_logs:
                blockers.append("Rule 6 Check: Checkpoint found but no corresponding training logs exist.")
                return ModelState.VALIDATION_REQUIRED, blockers

            # In future stages, we will verify model checkpoint test set metrics
            # For now, if checkpoint and logs exist, mark READY_FOR_INFERENCE
            return ModelState.READY_FOR_INFERENCE, []
        except Exception as e:
            logger.error(f"Error validating checkpoint: {e}")
            return ModelState.FAILED, [f"Checkpoint evaluation failed: {str(e)}"]

    def get_model_status(self) -> ModelStatusResponse:
        """Returns comprehensive model status and readiness details."""
        state, blockers = self.determine_model_state()
        checkpoint = self.get_active_checkpoint()
        checkpoint_path = str(checkpoint) if checkpoint else None

        arch_info = ModelArchitectureInfo(
            backbone=self.settings.active_backbone,
            radiomics_features=self.settings.radiomics_dim,
            semantic_features=self.settings.semantic_dim,
            total_fusion_dim=self.settings.fusion_dim,
            num_classes=self.settings.num_classes,
            target_classes=self.settings.target_classes,
        )

        can_infer = (state == ModelState.READY_FOR_INFERENCE) and (checkpoint is not None)
        can_train = (state == ModelState.READY_FOR_TRAINING)

        message = (
            "Model is AWAITING_DATA. Currently zero clinical datasets exist on disk. "
            "Strict adherence to Rules 4, 5, 6, and 7 prohibits phantom predictions or fake training metrics."
            if state == ModelState.AWAITING_DATA
            else f"Model is currently in state {state.value}."
        )

        return ModelStatusResponse(
            model_state=state,
            active_backbone=self.settings.active_backbone,
            weights_loaded=False,  # No weights loaded in zero-dataset state
            checkpoint_path=checkpoint_path,
            checkpoint_verified=can_infer,
            architecture=arch_info,
            can_infer=can_infer,
            can_train=can_train,
            training_history_available=False,
            message=message,
            blockers=blockers,
        )

    def get_overall_status(self) -> MLStatusResponse:
        """Returns aggregate status of the ML service."""
        model_status = self.get_model_status()
        dataset_status = self.get_dataset_status()

        high_level_status = "awaiting_data"
        if model_status.model_state == ModelState.READY_FOR_INFERENCE:
            high_level_status = "operational"
        elif model_status.model_state == ModelState.NOT_CONFIGURED:
            high_level_status = "unconfigured"
        elif model_status.model_state == ModelState.TRAINING:
            high_level_status = "training"
        elif model_status.model_state == ModelState.FAILED:
            high_level_status = "error"

        return MLStatusResponse(
            status=high_level_status,
            service=self.settings.service_name,
            version=self.settings.service_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_state=model_status.model_state,
            dataset_state=dataset_status.status,
            active_backbone=self.settings.active_backbone,
            device=self.settings.device,
            sample_count=dataset_status.sample_count,
            patient_count=dataset_status.patient_count,
            weights_loaded=model_status.weights_loaded,
            can_train=model_status.can_train,
            can_infer=model_status.can_infer,
            message=model_status.message,
        )

    def ensure_inference_ready(self) -> None:
        """
        Strict inference gatekeeper.
        Raises ModelNotReadyException if models are not fully trained and verified.
        """
        model_status = self.get_model_status()
        if not model_status.can_infer:
            raise ModelNotReadyException(
                message=(
                    f"Model inference is strictly prohibited: active model state is '{model_status.model_state.value}'. "
                    "Zero verified clinical checkpoints exist. Rule 6 & 7 violation avoided."
                ),
                current_state=model_status.model_state.value,
                details={"blockers": model_status.blockers},
            )
