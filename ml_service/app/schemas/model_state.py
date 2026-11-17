"""
OsteoGuard AI - Model Readiness State Machine Definitions.
Strictly defines the finite states governing model training and inference clearance.
"""

from enum import Enum


class ModelState(str, Enum):
    """
    Finite state enum representing the lifecycle of OsteoGuard AI models.
    Transition rules:
      NOT_CONFIGURED -> AWAITING_DATA (when config valid, data dirs checked)
      AWAITING_DATA -> DATA_VALIDATION_REQUIRED (when raw data is placed in dirs)
      DATA_VALIDATION_REQUIRED -> READY_FOR_TRAINING (when patient stratification & modality checks pass)
      READY_FOR_TRAINING -> TRAINING (when training loop begins)
      TRAINING -> TRAINED (when weights converge and checkpoints written)
      TRAINED -> VALIDATION_REQUIRED (when test split evaluation starts)
      VALIDATION_REQUIRED -> READY_FOR_INFERENCE (when benchmark criteria met)
      * -> FAILED (on unrecoverable pipeline failure)
    """
    NOT_CONFIGURED = "NOT_CONFIGURED"
    AWAITING_DATA = "AWAITING_DATA"
    DATA_VALIDATION_REQUIRED = "DATA_VALIDATION_REQUIRED"
    READY_FOR_TRAINING = "READY_FOR_TRAINING"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    VALIDATION_REQUIRED = "VALIDATION_REQUIRED"
    READY_FOR_INFERENCE = "READY_FOR_INFERENCE"
    FAILED = "FAILED"
