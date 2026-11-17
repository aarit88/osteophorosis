"""
OsteoGuard AI - Selective Prediction and Clinical Abstention Engine.

Implements safe automated abstention logic:
- Decision states: ACCEPT_PREDICTION, ABSTAIN_HUMAN_REVIEW_REQUIRED, INVALID_INPUT.
- Evaluates Normalized Shannon Entropy against the proposal's H > 0.60 candidate threshold.
- Evaluates Out-Of-Distribution (OOD) flag triggers.
- Dispatches transparent clinical abstention reason codes.

SAFETY & CLINICAL GOVERNANCE NOTICE:
    - An abstention is NOT a negative or normal diagnosis.
    - Demoting a high-uncertainty case to 'Human Review Required' prevents automation bias.
    - The H > 0.60 threshold is an unvalidated candidate engineering threshold;
      it has not been optimized for clinical diagnostic sensitivity or specificity.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
from pydantic import BaseModel, Field

from osteoguard.constants import CLASS_DISPLAY_NAMES, IDX_TO_CLASS, TARGET_CLASSES
from osteoguard.uncertainty.metrics import (
    DistributionDiagnostics,
    compute_distribution_diagnostics,
    compute_normalized_entropy,
    validate_probability_distribution,
)


class SelectiveDecision(str, Enum):
    ACCEPT_PREDICTION = "ACCEPT_PREDICTION"
    ABSTAIN_HUMAN_REVIEW_REQUIRED = "ABSTAIN_HUMAN_REVIEW_REQUIRED"
    INVALID_INPUT = "INVALID_INPUT"


class AbstentionReasonCode(str, Enum):
    NONE = "NONE"
    ELEVATED_ENTROPY = "ELEVATED_ENTROPY"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    AMBIGUOUS_TOP_MARGIN = "AMBIGUOUS_TOP_MARGIN"
    COMBINED_ENTROPY_AND_OOD = "COMBINED_ENTROPY_AND_OOD"
    INVALID_PROBABILITY_VECTOR = "INVALID_PROBABILITY_VECTOR"


class SelectivePredictionResult(BaseModel):
    """Container for selective prediction outcomes."""
    decision: SelectiveDecision = Field(..., description="Tri-state triage outcome")
    predicted_class_idx: Optional[int] = Field(None, description="Proposed predicted class index")
    predicted_class_label: Optional[str] = Field(None, description="Clinical taxonomy label")
    human_review_required: bool = Field(..., description="True if clinician verification is mandated")
    abstention_reason: AbstentionReasonCode = Field(..., description="Formal technical reason code")
    reason_description: str = Field(..., description="Human-readable safety explanation")
    entropy_threshold_used: float = Field(..., description="Applied entropy abstention threshold")
    applied_to_normalized: bool = Field(True, description="True if threshold was applied to normalized entropy")
    diagnostics: Optional[DistributionDiagnostics] = Field(None, description="Detailed probability diagnostics")
    disclaimer: str = Field(
        default=(
            "RESEARCH ONLY: Clinical abstention protocol. An abstention does NOT indicate absence "
            "of disease or healthy status. It mandates independent specialist review."
        )
    )


class SelectivePredictionEngine:
    """
    Automated triage gate enforcing the H > 0.60 abstention rule.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.60,
        margin_threshold: Optional[float] = None,
        use_normalized_entropy: bool = True,
    ):
        if entropy_threshold <= 0.0 or (use_normalized_entropy and entropy_threshold >= 1.0):
            raise ValueError(
                f"Entropy threshold for normalized scale must be in (0.0, 1.0), got {entropy_threshold}"
            )
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.use_normalized_entropy = use_normalized_entropy

    def evaluate(
        self,
        probabilities: Union[np.ndarray, List[float]],
        ood_flag: bool = False,
        class_labels: Optional[Sequence[str]] = TARGET_CLASSES,
    ) -> SelectivePredictionResult:
        """
        Evaluates input probabilities against the abstention criteria.

        Decision Rules:
        1. If probability vector is invalid: INVALID_INPUT.
        2. If ood_flag is True and entropy > threshold: COMBINED_ENTROPY_AND_OOD.
        3. If entropy > threshold: ELEVATED_ENTROPY -> ABSTAIN.
        4. If ood_flag is True: OUT_OF_DISTRIBUTION -> ABSTAIN.
        5. If margin_threshold is set and margin < margin_threshold: AMBIGUOUS_TOP_MARGIN -> ABSTAIN.
        6. Otherwise: ACCEPT_PREDICTION.
        """
        try:
            diag = compute_distribution_diagnostics(probabilities)
        except Exception as e:
            return SelectivePredictionResult(
                decision=SelectiveDecision.INVALID_INPUT,
                predicted_class_idx=None,
                predicted_class_label=None,
                human_review_required=True,
                abstention_reason=AbstentionReasonCode.INVALID_PROBABILITY_VECTOR,
                reason_description=f"Input rejected as invalid distribution: {str(e)}",
                entropy_threshold_used=self.entropy_threshold,
                applied_to_normalized=self.use_normalized_entropy,
                diagnostics=None,
            )

        tested_entropy = diag.normalized_entropy if self.use_normalized_entropy else diag.raw_entropy_nats
        # Strict inequality as formulated in KSCST Proposal: H > 0.60
        entropy_exceeded = tested_entropy > self.entropy_threshold

        margin_ambiguous = False
        if self.margin_threshold is not None:
            margin_ambiguous = diag.top_two_margin < self.margin_threshold

        pred_label = None
        if class_labels is not None and 0 <= diag.predicted_class_idx < len(class_labels):
            pred_label = class_labels[diag.predicted_class_idx]

        # Case 1: Both elevated entropy and OOD
        if entropy_exceeded and ood_flag:
            return SelectivePredictionResult(
                decision=SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED,
                predicted_class_idx=diag.predicted_class_idx,
                predicted_class_label=pred_label,
                human_review_required=True,
                abstention_reason=AbstentionReasonCode.COMBINED_ENTROPY_AND_OOD,
                reason_description=(
                    f"CRITICAL ABSTENTION: High entropy ({tested_entropy:.3f} > {self.entropy_threshold:.2f}) "
                    f"and Out-of-Distribution detector triggered simultaneously."
                ),
                entropy_threshold_used=self.entropy_threshold,
                applied_to_normalized=self.use_normalized_entropy,
                diagnostics=diag,
            )

        # Case 2: Elevated entropy
        if entropy_exceeded:
            return SelectivePredictionResult(
                decision=SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED,
                predicted_class_idx=diag.predicted_class_idx,
                predicted_class_label=pred_label,
                human_review_required=True,
                abstention_reason=AbstentionReasonCode.ELEVATED_ENTROPY,
                reason_description=(
                    f"ABSTENTION: Prediction entropy ({tested_entropy:.3f}) exceeds safety limit ({self.entropy_threshold:.2f}). "
                    f"Diagnostic probabilities are overly dispersed across differential targets."
                ),
                entropy_threshold_used=self.entropy_threshold,
                applied_to_normalized=self.use_normalized_entropy,
                diagnostics=diag,
            )

        # Case 3: OOD triggered
        if ood_flag:
            return SelectivePredictionResult(
                decision=SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED,
                predicted_class_idx=diag.predicted_class_idx,
                predicted_class_label=pred_label,
                human_review_required=True,
                abstention_reason=AbstentionReasonCode.OUT_OF_DISTRIBUTION,
                reason_description=(
                    "ABSTENTION: Anatomical feature representation lies outside training manifold bounds. "
                    "Mandatory clinical radiologist review required."
                ),
                entropy_threshold_used=self.entropy_threshold,
                applied_to_normalized=self.use_normalized_entropy,
                diagnostics=diag,
            )

        # Case 4: Ambiguous margin
        if margin_ambiguous:
            return SelectivePredictionResult(
                decision=SelectiveDecision.ABSTAIN_HUMAN_REVIEW_REQUIRED,
                predicted_class_idx=diag.predicted_class_idx,
                predicted_class_label=pred_label,
                human_review_required=True,
                abstention_reason=AbstentionReasonCode.AMBIGUOUS_TOP_MARGIN,
                reason_description=(
                    f"ABSTENTION: Top-two probability margin ({diag.top_two_margin:.3f}) below "
                    f"discriminative threshold ({self.margin_threshold:.2f})."
                ),
                entropy_threshold_used=self.entropy_threshold,
                applied_to_normalized=self.use_normalized_entropy,
                diagnostics=diag,
            )

        # Case 5: Accept
        return SelectivePredictionResult(
            decision=SelectiveDecision.ACCEPT_PREDICTION,
            predicted_class_idx=diag.predicted_class_idx,
            predicted_class_label=pred_label,
            human_review_required=False,
            abstention_reason=AbstentionReasonCode.NONE,
            reason_description="Verified high-confidence prediction. Entropy and manifold bounds satisfied.",
            entropy_threshold_used=self.entropy_threshold,
            applied_to_normalized=self.use_normalized_entropy,
            diagnostics=diag,
        )
