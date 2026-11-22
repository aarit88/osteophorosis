"""
OsteoGuard AI — Stage 35 Institutional Dataset Acquisition, Research Governance,
Acquisition Workflow State Machine, Cohort Protocol Validation, and Cohort Planning Service.

Operates cleanly in pure Python 3.10+ stdlib and reuses Stage 33/34 canonical modules:
- `ml_service.datasets.clinical_ontology` (`OSTEO_ONTOLOGY_V1.0_STAGE34`)
- `ml_service.app.services.adjudication_engine` (`AdjudicationEngine`)
- `ml_service.datasets.track_d_intake` (`validate_track_d_dataset`, `load_protected_holdout_registry`)

Strict Governance Guarantees:
1. Never marks a governance field verified merely because it is non-empty; requires valid status,
   evidence reference, authorized pseudonymous reviewer ID + role, and non-expired dates.
2. Enforces sequential cohort acquisition workflow transitions; blocks unauthorized reviewers,
   missing evidence references, skipped stages, or expired ethics/DUA approvals.
3. Keeps X-ray, bone scintigraphy, and CT strictly separated.
4. Strictly separates planned cohort targets from actual on-disk validated counts (`0` until real data arrives).
5. Preserves Stage 34 ontology boundaries and BTXRD Center 2 (N=549) / Center 3 (N=259) holdouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ml_service.datasets.clinical_ontology import (
    ONTOLOGY_VERSION,
    PLACEHOLDER_VALUES,
    PSEUDO_REVIEWER_ID_RE,
    AdjudicationStatus,
    OntologyTargetClass,
    adjudicate_diagnostic_label,
    validate_healthy_control_eligibility,
)
from ml_service.datasets.track_d_intake import (
    SUPPORTED_MODALITIES,
    TARGET_CLASSES,
    validate_track_d_dataset,
)
from ml_service.app.services.adjudication_engine import AdjudicationEngine


class AcquisitionStatus(str, Enum):
    NOT_RESEARCHED = "NOT_RESEARCHED"
    IDENTIFIED = "IDENTIFIED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    ACCESS_REQUESTED = "ACCESS_REQUESTED"
    APPROVED = "APPROVED"
    DATA_RECEIVED = "DATA_RECEIVED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


class WorkflowStage(str, Enum):
    SOURCE_IDENTIFIED = "SOURCE_IDENTIFIED"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    ETHICS_PENDING = "ETHICS_PENDING"
    DUA_PENDING = "DUA_PENDING"
    ACCESS_APPROVED = "ACCESS_APPROVED"
    DATA_RECEIVED = "DATA_RECEIVED"
    DEIDENTIFICATION_VERIFIED = "DEIDENTIFICATION_VERIFIED"
    LABEL_ADJUDICATION = "LABEL_ADJUDICATION"
    QUALITY_VALIDATION = "QUALITY_VALIDATION"
    COHORT_ACCEPTED = "COHORT_ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    BLOCKED = "BLOCKED"


class GovernanceRole(str, Enum):
    DATA_STEWARD = "DATA_STEWARD"
    PRINCIPAL_INVESTIGATOR = "PRINCIPAL_INVESTIGATOR"
    IRB_OFFICER = "IRB_OFFICER"
    DATA_PROTECTION_OFFICER = "DATA_PROTECTION_OFFICER"
    LEGAL_COUNSEL = "LEGAL_COUNSEL"
    CLINICAL_ADJUDICATOR = "CLINICAL_ADJUDICATOR"
    RADIOLOGIST = "RADIOLOGIST"


HAPPY_PATH_SEQUENCE: Tuple[WorkflowStage, ...] = (
    WorkflowStage.SOURCE_IDENTIFIED,
    WorkflowStage.SOURCE_VERIFIED,
    WorkflowStage.ETHICS_PENDING,
    WorkflowStage.DUA_PENDING,
    WorkflowStage.ACCESS_APPROVED,
    WorkflowStage.DATA_RECEIVED,
    WorkflowStage.DEIDENTIFICATION_VERIFIED,
    WorkflowStage.LABEL_ADJUDICATION,
    WorkflowStage.QUALITY_VALIDATION,
    WorkflowStage.COHORT_ACCEPTED,
)

EXCEPTION_STATES: Set[WorkflowStage] = {
    WorkflowStage.REJECTED,
    WorkflowStage.EXPIRED,
    WorkflowStage.REQUIRES_REVIEW,
    WorkflowStage.BLOCKED,
}

# Explicit allowed transitions (strict sequential ordering + controlled exception transitions)
ALLOWED_WORKFLOW_TRANSITIONS: Dict[WorkflowStage, Set[WorkflowStage]] = {
    WorkflowStage.SOURCE_IDENTIFIED: {
        WorkflowStage.SOURCE_VERIFIED,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.SOURCE_VERIFIED: {
        WorkflowStage.ETHICS_PENDING,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.ETHICS_PENDING: {
        WorkflowStage.DUA_PENDING,
        WorkflowStage.REJECTED,
        WorkflowStage.EXPIRED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.DUA_PENDING: {
        WorkflowStage.ACCESS_APPROVED,
        WorkflowStage.REJECTED,
        WorkflowStage.EXPIRED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.ACCESS_APPROVED: {
        WorkflowStage.DATA_RECEIVED,
        WorkflowStage.EXPIRED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.DATA_RECEIVED: {
        WorkflowStage.DEIDENTIFICATION_VERIFIED,
        WorkflowStage.REJECTED,
        WorkflowStage.EXPIRED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.DEIDENTIFICATION_VERIFIED: {
        WorkflowStage.LABEL_ADJUDICATION,
        WorkflowStage.REJECTED,
        WorkflowStage.EXPIRED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.LABEL_ADJUDICATION: {
        WorkflowStage.QUALITY_VALIDATION,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.QUALITY_VALIDATION: {
        WorkflowStage.COHORT_ACCEPTED,
        WorkflowStage.REJECTED,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.COHORT_ACCEPTED: {
        WorkflowStage.EXPIRED,
        WorkflowStage.REQUIRES_REVIEW,
    },
    WorkflowStage.REQUIRES_REVIEW: {
        WorkflowStage.SOURCE_VERIFIED,
        WorkflowStage.ETHICS_PENDING,
        WorkflowStage.DUA_PENDING,
        WorkflowStage.REJECTED,
        WorkflowStage.BLOCKED,
    },
    WorkflowStage.BLOCKED: {
        WorkflowStage.ETHICS_PENDING,
        WorkflowStage.DUA_PENDING,
        WorkflowStage.REQUIRES_REVIEW,
        WorkflowStage.REJECTED,
    },
    WorkflowStage.EXPIRED: {
        WorkflowStage.ETHICS_PENDING,
        WorkflowStage.DUA_PENDING,
        WorkflowStage.REJECTED,
    },
    WorkflowStage.REJECTED: set(),  # Terminal disqualification
}

# Minimum authorized governance roles required for sensitive workflow stage advances
TRANSITION_ROLE_REQUIREMENTS: Dict[WorkflowStage, Set[GovernanceRole]] = {
    WorkflowStage.SOURCE_VERIFIED: {
        GovernanceRole.DATA_STEWARD,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
        GovernanceRole.CLINICAL_ADJUDICATOR,
        GovernanceRole.RADIOLOGIST,
    },
    WorkflowStage.ETHICS_PENDING: {
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.DATA_STEWARD,
    },
    WorkflowStage.DUA_PENDING: {
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
    WorkflowStage.ACCESS_APPROVED: {
        GovernanceRole.DATA_PROTECTION_OFFICER,
        GovernanceRole.LEGAL_COUNSEL,
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
    WorkflowStage.DATA_RECEIVED: {
        GovernanceRole.DATA_STEWARD,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
    WorkflowStage.DEIDENTIFICATION_VERIFIED: {
        GovernanceRole.DATA_PROTECTION_OFFICER,
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
    WorkflowStage.LABEL_ADJUDICATION: {
        GovernanceRole.CLINICAL_ADJUDICATOR,
        GovernanceRole.RADIOLOGIST,
    },
    WorkflowStage.QUALITY_VALIDATION: {
        GovernanceRole.DATA_STEWARD,
        GovernanceRole.CLINICAL_ADJUDICATOR,
        GovernanceRole.RADIOLOGIST,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
    WorkflowStage.COHORT_ACCEPTED: {
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
        GovernanceRole.CLINICAL_ADJUDICATOR,
        GovernanceRole.RADIOLOGIST,
        GovernanceRole.IRB_OFFICER,
    },
    WorkflowStage.REJECTED: {
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.DATA_PROTECTION_OFFICER,
        GovernanceRole.LEGAL_COUNSEL,
        GovernanceRole.CLINICAL_ADJUDICATOR,
        GovernanceRole.RADIOLOGIST,
    },
    WorkflowStage.EXPIRED: {
        GovernanceRole.DATA_STEWARD,
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.DATA_PROTECTION_OFFICER,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
    WorkflowStage.REQUIRES_REVIEW: {
        GovernanceRole.DATA_STEWARD,
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.CLINICAL_ADJUDICATOR,
        GovernanceRole.RADIOLOGIST,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
    WorkflowStage.BLOCKED: {
        GovernanceRole.DATA_STEWARD,
        GovernanceRole.IRB_OFFICER,
        GovernanceRole.DATA_PROTECTION_OFFICER,
        GovernanceRole.LEGAL_COUNSEL,
        GovernanceRole.PRINCIPAL_INVESTIGATOR,
    },
}

WORKFLOW_TO_ACQUISITION_STATUS: Dict[WorkflowStage, AcquisitionStatus] = {
    WorkflowStage.SOURCE_IDENTIFIED: AcquisitionStatus.IDENTIFIED,
    WorkflowStage.SOURCE_VERIFIED: AcquisitionStatus.VERIFICATION_REQUIRED,
    WorkflowStage.ETHICS_PENDING: AcquisitionStatus.ACCESS_REQUESTED,
    WorkflowStage.DUA_PENDING: AcquisitionStatus.ACCESS_REQUESTED,
    WorkflowStage.ACCESS_APPROVED: AcquisitionStatus.APPROVED,
    WorkflowStage.DATA_RECEIVED: AcquisitionStatus.DATA_RECEIVED,
    WorkflowStage.DEIDENTIFICATION_VERIFIED: AcquisitionStatus.DATA_RECEIVED,
    WorkflowStage.LABEL_ADJUDICATION: AcquisitionStatus.DATA_RECEIVED,
    WorkflowStage.QUALITY_VALIDATION: AcquisitionStatus.DATA_RECEIVED,
    WorkflowStage.COHORT_ACCEPTED: AcquisitionStatus.VALIDATED,
    WorkflowStage.REJECTED: AcquisitionStatus.REJECTED,
    WorkflowStage.EXPIRED: AcquisitionStatus.VERIFICATION_REQUIRED,
    WorkflowStage.REQUIRES_REVIEW: AcquisitionStatus.VERIFICATION_REQUIRED,
    WorkflowStage.BLOCKED: AcquisitionStatus.VERIFICATION_REQUIRED,
}

APPROVED_FLAG_VALUES: Set[str] = {"APPROVED", "EXECUTED", "VERIFIED", "ACTIVE"}


@dataclass(frozen=True)
class WorkflowTransitionEntry:
    timestamp: str
    source_id: str
    previous_status: str
    new_status: str
    reviewer_pseudo_id: str
    reviewer_role: str
    reason: str
    evidence_reference: str


def _parse_iso_date(raw: Optional[str]) -> Optional[date]:
    if not raw or not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.lower() in PLACEHOLDER_VALUES:
        return None
    try:
        return date.fromisoformat(cleaned[:10])
    except ValueError:
        return None


def _is_non_placeholder(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip().lower() not in PLACEHOLDER_VALUES
    if isinstance(val, (list, dict)):
        return len(val) > 0
    return True


def evaluate_governance_record(
    source: Dict[str, Any],
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Audits a source's institutional governance record rigorously.
    Never treats non-empty strings alone as verified approval.
    """
    today = reference_date or datetime.now(timezone.utc).date()
    gov = source.get("governance_record") or {}
    source_id = str(source.get("source_id") or "")
    target_cohort = str(source.get("target_cohort") or "")

    irb = gov.get("irb_iec_approval") or {}
    irb_status = str(irb.get("status") or "").strip().upper()
    irb_ref = irb.get("protocol_ref")
    irb_approved = irb_status in APPROVED_FLAG_VALUES and _is_non_placeholder(irb_ref)

    dua = gov.get("dua") or {}
    dua_status = str(dua.get("status") or "").strip().upper()
    dua_ref = dua.get("agreement_ref")
    dua_approved = dua_status in APPROVED_FLAG_VALUES and _is_non_placeholder(dua_ref)

    inst_auth = gov.get("institutional_authorization") or {}
    inst_status = str(inst_auth.get("status") or "").strip().upper()
    inst_approved = inst_status in APPROVED_FLAG_VALUES and _is_non_placeholder(inst_auth.get("signoff_ref"))

    deid = gov.get("deidentification_approval") or {}
    deid_status = str(deid.get("status") or "").strip().upper()
    deid_approved = deid_status in APPROVED_FLAG_VALUES and _is_non_placeholder(deid.get("sop_ref"))

    protocol_id = gov.get("research_protocol_id")
    consent_status = str(gov.get("consent_or_waiver_status") or "").strip().upper()
    retention_req = gov.get("data_retention_requirements")
    purposes = gov.get("permitted_research_purposes") or []
    redistribution = gov.get("redistribution_restrictions")
    evidence_ref = gov.get("evidence_reference") or source.get("evidence_reference")
    approval_dt = _parse_iso_date(gov.get("approval_date"))
    expiry_dt = _parse_iso_date(gov.get("expiry_date"))

    reviewer_id = str(gov.get("reviewer_pseudo_id") or "").strip()
    reviewer_role = str(gov.get("reviewer_role") or "").strip().upper()
    valid_reviewer = bool(reviewer_id and PSEUDO_REVIEWER_ID_RE.match(reviewer_id))
    valid_role = reviewer_role in {r.value for r in GovernanceRole}

    is_expired = expiry_dt is not None and expiry_dt < today

    missing_or_pending: List[str] = []
    if not irb_approved:
        missing_or_pending.append("MISSING_OR_PENDING:irb_iec_approval")
    if not _is_non_placeholder(protocol_id):
        missing_or_pending.append("MISSING:research_protocol_id")
    if not dua_approved:
        missing_or_pending.append("MISSING_OR_PENDING:dua")
    if not inst_approved:
        missing_or_pending.append("MISSING_OR_PENDING:institutional_authorization")
    if not _is_non_placeholder(consent_status) or "PENDING" in consent_status or "REQUIRED" in consent_status:
        missing_or_pending.append("MISSING_OR_PENDING:consent_or_waiver_status")
    if not deid_approved:
        missing_or_pending.append("MISSING_OR_PENDING:deidentification_approval")
    if not _is_non_placeholder(retention_req):
        missing_or_pending.append("MISSING:data_retention_requirements")
    if not _is_non_placeholder(purposes):
        missing_or_pending.append("MISSING:permitted_research_purposes")
    if not _is_non_placeholder(redistribution):
        missing_or_pending.append("MISSING:redistribution_restrictions")
    if approval_dt is None:
        missing_or_pending.append("MISSING:approval_date")
    if expiry_dt is None:
        missing_or_pending.append("MISSING:expiry_date")
    elif is_expired:
        missing_or_pending.append(f"EXPIRED_GOVERNANCE_APPROVAL:{expiry_dt.isoformat()}")
    if not _is_non_placeholder(evidence_ref):
        missing_or_pending.append("MISSING:evidence_reference")
    if not valid_reviewer or not valid_role:
        missing_or_pending.append("MISSING_OR_INVALID:authorized_reviewer_signoff")

    if target_cohort == "oi" and "PEDIATRIC" in consent_status and "APPROVED" not in consent_status and "WAIVED" not in consent_status:
        missing_or_pending.append("PENDING:pediatric_consent_or_assent_clearance")

    checks = {
        "irb_iec_approved": irb_approved,
        "protocol_id_present": _is_non_placeholder(protocol_id),
        "dua_approved": dua_approved,
        "institutional_authorization_approved": inst_approved,
        "consent_or_waiver_cleared": (
            _is_non_placeholder(consent_status)
            and "PENDING" not in consent_status
            and "REQUIRED" not in consent_status
        ),
        "deidentification_approved": deid_approved,
        "retention_policy_present": _is_non_placeholder(retention_req),
        "permitted_purposes_present": _is_non_placeholder(purposes),
        "redistribution_restrictions_present": _is_non_placeholder(redistribution),
        "approval_date_valid": approval_dt is not None,
        "expiry_date_valid_and_active": expiry_dt is not None and not is_expired,
        "evidence_reference_present": _is_non_placeholder(evidence_ref),
        "authorized_reviewer_signoff": valid_reviewer and valid_role,
    }
    passed_checks = sum(1 for ok in checks.values() if ok)
    evidence_completeness_pct = round(100.0 * passed_checks / len(checks), 1)
    governance_verified = all(checks.values()) and not is_expired

    return {
        "source_id": source_id,
        "target_cohort": target_cohort,
        "governance_verified": governance_verified,
        "irb_iec_approved": irb_approved,
        "dua_approved": dua_approved,
        "is_expired": is_expired,
        "approval_date": approval_dt.isoformat() if approval_dt else None,
        "expiry_date": expiry_dt.isoformat() if expiry_dt else None,
        "evidence_reference": evidence_ref if _is_non_placeholder(evidence_ref) else None,
        "evidence_completeness_pct": evidence_completeness_pct,
        "checks": checks,
        "missing_or_pending": sorted(set(missing_or_pending)),
    }


def transition_workflow_state(
    source: Dict[str, Any],
    target_state: WorkflowStage | str,
    reviewer_pseudo_id: str,
    reviewer_role: GovernanceRole | str,
    reason: str,
    evidence_reference: str,
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Executes an audited transition on a dataset source's acquisition workflow.
    Raises ValueError or PermissionError if any transition rule, reviewer permission,
    evidence requirement, or approval expiry check is violated.
    """
    current_raw = str(source.get("workflow_state") or WorkflowStage.SOURCE_IDENTIFIED.value).strip().upper()
    current_state = WorkflowStage(current_raw)
    next_state = (
        target_state
        if isinstance(target_state, WorkflowStage)
        else WorkflowStage(str(target_state).strip().upper())
    )

    reviewer_clean = (reviewer_pseudo_id or "").strip()
    if not PSEUDO_REVIEWER_ID_RE.match(reviewer_clean):
        raise ValueError(
            "reviewer_pseudo_id must be a valid pseudonymous reviewer identifier (REV_*, ADJ_*, RAD_*, CLIN_*)."
        )

    role_enum = (
        reviewer_role
        if isinstance(reviewer_role, GovernanceRole)
        else GovernanceRole(str(reviewer_role).strip().upper())
    )

    if not (reason or "").strip():
        raise ValueError("Explicit governance rationale (`reason`) is required for every workflow transition.")
    if not _is_non_placeholder(evidence_reference):
        raise ValueError("Supporting `evidence_reference` is required and cannot be empty or placeholder.")

    allowed_targets = ALLOWED_WORKFLOW_TRANSITIONS.get(current_state, set())
    if next_state not in allowed_targets:
        raise ValueError(
            f"Invalid workflow transition from '{current_state.value}' to '{next_state.value}'. "
            f"Allowed targets: {sorted(s.value for s in allowed_targets)}."
        )

    required_roles = TRANSITION_ROLE_REQUIREMENTS.get(next_state, set())
    if role_enum not in required_roles:
        raise PermissionError(
            f"Unauthorized reviewer role '{role_enum.value}' for transition to '{next_state.value}'. "
            f"Authorized roles: {sorted(r.value for r in required_roles)}."
        )

    gov_audit = evaluate_governance_record(source, reference_date=reference_date)
    if gov_audit["is_expired"] and next_state not in {
        WorkflowStage.EXPIRED,
        WorkflowStage.REJECTED,
        WorkflowStage.ETHICS_PENDING,
        WorkflowStage.DUA_PENDING,
    }:
        raise ValueError(
            f"Cannot advance '{source.get('source_id')}' to '{next_state.value}': "
            f"governance approval expired on {gov_audit['expiry_date']}."
        )

    # Advancing to ACCESS_APPROVED or beyond requires full verified ethics + DUA + governance evidence
    gated_post_access = {
        WorkflowStage.ACCESS_APPROVED,
        WorkflowStage.DATA_RECEIVED,
        WorkflowStage.DEIDENTIFICATION_VERIFIED,
        WorkflowStage.LABEL_ADJUDICATION,
        WorkflowStage.QUALITY_VALIDATION,
        WorkflowStage.COHORT_ACCEPTED,
    }
    if next_state in gated_post_access:
        if not gov_audit["irb_iec_approved"]:
            raise ValueError("Cannot advance past DUA_PENDING without verified IRB/IEC approval and protocol_ref.")
        if not gov_audit["dua_approved"]:
            raise ValueError("Cannot advance past DUA_PENDING without executed DUA and agreement_ref.")
        if gov_audit["is_expired"] or gov_audit["expiry_date"] is None:
            raise ValueError("Cannot approve access without active non-expired governance expiry_date.")
        if not gov_audit["governance_verified"] and next_state in {
            WorkflowStage.ACCESS_APPROVED,
            WorkflowStage.COHORT_ACCEPTED,
        }:
            raise ValueError(
                "Incomplete governance verification: " + ", ".join(gov_audit["missing_or_pending"])
            )

    now_iso = datetime.now(timezone.utc).isoformat()
    entry = WorkflowTransitionEntry(
        timestamp=now_iso,
        source_id=str(source.get("source_id") or ""),
        previous_status=current_state.value,
        new_status=next_state.value,
        reviewer_pseudo_id=reviewer_clean,
        reviewer_role=role_enum.value,
        reason=reason.strip(),
        evidence_reference=evidence_reference.strip(),
    )

    updated = dict(source)
    updated["workflow_state"] = next_state.value
    updated["acquisition_status"] = WORKFLOW_TO_ACQUISITION_STATUS[next_state].value
    history = list(updated.get("workflow_history") or [])
    history.append(asdict(entry))
    updated["workflow_history"] = history
    return updated


def validate_cohort_protocol_record(
    record: Dict[str, Any],
    protocols: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validates a candidate patient/sample intake record against both:
    - Stage 34 centralized ontology & healthy-control eligibility (`adjudicate_diagnostic_label`)
    - Stage 35 cohort-specific acquisition protocols (modality separation, confirmation tiering,
      pediatric OI consent/genetics, Paget triad, and healthy anatomical matching).
    """
    adj = adjudicate_diagnostic_label(record)
    errors: List[str] = []
    warnings: List[str] = []

    raw_mod = str(record.get("imaging_modality") or "").strip().lower()
    if raw_mod not in SUPPORTED_MODALITIES:
        errors.append(f"PROTOCOL_MODALITY_ERROR:unsupported_modality:{raw_mod or 'missing'}")

    target_short = adj.get("short_class_code") or str(record.get("diagnostic_class") or "").strip().lower()
    if protocols and target_short in protocols:
        allowed_mods = protocols[target_short].get("allowed_modalities") or []
        if raw_mod and allowed_mods and raw_mod not in allowed_mods:
            errors.append(
                f"PROTOCOL_MODALITY_DISALLOWED_FOR_CLASS:{raw_mod}_not_in_{allowed_mods}"
            )

    ev_cat = str(record.get("diagnostic_evidence_category") or "").strip().upper()
    combined_meta = f"{record.get('dataset_source', '')} {record.get('diagnostic_evidence', '')} {record.get('image_path', '')}".lower()

    if target_short == "metastasis":
        tier = str(record.get("confirmation_tier") or "").strip().upper()
        if not tier:
            if ev_cat == "BIOPSY_HISTOPATHOLOGY" or str(record.get("histopathology_confirmation_status") or "").upper() == "CONFIRMED":
                tier = "HISTOPATHOLOGY_CONFIRMED"
            elif ev_cat in ("PET_CT_AND_RADIOLOGY", "MULTI_READER_CONSENSUS"):
                tier = "RADIOLOGY_AND_CLINICAL_CONCORDANCE"
            else:
                errors.append("METASTASIS_CONFIRMATION_TIER_UNSPECIFIED")
        if tier not in ("HISTOPATHOLOGY_CONFIRMED", "RADIOLOGY_AND_CLINICAL_CONCORDANCE"):
            errors.append(f"METASTASIS_INVALID_CONFIRMATION_TIER:{tier}")
        if tier == "HISTOPATHOLOGY_CONFIRMED" and str(record.get("histopathology_confirmation_status") or "").upper() != "CONFIRMED":
            errors.append("METASTASIS_HISTOPATHOLOGY_NOT_CONFIRMED")

    elif target_short == "paget":
        for banned_edu in ("radiopaedia", "medpix", "wikimedia"):
            if banned_edu in combined_meta:
                errors.append(f"PAGET_EDUCATIONAL_WEB_SOURCE_DISQUALIFIED:{banned_edu}")
        valid_paget_cats = {
            "RADIOLOGY_AND_CLINICAL_TRIAD",
            "SERUM_ALP_AND_RADIOGRAPH",
            "BIOPSY_HISTOPATHOLOGY",
            "MULTI_READER_CONSENSUS",
        }
        if ev_cat not in valid_paget_cats:
            errors.append(f"PAGET_MISSING_CLINICAL_RADIOLOGICAL_TRIAD_EVIDENCE:{ev_cat or 'MISSING'}")

    elif target_short == "oi":
        for banned_animal in ("mouse", "murine", "mus_musculus", "micro-ct", "microct", "3795019"):
            if banned_animal in combined_meta:
                errors.append(f"OI_ANIMAL_OR_MICROCT_DISQUALIFIED:{banned_animal}")
        track = str(record.get("confirmation_track") or "").strip().upper()
        if not track:
            if ev_cat == "GENETIC_COL1A1_COL1A2":
                track = "GENETIC_CONFIRMED"
            elif ev_cat in ("SPECIALIST_CLINICAL_DIAGNOSIS", "MULTI_READER_CONSENSUS", "RADIOLOGY_AND_CLINICAL_TRIAD"):
                track = "CLINICAL_RADIOLOGICAL_SILLENCE"
            else:
                errors.append("OI_CONFIRMATION_TRACK_UNSPECIFIED")
        if track not in ("GENETIC_CONFIRMED", "CLINICAL_RADIOLOGICAL_SILLENCE"):
            errors.append(f"OI_INVALID_CONFIRMATION_TRACK:{track}")

        age = record.get("patient_age_years")
        is_pediatric = bool(record.get("is_pediatric")) or (isinstance(age, (int, float)) and age < 18)
        if is_pediatric:
            ped_consent = str(record.get("pediatric_consent_status") or "").strip().upper()
            if ped_consent not in ("PARENTAL_CONSENT_AND_ASSENT_VERIFIED", "PEDIATRIC_IEC_WAIVER_APPROVED"):
                errors.append("OI_PEDIATRIC_CONSENT_OR_IEC_WAIVER_MISSING")

    elif target_short == "healthy":
        h_ok, h_failures = validate_healthy_control_eligibility(record)
        if not h_ok:
            errors.extend(h_failures)
        src_l = str(record.get("dataset_source") or "").lower()
        anat = str(record.get("anatomical_region") or "").lower()
        if "mura" in src_l and anat in ("pelvis", "spine", "lumbar_spine", "thoracic_spine", "femur", "tibia", "skull"):
            errors.append("HEALTHY_MURA_ANATOMICAL_SCOPE_VIOLATION:upper_extremity_only")

    if adj["adjudication_status"] != AdjudicationStatus.ACCEPTED.value:
        errors.append(f"ONTOLOGY_ADJUDICATION_NOT_ACCEPTED:{adj['adjudication_status']}")

    return {
        "valid": len(errors) == 0,
        "target_class": target_short or None,
        "modality": raw_mod or None,
        "adjudication_status": adj["adjudication_status"],
        "ontology_version": ONTOLOGY_VERSION,
        "protocol_errors": sorted(set(errors)),
        "warnings": warnings,
    }


class AcquisitionGovernanceService:
    """Centralized Stage 35 service coordinating sources, governance, workflow, and planning."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[3]
        self.manifests_dir = self.root_dir / "ml_service" / "data" / "manifests"
        self.registry_path = self.manifests_dir / "dataset_source_registry.json"
        self.protocols_path = self.manifests_dir / "cohort_acquisition_protocols.json"
        self.planning_path = self.manifests_dir / "cohort_planning_manifest.json"

    def load_registry(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            return {"schema_version": "1.0.0", "sources": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def load_protocols(self) -> Dict[str, Any]:
        if not self.protocols_path.exists():
            return {"protocols": {}}
        return json.loads(self.protocols_path.read_text(encoding="utf-8"))

    def load_planning(self) -> Dict[str, Any]:
        if not self.planning_path.exists():
            return {"planned_cohorts": []}
        return json.loads(self.planning_path.read_text(encoding="utf-8"))

    def _measure_actual_disk_readiness(self) -> Dict[str, Any]:
        validation = validate_track_d_dataset(self.root_dir)
        passed = validation.get("passed_records") or []
        matrix: Dict[str, Dict[str, int]] = {
            mod: {cls: 0 for cls in TARGET_CLASSES}
            for mod in SUPPORTED_MODALITIES
        }
        patient_sets: Dict[Tuple[str, str], Set[str]] = {
            (mod, cls): set()
            for mod in SUPPORTED_MODALITIES
            for cls in TARGET_CLASSES
        }
        for rec in passed:
            mod = str(rec.get("imaging_modality") or "").lower()
            cls = str(rec.get("diagnostic_class") or "").lower()
            pid = str(rec.get("patient_id") or "")
            if mod in matrix and cls in matrix[mod]:
                matrix[mod][cls] += 1
                if pid:
                    patient_sets[(mod, cls)].add(pid)

        return {
            "validation": validation,
            "class_modality_images": matrix,
            "class_modality_patients": {
                f"{mod}:{cls}": len(pids) for (mod, cls), pids in patient_sets.items()
            },
            "valid_samples": len(passed),
            "unique_patients": len({str(r.get("patient_id") or "") for r in passed if r.get("patient_id")}),
        }

    def get_sources_payload(self) -> Dict[str, Any]:
        reg = self.load_registry()
        sources = reg.get("sources") or []
        enriched: List[Dict[str, Any]] = []
        valid_statuses = {s.value for s in AcquisitionStatus}

        for src in sources:
            gov_audit = evaluate_governance_record(src)
            status = str(src.get("acquisition_status") or AcquisitionStatus.NOT_RESEARCHED.value)
            if status not in valid_statuses:
                status = AcquisitionStatus.VERIFICATION_REQUIRED.value
            enriched.append({
                "source_id": src.get("source_id"),
                "target_cohort": src.get("target_cohort"),
                "ontology_target_class": src.get("ontology_target_class"),
                "source_classification": src.get("source_classification"),
                "source_name": src.get("source_name"),
                "institution": src.get("institution"),
                "repository_url_or_contact_ref": src.get("repository_url_or_contact_ref"),
                "url_verified": bool(src.get("url_verified")),
                "available_modalities": list(src.get("available_modalities") or []),
                "anatomical_coverage": src.get("anatomical_coverage"),
                "diagnostic_confirmation_requirements": src.get("diagnostic_confirmation_requirements"),
                "verified_patient_count": src.get("verified_patient_count"),
                "verified_image_count": src.get("verified_image_count"),
                "license_and_usage_restrictions": src.get("license_and_usage_restrictions"),
                "access_requirements": src.get("access_requirements"),
                "ethics_approval_requirements": src.get("ethics_approval_requirements"),
                "dua_status": src.get("dua_status"),
                "deidentification_requirements": src.get("deidentification_requirements"),
                "evidence_reference": src.get("evidence_reference"),
                "verification_date": src.get("verification_date"),
                "acquisition_status": status,
                "workflow_state": src.get("workflow_state"),
                "governance_verified": gov_audit["governance_verified"],
                "evidence_completeness_pct": gov_audit["evidence_completeness_pct"],
                "blockers": list(src.get("blockers") or []),
            })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ontology_version": ONTOLOGY_VERSION,
            "total_sources": len(enriched),
            "identified_sources_count": sum(
                1 for s in enriched if s["source_classification"] != "DISQUALIFIED"
            ),
            "verified_external_or_in_tree_candidates_count": sum(
                1
                for s in enriched
                if s["source_classification"] in ("VERIFIED_EXTERNAL_CANDIDATE", "AUDITED_IN_TREE_SUBTRACK")
            ),
            "approved_sources_count": sum(
                1 for s in enriched if s["acquisition_status"] in ("APPROVED", "DATA_RECEIVED", "VALIDATED")
            ),
            "rejected_sources_count": sum(1 for s in enriched if s["acquisition_status"] == "REJECTED"),
            "sources": enriched,
            "disclaimer": "Registry clearly distinguishes proposed institutional partnerships from verified candidate sources. Zero approvals or patient counts are fabricated.",
        }

    def get_acquisition_status_payload(self) -> Dict[str, Any]:
        reg = self.load_registry()
        sources = reg.get("sources") or []
        actual = self._measure_actual_disk_readiness()
        protocols = (self.load_protocols().get("protocols") or {})

        cohorts: List[Dict[str, Any]] = []
        for cls in TARGET_CLASSES:
            cls_sources = [s for s in sources if s.get("target_cohort") == cls]
            active_sources = [s for s in cls_sources if s.get("acquisition_status") != "REJECTED"]
            approved_sources = [
                s for s in active_sources if s.get("acquisition_status") in ("APPROVED", "DATA_RECEIVED", "VALIDATED")
            ]
            received_images = sum(actual["class_modality_images"][mod][cls] for mod in SUPPORTED_MODALITIES)
            received_patients = sum(
                actual["class_modality_patients"].get(f"{mod}:{cls}", 0) for mod in SUPPORTED_MODALITIES
            )

            if received_patients > 0 and approved_sources:
                cohort_wf = WorkflowStage.QUALITY_VALIDATION.value
            elif approved_sources:
                cohort_wf = WorkflowStage.ACCESS_APPROVED.value
            elif any(s.get("workflow_state") == WorkflowStage.SOURCE_VERIFIED.value for s in active_sources):
                cohort_wf = WorkflowStage.ETHICS_PENDING.value
            else:
                cohort_wf = WorkflowStage.SOURCE_IDENTIFIED.value

            blockers: List[str] = []
            for s in active_sources:
                blockers.extend(s.get("blockers") or [])

            cohorts.append({
                "cohort_class": cls,
                "ontology_target_class": protocols.get(cls, {}).get("ontology_target_class", cls.upper()),
                "current_workflow_stage": cohort_wf,
                "identified_sources_count": len(active_sources),
                "verified_sources_count": sum(
                    1
                    for s in active_sources
                    if s.get("source_classification") in ("VERIFIED_EXTERNAL_CANDIDATE", "AUDITED_IN_TREE_SUBTRACK")
                ),
                "approved_sources_count": len(approved_sources),
                "disqualified_sources_count": len(cls_sources) - len(active_sources),
                "received_images_count": received_images,
                "validated_patients_count": received_patients,
                "modality_breakdown": {
                    mod: {
                        "images": actual["class_modality_images"][mod][cls],
                        "patients": actual["class_modality_patients"].get(f"{mod}:{cls}", 0),
                    }
                    for mod in SUPPORTED_MODALITIES
                },
                "outstanding_blockers": list(dict.fromkeys(blockers)),
            })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_acquisition_status": (
                "READY_FOR_TRAINING"
                if all(c["validated_patients_count"] > 0 for c in cohorts)
                else "BLOCKED — INSTITUTIONAL ETHICS, DUA & CLINICAL COHORTS PENDING"
            ),
            "workflow_stages": [s.value for s in HAPPY_PATH_SEQUENCE],
            "exception_states": sorted(s.value for s in EXCEPTION_STATES),
            "cohorts": cohorts,
            "total_validated_patients": actual["unique_patients"],
            "total_received_valid_images": actual["valid_samples"],
            "protected_holdouts": {
                "btxrd_center2_samples": 549,
                "btxrd_center3_samples": 259,
                "isolation_enforced": True,
            },
        }

    def get_governance_status_payload(self, reference_date: Optional[date] = None) -> Dict[str, Any]:
        reg = self.load_registry()
        sources = reg.get("sources") or []
        audits = [
            evaluate_governance_record(s, reference_date=reference_date)
            for s in sources
            if s.get("acquisition_status") != "REJECTED"
        ]

        pending_ethics = sum(1 for a in audits if not a["irb_iec_approved"])
        pending_duas = sum(1 for a in audits if not a["dua_approved"])
        approved_sources = sum(1 for a in audits if a["governance_verified"])
        expired_count = sum(1 for a in audits if a["is_expired"])
        avg_completeness = (
            round(sum(a["evidence_completeness_pct"] for a in audits) / len(audits), 1)
            if audits
            else 0.0
        )

        templates_dir = self.root_dir / "ml_service" / "data" / "templates"
        template_files = (
            sorted(p.name for p in templates_dir.glob("*.md"))
            if templates_dir.exists()
            else []
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "governance_gate_status": (
                "ALL_GOVERNANCE_VERIFIED"
                if audits and approved_sources == len(audits) and expired_count == 0
                else "BLOCKED — ETHICS, DUA & INSTITUTIONAL AUTHORIZATIONS PENDING"
            ),
            "active_sources_audited": len(audits),
            "approved_sources_count": approved_sources,
            "pending_ethics_approvals_count": pending_ethics,
            "pending_duas_count": pending_duas,
            "expired_approvals_count": expired_count,
            "mean_evidence_completeness_pct": avg_completeness,
            "templates_status": "DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW",
            "available_templates": template_files,
            "source_audits": audits,
            "protected_holdouts": {
                "btxrd_center2_samples": 549,
                "btxrd_center3_samples": 259,
                "isolation_enforced": True,
            },
        }

    def get_cohort_planning_payload(self) -> Dict[str, Any]:
        planning = self.load_planning()
        actual = self._measure_actual_disk_readiness()
        planned_items = planning.get("planned_cohorts") or []

        reconciled: List[Dict[str, Any]] = []
        total_planned_patients = 0
        total_planned_images = 0
        total_actual_patients = 0
        total_actual_images = 0

        for item in planned_items:
            cls = str(item.get("diagnostic_class") or "").lower()
            mod = str(item.get("imaging_modality") or "").lower()
            planned_pts = int(item.get("planned_patient_count") or 0)
            planned_imgs = int(item.get("planned_image_count") or 0)
            act_imgs = actual["class_modality_images"].get(mod, {}).get(cls, 0)
            act_pts = actual["class_modality_patients"].get(f"{mod}:{cls}", 0)

            total_planned_patients += planned_pts
            total_planned_images += planned_imgs
            total_actual_patients += act_pts
            total_actual_images += act_imgs

            reconciled.append({
                "plan_id": item.get("plan_id"),
                "diagnostic_class": cls,
                "ontology_target_class": item.get("ontology_target_class"),
                "imaging_modality": mod,
                "planned_patient_count": planned_pts,
                "planned_image_count": planned_imgs,
                "actual_received_patients": act_pts,
                "actual_validated_patients": act_pts,
                "actual_validated_images": act_imgs,
                "institution_distribution": item.get("institution_distribution"),
                "inclusion_criteria": item.get("inclusion_criteria"),
                "exclusion_criteria": item.get("exclusion_criteria"),
                "primary_endpoint": item.get("primary_endpoint"),
                "statistical_sample_size_justification_ref": item.get(
                    "statistical_sample_size_justification_ref",
                    "PENDING_BIOSTATISTICAL_AND_IEC_REVIEW",
                ),
                "proposed_split_allocation": item.get("proposed_split_allocation"),
                "data_gap_patients": max(0, planned_pts - act_pts),
            })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ontology_version": ONTOLOGY_VERSION,
            "synthetic_data_excluded": True,
            "statistical_sample_size_justification_ref": planning.get(
                "statistical_sample_size_justification_ref",
                "PENDING_BIOSTATISTICAL_AND_IEC_REVIEW",
            ),
            "statistical_adequacy_status": planning.get(
                "statistical_adequacy_status",
                "NOT_ESTABLISHED_IN_SOFTWARE",
            ),
            "statistical_disclaimer": planning.get("statistical_disclaimer"),
            "proposed_patient_level_split_allocation": planning.get(
                "proposed_patient_level_split_allocation"
            ),
            "totals": {
                "planned_patient_count": total_planned_patients,
                "planned_image_count": total_planned_images,
                "actual_received_patients": total_actual_patients,
                "actual_validated_patients": total_actual_patients,
                "actual_validated_images": total_actual_images,
            },
            "planned_cohorts": reconciled,
            "protected_holdouts": {
                "btxrd_center2_samples": 549,
                "btxrd_center3_samples": 259,
                "isolation_enforced": True,
            },
        }
