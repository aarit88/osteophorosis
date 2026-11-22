"""
OsteoGuard AI — Stage 34 Centralized Four-Class Clinical Label Ontology,
Diagnostic Adjudication Engine, and Healthy-Control Eligibility Validator.

Strict Clinical & Scientific Rules:
1. Canonical Four-Class Ontology (`OSTEO_ONTOLOGY_V1.0_STAGE34`):
   - SECONDARY_BONE_METASTASIS (metastasis)
   - PAGETS_DISEASE (paget)
   - OSTEOGENESIS_IMPERFECTA (oi)
   - HEALTHY_SKELETAL_CONTROL (healthy)
2. Zero Unsupported Equivalences:
   - Osteoporosis / osteopenia / rickets / osteomalacia NEVER map to OI (OUT_OF_SCOPE).
   - Osteoarthritis / degenerative sclerosis / fibrous dysplasia NEVER map to Paget's (OUT_OF_SCOPE).
   - Contralateral / fellow / unaffected limbs or "absence of fracture" NEVER map to Healthy (REJECTED).
   - Primary bone malignancies (osteosarcoma, Ewing sarcoma, chondrosarcoma, myeloma) NEVER map to Secondary Metastasis (OUT_OF_SCOPE).
3. Uncertain / equivocal diagnoses are assigned UNCERTAIN and NEVER converted to a target class.
4. Conflicting or unreviewed ambiguous diagnoses are assigned REQUIRES_ADJUDICATION for human review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple


ONTOLOGY_VERSION = "OSTEO_ONTOLOGY_V1.0_STAGE34"


class OntologyTargetClass(str, Enum):
    SECONDARY_BONE_METASTASIS = "SECONDARY_BONE_METASTASIS"
    PAGETS_DISEASE = "PAGETS_DISEASE"
    OSTEOGENESIS_IMPERFECTA = "OSTEOGENESIS_IMPERFECTA"
    HEALTHY_SKELETAL_CONTROL = "HEALTHY_SKELETAL_CONTROL"


class AdjudicationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNCERTAIN = "UNCERTAIN"
    REQUIRES_ADJUDICATION = "REQUIRES_ADJUDICATION"
    REJECTED = "REJECTED"


TARGET_SHORT_TO_ONTOLOGY: Dict[str, OntologyTargetClass] = {
    "metastasis": OntologyTargetClass.SECONDARY_BONE_METASTASIS,
    "paget": OntologyTargetClass.PAGETS_DISEASE,
    "oi": OntologyTargetClass.OSTEOGENESIS_IMPERFECTA,
    "healthy": OntologyTargetClass.HEALTHY_SKELETAL_CONTROL,
}

ONTOLOGY_TO_SHORT_CODE: Dict[OntologyTargetClass, str] = {
    v: k for k, v in TARGET_SHORT_TO_ONTOLOGY.items()
}

# Only strict canonical synonyms for each exact disease entity — ZERO cross-pathology equivalences.
STRICT_CANONICAL_SYNONYMS: Dict[str, OntologyTargetClass] = {
    "secondary_bone_metastasis": OntologyTargetClass.SECONDARY_BONE_METASTASIS,
    "secondary bone metastasis": OntologyTargetClass.SECONDARY_BONE_METASTASIS,
    "secondary_skeletal_metastasis": OntologyTargetClass.SECONDARY_BONE_METASTASIS,
    "secondary skeletal metastasis": OntologyTargetClass.SECONDARY_BONE_METASTASIS,
    "metastasis": OntologyTargetClass.SECONDARY_BONE_METASTASIS,
    "pagets_disease": OntologyTargetClass.PAGETS_DISEASE,
    "paget's disease": OntologyTargetClass.PAGETS_DISEASE,
    "paget's disease of bone": OntologyTargetClass.PAGETS_DISEASE,
    "pagets disease of bone": OntologyTargetClass.PAGETS_DISEASE,
    "paget_disease": OntologyTargetClass.PAGETS_DISEASE,
    "paget": OntologyTargetClass.PAGETS_DISEASE,
    "osteogenesis_imperfecta": OntologyTargetClass.OSTEOGENESIS_IMPERFECTA,
    "osteogenesis imperfecta": OntologyTargetClass.OSTEOGENESIS_IMPERFECTA,
    "oi": OntologyTargetClass.OSTEOGENESIS_IMPERFECTA,
    "healthy_skeletal_control": OntologyTargetClass.HEALTHY_SKELETAL_CONTROL,
    "healthy skeletal control": OntologyTargetClass.HEALTHY_SKELETAL_CONTROL,
    "healthy_control": OntologyTargetClass.HEALTHY_SKELETAL_CONTROL,
    "healthy control": OntologyTargetClass.HEALTHY_SKELETAL_CONTROL,
    "healthy": OntologyTargetClass.HEALTHY_SKELETAL_CONTROL,
}

# Conditions explicitly OUTSIDE the four target classes — always remain OUT_OF_SCOPE.
OUT_OF_SCOPE_CONDITIONS: Tuple[Tuple[str, str], ...] = (
    ("osteoporosis", "OUT_OF_SCOPE_METABOLIC_BONE_LOSS_NOT_OI"),
    ("senile_osteoporosis", "OUT_OF_SCOPE_METABOLIC_BONE_LOSS_NOT_OI"),
    ("postmenopausal_osteoporosis", "OUT_OF_SCOPE_METABOLIC_BONE_LOSS_NOT_OI"),
    ("osteopenia", "OUT_OF_SCOPE_LOW_BMD_NOT_OI"),
    ("rickets", "OUT_OF_SCOPE_METABOLIC_RICKETS_NOT_OI"),
    ("osteomalacia", "OUT_OF_SCOPE_METABOLIC_OSTEOMALACIA_NOT_OI"),
    ("osteoarthritis", "OUT_OF_SCOPE_DEGENERATIVE_JOINT_NOT_PAGET"),
    ("degenerative_joint", "OUT_OF_SCOPE_DEGENERATIVE_JOINT_NOT_PAGET"),
    ("degenerative sclerosis", "OUT_OF_SCOPE_DEGENERATIVE_SCLEROSIS_NOT_PAGET"),
    ("subchondral sclerosis", "OUT_OF_SCOPE_DEGENERATIVE_SCLEROSIS_NOT_PAGET"),
    ("fibrous_dysplasia", "OUT_OF_SCOPE_FIBROUS_DYSPLASIA_NOT_PAGET"),
    ("fibrous dysplasia", "OUT_OF_SCOPE_FIBROUS_DYSPLASIA_NOT_PAGET"),
    ("osteopetrosis", "OUT_OF_SCOPE_OSTEOPETROSIS_NOT_PAGET"),
    ("brown_tumor", "OUT_OF_SCOPE_HYPERPARATHYROID_NOT_PAGET"),
    ("brown tumor", "OUT_OF_SCOPE_HYPERPARATHYROID_NOT_PAGET"),
    ("hyperparathyroidism", "OUT_OF_SCOPE_HYPERPARATHYROID_NOT_PAGET"),
    ("osteosarcoma", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("ewing", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("chondrosarcoma", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("primary_malignant", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("primary malignant", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("primary bone malignancy", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("primary bone tumor", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("primary bone tumour", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("primary sarcoma", "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS"),
    ("multiple_myeloma", "OUT_OF_SCOPE_HEMATOLOGIC_MALIGNANCY_NOT_METASTASIS"),
    ("multiple myeloma", "OUT_OF_SCOPE_HEMATOLOGIC_MALIGNANCY_NOT_METASTASIS"),
    ("plasmacytoma", "OUT_OF_SCOPE_HEMATOLOGIC_MALIGNANCY_NOT_METASTASIS"),
    ("enchondroma", "OUT_OF_SCOPE_BENIGN_TUMOR"),
    ("osteochondroma", "OUT_OF_SCOPE_BENIGN_TUMOR"),
    ("giant cell tumor", "OUT_OF_SCOPE_BENIGN_OR_LOCAL_TUMOR"),
    ("traumatic fracture", "OUT_OF_SCOPE_TRAUMA_FRACTURE"),
)

UNCERTAIN_DIAGNOSIS_KEYWORDS: Tuple[str, ...] = (
    "uncertain",
    "indeterminate",
    "borderline",
    "equivocal",
    "atypical",
    "suspicious",
    "unconfirmed",
    "rule out",
    "rule_out",
    "r/o",
    "pending",
    "inconclusive",
    "possible",
    "probable",
    "suspected",
    "differential only",
)

CONTRALATERAL_KEYWORDS: Tuple[str, ...] = (
    "contralateral",
    "opposite side",
    "opposite-side",
    "opposite_side",
    "fellow limb",
    "fellow_limb",
    "unaffected limb",
    "unaffected_limb",
    "non-lesion limb",
    "non lesion limb",
    "non_lesion_limb",
    "trauma_contralateral",
    "uninvolved limb",
)

INSUFFICIENT_HEALTHY_EVIDENCE_PATTERNS: Tuple[str, ...] = (
    "absence of finding",
    "absent finding",
    "no finding",
    "no fracture detected",
    "non-fractured",
    "non fractured",
    "unremarkable fracture screen",
    "missing medical history",
    "incomplete medical history",
    "no diagnosis recorded",
    "assumed healthy",
    "presumed normal",
)

PLACEHOLDER_VALUES: Set[str] = {
    "",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "missing",
    "unverified",
    "pending",
    "tbd",
    "not_documented",
    "unspecified",
}

GENERIC_ANATOMICAL_PLACEHOLDERS: Set[str] = {
    "",
    "skeleton",
    "bone",
    "unspecified",
    "unknown",
    "none",
    "general",
    "n/a",
}

VALID_EVIDENCE_CATEGORIES: Set[str] = {
    "BIOPSY_HISTOPATHOLOGY",
    "PET_CT_AND_RADIOLOGY",
    "RADIOLOGY_AND_CLINICAL_TRIAD",
    "SERUM_ALP_AND_RADIOGRAPH",
    "GENETIC_COL1A1_COL1A2",
    "SPECIALIST_CLINICAL_DIAGNOSIS",
    "RADIOLOGIST_NORMAL_CONFIRMATION",
    "MULTI_READER_CONSENSUS",
}

PROVENANCE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "source_report_ref",
    "diagnostic_evidence_category",
    "label_provenance",
    "label_ontology_version",
    "radiology_report_confirmation_status",
    "clinical_adjudication_status",
)

PSEUDO_REVIEWER_ID_RE = re.compile(r"^(?:REV|ADJ|RAD|CLIN)_[A-Za-z0-9_-]{2,}$")


def normalize_diagnosis_term(raw_label: str) -> str:
    cleaned = re.sub(r"[\s\-]+", "_", (raw_label or "").strip().lower())
    return cleaned.strip("_")


def _contains_token(haystack: str, token: str) -> bool:
    norm_h = haystack.lower().replace("-", " ").replace("_", " ")
    norm_t = token.lower().replace("-", " ").replace("_", " ")
    return norm_t in norm_h


def validate_healthy_control_eligibility(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Enforces strict eligibility criteria for HEALTHY_SKELETAL_CONTROL.
    Rejects automatic healthy classification on contralateral limbs, skeletal comorbidities,
    missing diagnostic report, missing anatomical region, missing cohort selection criteria,
    or negative-absence reasoning.
    """
    failures: List[str] = []

    src_diag = str(record.get("source_diagnosis") or record.get("diagnostic_class") or "")
    evidence = str(record.get("diagnostic_evidence") or "")
    path_str = str(record.get("image_path") or "")
    notes = str(record.get("clinical_notes_summary") or "")
    combined = f"{src_diag} {evidence} {path_str} {notes}".lower()

    # 1. Opposite-side / contralateral limb rejection
    if any(_contains_token(combined, tok) for tok in CONTRALATERAL_KEYWORDS):
        failures.append("HEALTHY_ELIGIBILITY_FAILED:CONTRALATERAL_OR_FELLOW_LIMB")

    # 2. Relevant skeletal diagnosis or missing/incomplete skeletal history
    skeletal_history = str(record.get("patient_skeletal_history") or "").strip().lower()
    if skeletal_history in PLACEHOLDER_VALUES or "incomplete" in skeletal_history or "missing" in skeletal_history:
        failures.append("HEALTHY_ELIGIBILITY_FAILED:INFERRED_FROM_ABSENCE_OR_INCOMPLETE_HISTORY")
    elif skeletal_history not in ("none_documented", "no_skeletal_pathology", "normal_skeletal_history"):
        if any(
            kw in skeletal_history
            for kw in (
                "metastasis", "paget", "oi", "osteogenesis", "osteoporosis", "osteopenia",
                "osteoarthritis", "sarcoma", "myeloma", "fracture", "dysplasia", "tumor", "tumour",
                "cancer", "carcinoma", "pathology",
            )
        ):
            failures.append("HEALTHY_ELIGIBILITY_FAILED:PATIENT_HAS_RELEVANT_SKELETAL_DIAGNOSIS")
    if any(
        _contains_token(combined, kw)
        for kw in ("cancer staging", "trauma patient", "known metastasis", "known osteoporosis", "osteopenia")
    ):
        failures.append("HEALTHY_ELIGIBILITY_FAILED:PATIENT_HAS_RELEVANT_SKELETAL_DIAGNOSIS")

    # 3. Diagnostic / radiology report must be explicitly referenced and confirmed
    report_ref = str(record.get("source_report_ref") or "").strip().lower()
    rad_status = str(record.get("radiology_report_confirmation_status") or "").strip().upper()
    if report_ref in PLACEHOLDER_VALUES or rad_status != "CONFIRMED":
        failures.append("HEALTHY_ELIGIBILITY_FAILED:MISSING_OR_UNCONFIRMED_DIAGNOSTIC_REPORT")

    # 4. Anatomical region must be specifically documented
    anat = str(record.get("anatomical_region") or "").strip().lower()
    if anat in GENERIC_ANATOMICAL_PLACEHOLDERS:
        failures.append("HEALTHY_ELIGIBILITY_FAILED:ANATOMICAL_REGION_NOT_DOCUMENTED")

    # 5. Source cohort control-selection criteria must be documented
    criteria = str(record.get("control_selection_criteria") or "").strip().lower()
    if criteria in PLACEHOLDER_VALUES:
        failures.append("HEALTHY_ELIGIBILITY_FAILED:MISSING_COHORT_CONTROL_SELECTION_CRITERIA")

    # 6. Positive evidence required — cannot infer health from absence of finding or incomplete history
    ev_lower = evidence.strip().lower()
    if ev_lower in PLACEHOLDER_VALUES:
        failures.append("HEALTHY_ELIGIBILITY_FAILED:INSUFFICIENT_POSITIVE_EVIDENCE")
    elif any(_contains_token(combined, pat) for pat in INSUFFICIENT_HEALTHY_EVIDENCE_PATTERNS):
        failures.append("HEALTHY_ELIGIBILITY_FAILED:INFERRED_FROM_ABSENCE_OR_INCOMPLETE_HISTORY")
    elif not any(
        kw in ev_lower
        for kw in ("normal", "intact", "asymptomatic", "physiological", "healthy")
    ):
        failures.append("HEALTHY_ELIGIBILITY_FAILED:NO_EXPLICIT_NORMAL_SKELETAL_FINDING")

    return len(failures) == 0, sorted(set(failures))


def check_label_provenance_completeness(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Checks Stage 34 diagnostic provenance & confirmation metadata completeness."""
    missing: List[str] = []
    for field in PROVENANCE_REQUIRED_FIELDS:
        val = str(record.get(field) or "").strip()
        if val.lower() in PLACEHOLDER_VALUES:
            missing.append(f"MISSING_PROVENANCE:{field}")

    ev_cat = str(record.get("diagnostic_evidence_category") or "").strip().upper()
    if ev_cat and ev_cat not in VALID_EVIDENCE_CATEGORIES:
        missing.append(f"INVALID_EVIDENCE_CATEGORY:{ev_cat}")

    ont_ver = str(record.get("label_ontology_version") or "").strip()
    if ont_ver and ont_ver != ONTOLOGY_VERSION:
        missing.append(f"ONTOLOGY_VERSION_MISMATCH:{ont_ver}")

    rev_id = str(record.get("reviewer_pseudo_id") or "").strip()
    if rev_id and not PSEUDO_REVIEWER_ID_RE.match(rev_id):
        missing.append("NON_PSEUDONYMIZED_REVIEWER_ID")

    return len(missing) == 0, missing


def adjudicate_diagnostic_label(
    record: Dict[str, Any],
    patient_conflicting_diagnoses: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Centralized Stage 34 Label Adjudication Engine.
    Evaluates source diagnosis, target class, uncertainty, out-of-scope conditions,
    conflicting diagnoses, diagnostic evidence, provenance, and healthy-control eligibility.
    Never converts uncertain or out-of-scope diagnoses into any of the four classes.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_source_diag = str(record.get("source_diagnosis") or record.get("diagnostic_class") or "").strip()
    raw_target_cls = str(record.get("diagnostic_class") or "").strip()
    evidence = str(record.get("diagnostic_evidence") or "").strip()
    path_str = str(record.get("image_path") or "").strip()

    norm_source = normalize_diagnosis_term(raw_source_diag)
    norm_target = normalize_diagnosis_term(raw_target_cls)

    evidence_source = str(
        record.get("source_report_ref")
        or record.get("label_provenance")
        or record.get("dataset_source")
        or ""
    ).strip()

    # Determine candidate target class strictly from canonical ontology only
    candidate_enum: Optional[OntologyTargetClass] = None
    if raw_target_cls.strip().upper() in {e.value for e in OntologyTargetClass}:
        candidate_enum = OntologyTargetClass(raw_target_cls.strip().upper())
    elif norm_target in STRICT_CANONICAL_SYNONYMS:
        candidate_enum = STRICT_CANONICAL_SYNONYMS[norm_target]
    elif norm_source in STRICT_CANONICAL_SYNONYMS:
        candidate_enum = STRICT_CANONICAL_SYNONYMS[norm_source]

    reviewer_id = str(record.get("reviewer_pseudo_id") or "").strip()
    review_date = str(record.get("review_date") or "").strip()
    clin_adj_status = str(record.get("clinical_adjudication_status") or "").strip().upper()
    conflict_flag = bool(record.get("conflicting_diagnosis_indicator")) or (
        patient_conflicting_diagnoses is not None and len(patient_conflicting_diagnoses) > 1
    )

    def _result(
        status: AdjudicationStatus,
        reason: str,
        reviewer_status: str,
        candidate: Optional[OntologyTargetClass] = candidate_enum,
        healthy_failures: Optional[List[str]] = None,
        provenance_missing: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "original_source_diagnosis": raw_source_diag or raw_target_cls or "MISSING",
            "normalized_diagnosis": norm_source or norm_target or "missing",
            "candidate_target_class": candidate.value if candidate else None,
            "short_class_code": ONTOLOGY_TO_SHORT_CODE[candidate] if candidate else None,
            "evidence_source": evidence_source or "MISSING",
            "adjudication_status": status.value,
            "reason": reason,
            "reviewer_status": reviewer_status,
            "reviewer_pseudo_id": reviewer_id or "UNASSIGNED",
            "review_date": review_date or "UNREVIEWED",
            "audit_timestamp": now_iso,
            "ontology_version": ONTOLOGY_VERSION,
            "healthy_eligibility_failures": healthy_failures or [],
            "provenance_missing_fields": provenance_missing or [],
        }

    combined_diag_text = f"{raw_source_diag} {raw_target_cls} {evidence} {path_str}".lower()

    # 1. Explicit Uncertain / Equivocal Diagnosis Guard (NEVER converted to a target class)
    if any(_contains_token(combined_diag_text, kw) for kw in UNCERTAIN_DIAGNOSIS_KEYWORDS):
        return _result(
            AdjudicationStatus.UNCERTAIN,
            "Diagnosis contains uncertainty/equivocal/rule-out terminology; human expert review required.",
            "HUMAN_REVIEW_REQUIRED",
            candidate=None,
        )

    # 2. Explicit Out-of-Scope Condition Check (Osteoporosis != OI, Osteoarthritis != Paget, Primary Malignancy != Metastasis)
    diag_evidence_text = f"{raw_source_diag} {raw_target_cls} {evidence}"
    for oos_tok, oos_code in OUT_OF_SCOPE_CONDITIONS:
        if _contains_token(diag_evidence_text, oos_tok):
            return _result(
                AdjudicationStatus.OUT_OF_SCOPE,
                f"{oos_code}: Source pathology '{oos_tok}' is outside the 4-class Track D ontology and must never be auto-mapped.",
                "AUTOMATED_EXCLUSION",
                candidate=None,
            )

    # 3. Healthy-Control strict contralateral / negative-absence check when targeting HEALTHY_SKELETAL_CONTROL
    if candidate_enum == OntologyTargetClass.HEALTHY_SKELETAL_CONTROL:
        h_ok, h_failures = validate_healthy_control_eligibility(record)
        if not h_ok and any(
            f in h_failures
            for f in (
                "HEALTHY_ELIGIBILITY_FAILED:CONTRALATERAL_OR_FELLOW_LIMB",
                "HEALTHY_ELIGIBILITY_FAILED:PATIENT_HAS_RELEVANT_SKELETAL_DIAGNOSIS",
                "HEALTHY_ELIGIBILITY_FAILED:INFERRED_FROM_ABSENCE_OR_INCOMPLETE_HISTORY",
            )
        ):
            return _result(
                AdjudicationStatus.REJECTED,
                "; ".join(h_failures),
                "AUTOMATED_REJECTION",
                healthy_failures=h_failures,
            )

    # 4. Any diagnosis outside the canonical 4 classes stays OUT_OF_SCOPE
    if candidate_enum is None:
        return _result(
            AdjudicationStatus.OUT_OF_SCOPE,
            f"OUT_OF_SCOPE_DIAGNOSIS: '{raw_source_diag or raw_target_cls or 'empty'}' is not a canonical Track D ontology class.",
            "AUTOMATED_EXCLUSION",
            candidate=None,
        )

    # If raw source_diagnosis was explicitly a different non-canonical disease term from target_class, keep OUT_OF_SCOPE
    if raw_source_diag and norm_source not in STRICT_CANONICAL_SYNONYMS and raw_source_diag.strip().upper() not in {
        e.value for e in OntologyTargetClass
    }:
        return _result(
            AdjudicationStatus.OUT_OF_SCOPE,
            f"OUT_OF_SCOPE_SOURCE_DIAGNOSIS: Source diagnosis '{raw_source_diag}' is not equivalent to {candidate_enum.value}.",
            "AUTOMATED_EXCLUSION",
            candidate=None,
        )

    # 5. Conflicting Diagnosis or Explicit Pending Human Adjudication
    if conflict_flag or clin_adj_status in ("PENDING_ADJUDICATION", "REQUIRES_ADJUDICATION", "CONFLICTING", "DISCREPANT"):
        return _result(
            AdjudicationStatus.REQUIRES_ADJUDICATION,
            "Conflicting patient diagnosis history or clinical adjudication pending authorized human reviewer sign-off.",
            "HUMAN_REVIEW_REQUIRED",
        )

    # 6. Remaining Healthy-Control Eligibility Validation (report ref, anatomical site, criteria, normal finding)
    if candidate_enum == OntologyTargetClass.HEALTHY_SKELETAL_CONTROL:
        h_ok, h_failures = validate_healthy_control_eligibility(record)
        if not h_ok:
            return _result(
                AdjudicationStatus.REJECTED,
                "; ".join(h_failures),
                "AUTOMATED_REJECTION",
                healthy_failures=h_failures,
            )

    # 6. Required Documented Diagnostic Evidence & Confirmation Checks for Pathology Classes
    if evidence.lower() in PLACEHOLDER_VALUES:
        return _result(
            AdjudicationStatus.REJECTED,
            f"MISSING_DIAGNOSTIC_EVIDENCE: {candidate_enum.value} requires explicit documented diagnostic evidence.",
            "AUTOMATED_REJECTION",
        )

    rad_conf = str(record.get("radiology_report_confirmation_status") or "").strip().upper()
    histo_conf = str(record.get("histopathology_confirmation_status") or "").strip().upper()

    if candidate_enum == OntologyTargetClass.SECONDARY_BONE_METASTASIS:
        if rad_conf != "CONFIRMED" and histo_conf != "CONFIRMED":
            return _result(
                AdjudicationStatus.REQUIRES_ADJUDICATION,
                "SECONDARY_BONE_METASTASIS requires confirmed histopathology or radiology/PET-CT report verification.",
                "HUMAN_REVIEW_REQUIRED",
            )
    elif candidate_enum in (OntologyTargetClass.PAGETS_DISEASE, OntologyTargetClass.OSTEOGENESIS_IMPERFECTA):
        if rad_conf != "CONFIRMED" and histo_conf != "CONFIRMED":
            return _result(
                AdjudicationStatus.REQUIRES_ADJUDICATION,
                f"{candidate_enum.value} requires confirmed radiology or specialist diagnostic report verification.",
                "HUMAN_REVIEW_REQUIRED",
            )

    # 7. Label Provenance & Authorized Reviewer Verification
    prov_ok, prov_missing = check_label_provenance_completeness(record)
    if not prov_ok:
        return _result(
            AdjudicationStatus.REJECTED,
            "; ".join(prov_missing),
            "UNREVIEWED",
            provenance_missing=prov_missing,
        )

    if clin_adj_status not in ("ADJUDICATED_CONFIRMED", "CONFIRMED", "VERIFIED", "ACCEPTED"):
        return _result(
            AdjudicationStatus.REQUIRES_ADJUDICATION,
            f"clinical_adjudication_status='{clin_adj_status or 'MISSING'}' requires authorized human adjudication confirmation.",
            "HUMAN_REVIEW_REQUIRED",
        )

    if not reviewer_id or not PSEUDO_REVIEWER_ID_RE.match(reviewer_id) or review_date.lower() in PLACEHOLDER_VALUES:
        return _result(
            AdjudicationStatus.REQUIRES_ADJUDICATION,
            "Authorized reviewer pseudonymous ID (REV_*/RAD_*/CLIN_*) and review_date required before acceptance.",
            "HUMAN_REVIEW_REQUIRED",
        )

    return _result(
        AdjudicationStatus.ACCEPTED,
        f"Verified {candidate_enum.value} with documented evidence, provenance, and reviewer sign-off.",
        "VERIFIED_REVIEWER",
    )
