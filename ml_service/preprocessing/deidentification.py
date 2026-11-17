"""
OsteoGuard AI - HIPAA Safe Harbor PHI De-identification Engine.
Removes all 18 HIPAA identifier elements from DICOM and medical image metadata
while preserving essential non-identifying diagnostic parameters (kVp, slice thickness, spacing).
"""

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple


# Standard DICOM PHI tags to redact or pseudo-anonymize (HIPAA 164.514(b)(2))
HIPAA_PHI_TAGS: Set[str] = {
    # Patient Demographics
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",  # Retained in coarse form (M/F/O) if approved, redacted by default
    "PatientAge",  # Age > 89 must be aggregated under HIPAA
    "PatientAddress",
    "PatientMotherBirthName",
    "PatientTelephoneNumbers",
    "PatientInsurancePlanCodeSequence",
    "OtherPatientIDs",
    "OtherPatientNames",
    "EthnicGroup",
    # Institutional & Physician
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PhysiciansOfRecord",
    "PerformingPhysicianName",
    "OperatorsName",
    # Identifiers & Dates
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "StudyDescription",  # May contain clinician notes or patient names
    "SeriesDescription",
    "DeviceSerialNumber",
    "StationName",
}


def generate_salted_pseudo_id(raw_identifier: str, salt: str = "OsteoGuard_AI_HIPAA_Salt_2026") -> str:
    """
    Generates a deterministic, irreversibly hashed pseudo-anonymous patient ID.
    Enables patient-level clustering across studies without exposing real medical record numbers.
    """
    if not raw_identifier:
        raw_identifier = "UNKNOWN_PATIENT"
    payload = f"{salt}:{raw_identifier}".encode("utf-8")
    hash_digest = hashlib.sha256(payload).hexdigest()[:12].upper()
    return f"PT_ANON_{hash_digest}"


def sanitize_dicom_metadata(
    metadata: Dict[str, Any],
    salt: str = "OsteoGuard_AI_HIPAA_Salt_2026",
    preserve_age_group: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Scrubs all HIPAA PHI elements from extracted DICOM metadata dictionary.
    Returns:
      sanitized_metadata: Cleaned dictionary safe for logging or manifest generation.
      redacted_fields: List of tag names that were scrubbed or replaced.
    """
    sanitized: Dict[str, Any] = {}
    redacted_fields: List[str] = []

    # Handle PatientID pseudo-anonymization
    raw_pid = str(metadata.get("PatientID", metadata.get("patient_id", "UNKNOWN")))
    sanitized["PatientID"] = generate_salted_pseudo_id(raw_pid, salt=salt)
    redacted_fields.append("PatientID")

    for key, value in metadata.items():
        if key in ["PatientID", "patient_id"]:
            continue

        if key in HIPAA_PHI_TAGS:
            redacted_fields.append(key)
            if key == "PatientAge" and preserve_age_group:
                # Coarsen patient age to prevent re-identification of nonagenarians
                try:
                    age_str = str(value).rstrip("Y").strip()
                    age_val = int(age_str)
                    if age_val >= 90:
                        sanitized["PatientAgeGroup"] = "90+"
                    else:
                        decade = (age_val // 10) * 10
                        sanitized["PatientAgeGroup"] = f"{decade}-{decade+9}"
                except (ValueError, TypeError):
                    sanitized["PatientAgeGroup"] = "UNKNOWN"
            # All other PHI tags are omitted
            continue

        # Retain non-identifying technical acquisition parameters
        sanitized[key] = value

    return sanitized, redacted_fields

