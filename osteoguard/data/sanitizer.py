"""
OsteoGuard AI - Patient Privacy & Metadata Sanitization Engine.
Enforces Rule 10: Protect patient privacy and exclude identifying metadata
from training artifacts, complying with HIPAA Safe Harbor and GDPR standards.
"""

import hashlib
from typing import Any, Dict, List, Optional, Set
from PIL import Image
import numpy as np


# Standard HIPAA Safe Harbor Protected Health Information (PHI) identifiers
PHI_DICOM_TAGS: Set[str] = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
    "PatientAddress", "PatientWeight", "MedicalRecordLocator", "EthnicGroup",
    "Occupation", "AdditionalPatientHistory", "ReferringPhysicianName",
    "ConsultingPhysicianName", "PhysiciansOfRecord", "NameOfPhysiciansReadingStudy",
    "OperatorsName", "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "StationName", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime", "AccessionNumber",
    "StudyID", "DeviceSerialNumber", "ProtocolName",
}


class PatientPrivacySanitizer:
    """
    Sanitizes radiological files, metadata, and patient identifiers
    to guarantee zero leakage of Protected Health Information into artifacts.
    """

    def __init__(self, salt: str = "OsteoGuard_HIPAA_Salt_2026"):
        self.salt = salt

    def generate_pseudonym(self, raw_patient_id: str) -> str:
        """
        Generates a deterministic irreversible pseudonym for patient grouping.
        Allows multi-slice clustering without exposing actual hospital IDs.
        """
        combined = f"{self.salt}_{raw_patient_id.strip()}".encode("utf-8")
        digest = hashlib.sha256(combined).hexdigest()
        return f"PT_{digest[:12].upper()}"

    def sanitize_metadata_dict(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips all PHI tags from a metadata dictionary, retaining only
        clinically essential physical parameters (e.g. slice thickness, pixel spacing, modality).
        """
        sanitized = {}
        safe_keys = {
            "Modality", "SliceThickness", "PixelSpacing", "RescaleSlope",
            "RescaleIntercept", "WindowCenter", "WindowWidth", "Rows", "Columns",
            "BitsAllocated", "BitsStored", "HighBit", "PhotometricInterpretation",
        }

        for k, v in raw_metadata.items():
            if k in safe_keys:
                sanitized[k] = v
            elif k in PHI_DICOM_TAGS:
                continue  # Exclude identifying PHI
            elif any(term in k.lower() for term in ["name", "date", "time", "id", "hospital", "doctor", "address"]):
                continue  # Defensive exclusion

        # Attach pseudonym if PatientID was present
        if "PatientID" in raw_metadata:
            sanitized["PatientPseudonym"] = self.generate_pseudonym(str(raw_metadata["PatientID"]))

        return sanitized

    @staticmethod
    def strip_image_exif(pil_image: Image.Image) -> Image.Image:
        """Removes EXIF, GPS, and camera metadata from image files."""
        clean_img = Image.new(pil_image.mode, pil_image.size)
        clean_img.putdata(list(pil_image.getdata()))
        return clean_img

