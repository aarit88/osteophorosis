"""
Unit tests for Patient Privacy Sanitizer and Patient-Level Partitioning.
Enforces Rule 10 (Protect patient privacy) and guarantees zero data leakage across splits.
"""

import pytest
from PIL import Image

from osteoguard.data.sanitizer import PatientPrivacySanitizer
from osteoguard.data.partitioning import (
    PatientLevelPartitioner,
    SampleRecord,
)


def test_patient_privacy_sanitizer():
    """Verify stripping of Protected Health Information and pseudonym generation."""
    sanitizer = PatientPrivacySanitizer(salt="TestSalt2026")

    raw_metadata = {
        "PatientName": "Doe^John",
        "PatientID": "HOSP_123456",
        "PatientBirthDate": "19600101",
        "ReferringPhysicianName": "Dr. Smith",
        "InstitutionName": "Central Memorial Hospital",
        "Modality": "CR",
        "SliceThickness": 2.5,
        "PixelSpacing": [0.143, 0.143],
    }

    clean_meta = sanitizer.sanitize_metadata_dict(raw_metadata)

    # Identifiers must be eliminated
    assert "PatientName" not in clean_meta
    assert "PatientBirthDate" not in clean_meta
    assert "ReferringPhysicianName" not in clean_meta
    assert "InstitutionName" not in clean_meta

    # Clinical physical parameters must be preserved
    assert clean_meta["Modality"] == "CR"
    assert clean_meta["SliceThickness"] == 2.5
    assert clean_meta["PixelSpacing"] == [0.143, 0.143]

    # Irreversible pseudonym must be present
    assert "PatientPseudonym" in clean_meta
    assert clean_meta["PatientPseudonym"].startswith("PT_")


def test_patient_level_partitioning_zero_leakage():
    """
    Verify that patient-level partitioning strictly prevents patient leakage
    across train, validation, and test subsets.
    """
    partitioner = PatientLevelPartitioner(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Simulate 40 patients (10 per class) with multiple scans per patient
    samples = []
    classes = ["metastasis", "paget", "oi", "healthy"]

    for c_idx, cls in enumerate(classes):
        for p_idx in range(10):
            p_id = f"PAT_{cls}_{p_idx:02d}"
            # 3 scans/slices per patient
            for s_idx in range(3):
                samples.append(
                    SampleRecord(
                        sample_id=f"{p_id}_slice_{s_idx}",
                        patient_id=p_id,
                        label=cls,
                        file_path=f"dummy/path/{p_id}_{s_idx}.png",
                    )
                )

    assert len(samples) == 120  # 40 patients * 3 scans

    split = partitioner.partition(samples)

    # 1. Total samples conserved
    total_split_samples = len(split.train_samples) + len(split.val_samples) + len(split.test_samples)
    assert total_split_samples == len(samples)

    # 2. Strict patient disjointness (ZERO DATA LEAKAGE)
    assert split.train_patients.isdisjoint(split.val_patients), "Patient leakage between train & val"
    assert split.train_patients.isdisjoint(split.test_patients), "Patient leakage between train & test"
    assert split.val_patients.isdisjoint(split.test_patients), "Patient leakage between val & test"

    # 3. Every sample in train must come from a train patient
    for s in split.train_samples:
        assert s.patient_id in split.train_patients
        assert s.patient_id not in split.val_patients
        assert s.patient_id not in split.test_patients

