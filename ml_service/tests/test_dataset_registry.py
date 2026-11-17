"""
Tests for OsteoGuard AI Dataset Registry, Provenance Verification, and Negative Mapping Rules.
"""

import csv
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from ml_service.datasets.registry_verifier import (
    DatasetRegistry,
    NegativeMappingViolation,
    compute_file_sha256,
    generate_dataset_manifest,
)


@pytest.fixture
def registry():
    """Provides DatasetRegistry instance loaded from ml_service/data/dataset_registry.yaml."""
    return DatasetRegistry()


def test_registry_yaml_structure(registry: DatasetRegistry):
    """Verifies that dataset_registry.yaml contains required root keys and metadata."""
    assert registry.version == "1.0.0"
    assert registry.is_dataset_ready is False  # Strict Rule: Must remain False

    target_classes = registry.get_target_classes()
    assert len(target_classes) == 4
    expected_classes = {
        "secondary_bone_metastasis",
        "pagets_disease",
        "osteogenesis_imperfecta",
        "healthy_control",
    }
    assert set(target_classes.keys()) == expected_classes

    for name, spec in target_classes.items():
        assert "canonical_name" in spec
        assert "short_code" in spec
        assert "icd10_code" in spec
        assert "clinical_definition" in spec
        assert "strict_exclusion_rules" in spec
        assert len(spec["strict_exclusion_rules"]) >= 3


def test_registry_sources_coverage(registry: DatasetRegistry):
    """Verifies all 9 registered sources with mandatory schema fields and access tiers."""
    sources = registry.get_sources()
    assert len(sources) == 9

    required_fields = [
        "source_id", "target_class", "short_code", "title", "url",
        "available_format", "available_modality", "available_labels",
        "patient_level_metadata", "segmentation_annotations",
        "download_method", "access_category", "license",
        "intended_ml_use", "known_limitations"
    ]

    for src in sources:
        for f in required_fields:
            assert f in src, f"Source {src.get('source_id')} missing field: {f}"

    public_sources = registry.get_public_sources()
    restricted_sources = registry.get_restricted_sources()

    assert len(public_sources) == 5
    assert len(restricted_sources) == 4

    public_ids = {s["source_id"] for s in public_sources}
    restricted_ids = {s["source_id"] for s in restricted_sources}

    expected_public = {
        "ZENODO_BONE_METS_13900966",
        "WIKIMEDIA_PAGET_COMMONS",
        "PMC_11324408_OI",
        "FIGSHARE_OI_COLLECTION_3795019",
        "FIGSHARE_NORMAL_SKELETAL_27865398",
    }
    expected_restricted = {
        "TCIA_SPINE_METS_CT_SEG",
        "NLM_MEDPIX_PAGET",
        "STANFORD_AIMI_MURA",
        "TCIA_HEALTHY_TOTAL_BODY_CTS",
    }

    assert public_ids == expected_public
    assert restricted_ids == expected_restricted


def test_strict_negative_mapping_rules(registry: DatasetRegistry):
    """
    Exhaustively verifies negative mapping rule validation:
      - Rule 2.1: Primary bone tumours -> secondary metastasis
      - Rule 2.2: Generic abnormal -> Paget's
      - Rule 2.3: Osteoporosis -> OI
      - Rule 2.4: Unaffected scans -> Healthy
    """
    # Rule 2.1: Primary bone tumors MUST NOT map to metastasis
    valid, err = registry.validate_pathology_mapping("osteosarcoma_tibia.png", "secondary_bone_metastasis")
    assert not valid
    assert "Rule 2.1 Violation" in err

    valid, err = registry.validate_pathology_mapping("ewings_sarcoma_femur.png", "metastasis")
    assert not valid
    assert "Rule 2.1 Violation" in err

    valid, err = registry.validate_pathology_mapping("multiple_myeloma_spine.png", "secondary_bone_metastasis")
    assert not valid
    assert "Rule 2.1 Violation" in err

    # Rule 2.2: Generic abnormal MUST NOT map to Paget's
    valid, err = registry.validate_pathology_mapping("severe_osteoarthritis_knee.png", "pagets_disease")
    assert not valid
    assert "Rule 2.2 Violation" in err

    valid, err = registry.validate_pathology_mapping("fibrous_dysplasia_cranium.png", "paget")
    assert not valid
    assert "Rule 2.2 Violation" in err

    # Rule 2.3: Osteoporosis MUST NOT map to Osteogenesis Imperfecta
    valid, err = registry.validate_pathology_mapping("senile_osteoporosis_spine.png", "osteogenesis_imperfecta")
    assert not valid
    assert "Rule 2.3 Violation" in err

    valid, err = registry.validate_pathology_mapping("rickets_wrist.png", "oi")
    assert not valid
    assert "Rule 2.3 Violation" in err

    # Rule 2.4: Unaffected limb MUST NOT be assumed healthy
    valid, err = registry.validate_pathology_mapping("contralateral_unaffected_femur.png", "healthy_control")
    assert not valid
    assert "Rule 2.4 Violation" in err

    # Legitimate labels MUST pass
    valid, err = registry.validate_pathology_mapping("secondary_breast_mets_l1.png", "secondary_bone_metastasis")
    assert valid and err is None

    valid, err = registry.validate_pathology_mapping("pagets_cotton_wool_skull.png", "pagets_disease")
    assert valid and err is None

    valid, err = registry.validate_pathology_mapping("osteogenesis_imperfecta_type4_femur.png", "osteogenesis_imperfecta")
    assert valid and err is None

    valid, err = registry.validate_pathology_mapping("asymptomatic_normal_spine.png", "healthy_control")
    assert valid and err is None


def test_sha256_computation(tmp_path: Path):
    """Verifies that compute_file_sha256 computes deterministic cryptographic hashes."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("OsteoGuard AI Medical Imaging Checksum", encoding="utf-8")

    digest1 = compute_file_sha256(sample_file)
    digest2 = compute_file_sha256(sample_file)

    assert digest1 == digest2
    assert len(digest1) == 64
    assert isinstance(digest1, str)


def test_manifest_generator_workflow(tmp_path: Path, registry: DatasetRegistry):
    """Verifies generate_dataset_manifest indexes directory and writes structured CSV."""
    raw_dir = tmp_path / "raw"
    csv_out = tmp_path / "manifest.csv"

    # Create one valid sample for each class
    classes = ["metastasis", "paget", "oi", "healthy"]
    for c in classes:
        c_dir = raw_dir / c
        c_dir.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((64, 64), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(c_dir / f"PT_{c}_01_image.png")

    count = generate_dataset_manifest(raw_dir, csv_out, registry)
    assert count == 4
    assert csv_out.exists()

    with open(csv_out, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 4
        for row in reader:
            assert row["patient_id"].startswith("PT_")
            assert len(row["checksum_sha256"]) == 64
            assert row["modality"] == "xray"
            assert row["license"] == "CC-BY-4.0"


def test_manifest_generator_rejects_negative_mapping_violation(tmp_path: Path, registry: DatasetRegistry):
    """Verifies that generate_dataset_manifest raises NegativeMappingViolation on rule violation."""
    raw_dir = tmp_path / "raw"
    csv_out = tmp_path / "manifest.csv"

    # Put osteosarcoma inside metastasis
    c_dir = raw_dir / "metastasis"
    c_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((64, 64), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(c_dir / "PT_osteosarcoma_primary_01.png")

    with pytest.raises(NegativeMappingViolation) as exc_info:
        generate_dataset_manifest(raw_dir, csv_out, registry)

    assert "Rule 2.1 Violation" in str(exc_info.value)

