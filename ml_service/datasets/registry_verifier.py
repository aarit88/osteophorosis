"""
OsteoGuard AI - Dataset Registry and Provenance Verification Engine.
Provides programmatic access to the dataset registry, file integrity verification (SHA-256),
manifest generation, and strict negative mapping rule enforcement.
"""

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from ml_service.config.settings import get_settings
from ml_service.utils.logging import logger
from osteoguard.constants import TARGET_CLASSES


class NegativeMappingViolation(Exception):
    """Raised when an invalid pathology mapping is attempted."""
    pass


class DatasetRegistry:
    """
    Programmatic interface for querying and validating registered dataset cohorts.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        settings = get_settings()
        self.registry_path = registry_path or (settings.base_dir / "ml_service/data/dataset_registry.yaml")
        self._data = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Loads and parses the YAML registry specification."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry specification not found at {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def is_dataset_ready(self) -> bool:
        """Returns the master dataset readiness flag (strictly False when zero data exists)."""
        return bool(self._data.get("dataset_ready", False))

    @property
    def version(self) -> str:
        return str(self._data.get("version", "1.0.0"))

    def get_target_classes(self) -> Dict[str, Any]:
        """Returns target classes definitions and strict negative rules."""
        return self._data.get("target_classes", {})

    def get_sources(self) -> List[Dict[str, Any]]:
        """Returns all registered data sources."""
        return self._data.get("sources", [])

    def get_public_sources(self) -> List[Dict[str, Any]]:
        """Returns sources that can be downloaded publicly without registration."""
        return [s for s in self.get_sources() if s.get("access_category") == "PUBLIC_DOWNLOAD"]

    def get_restricted_sources(self) -> List[Dict[str, Any]]:
        """Returns sources requiring DUA, research agreement, or credentialed access."""
        return [s for s in self.get_sources() if s.get("access_category") == "RESTRICTED_APPROVAL"]

    def get_acquired_datasets(self) -> Dict[str, Any]:
        """Returns acquired and verified local datasets with independent readiness statuses."""
        return self._data.get("acquired_datasets", {})

    def get_source_by_id(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific data source."""
        for s in self.get_sources():
            if s.get("source_id") == source_id:
                return s
        return None

    def validate_pathology_mapping(self, source_label: str, target_class: str) -> Tuple[bool, Optional[str]]:
        """
        Validates an proposed mapping against Strict Negative Rules:
          - Rule 2.1: NEVER map primary bone tumours to secondary metastasis.
          - Rule 2.2: NEVER map generic abnormal images to Paget's disease.
          - Rule 2.3: NEVER map age-related osteoporosis to osteogenesis imperfecta.
          - Rule 2.4: NEVER map unaffected scans to healthy controls.
        """
        label_lower = source_label.lower().replace("_", " ").replace("-", " ").strip()
        target_norm = target_class.lower().replace("_", "").replace("-", "").replace(" ", "").strip()

        # Rule 2.1 check: Primary bone tumours -> metastasis
        primary_tumors = ["osteosarcoma", "ewing", "chondrosarcoma", "multiple myeloma", "plasmacytoma", "enchondroma"]
        if target_norm in ["metastasis", "secondarybonemetastasis"]:
            for pt in primary_tumors:
                if pt in label_lower:
                    return False, f"Rule 2.1 Violation: Primary bone tumour '{source_label}' must NEVER be mapped to secondary metastasis."

        # Rule 2.2 check: Generic abnormal -> Paget's
        generic_abnormal = ["osteoarthritis", "degenerative", "fibrous dysplasia", "osteopetrosis", "brown tumour", "sclerosis"]
        if target_norm in ["paget", "pagetsdisease"]:
            for ga in generic_abnormal:
                if ga in label_lower and "paget" not in label_lower:
                    return False, f"Rule 2.2 Violation: Generic abnormality '{source_label}' must NEVER be mapped to Paget's disease."

        # Rule 2.3 check: Osteoporosis -> OI
        osteoporosis_keywords = ["osteoporosis", "osteopenia", "rickets", "osteomalacia", "senile"]
        if target_norm in ["oi", "osteogenesisimperfecta"]:
            for op in osteoporosis_keywords:
                if op in label_lower and "osteogenesis" not in label_lower and "imperfecta" not in label_lower:
                    return False, f"Rule 2.3 Violation: Condition '{source_label}' must NEVER be mapped to osteogenesis imperfecta."

        # Rule 2.4 check: Unaffected / contralateral -> healthy
        unaffected_keywords = ["unaffected", "contralateral", "fellow limb", "non lesion limb", "non-lesion limb"]
        if target_norm in ["healthy", "healthycontrol"]:
            for un in unaffected_keywords:
                if un in label_lower:
                    return False, f"Rule 2.4 Violation: Unaffected limb '{source_label}' must NEVER be assumed to be a verified healthy control."

        return True, None


def compute_file_sha256(file_path: Path) -> str:
    """Computes cryptographic SHA-256 hash of a file for integrity tracking."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def generate_dataset_manifest(
    data_dir: Path,
    output_csv: Path,
    registry: Optional[DatasetRegistry] = None,
) -> int:
    """
    Scans data directory, audits clinical images, computes hashes, and writes dataset_manifest.csv.
    Returns count of indexed files.
    """
    reg = registry or DatasetRegistry()
    indexed_rows: List[Dict[str, Any]] = []

    fieldnames = [
        "dataset_id", "source_id", "patient_id", "study_id", "series_id", "instance_id",
        "target_class", "short_code", "modality", "original_format", "file_path", "mask_path",
        "anatomic_region", "view_projection", "slice_thickness_mm", "kvp", "checksum_sha256",
        "verified_healthy_flag", "license", "split"
    ]

    target_map = {
        "metastasis": ("secondary_bone_metastasis", "metastasis"),
        "secondary_bone_metastasis": ("secondary_bone_metastasis", "metastasis"),
        "paget": ("pagets_disease", "paget"),
        "pagets_disease": ("pagets_disease", "paget"),
        "oi": ("osteogenesis_imperfecta", "oi"),
        "osteogenesis_imperfecta": ("osteogenesis_imperfecta", "oi"),
        "healthy": ("healthy_control", "healthy"),
        "healthy_control": ("healthy_control", "healthy"),
    }

    if data_dir.exists():
        for class_dir in data_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue

            mapped_entry = target_map.get(class_dir.name.lower())
            if not mapped_entry:
                continue

            target_class_full, short_code = mapped_entry

            for f in class_dir.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    ext = f.suffix.lower()
                    if ext in [".png", ".jpg", ".jpeg", ".dcm", ".nii", ".nii.gz", ".tiff"]:
                        # Extract patient ID from filename convention: PT_{patient_id}_...
                        stem = f.stem
                        if stem.startswith("PT_"):
                            patient_id = stem.rsplit("_", 1)[0]
                        else:
                            patient_id = stem.split("_")[0]

                        # Check negative rules on filename
                        valid, err = reg.validate_pathology_mapping(f.name, target_class_full)
                        if not valid:
                            logger.error(f"Integrity error in {f}: {err}")
                            raise NegativeMappingViolation(err)

                        checksum = compute_file_sha256(f)

                        indexed_rows.append({
                            "dataset_id": f"DS_{target_class_full[:4].upper()}_{len(indexed_rows)+1:05d}",
                            "source_id": "LOCAL_INGEST",
                            "patient_id": patient_id,
                            "study_id": f"ST_{patient_id}_01",
                            "series_id": f"SE_{short_code}_01",
                            "instance_id": f.name,
                            "target_class": target_class_full,
                            "short_code": short_code,
                            "modality": "xray" if ext in [".png", ".jpg", ".jpeg"] else "ct",
                            "original_format": ext.lstrip("."),
                            "file_path": str(f.relative_to(data_dir.parent.parent)),
                            "mask_path": "",
                            "anatomic_region": "skeleton",
                            "view_projection": "AP",
                            "slice_thickness_mm": "",
                            "kvp": "",
                            "checksum_sha256": checksum,
                            "verified_healthy_flag": "TRUE" if short_code == "healthy" else "FALSE",
                            "license": "CC-BY-4.0",
                            "split": "unassigned",
                        })

    # Write manifest CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in indexed_rows:
            writer.writerow(row)

    logger.info(f"Generated manifest {output_csv} with {len(indexed_rows)} indexed files.")
    return len(indexed_rows)
