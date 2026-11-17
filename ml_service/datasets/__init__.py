"""
OsteoGuard AI Datasets & Ingestion Subsystem.
"""

from ml_service.datasets.auditor import (
    DatasetAuditor,
    compute_hamming_distance,
    compute_perceptual_hash,
)
from ml_service.datasets.partitioner import (
    PatientLeakageError,
    PatientLevelPartitioner,
    SPLIT_MANIFEST_COLUMNS,
    TrainingIsolationError,
    assert_training_only_fit,
    verify_patient_leakage,
)
from ml_service.datasets.registry_verifier import (
    DatasetRegistry,
    NegativeMappingViolation,
    compute_file_sha256,
    generate_dataset_manifest,
)

from ml_service.datasets.fracatlas_dataset import (
    FracAtlasClassificationDataset,
    create_fracatlas_dataloaders,
    get_fracatlas_transforms,
)

__all__ = [
    "DatasetAuditor",
    "compute_hamming_distance",
    "compute_perceptual_hash",
    "PatientLeakageError",
    "PatientLevelPartitioner",
    "SPLIT_MANIFEST_COLUMNS",
    "TrainingIsolationError",
    "assert_training_only_fit",
    "verify_patient_leakage",
    "DatasetRegistry",
    "NegativeMappingViolation",
    "compute_file_sha256",
    "generate_dataset_manifest",
    "FracAtlasClassificationDataset",
    "create_fracatlas_dataloaders",
    "get_fracatlas_transforms",
]
