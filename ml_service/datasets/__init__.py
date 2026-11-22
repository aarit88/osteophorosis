"""
OsteoGuard AI Datasets & Ingestion Subsystem.
"""

from ml_service.datasets.clinical_ontology import (
    AdjudicationStatus,
    ONTOLOGY_VERSION,
    OntologyTargetClass,
    adjudicate_diagnostic_label,
    check_label_provenance_completeness,
    validate_healthy_control_eligibility,
)
from ml_service.datasets.track_d_intake import (
    evaluate_track_d_readiness,
    generate_leakage_safe_splits,
    inspect_image_file,
    run_track_d_intake_pipeline,
    validate_track_d_dataset,
    verify_cross_split_leakage,
)

__all__ = [
    "AdjudicationStatus",
    "ONTOLOGY_VERSION",
    "OntologyTargetClass",
    "adjudicate_diagnostic_label",
    "check_label_provenance_completeness",
    "validate_healthy_control_eligibility",
    "evaluate_track_d_readiness",
    "generate_leakage_safe_splits",
    "inspect_image_file",
    "run_track_d_intake_pipeline",
    "validate_track_d_dataset",
    "verify_cross_split_leakage",
]

try:  # Optional ML-stack dependencies (imagehash, PIL, numpy, torch, pyyaml)
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

    __all__.extend([
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
    ])
except ImportError:  # pragma: no cover - stdlib-only AI Studio container
    pass

