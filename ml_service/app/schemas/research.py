"""
OsteoGuard AI - Research API Schemas (Stage 31).

Defines Pydantic V2 schemas for:
- Research Tracks (Tracks A, B, C, D)
- Model Evidence Registry
- Architecture Readiness & Component Status
- Dataset & Governance Status
- Radiomics, Biomechanics, Explainability, and Uncertainty Demonstrations
- End-to-End Synthetic Research Workbench
- Blocked Diagnostic Inference Handlers
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TrackID(str, Enum):
    TRACK_A = "track_a"
    TRACK_B = "track_b"
    TRACK_C = "track_c"
    TRACK_D = "track_d"


class ReadinessStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIALLY_IMPLEMENTED = "PARTIALLY_IMPLEMENTED"
    ARCHITECTURALLY_SPECIFIED = "ARCHITECTURALLY_SPECIFIED"
    BLOCKED_BY_DATA = "BLOCKED_BY_DATA"
    NOT_VERIFIED = "NOT_VERIFIED"


class ResearchTrackDetail(BaseModel):
    track_id: TrackID
    title: str
    modality: str
    task_type: str
    target_classes: List[str]
    status: str
    historical_metrics: Dict[str, Any]
    limitations: List[str]
    governance_clearance: str
    disclaimer: str


class ResearchTracksResponse(BaseModel):
    timestamp: str
    active_tracks_count: int
    tracks: List[ResearchTrackDetail]
    disclaimer: str


class ModelEvidenceItem(BaseModel):
    model_id: str
    track: str
    dataset_name: str
    modality: str
    architecture: str
    checkpoint_hash: str
    training_status: str
    validation_evidence: Dict[str, Any]
    external_holdout_status: str
    generalization_findings: str
    clinical_deployment_status: str
    limitations: List[str]


class EvidenceRegistryResponse(BaseModel):
    timestamp: str
    total_models: int
    models: List[ModelEvidenceItem]
    frozen_checkpoint_verified: bool
    disclaimer: str


class ArchitectureComponentStatus(BaseModel):
    component_name: str
    layer_group: str
    status: ReadinessStatus
    implemented_features: str
    unresolved_gaps: str
    governance_dependency: str


class ArchitectureReadinessResponse(BaseModel):
    timestamp: str
    system_readiness: str
    components: List[ArchitectureComponentStatus]
    fusion_contract_status: str
    inference_eligibility: str
    disclaimer: str


class DatasetGovernanceItem(BaseModel):
    cohort_name: str
    target_class: str
    intended_modality: str
    source_status: str
    patient_count_estimate: str
    acquisition_blockers: List[str]
    icmr_iec_required: bool
    dua_status: str


class DatasetGovernanceResponse(BaseModel):
    timestamp: str
    target_classes: List[str]
    cohorts: List[DatasetGovernanceItem]
    overall_acquisition_gate: str
    disclaimer: str


class RadiomicsDemoRequest(BaseModel):
    phantom_type: Literal["checkerboard", "linear_gradient", "annulus", "uniform"] = Field(default="checkerboard")
    grid_size: int = Field(default=32, ge=8, le=128)
    spacing_mm: float = Field(default=1.0, ge=0.1, le=5.0)


class RadiomicsDemoResponse(BaseModel):
    phantom_type: str
    grid_size: int
    spacing_mm: float
    verified_features_count: int
    unverified_features_count: int
    total_features_count: int
    specification_status: str
    verified_features_sample: Dict[str, float]
    unverified_features_sample: Dict[str, float]
    disclaimer: str


class BiomechanicsGeometryRequest(BaseModel):
    phantom_type: Literal[
        "intact_circular_ring",
        "uniform_cortical_thinning",
        "symmetric_cortical_defect",
        "eccentric_cortical_defect",
        "lytic_cavity",
        "multiple_small_defects",
        "non_circular_elliptical_annulus",
        "anisotropic_spacing_annulus",
    ] = Field(default="intact_circular_ring", description="One of the registered synthetic cross-section phantoms")
    voxel_size_mm: float = Field(default=1.0, ge=0.1, le=5.0)


class BiomechanicsGeometryResponse(BaseModel):
    phantom_type: str
    voxel_size_mm: float
    pixel_spacing_mm: List[float]
    area_mm2: float
    centroid_x_mm: float
    centroid_y_mm: float
    I_xx_mm4: float
    I_yy_mm4: float
    I_xy_mm4: float
    polar_J_mm4: float
    area_retention_ratio: float
    bending_retention_ratio_x: float
    bending_retention_ratio_y: float
    torsion_retention_ratio: float
    warning: str


class BiomechanicsMirelsRequest(BaseModel):
    site: int = Field(..., ge=1, le=3, description="1: Upper extremity, 2: Lower extremity, 3: Peritrochanteric")
    pain: int = Field(..., ge=1, le=3, description="1: Mild, 2: Moderate, 3: Functional")
    radiographic_nature: int = Field(..., ge=1, le=3, description="1: Blastic, 2: Mixed, 3: Lytic")
    lesion_size: int = Field(..., ge=1, le=3, description="1: <1/3 cortical, 2: 1/3-2/3, 3: >2/3 cortical")


class BiomechanicsMirelsResponse(BaseModel):
    total_score: int
    fracture_risk_percent: float
    recommendation_tier: str
    prophylactic_fixation_indicated: bool
    clinical_guidance: str
    disclaimer: str


class ExplainabilityDemoRequest(BaseModel):
    target_class: int = Field(default=1, ge=0, le=1)
    shift_horizontal: int = Field(default=0, ge=-8, le=8)


class ExplainabilityDemoResponse(BaseModel):
    synthetic_input_shape: List[int]
    target_class: int
    active_region_ratio: float
    peak_coordinates: List[int]
    heatmap_preview_base64: Optional[str] = None
    heatmap_grid_sample: List[List[float]]
    method: str
    shap_status: str
    warning: str


class UncertaintyDemoRequest(BaseModel):
    distribution_preset: Optional[str] = Field(default="highly_confident", description="Preset canonical distribution")
    custom_probabilities: Optional[List[float]] = Field(default=None, description="Optional custom 4-class vector")
    ood_flag: bool = Field(default=False, description="Simulate Out-Of-Distribution trigger")


class UncertaintyDemoResponse(BaseModel):
    probabilities: List[float]
    predicted_class_idx: int
    predicted_class_name: str
    max_probability: float
    top_two_margin: float
    shannon_entropy_nats: float
    normalized_entropy: float
    candidate_threshold: float
    decision: str
    human_review_required: bool
    abstention_reason: str
    reason_description: str
    ood_detected: bool
    clinical_guidance: str
    disclaimer: str


class SyntheticWorkbenchRunRequest(BaseModel):
    phantom_type: str = Field(default="checkerboard")
    bone_geometry_type: str = Field(default="intact_femur_diaphysis")
    mirels_site: int = Field(default=2)
    mirels_pain: int = Field(default=2)
    mirels_nature: int = Field(default=3)
    mirels_size: int = Field(default=2)
    uncertainty_preset: str = Field(default="moderately_confident")
    ood_simulation: bool = Field(default=False)


class SyntheticWorkbenchRunResponse(BaseModel):
    timestamp: str
    status: str
    radiomics_panel: RadiomicsDemoResponse
    biomechanics_geometry_panel: BiomechanicsGeometryResponse
    biomechanics_mirels_panel: BiomechanicsMirelsResponse
    explainability_panel: ExplainabilityDemoResponse
    uncertainty_panel: UncertaintyDemoResponse
    four_class_system_status: str
    outstanding_clinical_requirements: List[str]
    disclaimer: str


class BlockedInferenceResponse(BaseModel):
    status: str = "BLOCKED"
    error_code: str = "FOUR_CLASS_MODEL_NOT_READY"
    message: str = (
        "Clinical diagnostic inference is disabled. The OsteoGuard four-class diagnostic model "
        "cannot be deployed or executed because approved clinical datasets for Paget's disease, "
        "human osteogenesis imperfecta (OI), and matched healthy controls have not been acquired."
    )
    target_classes: List[str] = [
        "Secondary bone metastasis",
        "Paget's disease of bone",
        "Osteogenesis imperfecta (OI)",
        "Healthy skeletal control",
    ]
    blockers: List[str] = [
        "Unavailability of open-access human Paget's disease radiographs",
        "Human osteogenesis imperfecta datasets pending institutional ethics and data transfer agreements",
        "Matched healthy skeletal controls pending clinical site harmonization",
        "Unresolved 21 engineering radiomics features (only 81/102 IBSI features verified)",
        "Absence of institutional review board (IRB/IEC) approved clinical validation protocol",
    ]
    mandatory_safeguard: str = (
        "Under research governance rules, synthetic phantoms must not be fed into clinical models, "
        "and preliminary sub-track models (FracAtlas, Bone Scan, BTXRD) must never be remapped into four-class predictions."
    )


class TrackDDatasetReadinessResponse(BaseModel):
    timestamp: str
    track: str = "track_d"
    readiness_status: str
    ready_for_training: bool
    data_origin: str
    synthetic_data_excluded: bool = True
    total_inspected_samples: int
    valid_samples: int
    quarantined_samples: int
    unique_patients: int
    institution_count: int
    institutions: List[str]
    dataset_sources: List[str]
    permissions_verified: List[str]
    ethics_references: List[str]
    class_counts: Dict[str, int]
    modality_counts: Dict[str, int]
    class_modality_matrix: Dict[str, Dict[str, int]]
    missing_metadata_count: int
    duplicate_samples_count: int
    manifest_schema_validation: str = "NOT_TESTED"
    ontology_version: str = "OSTEO_ONTOLOGY_V1.0_STAGE34"
    accepted_by_class: Dict[str, int]
    adjudication_summary: Dict[str, int]
    healthy_control_eligibility_failures: int = 0
    conflicting_diagnoses_count: int = 0
    label_provenance_completeness_pct: float = 0.0
    split_distribution: Dict[str, int]
    gates: Dict[str, bool]
    protected_holdouts: Dict[str, Any]
    blockers: List[str]
    outstanding_requirements: List[str]
    clinical_sample_size_justification_note: str
    quarantine_reasons: Dict[str, int]


class DatasetSourcesResponse(BaseModel):
    timestamp: str
    ontology_version: str
    total_sources: int
    identified_sources_count: int
    verified_external_or_in_tree_candidates_count: int
    approved_sources_count: int
    rejected_sources_count: int
    sources: List[Dict[str, Any]]
    disclaimer: str


class DatasetAcquisitionStatusResponse(BaseModel):
    timestamp: str
    overall_acquisition_status: str
    workflow_stages: List[str]
    exception_states: List[str]
    cohorts: List[Dict[str, Any]]
    total_validated_patients: int
    total_received_valid_images: int
    protected_holdouts: Dict[str, Any]


class DatasetGovernanceStatusResponse(BaseModel):
    timestamp: str
    governance_gate_status: str
    active_sources_audited: int
    approved_sources_count: int
    pending_ethics_approvals_count: int
    pending_duas_count: int
    expired_approvals_count: int
    mean_evidence_completeness_pct: float
    templates_status: str
    available_templates: List[str]
    source_audits: List[Dict[str, Any]]
    protected_holdouts: Dict[str, Any]


class CohortPlanningResponse(BaseModel):
    timestamp: str
    ontology_version: str
    synthetic_data_excluded: bool = True
    statistical_sample_size_justification_ref: str
    statistical_adequacy_status: str
    statistical_disclaimer: Optional[str] = None
    proposed_patient_level_split_allocation: Optional[Dict[str, Any]] = None
    totals: Dict[str, int]
    planned_cohorts: List[Dict[str, Any]]
    protected_holdouts: Dict[str, Any]


class WorkflowTransitionRequest(BaseModel):
    source_id: str
    target_state: str
    reviewer_pseudo_id: str
    reviewer_role: str
    reason: str
    evidence_reference: str


class ReviewerAdjudicationDecisionRequest(BaseModel):
    sample_id: str = Field(min_length=1, max_length=160)
    decision: str = Field(pattern="^(ACCEPTED|REJECTED|OUT_OF_SCOPE|UNCERTAIN|REQUIRES_ADJUDICATION)$")
    reviewer_pseudo_id: str = Field(min_length=3, max_length=80)
    rationale: str = Field(min_length=1, max_length=2000)
    resolved_target_class: Optional[str] = None
    source_report_ref: Optional[str] = None
    diagnostic_evidence_category: Optional[str] = None
    diagnostic_evidence: Optional[str] = None
    radiology_report_confirmation_status: Optional[str] = None
    histopathology_confirmation_status: Optional[str] = None
    review_date: Optional[str] = None



