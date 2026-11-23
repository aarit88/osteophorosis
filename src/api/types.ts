export type ModalityType = 'xray' | 'ct' | 'bone_scan';

export type BonePathologyClass = 'metastasis' | 'paget' | 'oi' | 'healthy';

export type UncertaintyLevel = 'low' | 'medium' | 'high';

export type FractureRiskLevel = 'low' | 'medium' | 'high';

export interface ClassProbabilities {
  metastasis: number;
  paget: number;
  oi: number;
  healthy: number;
}

export interface PredictionResult {
  topClass: BonePathologyClass;
  confidence: number; // 0.0 - 1.0
  probabilities: ClassProbabilities;
  uncertainty: UncertaintyLevel;
}

export interface UncertaintyDetail {
  entropy: number; // Normalized Shannon entropy 0.0 - 1.0
  confidenceInterval: [number, number]; // 95% Confidence Interval [lower, upper]
  oodFlag: boolean; // Out-of-Distribution detector heuristic trigger
  calibrationBin: {
    predicted: number;
    observed: number;
  };
}

export interface ShapFeature {
  name: string;
  value: number; // Signed float (-1.0 to +1.0)
}

export interface Counterfactual {
  feature: string;
  deltaDirection: 'increase' | 'decrease';
  percentChange: number;
  flipTo: BonePathologyClass;
  probabilityShift: number;
}

export interface FractureRiskResult {
  level: FractureRiskLevel;
  percent: number; // 0 - 100
}

export interface AnalyzeResponse {
  requestId: string;
  timestamp: string;
  modality: ModalityType;
  imageUrl: string;
  gradcamUrl: string;
  prediction: PredictionResult;
  uncertaintyDetail: UncertaintyDetail;
  shap: ShapFeature[];
  counterfactuals: Counterfactual[];
  fractureRisk: FractureRiskResult;
  disclaimer: string;
}

export interface HealthResponse {
  status: 'ok' | 'error';
  mode: 'simulation' | 'research' | 'production' | 'unavailable';
  clinical_readiness?: 'blocked' | 'ready';
  error_code?: string;
}

export interface AnalyzePayload {
  image: File | Blob;
  modality: ModalityType;
}

// ━━━ Clinician Feedback & Audit Types (EU AI Act Art. 14) ━━━
export type ClinicianDecision = 'agree' | 'disagree' | 'unsure';

export type FeedbackReasonCode =
  | 'artifact'
  | 'insufficient_context'
  | 'missed_lesion'
  | 'class_confusion'
  | 'other';

export interface FeedbackPayload {
  requestId: string;
  decision: ClinicianDecision;
  predictedClass: BonePathologyClass;
  reasonCode?: FeedbackReasonCode;
  reasonText?: string;
  timestamp: string;
}

export interface FeedbackRecord extends FeedbackPayload {
  feedbackId: string;
}

export interface FeedbackStats {
  total: number;
  agree: number;
  disagree: number;
  unsure: number;
  disagreementRate: number;
  byClass: Record<BonePathologyClass, { agree: number; disagree: number; unsure: number }>;
}

// ━━━ Telemetry & Event Logging Types ━━━
export interface EventPayload {
  type: string;
  requestId?: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

// ━━━ Asynchronous Inference Job Types ━━━
export interface JobStatusResponse {
  id: string;
  status: 'queued' | 'processing' | 'done' | 'failed';
  progress: number;
  resultUrl?: string;
}

// ━━━ Production Error Unions ━━━
export abstract class OsteoGuardError extends Error {
  abstract readonly code: string;
}

export class NetworkError extends OsteoGuardError {
  readonly code = 'NETWORK_ERROR';
  constructor(message: string) {
    super(message);
    this.name = 'NetworkError';
  }
}

export class TimeoutError extends OsteoGuardError {
  readonly code = 'TIMEOUT_ERROR';
  constructor(message = 'Request timed out after 10000ms') {
    super(message);
    this.name = 'TimeoutError';
  }
}

export class ValidationError extends OsteoGuardError {
  readonly code = 'VALIDATION_ERROR';
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class ServerError extends OsteoGuardError {
  readonly code = 'SERVER_ERROR';
  readonly status: number;
  constructor(message: string, status = 500) {
    super(message);
    this.name = 'ServerError';
    this.status = status;
  }
}

// ━━━ Research Prototype & Multi-Track Types (Stage 31) ━━━

export interface ResearchTrackDetail {
  track_id: 'track_a' | 'track_b' | 'track_c' | 'track_d';
  title: string;
  modality: string;
  task_type: string;
  target_classes: string[];
  status: string;
  historical_metrics: Record<string, any>;
  limitations: string[];
  governance_clearance: string;
  disclaimer: string;
}

export interface ResearchTracksResponse {
  timestamp: string;
  active_tracks_count: number;
  tracks: ResearchTrackDetail[];
  disclaimer: string;
}

export interface ModelEvidenceItem {
  model_id: string;
  track: string;
  dataset_name: string;
  modality: string;
  architecture: string;
  checkpoint_hash: string;
  training_status: string;
  validation_evidence: Record<string, any>;
  external_holdout_status: string;
  generalization_findings: string;
  clinical_deployment_status: string;
  limitations: string[];
}

export interface EvidenceRegistryResponse {
  timestamp: string;
  total_models: number;
  models: ModelEvidenceItem[];
  frozen_checkpoint_verified: boolean;
  disclaimer: string;
}

export interface ArchitectureComponentStatus {
  component_name: string;
  layer_group: string;
  status: 'IMPLEMENTED' | 'PARTIALLY_IMPLEMENTED' | 'ARCHITECTURALLY_SPECIFIED' | 'BLOCKED_BY_DATA' | 'NOT_VERIFIED';
  implemented_features: string;
  unresolved_gaps: string;
  governance_dependency: string;
}

export interface ArchitectureReadinessResponse {
  timestamp: string;
  system_readiness: string;
  components: ArchitectureComponentStatus[];
  fusion_contract_status: string;
  inference_eligibility: string;
  disclaimer: string;
}

export interface DatasetGovernanceItem {
  cohort_name: string;
  target_class: string;
  intended_modality: string;
  source_status: string;
  patient_count_estimate: string;
  acquisition_blockers: string[];
  icmr_iec_required: boolean;
  dua_status: string;
  disk_folder?: string;
  disk_sample_count?: number;
}

export interface DatasetGovernanceResponse {
  timestamp: string;
  target_classes: string[];
  cohorts: DatasetGovernanceItem[];
  overall_acquisition_gate: string;
  disclaimer: string;
}

export interface TrackDDatasetReadinessResponse {
  timestamp: string;
  track: 'track_d';
  readiness_status: 'NOT_AVAILABLE' | 'BLOCKED' | 'READY_FOR_TRAINING';
  ready_for_training: boolean;
  data_origin: 'REAL_CLINICAL_DATA_UNAVAILABLE' | 'REAL_DISK_DATA_PRESENT';
  synthetic_data_excluded: true;
  total_inspected_samples: number;
  valid_samples: number;
  quarantined_samples: number;
  unique_patients: number;
  institution_count: number;
  institutions: string[];
  dataset_sources: string[];
  permissions_verified: string[];
  ethics_references: string[];
  class_counts: Record<'metastasis' | 'paget' | 'oi' | 'healthy', number>;
  modality_counts: Record<'xray' | 'bone_scan' | 'ct', number>;
  class_modality_matrix: Record<
    'xray' | 'bone_scan' | 'ct',
    Record<'metastasis' | 'paget' | 'oi' | 'healthy', number>
  >;
  missing_metadata_count: number;
  duplicate_samples_count: number;
  manifest_schema_validation: string;
  ontology_version: string;
  accepted_by_class: Record<
    'SECONDARY_BONE_METASTASIS' | 'PAGETS_DISEASE' | 'OSTEOGENESIS_IMPERFECTA' | 'HEALTHY_SKELETAL_CONTROL',
    number
  >;
  adjudication_summary: Record<
    'ACCEPTED' | 'OUT_OF_SCOPE' | 'UNCERTAIN' | 'REQUIRES_ADJUDICATION' | 'REJECTED',
    number
  >;
  healthy_control_eligibility_failures: number;
  conflicting_diagnoses_count: number;
  label_provenance_completeness_pct: number;
  split_distribution: {
    train: number;
    val: number;
    test: number;
    external_test: number;
  };
  gates: {
    provenance_documented: boolean;
    permissions_verified: boolean;
    deidentification_satisfied: boolean;
    diagnostic_labels_documented: boolean;
    patient_level_split_integrity: boolean;
    image_quality_verified: boolean;
    class_and_modality_coverage: boolean;
  };
  protected_holdouts: {
    btxrd_center2_samples: number;
    btxrd_center3_samples: number;
    isolation_enforced: boolean;
  };
  blockers: string[];
  outstanding_requirements: string[];
  clinical_sample_size_justification_note: string;
  quarantine_reasons: Record<string, number>;
}

export interface ClinicalTrainingStatusResponse {
  disk: ClinicalDiskReadiness;
  minImagesPerClass: number;
  cohortsAcquired: number;
  totalRawSamples: number;
  trackDState: 'BLOCKED' | 'READY_FOR_TRAINING' | 'TRAINED_VERIFIED';
  activeCheckpoint: CheckpointArtifactInfo | null;
  trainingSummary: Record<string, any> | null;
  trainingCommand: string;
  serveCommand: string;
}

export interface RadiomicsDemoResponse {
  phantom_type: string;
  grid_size: number;
  spacing_mm: number;
  verified_features_count: number;
  unverified_features_count: number;
  total_features_count: number;
  specification_status: string;
  verified_features_sample: Record<string, number>;
  unverified_features_sample: Record<string, number>;
  disclaimer: string;
}

export interface BiomechanicsGeometryResponse {
  phantom_type: string;
  voxel_size_mm: number;
  pixel_spacing_mm: number[];
  area_mm2: number;
  centroid_x_mm: number;
  centroid_y_mm: number;
  I_xx_mm4: number;
  I_yy_mm4: number;
  I_xy_mm4: number;
  polar_J_mm4: number;
  area_retention_ratio: number;
  bending_retention_ratio_x: number;
  bending_retention_ratio_y: number;
  torsion_retention_ratio: number;
  warning: string;
}

export interface BiomechanicsMirelsResponse {
  total_score: number;
  fracture_risk_percent: number;
  recommendation_tier: string;
  prophylactic_fixation_indicated: boolean;
  clinical_guidance: string;
  disclaimer: string;
}

export interface ExplainabilityDemoResponse {
  synthetic_input_shape: number[];
  target_class: number;
  active_region_ratio: number;
  peak_coordinates: number[];
  heatmap_preview_base64?: string;
  heatmap_grid_sample: number[][];
  method: string;
  shap_status: string;
  warning: string;
}

export interface UncertaintyDemoResponse {
  probabilities: number[];
  predicted_class_idx: number;
  predicted_class_name: string;
  max_probability: number;
  top_two_margin: number;
  shannon_entropy_nats: number;
  normalized_entropy: number;
  candidate_threshold: number;
  decision: string;
  human_review_required: boolean;
  abstention_reason: string;
  reason_description: string;
  ood_detected: boolean;
  clinical_guidance: string;
  disclaimer: string;
}

export interface SyntheticWorkbenchRunResponse {
  timestamp: string;
  status: string;
  radiomics_panel: RadiomicsDemoResponse;
  biomechanics_geometry_panel: BiomechanicsGeometryResponse;
  biomechanics_mirels_panel: BiomechanicsMirelsResponse;
  explainability_panel: ExplainabilityDemoResponse;
  uncertainty_panel: UncertaintyDemoResponse;
  four_class_system_status: string;
  outstanding_clinical_requirements: string[];
  disclaimer: string;
}

export interface BlockedInferenceResponse {
  status: string;
  error_code: string;
  message: string;
  target_classes: string[];
  blockers: string[];
  mandatory_safeguard: string;
}

export interface TrainedModelPlot {
  title: string;
  url: string;
  caption: string;
}

export interface TrainedPredictionSample {
  sample_id: string;
  cohort: string;
  metadata: string;
  ground_truth: string;
  predicted_label: string;
  probability: number;
  is_correct: boolean;
}

export interface TrainedModelArtifactSummary {
  model_id: string;
  track: string;
  title: string;
  architecture: string;
  modality: string;
  total_holdout_evaluated: number;
  headline_metrics: Record<string, number>;
  plots: TrainedModelPlot[];
  sample_predictions: TrainedPredictionSample[];
}

export interface TrainedModelsOverviewResponse {
  timestamp: string;
  trained_models_count: number;
  total_holdout_samples_evaluated: number;
  models: TrainedModelArtifactSummary[];
  disclaimer: string;
}

export interface CheckpointArtifactInfo {
  filename: string;
  sizeBytes: number;
  sha256: string;
}

export interface ClinicalDiskReadiness {
  checkpointsDir: string;
  checkpointsFound: CheckpointArtifactInfo[];
  trainingLogsCount: number;
  rawDataDir: string;
  rawClassCounts: Record<'metastasis' | 'paget' | 'oi' | 'healthy', number>;
  isFourClassReady: boolean;
}

export interface PublicClinicalReadinessSummary {
  checkpointCount: number;
  trainingLogsAvailable: boolean;
  clinicalDatasetAvailable: boolean;
  isFourClassReady: boolean;
}

export interface SecurityStatusResponse {
  timestamp: string;
  encryptionAtRest: {
    enabled: boolean;
    algorithm: string;
    keySource: 'environment' | 'local_0600_keyfile';
    feedbackEncryptedCount: number;
    feedbackPlaintextCount: number;
    eventsEncryptedCount: number;
    eventsPlaintextCount: number;
  };
  ephemeralMedia: {
    ramOnly: boolean;
    exifStrippingActive: boolean;
    diskUploadsRemaining: number;
    ttlSeconds: number;
    activeRamAssets: number;
  };
  gatewayTransport: {
    hstsHeaderReady: boolean;
    strictNoStoreApiCache: boolean;
    bearerAuthRequired: boolean;
    hmacSigningConfigured: boolean;
    mlBackendConfigured: boolean;
    mlBackendReachable: boolean;
    mlBackendCanInfer: boolean;
    strictClinicalMode: boolean;
  };
  clinicalReadiness: PublicClinicalReadinessSummary;
}

// ━━━ Stage 35 Institutional Acquisition, Governance & Cohort Planning Types ━━━

export interface DatasetRegistrySourceItem {
  source_id: string;
  target_cohort: 'metastasis' | 'paget' | 'oi' | 'healthy';
  ontology_target_class: string;
  source_classification: 'PROPOSED_INSTITUTIONAL' | 'VERIFIED_EXTERNAL_CANDIDATE' | 'AUDITED_IN_TREE_SUBTRACK' | 'DISQUALIFIED';
  source_name: string;
  institution: string;
  repository_url_or_contact_ref: string;
  url_verified: boolean;
  available_modalities: Array<'xray' | 'bone_scan' | 'ct'>;
  anatomical_coverage: string;
  diagnostic_confirmation_requirements: string;
  verified_patient_count: number | null;
  verified_image_count: number | null;
  license_and_usage_restrictions: string;
  access_requirements: string;
  ethics_approval_requirements: string;
  dua_status: string;
  deidentification_requirements: string;
  evidence_reference: string | null;
  verification_date: string | null;
  acquisition_status: string;
  workflow_state: string;
  governance_verified: boolean;
  evidence_completeness_pct: number;
  blockers: string[];
}

export interface DatasetSourcesResponse {
  timestamp: string;
  ontology_version: string;
  total_sources: number;
  identified_sources_count: number;
  verified_external_or_in_tree_candidates_count: number;
  approved_sources_count: number;
  rejected_sources_count: number;
  sources: DatasetRegistrySourceItem[];
  disclaimer: string;
}

export interface CohortAcquisitionItem {
  cohort_class: 'metastasis' | 'paget' | 'oi' | 'healthy';
  ontology_target_class: string;
  current_workflow_stage: string;
  identified_sources_count: number;
  verified_sources_count: number;
  approved_sources_count: number;
  disqualified_sources_count: number;
  received_images_count: number;
  validated_patients_count: number;
  modality_breakdown: Record<'xray' | 'bone_scan' | 'ct', { images: number; patients: number }>;
  outstanding_blockers: string[];
}

export interface DatasetAcquisitionStatusResponse {
  timestamp: string;
  overall_acquisition_status: string;
  workflow_stages: string[];
  exception_states: string[];
  cohorts: CohortAcquisitionItem[];
  total_validated_patients: number;
  total_received_valid_images: number;
  protected_holdouts: {
    btxrd_center2_samples: number;
    btxrd_center3_samples: number;
    isolation_enforced: boolean;
  };
}

export interface SourceGovernanceAuditItem {
  source_id: string;
  target_cohort: string;
  governance_verified: boolean;
  irb_iec_approved: boolean;
  dua_approved: boolean;
  is_expired: boolean;
  approval_date: string | null;
  expiry_date: string | null;
  evidence_reference: string | null;
  evidence_completeness_pct: number;
  checks: Record<string, boolean>;
  missing_or_pending: string[];
}

export interface DatasetGovernanceStatusResponse {
  timestamp: string;
  governance_gate_status: string;
  active_sources_audited: number;
  approved_sources_count: number;
  pending_ethics_approvals_count: number;
  pending_duas_count: number;
  expired_approvals_count: number;
  mean_evidence_completeness_pct: number;
  templates_status: string;
  available_templates: string[];
  source_audits: SourceGovernanceAuditItem[];
  protected_holdouts: {
    btxrd_center2_samples: number;
    btxrd_center3_samples: number;
    isolation_enforced: boolean;
  };
}

export interface CohortPlanRow {
  plan_id: string;
  diagnostic_class: 'metastasis' | 'paget' | 'oi' | 'healthy';
  ontology_target_class: string;
  imaging_modality: 'xray' | 'bone_scan' | 'ct';
  planned_patient_count: number;
  planned_image_count: number;
  actual_received_patients: number;
  actual_validated_patients: number;
  actual_validated_images: number;
  institution_distribution: string;
  inclusion_criteria: string;
  exclusion_criteria: string;
  primary_endpoint: string;
  statistical_sample_size_justification_ref: string;
  proposed_split_allocation: { train_pct: number; val_pct: number; test_pct: number };
  data_gap_patients: number;
}

export interface CohortPlanningResponse {
  timestamp: string;
  ontology_version: string;
  synthetic_data_excluded: boolean;
  statistical_sample_size_justification_ref: string;
  statistical_adequacy_status: string;
  statistical_disclaimer: string;
  proposed_patient_level_split_allocation: {
    partition_unit: string;
    train_pct: number;
    val_pct: number;
    internal_test_pct: number;
    external_site_holdout_note: string;
    strict_rules: string[];
  } | null;
  totals: {
    planned_patient_count: number;
    planned_image_count: number;
    actual_received_patients: number;
    actual_validated_patients: number;
    actual_validated_images: number;
  };
  planned_cohorts: CohortPlanRow[];
  protected_holdouts: {
    btxrd_center2_samples: number;
    btxrd_center3_samples: number;
    isolation_enforced: boolean;
  };
}


