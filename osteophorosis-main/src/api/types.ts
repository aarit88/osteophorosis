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
  mode: 'simulation' | 'production';
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
}

export interface DatasetGovernanceResponse {
  timestamp: string;
  target_classes: string[];
  cohorts: DatasetGovernanceItem[];
  overall_acquisition_gate: string;
  disclaimer: string;
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

