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

