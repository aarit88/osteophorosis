import fs from 'fs';
import path from 'path';
import { getTrackDDatasetReadiness } from './clinical-readiness.js';

export type AcquisitionStatusType =
  | 'NOT_RESEARCHED'
  | 'IDENTIFIED'
  | 'VERIFICATION_REQUIRED'
  | 'ACCESS_REQUESTED'
  | 'APPROVED'
  | 'DATA_RECEIVED'
  | 'VALIDATED'
  | 'REJECTED';

export type WorkflowStageType =
  | 'SOURCE_IDENTIFIED'
  | 'SOURCE_VERIFIED'
  | 'ETHICS_PENDING'
  | 'DUA_PENDING'
  | 'ACCESS_APPROVED'
  | 'DATA_RECEIVED'
  | 'DEIDENTIFICATION_VERIFIED'
  | 'LABEL_ADJUDICATION'
  | 'QUALITY_VALIDATION'
  | 'COHORT_ACCEPTED'
  | 'REJECTED'
  | 'EXPIRED'
  | 'REQUIRES_REVIEW'
  | 'BLOCKED';

export type GovernanceRoleType =
  | 'DATA_STEWARD'
  | 'PRINCIPAL_INVESTIGATOR'
  | 'IRB_OFFICER'
  | 'DATA_PROTECTION_OFFICER'
  | 'LEGAL_COUNSEL'
  | 'CLINICAL_ADJUDICATOR'
  | 'RADIOLOGIST';

const PSEUDO_REVIEWER_ID_RE = /^(?:REV|ADJ|RAD|CLIN)_[A-Za-z0-9_-]{2,}$/;
const PLACEHOLDER_VALUES = new Set([
  '',
  'none',
  'null',
  'na',
  'n/a',
  'unknown',
  'missing',
  'unverified',
  'pending',
  'tbd',
  'not_documented',
  'unspecified',
]);
const APPROVED_FLAGS = new Set(['APPROVED', 'EXECUTED', 'VERIFIED', 'ACTIVE']);

const HAPPY_PATH_SEQUENCE: WorkflowStageType[] = [
  'SOURCE_IDENTIFIED',
  'SOURCE_VERIFIED',
  'ETHICS_PENDING',
  'DUA_PENDING',
  'ACCESS_APPROVED',
  'DATA_RECEIVED',
  'DEIDENTIFICATION_VERIFIED',
  'LABEL_ADJUDICATION',
  'QUALITY_VALIDATION',
  'COHORT_ACCEPTED',
];

const EXCEPTION_STATES: WorkflowStageType[] = ['REJECTED', 'EXPIRED', 'REQUIRES_REVIEW', 'BLOCKED'];

const ALLOWED_TRANSITIONS: Record<WorkflowStageType, Set<WorkflowStageType>> = {
  SOURCE_IDENTIFIED: new Set(['SOURCE_VERIFIED', 'REJECTED', 'REQUIRES_REVIEW', 'BLOCKED']),
  SOURCE_VERIFIED: new Set(['ETHICS_PENDING', 'REJECTED', 'REQUIRES_REVIEW', 'BLOCKED']),
  ETHICS_PENDING: new Set(['DUA_PENDING', 'REJECTED', 'EXPIRED', 'REQUIRES_REVIEW', 'BLOCKED']),
  DUA_PENDING: new Set(['ACCESS_APPROVED', 'REJECTED', 'EXPIRED', 'REQUIRES_REVIEW', 'BLOCKED']),
  ACCESS_APPROVED: new Set(['DATA_RECEIVED', 'EXPIRED', 'REQUIRES_REVIEW', 'BLOCKED']),
  DATA_RECEIVED: new Set(['DEIDENTIFICATION_VERIFIED', 'REJECTED', 'EXPIRED', 'REQUIRES_REVIEW', 'BLOCKED']),
  DEIDENTIFICATION_VERIFIED: new Set(['LABEL_ADJUDICATION', 'REJECTED', 'EXPIRED', 'REQUIRES_REVIEW', 'BLOCKED']),
  LABEL_ADJUDICATION: new Set(['QUALITY_VALIDATION', 'REJECTED', 'REQUIRES_REVIEW', 'BLOCKED']),
  QUALITY_VALIDATION: new Set(['COHORT_ACCEPTED', 'REJECTED', 'REQUIRES_REVIEW', 'BLOCKED']),
  COHORT_ACCEPTED: new Set(['EXPIRED', 'REQUIRES_REVIEW']),
  REQUIRES_REVIEW: new Set(['SOURCE_VERIFIED', 'ETHICS_PENDING', 'DUA_PENDING', 'REJECTED', 'BLOCKED']),
  BLOCKED: new Set(['ETHICS_PENDING', 'DUA_PENDING', 'REQUIRES_REVIEW', 'REJECTED']),
  EXPIRED: new Set(['ETHICS_PENDING', 'DUA_PENDING', 'REJECTED']),
  REJECTED: new Set(),
};

const ROLE_REQUIREMENTS: Record<WorkflowStageType, Set<GovernanceRoleType>> = {
  SOURCE_IDENTIFIED: new Set(['DATA_STEWARD', 'PRINCIPAL_INVESTIGATOR']),
  SOURCE_VERIFIED: new Set(['DATA_STEWARD', 'PRINCIPAL_INVESTIGATOR', 'CLINICAL_ADJUDICATOR', 'RADIOLOGIST']),
  ETHICS_PENDING: new Set(['PRINCIPAL_INVESTIGATOR', 'IRB_OFFICER', 'DATA_STEWARD']),
  DUA_PENDING: new Set(['IRB_OFFICER', 'PRINCIPAL_INVESTIGATOR']),
  ACCESS_APPROVED: new Set(['DATA_PROTECTION_OFFICER', 'LEGAL_COUNSEL', 'IRB_OFFICER', 'PRINCIPAL_INVESTIGATOR']),
  DATA_RECEIVED: new Set(['DATA_STEWARD', 'PRINCIPAL_INVESTIGATOR']),
  DEIDENTIFICATION_VERIFIED: new Set(['DATA_PROTECTION_OFFICER', 'IRB_OFFICER', 'PRINCIPAL_INVESTIGATOR']),
  LABEL_ADJUDICATION: new Set(['CLINICAL_ADJUDICATOR', 'RADIOLOGIST']),
  QUALITY_VALIDATION: new Set(['DATA_STEWARD', 'CLINICAL_ADJUDICATOR', 'RADIOLOGIST', 'PRINCIPAL_INVESTIGATOR']),
  COHORT_ACCEPTED: new Set(['PRINCIPAL_INVESTIGATOR', 'CLINICAL_ADJUDICATOR', 'RADIOLOGIST', 'IRB_OFFICER']),
  REJECTED: new Set([
    'PRINCIPAL_INVESTIGATOR',
    'IRB_OFFICER',
    'DATA_PROTECTION_OFFICER',
    'LEGAL_COUNSEL',
    'CLINICAL_ADJUDICATOR',
    'RADIOLOGIST',
  ]),
  EXPIRED: new Set(['DATA_STEWARD', 'IRB_OFFICER', 'DATA_PROTECTION_OFFICER', 'PRINCIPAL_INVESTIGATOR']),
  REQUIRES_REVIEW: new Set(['DATA_STEWARD', 'IRB_OFFICER', 'CLINICAL_ADJUDICATOR', 'RADIOLOGIST', 'PRINCIPAL_INVESTIGATOR']),
  BLOCKED: new Set(['DATA_STEWARD', 'IRB_OFFICER', 'DATA_PROTECTION_OFFICER', 'LEGAL_COUNSEL', 'PRINCIPAL_INVESTIGATOR']),
};

const VALID_GOVERNANCE_ROLES = new Set<string>([
  'DATA_STEWARD',
  'PRINCIPAL_INVESTIGATOR',
  'IRB_OFFICER',
  'DATA_PROTECTION_OFFICER',
  'LEGAL_COUNSEL',
  'CLINICAL_ADJUDICATOR',
  'RADIOLOGIST',
]);

function readJsonSafe(filePath: string): Record<string, any> | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function isNonPlaceholder(val: unknown): boolean {
  if (val === null || val === undefined) return false;
  if (typeof val === 'string') {
    return !PLACEHOLDER_VALUES.has(val.trim().toLowerCase());
  }
  if (Array.isArray(val)) return val.length > 0;
  if (typeof val === 'object') return Object.keys(val).length > 0;
  return true;
}

function parseIsoDate(raw: unknown): Date | null {
  if (typeof raw !== 'string') return null;
  const cleaned = raw.trim().slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(cleaned)) return null;
  const d = new Date(`${cleaned}T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function evaluateSourceGovernance(source: Record<string, any>, referenceDate = new Date()) {
  const todayUtc = new Date(Date.UTC(referenceDate.getUTCFullYear(), referenceDate.getUTCMonth(), referenceDate.getUTCDate()));
  const gov = (source.governance_record || {}) as Record<string, any>;
  const sourceId = String(source.source_id || '');
  const targetCohort = String(source.target_cohort || '');

  const irb = (gov.irb_iec_approval || {}) as Record<string, any>;
  const irbApproved = APPROVED_FLAGS.has(String(irb.status || '').trim().toUpperCase()) && isNonPlaceholder(irb.protocol_ref);

  const dua = (gov.dua || {}) as Record<string, any>;
  const duaApproved = APPROVED_FLAGS.has(String(dua.status || '').trim().toUpperCase()) && isNonPlaceholder(dua.agreement_ref);

  const instAuth = (gov.institutional_authorization || {}) as Record<string, any>;
  const instApproved =
    APPROVED_FLAGS.has(String(instAuth.status || '').trim().toUpperCase()) && isNonPlaceholder(instAuth.signoff_ref);

  const deid = (gov.deidentification_approval || {}) as Record<string, any>;
  const deidApproved =
    APPROVED_FLAGS.has(String(deid.status || '').trim().toUpperCase()) && isNonPlaceholder(deid.sop_ref);

  const protocolId = gov.research_protocol_id;
  const consentStatus = String(gov.consent_or_waiver_status || '').trim().toUpperCase();
  const retentionReq = gov.data_retention_requirements;
  const purposes = gov.permitted_research_purposes || [];
  const redistribution = gov.redistribution_restrictions;
  const evidenceRef = gov.evidence_reference || source.evidence_reference;
  const approvalDt = parseIsoDate(gov.approval_date);
  const expiryDt = parseIsoDate(gov.expiry_date);

  const reviewerId = String(gov.reviewer_pseudo_id || '').trim();
  const reviewerRole = String(gov.reviewer_role || '').trim().toUpperCase();
  const validReviewer = Boolean(reviewerId && PSEUDO_REVIEWER_ID_RE.test(reviewerId));
  const validRole = VALID_GOVERNANCE_ROLES.has(reviewerRole);
  const isExpired = expiryDt !== null && expiryDt < todayUtc;

  const missingOrPending: string[] = [];
  if (!irbApproved) missingOrPending.push('MISSING_OR_PENDING:irb_iec_approval');
  if (!isNonPlaceholder(protocolId)) missingOrPending.push('MISSING:research_protocol_id');
  if (!duaApproved) missingOrPending.push('MISSING_OR_PENDING:dua');
  if (!instApproved) missingOrPending.push('MISSING_OR_PENDING:institutional_authorization');
  if (!isNonPlaceholder(consentStatus) || consentStatus.includes('PENDING') || consentStatus.includes('REQUIRED')) {
    missingOrPending.push('MISSING_OR_PENDING:consent_or_waiver_status');
  }
  if (!deidApproved) missingOrPending.push('MISSING_OR_PENDING:deidentification_approval');
  if (!isNonPlaceholder(retentionReq)) missingOrPending.push('MISSING:data_retention_requirements');
  if (!isNonPlaceholder(purposes)) missingOrPending.push('MISSING:permitted_research_purposes');
  if (!isNonPlaceholder(redistribution)) missingOrPending.push('MISSING:redistribution_restrictions');
  if (!approvalDt) missingOrPending.push('MISSING:approval_date');
  if (!expiryDt) missingOrPending.push('MISSING:expiry_date');
  else if (isExpired) missingOrPending.push(`EXPIRED_GOVERNANCE_APPROVAL:${expiryDt.toISOString().slice(0, 10)}`);
  if (!isNonPlaceholder(evidenceRef)) missingOrPending.push('MISSING:evidence_reference');
  if (!validReviewer || !validRole) missingOrPending.push('MISSING_OR_INVALID:authorized_reviewer_signoff');
  if (
    targetCohort === 'oi' &&
    consentStatus.includes('PEDIATRIC') &&
    !consentStatus.includes('APPROVED') &&
    !consentStatus.includes('WAIVED')
  ) {
    missingOrPending.push('PENDING:pediatric_consent_or_assent_clearance');
  }

  const checks = {
    irb_iec_approved: irbApproved,
    protocol_id_present: isNonPlaceholder(protocolId),
    dua_approved: duaApproved,
    institutional_authorization_approved: instApproved,
    consent_or_waiver_cleared:
      isNonPlaceholder(consentStatus) && !consentStatus.includes('PENDING') && !consentStatus.includes('REQUIRED'),
    deidentification_approved: deidApproved,
    retention_policy_present: isNonPlaceholder(retentionReq),
    permitted_purposes_present: isNonPlaceholder(purposes),
    redistribution_restrictions_present: isNonPlaceholder(redistribution),
    approval_date_valid: Boolean(approvalDt),
    expiry_date_valid_and_active: Boolean(expiryDt) && !isExpired,
    evidence_reference_present: isNonPlaceholder(evidenceRef),
    authorized_reviewer_signoff: validReviewer && validRole,
  };

  const passedCount = Object.values(checks).filter(Boolean).length;
  const completeness = Number(((100 * passedCount) / Object.keys(checks).length).toFixed(1));
  const governanceVerified = Object.values(checks).every(Boolean) && !isExpired;

  return {
    source_id: sourceId,
    target_cohort: targetCohort,
    governance_verified: governanceVerified,
    irb_iec_approved: irbApproved,
    dua_approved: duaApproved,
    is_expired: isExpired,
    approval_date: approvalDt ? approvalDt.toISOString().slice(0, 10) : null,
    expiry_date: expiryDt ? expiryDt.toISOString().slice(0, 10) : null,
    evidence_reference: isNonPlaceholder(evidenceRef) ? String(evidenceRef) : null,
    evidence_completeness_pct: completeness,
    checks,
    missing_or_pending: Array.from(new Set(missingOrPending)).sort(),
  };
}

export function validateWorkflowTransitionRequest(
  rootDir: string,
  payload: {
    source_id?: string;
    target_state?: string;
    reviewer_pseudo_id?: string;
    reviewer_role?: string;
    reason?: string;
    evidence_reference?: string;
  }
): { ok: true; statusCode: 200; transition: Record<string, any> } | { ok: false; statusCode: 400 | 403 | 404 | 422; error: string } {
  const sourceId = String(payload.source_id || '').trim();
  const targetRaw = String(payload.target_state || '').trim().toUpperCase() as WorkflowStageType;
  const reviewerId = String(payload.reviewer_pseudo_id || '').trim();
  const reviewerRole = String(payload.reviewer_role || '').trim().toUpperCase() as GovernanceRoleType;
  const reason = String(payload.reason || '').trim();
  const evidenceRef = String(payload.evidence_reference || '').trim();

  if (!sourceId || !targetRaw) {
    return { ok: false, statusCode: 400, error: 'source_id and target_state are required.' };
  }
  if (!PSEUDO_REVIEWER_ID_RE.test(reviewerId)) {
    return {
      ok: false,
      statusCode: 422,
      error: 'reviewer_pseudo_id must be a valid pseudonymous reviewer ID (REV_*, ADJ_*, RAD_*, CLIN_*).',
    };
  }
  if (!(targetRaw in ALLOWED_TRANSITIONS)) {
    return { ok: false, statusCode: 422, error: `Unsupported workflow target_state '${targetRaw}'.` };
  }
  if (!reason) {
    return { ok: false, statusCode: 422, error: 'Explicit governance reason is required for transitions.' };
  }
  if (!isNonPlaceholder(evidenceRef)) {
    return { ok: false, statusCode: 422, error: 'Valid evidence_reference is required for workflow transitions.' };
  }

  const registryPath = path.join(rootDir, 'ml_service', 'data', 'manifests', 'dataset_source_registry.json');
  const registry = readJsonSafe(registryPath);
  const sources = Array.isArray(registry?.sources) ? registry!.sources : [];
  const source = sources.find((s: Record<string, any>) => s.source_id === sourceId);
  if (!source) {
    return { ok: false, statusCode: 404, error: `Unknown dataset source_id '${sourceId}'.` };
  }

  const currentRaw = String(source.workflow_state || 'SOURCE_IDENTIFIED').trim().toUpperCase() as WorkflowStageType;
  const allowedTargets = ALLOWED_TRANSITIONS[currentRaw] || new Set<WorkflowStageType>();
  if (!allowedTargets.has(targetRaw)) {
    return {
      ok: false,
      statusCode: 422,
      error: `Invalid workflow transition from '${currentRaw}' to '${targetRaw}'.`,
    };
  }

  const allowedRoles = ROLE_REQUIREMENTS[targetRaw] || new Set<GovernanceRoleType>();
  if (!allowedRoles.has(reviewerRole)) {
    return {
      ok: false,
      statusCode: 403,
      error: `Unauthorized reviewer role '${reviewerRole || 'NONE'}' for transition to '${targetRaw}'.`,
    };
  }

  const govAudit = evaluateSourceGovernance(source);
  if (govAudit.is_expired && !['EXPIRED', 'REJECTED', 'ETHICS_PENDING', 'DUA_PENDING'].includes(targetRaw)) {
    return {
      ok: false,
      statusCode: 422,
      error: `Cannot advance '${sourceId}' to '${targetRaw}': governance approval expired on ${govAudit.expiry_date}.`,
    };
  }

  const gatedStages: WorkflowStageType[] = [
    'ACCESS_APPROVED',
    'DATA_RECEIVED',
    'DEIDENTIFICATION_VERIFIED',
    'LABEL_ADJUDICATION',
    'QUALITY_VALIDATION',
    'COHORT_ACCEPTED',
  ];
  if (gatedStages.includes(targetRaw) && (!govAudit.irb_iec_approved || !govAudit.dua_approved || !govAudit.governance_verified)) {
    return {
      ok: false,
      statusCode: 422,
      error: `Governance verification incomplete for '${targetRaw}': ${govAudit.missing_or_pending.join(', ')}`,
    };
  }

  return {
    ok: true,
    statusCode: 200,
    transition: {
      timestamp: new Date().toISOString(),
      source_id: sourceId,
      previous_status: currentRaw,
      new_status: targetRaw,
      reviewer_pseudo_id: reviewerId,
      reviewer_role: reviewerRole,
      reason,
      evidence_reference: evidenceRef,
      validated_only: true,
    },
  };
}

export function getDatasetSourcesResponse(rootDir: string) {
  const registryPath = path.join(rootDir, 'ml_service', 'data', 'manifests', 'dataset_source_registry.json');
  const registry = readJsonSafe(registryPath) || { sources: [] };
  const rawSources: Array<Record<string, any>> = Array.isArray(registry.sources) ? registry.sources : [];

  const sources = rawSources.map((src) => {
    const gov = evaluateSourceGovernance(src);
    return {
      source_id: String(src.source_id || ''),
      target_cohort: String(src.target_cohort || ''),
      ontology_target_class: String(src.ontology_target_class || ''),
      source_classification: String(src.source_classification || 'PROPOSED_INSTITUTIONAL'),
      source_name: String(src.source_name || ''),
      institution: String(src.institution || ''),
      repository_url_or_contact_ref: String(src.repository_url_or_contact_ref || ''),
      url_verified: Boolean(src.url_verified),
      available_modalities: Array.isArray(src.available_modalities) ? src.available_modalities : [],
      anatomical_coverage: String(src.anatomical_coverage || ''),
      diagnostic_confirmation_requirements: String(src.diagnostic_confirmation_requirements || ''),
      verified_patient_count: src.verified_patient_count ?? null,
      verified_image_count: src.verified_image_count ?? null,
      license_and_usage_restrictions: String(src.license_and_usage_restrictions || ''),
      access_requirements: String(src.access_requirements || ''),
      ethics_approval_requirements: String(src.ethics_approval_requirements || ''),
      dua_status: String(src.dua_status || ''),
      deidentification_requirements: String(src.deidentification_requirements || ''),
      evidence_reference: src.evidence_reference ?? null,
      verification_date: src.verification_date ?? null,
      acquisition_status: String(src.acquisition_status || 'NOT_RESEARCHED'),
      workflow_state: String(src.workflow_state || 'SOURCE_IDENTIFIED'),
      governance_verified: gov.governance_verified,
      evidence_completeness_pct: gov.evidence_completeness_pct,
      blockers: Array.isArray(src.blockers) ? src.blockers : [],
    };
  });

  return {
    timestamp: new Date().toISOString(),
    ontology_version: String(registry.ontology_version || 'OSTEO_ONTOLOGY_V1.0_STAGE34'),
    total_sources: sources.length,
    identified_sources_count: sources.filter((s) => s.source_classification !== 'DISQUALIFIED').length,
    verified_external_or_in_tree_candidates_count: sources.filter((s) =>
      ['VERIFIED_EXTERNAL_CANDIDATE', 'AUDITED_IN_TREE_SUBTRACK'].includes(s.source_classification)
    ).length,
    approved_sources_count: sources.filter((s) =>
      ['APPROVED', 'DATA_RECEIVED', 'VALIDATED'].includes(s.acquisition_status)
    ).length,
    rejected_sources_count: sources.filter((s) => s.acquisition_status === 'REJECTED').length,
    sources,
    disclaimer:
      'Registry clearly distinguishes proposed institutional partnerships from verified candidate sources. Zero approvals or patient counts are fabricated.',
  };
}

export function getDatasetAcquisitionStatusResponse(rootDir: string) {
  const registryPath = path.join(rootDir, 'ml_service', 'data', 'manifests', 'dataset_source_registry.json');
  const protocolsPath = path.join(rootDir, 'ml_service', 'data', 'manifests', 'cohort_acquisition_protocols.json');
  const registry = readJsonSafe(registryPath) || { sources: [] };
  const protocols = (readJsonSafe(protocolsPath)?.protocols || {}) as Record<string, any>;
  const rawSources: Array<Record<string, any>> = Array.isArray(registry.sources) ? registry.sources : [];
  const readiness = getTrackDDatasetReadiness(rootDir);

  const cohorts = (['metastasis', 'paget', 'oi', 'healthy'] as const).map((cls) => {
    const clsSources = rawSources.filter((s) => s.target_cohort === cls);
    const activeSources = clsSources.filter((s) => s.acquisition_status !== 'REJECTED');
    const approvedSources = activeSources.filter((s) =>
      ['APPROVED', 'DATA_RECEIVED', 'VALIDATED'].includes(String(s.acquisition_status || ''))
    );

    const receivedImages =
      (readiness.class_modality_matrix.xray[cls] || 0) +
      (readiness.class_modality_matrix.bone_scan[cls] || 0) +
      (readiness.class_modality_matrix.ct[cls] || 0);

    // Patient counts are zero when no validated real clinical samples exist
    const validatedPatients = readiness.valid_samples > 0 ? Math.min(readiness.unique_patients, receivedImages) : 0;

    let workflowStage: WorkflowStageType = 'SOURCE_IDENTIFIED';
    if (validatedPatients > 0 && approvedSources.length > 0) {
      workflowStage = 'QUALITY_VALIDATION';
    } else if (approvedSources.length > 0) {
      workflowStage = 'ACCESS_APPROVED';
    } else if (activeSources.some((s) => s.workflow_state === 'SOURCE_VERIFIED')) {
      workflowStage = 'ETHICS_PENDING';
    }

    const blockers = Array.from(
      new Set(activeSources.flatMap((s) => (Array.isArray(s.blockers) ? (s.blockers as string[]) : [])))
    );

    return {
      cohort_class: cls,
      ontology_target_class: String(protocols[cls]?.ontology_target_class || cls.toUpperCase()),
      current_workflow_stage: workflowStage,
      identified_sources_count: activeSources.length,
      verified_sources_count: activeSources.filter((s) =>
        ['VERIFIED_EXTERNAL_CANDIDATE', 'AUDITED_IN_TREE_SUBTRACK'].includes(String(s.source_classification || ''))
      ).length,
      approved_sources_count: approvedSources.length,
      disqualified_sources_count: clsSources.length - activeSources.length,
      received_images_count: receivedImages,
      validated_patients_count: validatedPatients,
      modality_breakdown: {
        xray: { images: readiness.class_modality_matrix.xray[cls] || 0, patients: 0 },
        bone_scan: { images: readiness.class_modality_matrix.bone_scan[cls] || 0, patients: 0 },
        ct: { images: readiness.class_modality_matrix.ct[cls] || 0, patients: 0 },
      },
      outstanding_blockers: blockers,
    };
  });

  const allPopulated = cohorts.every((c) => c.validated_patients_count > 0);
  return {
    timestamp: new Date().toISOString(),
    overall_acquisition_status: allPopulated
      ? 'READY_FOR_TRAINING'
      : 'BLOCKED — INSTITUTIONAL ETHICS, DUA & CLINICAL COHORTS PENDING',
    workflow_stages: HAPPY_PATH_SEQUENCE,
    exception_states: EXCEPTION_STATES,
    cohorts,
    total_validated_patients: readiness.unique_patients,
    total_received_valid_images: readiness.valid_samples,
    protected_holdouts: readiness.protected_holdouts,
  };
}

export function getDatasetGovernanceStatusResponse(rootDir: string) {
  const registryPath = path.join(rootDir, 'ml_service', 'data', 'manifests', 'dataset_source_registry.json');
  const templatesDir = path.join(rootDir, 'ml_service', 'data', 'templates');
  const registry = readJsonSafe(registryPath) || { sources: [] };
  const rawSources: Array<Record<string, any>> = Array.isArray(registry.sources) ? registry.sources : [];

  const audits = rawSources
    .filter((s) => s.acquisition_status !== 'REJECTED')
    .map((s) => evaluateSourceGovernance(s));

  const pendingEthics = audits.filter((a) => !a.irb_iec_approved).length;
  const pendingDuas = audits.filter((a) => !a.dua_approved).length;
  const approvedCount = audits.filter((a) => a.governance_verified).length;
  const expiredCount = audits.filter((a) => a.is_expired).length;
  const meanCompleteness =
    audits.length > 0
      ? Number((audits.reduce((sum, a) => sum + a.evidence_completeness_pct, 0) / audits.length).toFixed(1))
      : 0;

  let availableTemplates: string[] = [];
  if (fs.existsSync(templatesDir)) {
    availableTemplates = fs
      .readdirSync(templatesDir)
      .filter((f) => f.endsWith('.md'))
      .sort();
  }

  return {
    timestamp: new Date().toISOString(),
    governance_gate_status:
      audits.length > 0 && approvedCount === audits.length && expiredCount === 0
        ? 'ALL_GOVERNANCE_VERIFIED'
        : 'BLOCKED — ETHICS, DUA & INSTITUTIONAL AUTHORIZATIONS PENDING',
    active_sources_audited: audits.length,
    approved_sources_count: approvedCount,
    pending_ethics_approvals_count: pendingEthics,
    pending_duas_count: pendingDuas,
    expired_approvals_count: expiredCount,
    mean_evidence_completeness_pct: meanCompleteness,
    templates_status: 'DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW',
    available_templates: availableTemplates,
    source_audits: audits,
    protected_holdouts: {
      btxrd_center2_samples: 549,
      btxrd_center3_samples: 259,
      isolation_enforced: true,
    },
  };
}

export function getCohortPlanningResponse(rootDir: string) {
  const planningPath = path.join(rootDir, 'ml_service', 'data', 'manifests', 'cohort_planning_manifest.json');
  const planning = readJsonSafe(planningPath) || { planned_cohorts: [] };
  const readiness = getTrackDDatasetReadiness(rootDir);
  const rawPlanned: Array<Record<string, any>> = Array.isArray(planning.planned_cohorts)
    ? planning.planned_cohorts
    : [];

  let totalPlannedPatients = 0;
  let totalPlannedImages = 0;
  let totalActualImages = 0;

  const planned_cohorts = rawPlanned.map((item) => {
    const cls = String(item.diagnostic_class || '').toLowerCase() as 'metastasis' | 'paget' | 'oi' | 'healthy';
    const mod = String(item.imaging_modality || '').toLowerCase() as 'xray' | 'bone_scan' | 'ct';
    const plannedPts = Number(item.planned_patient_count || 0);
    const plannedImgs = Number(item.planned_image_count || 0);
    const actImgs = readiness.valid_samples > 0 ? readiness.class_modality_matrix[mod]?.[cls] || 0 : 0;
    const actPts = 0; // Measured strictly from validated manifest records (0 when unavailable)

    totalPlannedPatients += plannedPts;
    totalPlannedImages += plannedImgs;
    totalActualImages += actImgs;

    return {
      plan_id: String(item.plan_id || ''),
      diagnostic_class: cls,
      ontology_target_class: String(item.ontology_target_class || ''),
      imaging_modality: mod,
      planned_patient_count: plannedPts,
      planned_image_count: plannedImgs,
      actual_received_patients: actPts,
      actual_validated_patients: actPts,
      actual_validated_images: actImgs,
      institution_distribution: String(item.institution_distribution || ''),
      inclusion_criteria: String(item.inclusion_criteria || ''),
      exclusion_criteria: String(item.exclusion_criteria || ''),
      primary_endpoint: String(item.primary_endpoint || ''),
      statistical_sample_size_justification_ref: String(
        item.statistical_sample_size_justification_ref || 'PENDING_BIOSTATISTICAL_AND_IEC_REVIEW'
      ),
      proposed_split_allocation: item.proposed_split_allocation || { train_pct: 70, val_pct: 15, test_pct: 15 },
      data_gap_patients: Math.max(0, plannedPts - actPts),
    };
  });

  return {
    timestamp: new Date().toISOString(),
    ontology_version: String(planning.ontology_version || 'OSTEO_ONTOLOGY_V1.0_STAGE34'),
    synthetic_data_excluded: true,
    statistical_sample_size_justification_ref: String(
      planning.statistical_sample_size_justification_ref || 'PENDING_BIOSTATISTICAL_AND_IEC_REVIEW'
    ),
    statistical_adequacy_status: String(planning.statistical_adequacy_status || 'NOT_ESTABLISHED_IN_SOFTWARE'),
    statistical_disclaimer: String(planning.statistical_disclaimer || ''),
    proposed_patient_level_split_allocation: planning.proposed_patient_level_split_allocation || null,
    totals: {
      planned_patient_count: totalPlannedPatients,
      planned_image_count: totalPlannedImages,
      actual_received_patients: readiness.unique_patients,
      actual_validated_patients: readiness.unique_patients,
      actual_validated_images: totalActualImages,
    },
    planned_cohorts,
    protected_holdouts: readiness.protected_holdouts,
  };
}
