import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'path';
import {
  evaluateSourceGovernance,
  getCohortPlanningResponse,
  getDatasetAcquisitionStatusResponse,
  getDatasetGovernanceStatusResponse,
  getDatasetSourcesResponse,
  validateWorkflowTransitionRequest,
} from './dataset-governance';

const REPO_ROOT = path.resolve(process.cwd());

test('Stage 35 dataset source registry distinguishes proposed vs verified sources with 0 fabricated approvals', () => {
  const payload = getDatasetSourcesResponse(REPO_ROOT);
  assert.equal(payload.ontology_version, 'OSTEO_ONTOLOGY_V1.0_STAGE34');
  assert.ok(payload.total_sources >= 8);
  assert.equal(payload.approved_sources_count, 0);
  assert.ok(payload.rejected_sources_count >= 4);

  const byId = new Map(payload.sources.map((s) => [s.source_id, s]));
  assert.equal(byId.get('SRC_OI_FIGSHARE_MOUSE_REJECTED')?.acquisition_status, 'REJECTED');
  assert.equal(byId.get('SRC_PAGET_WEB_EDUCATIONAL_REJECTED')?.acquisition_status, 'REJECTED');
  assert.equal(byId.get('SRC_METS_BTXRD_PRIMARY_MALIGNANT_REJECTED')?.acquisition_status, 'REJECTED');
  assert.equal(byId.get('SRC_HEALTHY_FRACATLAS_TRAUMA_REJECTED')?.acquisition_status, 'REJECTED');
  assert.equal(byId.get('SRC_PAGET_HOSP_PACS_XRAY')?.url_verified, false);
  assert.deepEqual(byId.get('SRC_METS_TCIA_SPINE_CT')?.available_modalities, ['ct']);
  assert.deepEqual(byId.get('SRC_METS_IICS_UNA_BONESCAN')?.available_modalities, ['bone_scan']);
});

test('Stage 35 acquisition, governance, and cohort planning strictly report 0 validated/received data', () => {
  const acq = getDatasetAcquisitionStatusResponse(REPO_ROOT);
  assert.ok(acq.overall_acquisition_status.startsWith('BLOCKED'));
  assert.equal(acq.total_validated_patients, 0);
  assert.equal(acq.total_received_valid_images, 0);
  assert.equal(acq.protected_holdouts.btxrd_center2_samples, 549);
  assert.equal(acq.protected_holdouts.btxrd_center3_samples, 259);

  const gov = getDatasetGovernanceStatusResponse(REPO_ROOT);
  assert.ok(gov.governance_gate_status.startsWith('BLOCKED'));
  assert.equal(gov.approved_sources_count, 0);
  assert.equal(gov.available_templates.length, 7);
  assert.equal(gov.templates_status, 'DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW');

  const plan = getCohortPlanningResponse(REPO_ROOT);
  assert.equal(plan.synthetic_data_excluded, true);
  assert.equal(plan.statistical_sample_size_justification_ref, 'PENDING_BIOSTATISTICAL_AND_IEC_REVIEW');
  assert.equal(plan.totals.planned_patient_count, 940);
  assert.equal(plan.totals.planned_image_count, 2270);
  assert.equal(plan.totals.actual_received_patients, 0);
  assert.equal(plan.totals.actual_validated_patients, 0);
  assert.equal(plan.totals.actual_validated_images, 0);
});

test('evaluateSourceGovernance blocks pending strings, missing DUA, and expired dates', () => {
  const audit = evaluateSourceGovernance(
    {
      source_id: 'SRC_TEST_EXPIRED',
      target_cohort: 'oi',
      governance_record: {
        irb_iec_approval: { status: 'APPROVED', protocol_ref: 'IEC-123' },
        research_protocol_id: 'OG-1',
        dua: { status: 'PENDING', agreement_ref: 'DRAFT_ONLY' },
        institutional_authorization: { status: 'APPROVED', signoff_ref: 'SIG-1' },
        consent_or_waiver_status: 'PEDIATRIC_CONSENT_PENDING',
        deidentification_approval: { status: 'APPROVED', sop_ref: 'SOP-1' },
        data_retention_requirements: '5 years',
        permitted_research_purposes: ['Research'],
        redistribution_restrictions: 'None',
        approval_date: '2025-01-01',
        expiry_date: '2025-12-31',
        evidence_reference: 'EVIDENCE_1',
        reviewer_pseudo_id: 'REV_IRB_01',
        reviewer_role: 'IRB_OFFICER',
      },
    },
    new Date('2026-11-22T00:00:00Z')
  );

  assert.equal(audit.irb_iec_approved, true);
  assert.equal(audit.dua_approved, false);
  assert.equal(audit.is_expired, true);
  assert.equal(audit.governance_verified, false);
  assert.ok(audit.missing_or_pending.includes('MISSING_OR_PENDING:dua'));
  assert.ok(audit.missing_or_pending.includes('PENDING:pediatric_consent_or_assent_clearance'));
  assert.ok(audit.missing_or_pending.includes('EXPIRED_GOVERNANCE_APPROVAL:2025-12-31'));
});

test('validateWorkflowTransitionRequest enforces sequential stage ordering, reviewer role, and evidence ref', () => {
  // Missing evidence -> 422
  const missingEv = validateWorkflowTransitionRequest(REPO_ROOT, {
    source_id: 'SRC_PAGET_HOSP_PACS_XRAY',
    target_state: 'SOURCE_VERIFIED',
    reviewer_pseudo_id: 'REV_DATA_01',
    reviewer_role: 'DATA_STEWARD',
    reason: 'Checking source',
    evidence_reference: 'PENDING',
  });
  assert.equal(missingEv.ok, false);
  assert.equal(missingEv.statusCode, 422);

  // Skipped stage -> 422
  const skipped = validateWorkflowTransitionRequest(REPO_ROOT, {
    source_id: 'SRC_PAGET_HOSP_PACS_XRAY',
    target_state: 'ACCESS_APPROVED',
    reviewer_pseudo_id: 'REV_IRB_01',
    reviewer_role: 'IRB_OFFICER',
    reason: 'Skipping ethics and DUA',
    evidence_reference: 'EVIDENCE_DOC_01',
  });
  assert.equal(skipped.ok, false);
  assert.equal(skipped.statusCode, 422);

  // Unauthorized role -> 403
  const unauthRole = validateWorkflowTransitionRequest(REPO_ROOT, {
    source_id: 'SRC_METS_TCIA_SPINE_CT',
    target_state: 'ETHICS_PENDING',
    reviewer_pseudo_id: 'REV_RAD_01',
    reviewer_role: 'RADIOLOGIST',
    reason: 'Advancing ethics',
    evidence_reference: 'IRB_FILING_01',
  });
  assert.equal(unauthRole.ok, false);
  assert.equal(unauthRole.statusCode, 403);

  // Valid authorized step passes validation
  const validStep = validateWorkflowTransitionRequest(REPO_ROOT, {
    source_id: 'SRC_METS_TCIA_SPINE_CT',
    target_state: 'ETHICS_PENDING',
    reviewer_pseudo_id: 'REV_IRB_01',
    reviewer_role: 'IRB_OFFICER',
    reason: 'TCIA local IEC submission filed',
    evidence_reference: 'IEC_SUBMISSION_2026_11',
  });
  assert.equal(validStep.ok, true);
  assert.equal(validStep.statusCode, 200);
});
