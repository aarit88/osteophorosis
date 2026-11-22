import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { GovernancePage } from '../src/pages/GovernancePage';
import * as apiClient from '../src/api/client';

describe('GovernancePage Stage 33 Track D Dataset Readiness Dashboard', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders NOT_AVAILABLE status and zero counts cleanly when no real Track D data exists', async () => {
    vi.spyOn(apiClient, 'getDatasetGovernance').mockResolvedValue({
      timestamp: '2026-11-22T00:00:00Z',
      target_classes: ['metastasis', 'paget', 'oi', 'healthy'],
      cohorts: [],
      overall_acquisition_gate: 'BLOCKED (0 / 4 verified cohorts)',
      disclaimer: 'Research governance only.',
    });

    vi.spyOn(apiClient, 'getTrackDDatasetReadiness').mockResolvedValue({
      timestamp: '2026-11-22T00:00:00Z',
      track: 'track_d',
      readiness_status: 'NOT_AVAILABLE',
      ready_for_training: false,
      data_origin: 'REAL_CLINICAL_DATA_UNAVAILABLE',
      synthetic_data_excluded: true,
      total_inspected_samples: 0,
      valid_samples: 0,
      quarantined_samples: 0,
      unique_patients: 0,
      institution_count: 0,
      institutions: [],
      dataset_sources: [],
      permissions_verified: [],
      ethics_references: [],
      class_counts: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
      modality_counts: { xray: 0, bone_scan: 0, ct: 0 },
      class_modality_matrix: {
        xray: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
        bone_scan: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
        ct: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
      },
      missing_metadata_count: 0,
      duplicate_samples_count: 0,
      ontology_version: 'OSTEO_ONTOLOGY_V1.0_STAGE34',
      accepted_by_class: {
        SECONDARY_BONE_METASTASIS: 0,
        PAGETS_DISEASE: 0,
        OSTEOGENESIS_IMPERFECTA: 0,
        HEALTHY_SKELETAL_CONTROL: 0,
      },
      adjudication_summary: {
        ACCEPTED: 0,
        OUT_OF_SCOPE: 0,
        UNCERTAIN: 0,
        REQUIRES_ADJUDICATION: 0,
        REJECTED: 0,
      },
      healthy_control_eligibility_failures: 0,
      conflicting_diagnoses_count: 0,
      label_provenance_completeness_pct: 0,
      split_distribution: { train: 0, val: 0, test: 0, external_test: 0 },
      gates: {
        provenance_documented: false,
        permissions_verified: false,
        deidentification_satisfied: false,
        diagnostic_labels_documented: false,
        patient_level_split_integrity: false,
        image_quality_verified: false,
        class_and_modality_coverage: false,
      },
      protected_holdouts: {
        btxrd_center2_samples: 549,
        btxrd_center3_samples: 259,
        isolation_enforced: true,
      },
      blockers: ['Zero real Track D clinical dataset records found.'],
      outstanding_requirements: [
        "Paget's Disease of Bone: Retrospective Hospital Radiology PACS IRB/IEC approval + DUA.",
      ],
      clinical_sample_size_justification_note:
        'No arbitrary minimum clinical sample count is fabricated in software.',
      quarantine_reasons: {},
    });

    vi.spyOn(apiClient, 'getDatasetSources').mockResolvedValue({
      timestamp: '2026-11-22T00:00:00Z',
      ontology_version: 'OSTEO_ONTOLOGY_V1.0_STAGE34',
      total_sources: 2,
      identified_sources_count: 1,
      verified_external_or_in_tree_candidates_count: 1,
      approved_sources_count: 0,
      rejected_sources_count: 1,
      sources: [
        {
          source_id: 'SRC_PAGET_HOSP_PACS_XRAY',
          target_cohort: 'paget',
          ontology_target_class: 'PAGETS_DISEASE',
          source_classification: 'PROPOSED_INSTITUTIONAL',
          source_name: "Hospital MSK Radiology PACS Paget's Cohort (Proposed)",
          institution: 'Academic Medical Center (Proposed)',
          repository_url_or_contact_ref: 'INSTITUTIONAL_OUTREACH_REQUIRED',
          url_verified: false,
          available_modalities: ['xray'],
          anatomical_coverage: 'Pelvis, spine, femur',
          diagnostic_confirmation_requirements: 'Radiographic triad + serum ALP',
          verified_patient_count: null,
          verified_image_count: null,
          license_and_usage_restrictions: 'Bilateral DUA required',
          access_requirements: 'IRB + DUA',
          ethics_approval_requirements: 'Mandatory IEC/IRB',
          dua_status: 'NOT_SUBMITTED',
          deidentification_requirements: 'Safe Harbor',
          evidence_reference: 'ml_artifacts/clinical_collaboration_preparation/paget_cohort_specification.md',
          verification_date: null,
          acquisition_status: 'IDENTIFIED',
          workflow_state: 'SOURCE_IDENTIFIED',
          governance_verified: false,
          evidence_completeness_pct: 30.8,
          blockers: ['Zero open-access human Paget radiograph datasets exist.'],
        },
      ],
      disclaimer: 'Proposed vs verified sources strictly separated.',
    });

    vi.spyOn(apiClient, 'getDatasetAcquisitionStatus').mockResolvedValue({
      timestamp: '2026-11-22T00:00:00Z',
      overall_acquisition_status: 'BLOCKED — INSTITUTIONAL ETHICS, DUA & CLINICAL COHORTS PENDING',
      workflow_stages: ['SOURCE_IDENTIFIED', 'SOURCE_VERIFIED', 'ETHICS_PENDING', 'COHORT_ACCEPTED'],
      exception_states: ['REJECTED', 'EXPIRED', 'REQUIRES_REVIEW', 'BLOCKED'],
      cohorts: [
        {
          cohort_class: 'paget',
          ontology_target_class: 'PAGETS_DISEASE',
          current_workflow_stage: 'SOURCE_IDENTIFIED',
          identified_sources_count: 1,
          verified_sources_count: 0,
          approved_sources_count: 0,
          disqualified_sources_count: 1,
          received_images_count: 0,
          validated_patients_count: 0,
          modality_breakdown: {
            xray: { images: 0, patients: 0 },
            bone_scan: { images: 0, patients: 0 },
            ct: { images: 0, patients: 0 },
          },
          outstanding_blockers: ['Zero open-access human Paget radiograph datasets exist.'],
        },
      ],
      total_validated_patients: 0,
      total_received_valid_images: 0,
      protected_holdouts: {
        btxrd_center2_samples: 549,
        btxrd_center3_samples: 259,
        isolation_enforced: true,
      },
    });

    vi.spyOn(apiClient, 'getDatasetGovernanceStatus').mockResolvedValue({
      timestamp: '2026-11-22T00:00:00Z',
      governance_gate_status: 'BLOCKED — ETHICS, DUA & INSTITUTIONAL AUTHORIZATIONS PENDING',
      active_sources_audited: 8,
      approved_sources_count: 0,
      pending_ethics_approvals_count: 8,
      pending_duas_count: 8,
      expired_approvals_count: 0,
      mean_evidence_completeness_pct: 32.7,
      templates_status: 'DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW',
      available_templates: ['01_institutional_research_collaboration_request.md'],
      source_audits: [],
      protected_holdouts: {
        btxrd_center2_samples: 549,
        btxrd_center3_samples: 259,
        isolation_enforced: true,
      },
    });

    vi.spyOn(apiClient, 'getCohortPlanning').mockResolvedValue({
      timestamp: '2026-11-22T00:00:00Z',
      ontology_version: 'OSTEO_ONTOLOGY_V1.0_STAGE34',
      synthetic_data_excluded: true,
      statistical_sample_size_justification_ref: 'PENDING_BIOSTATISTICAL_AND_IEC_REVIEW',
      statistical_adequacy_status: 'NOT_ESTABLISHED_IN_SOFTWARE',
      statistical_disclaimer: 'Planned targets require formal biostatistical review.',
      proposed_patient_level_split_allocation: {
        partition_unit: 'patient_id',
        train_pct: 70,
        val_pct: 15,
        internal_test_pct: 15,
        external_site_holdout_note: 'External holdout isolated.',
        strict_rules: ['Zero patient overlap'],
      },
      totals: {
        planned_patient_count: 940,
        planned_image_count: 2270,
        actual_received_patients: 0,
        actual_validated_patients: 0,
        actual_validated_images: 0,
      },
      planned_cohorts: [
        {
          plan_id: 'PLAN_PAGET_XRAY',
          diagnostic_class: 'paget',
          ontology_target_class: 'PAGETS_DISEASE',
          imaging_modality: 'xray',
          planned_patient_count: 150,
          planned_image_count: 350,
          actual_received_patients: 0,
          actual_validated_patients: 0,
          actual_validated_images: 0,
          institution_distribution: 'Multi-center (planned)',
          inclusion_criteria: 'Radiographic triad + ALP',
          exclusion_criteria: 'Osteoarthritis',
          primary_endpoint: 'Macro-F1',
          statistical_sample_size_justification_ref: 'PENDING_BIOSTATISTICAL_AND_IEC_REVIEW',
          proposed_split_allocation: { train_pct: 70, val_pct: 15, test_pct: 15 },
          data_gap_patients: 150,
        },
      ],
      protected_holdouts: {
        btxrd_center2_samples: 549,
        btxrd_center3_samples: 259,
        isolation_enforced: true,
      },
    });

    render(<GovernancePage />);

    await waitFor(() => {
      expect(screen.getByText('NOT_AVAILABLE')).toBeInTheDocument();
    });

    expect(screen.getByText(/0 REAL SAMPLES — SYNTHETIC DEMO EXCLUDED/i)).toBeInTheDocument();
    expect(screen.getByText(/BTXRD Center 2 \(N=549\) & Center 3 \(N=259\) Protected Holdout/i)).toBeInTheDocument();
    expect(screen.getByText(/Sample Counts by Class/i)).toBeInTheDocument();
    expect(screen.getByText(/Sample Counts by Modality/i)).toBeInTheDocument();
    expect(screen.getByText(/Class × Modality Separation Matrix/i)).toBeInTheDocument();
    expect(screen.getByText(/Permission, Provenance & Readiness Gate Checks/i)).toBeInTheDocument();
    expect(screen.getByText(/Clinical Label Ontology & Diagnostic Adjudication Audit/i)).toBeInTheDocument();
    expect(screen.getByText(/OSTEO_ONTOLOGY_V1\.0_STAGE34/i)).toBeInTheDocument();

    // Stage 35 Dashboard Assertions
    expect(screen.getByText(/Institutional Dataset Acquisition & Governance Registry/i)).toBeInTheDocument();
    expect(screen.getByText(/BLOCKED — ETHICS, DUA & INSTITUTIONAL AUTHORIZATIONS PENDING/i)).toBeInTheDocument();
    expect(screen.getByText(/Planned vs Actual Cohort Planning Matrix/i)).toBeInTheDocument();
    expect(screen.getByText(/940 patients \/ 2270 scans/i)).toBeInTheDocument();
    expect(screen.getByText(/DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW/i)).toBeInTheDocument();
  });
});

