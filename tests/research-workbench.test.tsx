import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ResearchWorkbenchPage } from '../src/pages/ResearchWorkbenchPage';
import type { ResearchTracksResponse, SyntheticWorkbenchRunResponse, TrackDDatasetReadinessResponse } from '../src/api/types';

const mocks = vi.hoisted(() => ({
  getResearchTracks: vi.fn(),
  getTrackDDatasetReadiness: vi.fn(),
  runSyntheticWorkbench: vi.fn(),
}));

vi.mock('../src/api/client', () => mocks);

const tracks: ResearchTracksResponse = {
  timestamp: '2026-11-24T00:00:00Z',
  active_tracks_count: 2,
  tracks: [
    { track_id: 'track_a', title: 'FracAtlas fracture research', modality: 'X-ray', task_type: 'Binary fracture research', target_classes: ['fracture', 'no fracture'], status: 'HISTORICAL_RESEARCH_FROZEN', historical_metrics: {}, limitations: ['Historical research only.'], governance_clearance: 'RESEARCH_ONLY', disclaimer: 'Not a clinical diagnostic.' },
    { track_id: 'track_d', title: 'Four-class development', modality: 'X-ray, bone scintigraphy, CT', task_type: 'Four-class research', target_classes: ['SECONDARY_BONE_METASTASIS', 'PAGETS_DISEASE', 'OSTEOGENESIS_IMPERFECTA', 'HEALTHY_SKELETAL_CONTROL'], status: 'BLOCKED_BY_CLINICAL_DATA', historical_metrics: {}, limitations: ['No validated four-class model.'], governance_clearance: 'BLOCKED', disclaimer: 'Research use only.' },
  ],
  disclaimer: 'Research metadata only.',
};

const readiness = {
  timestamp: '2026-11-24T00:00:00Z', track: 'track_d', readiness_status: 'NOT_AVAILABLE', ready_for_training: false,
  data_origin: 'REAL_CLINICAL_DATA_UNAVAILABLE', synthetic_data_excluded: true,
  total_inspected_samples: 0, valid_samples: 0, quarantined_samples: 0, unique_patients: 0, institution_count: 0,
  institutions: [], dataset_sources: [], permissions_verified: [], ethics_references: [],
  class_counts: { metastasis: 0, paget: 0, oi: 0, healthy: 0 }, modality_counts: { xray: 0, bone_scan: 0, ct: 0 },
  class_modality_matrix: { xray: { metastasis: 0, paget: 0, oi: 0, healthy: 0 }, bone_scan: { metastasis: 0, paget: 0, oi: 0, healthy: 0 }, ct: { metastasis: 0, paget: 0, oi: 0, healthy: 0 } },
  missing_metadata_count: 0, duplicate_samples_count: 0, manifest_schema_validation: 'NOT_AVAILABLE', ontology_version: '1',
  accepted_by_class: { SECONDARY_BONE_METASTASIS: 0, PAGETS_DISEASE: 0, OSTEOGENESIS_IMPERFECTA: 0, HEALTHY_SKELETAL_CONTROL: 0 },
  adjudication_summary: {}, blockers: ['Genuine authorized clinical data are unavailable.'],
} as unknown as TrackDDatasetReadinessResponse;

const syntheticDemo = {
  timestamp: '2026-11-24T00:00:00Z', status: 'COMPLETED', disclaimer: 'Synthetic mathematical demonstration only.',
  radiomics_panel: { phantom_type: 'synthetic-grid', grid_size: 8, spacing_mm: 1, verified_features_count: 12, unverified_features_count: 1, total_features_count: 13, specification_status: 'test fixture', verified_features_sample: {}, unverified_features_sample: {}, disclaimer: 'Synthetic fixture.' },
  biomechanics_geometry_panel: { phantom_type: 'synthetic geometry', voxel_size_mm: 1, area_mm2: 12.5, centroid_x_mm: 0, centroid_y_mm: 0, I_xx_mm4: 1, I_yy_mm4: 1, I_xy_mm4: 0, polar_J_mm4: 2, area_retention_ratio: 1, bending_retention_ratio_x: 1, bending_retention_ratio_y: 1, torsion_retention_ratio: 1, warning: 'Synthetic fixture.' },
  biomechanics_mirels_panel: { total_score: 6, fracture_risk_percent: 0, recommendation_tier: 'TEST_ONLY', prophylactic_fixation_indicated: false, clinical_guidance: 'Synthetic fixture.', disclaimer: 'Synthetic fixture.' },
  explainability_panel: { synthetic_input_shape: [1, 1, 8, 8], target_class: 0, active_region_ratio: 0, peak_coordinates: [0, 0], heatmap_grid_sample: [], method: 'synthetic fixture method', shap_status: 'NOT_RUN', warning: 'Synthetic fixture.' },
  uncertainty_panel: { probabilities: [], predicted_class_idx: 0, predicted_class_name: 'hidden from Workbench', max_probability: 0, top_two_margin: 0, shannon_entropy_nats: 0, normalized_entropy: 0.25, candidate_threshold: 0.6, decision: 'DEMONSTRATION_ONLY', human_review_required: false, abstention_reason: 'Synthetic fixture.', reason_description: 'Synthetic fixture.', ood_detected: false, clinical_guidance: 'Synthetic fixture.', disclaimer: 'Synthetic fixture.' },
} as SyntheticWorkbenchRunResponse;

function renderPage() {
  return render(<MemoryRouter><ResearchWorkbenchPage /></MemoryRouter>);
}

describe('research workbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getResearchTracks.mockResolvedValue(tracks);
    mocks.getTrackDDatasetReadiness.mockResolvedValue(readiness);
    mocks.runSyntheticWorkbench.mockResolvedValue(syntheticDemo);
  });

  it('loads the live track registry and displays modality and status without presenting historical metrics', async () => {
    renderPage();
    expect(await screen.findByRole('heading', { name: 'FracAtlas fracture research' })).toBeInTheDocument();
    expect(screen.getByText('X-ray')).toBeInTheDocument();
    expect(screen.getByText('HISTORICAL RESEARCH FROZEN')).toBeInTheDocument();
    expect(mocks.getResearchTracks).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/historical accuracy|patient count/i)).not.toBeInTheDocument();
  });

  it('checks Track D readiness and disables synthetic submission without creating prediction output', async () => {
    renderPage();
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    fireEvent.change(screen.getByRole('combobox', { name: 'Research track' }), { target: { value: 'track_d' } });
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(await screen.findByText('Genuine authorized clinical data are unavailable.')).toBeInTheDocument();
    expect(mocks.getTrackDDatasetReadiness).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Run synthetic demonstration' })).toBeDisabled();
    expect(screen.queryByText(/predicted class|probability|sample prediction/i)).not.toBeInTheDocument();
    expect(mocks.runSyntheticWorkbench).not.toHaveBeenCalled();
    expect(screen.getByRole('link', { name: /Review governance/ })).toHaveAttribute('href', '/governance');
  });

  it('runs the existing synthetic endpoint once and labels its result as a demonstration', async () => {
    renderPage();
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    fireEvent.click(screen.getByRole('button', { name: 'Run synthetic demonstration' }));
    expect(await screen.findByText('Synthetic demonstration completed')).toBeInTheDocument();
    expect(mocks.runSyntheticWorkbench).toHaveBeenCalledTimes(1);
    expect(mocks.runSyntheticWorkbench).toHaveBeenCalledWith({});
    expect(screen.getByText(/not patient results, model predictions, validation metrics/i)).toBeInTheDocument();
    expect(screen.getByText('synthetic-grid · 8 × 8')).toBeInTheDocument();
    expect(screen.queryByText('hidden from Workbench')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reset' })).toBeInTheDocument();
  });

  it('prevents repeated submissions while the demonstration request is in flight', async () => {
    let resolveRequest: (value: SyntheticWorkbenchRunResponse) => void = () => undefined;
    mocks.runSyntheticWorkbench.mockImplementationOnce(() => new Promise((resolve) => { resolveRequest = resolve; }));
    renderPage();
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    const button = screen.getByRole('button', { name: 'Run synthetic demonstration' });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(mocks.runSyntheticWorkbench).toHaveBeenCalledTimes(1);
    resolveRequest(syntheticDemo);
    expect(await screen.findByText('Synthetic demonstration completed')).toBeInTheDocument();
  });

  it('shows service errors and offers retry without fabricating a successful result', async () => {
    mocks.runSyntheticWorkbench.mockRejectedValueOnce(new Error('Backend unavailable'));
    renderPage();
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    fireEvent.click(screen.getByRole('button', { name: 'Run synthetic demonstration' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Backend unavailable');
    expect(screen.queryByText('Synthetic demonstration completed')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('Synthetic demonstration completed')).toBeInTheDocument();
  });

  it('handles unavailable track and readiness endpoints with actionable retry controls', async () => {
    mocks.getResearchTracks.mockRejectedValueOnce(new Error('Gateway offline'));
    const { rerender } = renderPage();
    expect(await screen.findByRole('alert')).toHaveTextContent('Gateway offline');
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    mocks.getTrackDDatasetReadiness.mockRejectedValueOnce(new Error('Readiness offline'));
    fireEvent.change(screen.getByRole('combobox', { name: 'Research track' }), { target: { value: 'track_d' } });
    expect(await screen.findByRole('alert')).toHaveTextContent('Readiness offline');
    fireEvent.click(screen.getByRole('button', { name: 'Retry readiness check' }));
    await waitFor(() => expect(mocks.getTrackDDatasetReadiness).toHaveBeenCalledTimes(2));
    rerender(<MemoryRouter><ResearchWorkbenchPage /></MemoryRouter>);
  });

  it('exposes links to tracks, governance, evidence, results, and research methods', async () => {
    renderPage();
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    for (const [name, href] of [['Tracks', '/tracks'], ['Governance', '/governance'], ['Evidence', '/evidence'], ['Results', '/results']]) {
      expect(screen.getAllByRole('link', { name: new RegExp(name) })[0]).toHaveAttribute('href', href);
    }
    expect(screen.getByText(/track-specific image upload and inference are not connected/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/image upload|study upload/i)).not.toBeInTheDocument();
  });
});
