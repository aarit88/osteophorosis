import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ResearchWorkbenchPage } from '../src/pages/ResearchWorkbenchPage';
import type { ResearchTracksResponse, TrackAInferenceResponse, TrackAStatusResponse, TrackDDatasetReadinessResponse } from '../src/api/types';

const mocks = vi.hoisted(() => ({
  getResearchTracks: vi.fn(),
  getTrackDDatasetReadiness: vi.fn(),
  getTrackAStatus: vi.fn(),
  runTrackAInference: vi.fn(),
  runSyntheticWorkbench: vi.fn(),
}));
vi.mock('../src/api/client', () => mocks);

const tracks: ResearchTracksResponse = {
  timestamp: '2026-11-24T00:00:00Z', active_tracks_count: 2,
  tracks: [
    { track_id: 'track_a', title: 'FracAtlas fracture research', modality: 'X-ray', task_type: 'Binary fracture research', target_classes: ['fracture', 'no fracture'], status: 'HISTORICAL_RESEARCH_FROZEN', historical_metrics: {}, limitations: ['Historical research only.'], governance_clearance: 'RESEARCH_ONLY', disclaimer: 'Not a clinical diagnostic.' },
    { track_id: 'track_d', title: 'Four-class development', modality: 'X-ray, bone scintigraphy, CT', task_type: 'Four-class research', target_classes: ['SECONDARY_BONE_METASTASIS', 'PAGETS_DISEASE', 'OSTEOGENESIS_IMPERFECTA', 'HEALTHY_SKELETAL_CONTROL'], status: 'BLOCKED_BY_CLINICAL_DATA', historical_metrics: {}, limitations: ['No validated four-class model.'], governance_clearance: 'BLOCKED', disclaimer: 'Research use only.' },
  ], disclaimer: 'Research metadata only.',
};
const readiness = { readiness_status: 'NOT_AVAILABLE', blockers: ['Genuine authorized clinical data are unavailable.'] } as TrackDDatasetReadinessResponse;
const trackAUnavailable: TrackAStatusResponse = {
  track_id: 'track_a', model_name: 'FracAtlas ResNet-101 binary fracture research classifier', modality: 'xray',
  input_size: [224, 224], classes: ['candidate_control_radiograph', 'acute_bone_fracture'],
  checkpoint_available: false, checkpoint_size_bytes: null, status: 'CHECKPOINT_UNAVAILABLE',
  error_code: 'TRACK_A_CHECKPOINT_NOT_AVAILABLE', clinical_use: false,
};
const trackAReady: TrackAStatusResponse = { ...trackAUnavailable, checkpoint_available: true, checkpoint_size_bytes: 1234, status: 'CHECKPOINT_PRESENT_LOAD_ON_RUN', error_code: null };
const inferenceFixture: TrackAInferenceResponse = {
  track_id: 'track_a', track: 'Track A FracAtlas research fixture', model_version: '1.0.0', checkpoint_epoch: 9,
  checkpoint_sha256: 'a'.repeat(64), checkpoint_size_bytes: 1234, modality: 'xray',
  prob_non_fractured: 0.2, prob_fractured: 0.8, active_threshold: 0.31,
  predicted_class_idx: 1, predicted_class_name: 'acute_bone_fracture', operating_point: 'high_sensitivity_screening',
  timestamp: '2026-11-25T00:00:00Z', research_notice: 'Synthetic UI test response only.', disclaimer: 'Synthetic UI test response only.',
};
const renderPage = () => render(<MemoryRouter><ResearchWorkbenchPage /></MemoryRouter>);

describe('research workbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getResearchTracks.mockResolvedValue(tracks);
    mocks.getTrackDDatasetReadiness.mockResolvedValue(readiness);
    mocks.getTrackAStatus.mockResolvedValue(trackAUnavailable);
  });

  it('loads the live track registry and displays modality/status without exposing metrics', async () => {
    renderPage();
    expect(await screen.findByRole('heading', { name: 'FracAtlas fracture research' })).toBeInTheDocument();
    expect(screen.getByText('X-ray')).toBeInTheDocument();
    expect(screen.getByText('HISTORICAL RESEARCH FROZEN')).toBeInTheDocument();
    expect(mocks.getResearchTracks).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/historical accuracy|patient count/i)).not.toBeInTheDocument();
  });

  it('checks Track D readiness and preserves the backend 503 blocker without synthetic fallback', async () => {
    renderPage();
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    fireEvent.change(screen.getByRole('combobox', { name: 'Research track' }), { target: { value: 'track_d' } });
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY', { selector: 'p' })).toBeInTheDocument();
    expect(await screen.findByText('Genuine authorized clinical data are unavailable.')).toBeInTheDocument();
    expect(mocks.getTrackDDatasetReadiness).toHaveBeenCalledTimes(1);
    expect(mocks.runTrackAInference).not.toHaveBeenCalled();
    expect(screen.getByRole('link', { name: /Review governance/ })).toHaveAttribute('href', '/governance');
  });

  it('shows module availability and links cleanly without firing inference prematurely', async () => {
    renderPage();
    expect(await screen.findByRole('heading', { name: /Interactive XAI & Diagnostic Research Modules/i })).toBeInTheDocument();
    for (const [name, href] of [['Radiomics', '/radiomics'], ['Biomechanics', '/biomechanics'], ['Explainability', '/explainability'], ['Uncertainty', '/uncertainty']]) {
      const article = screen.getByRole('heading', { name }).closest('article');
      expect(article && within(article).getByRole('link', { name: /Open interactive module/i })).toHaveAttribute('href', href);
    }
    expect(mocks.runTrackAInference).not.toHaveBeenCalled();
  });

  it('keeps the local X-ray portal visible and blocks inference when the real checkpoint is missing', async () => {
    renderPage();
    expect(await screen.findByText(/Track A scan is unavailable: no checkpoint file is configured/)).toBeInTheDocument();
    const input = screen.getByLabelText('Select a local X-ray image') as HTMLInputElement;
    expect(input).toBeEnabled();
    fireEvent.change(input, { target: { files: [new File(['synthetic image fixture'], 'synthetic-xray.png', { type: 'image/png' })] } });
    expect(await screen.findByText(/X-ray selected/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run X-ray scan' })).toBeDisabled();
    expect(mocks.runTrackAInference).not.toHaveBeenCalled();
    expect(screen.queryByText('synthetic-xray.png')).not.toBeInTheDocument();
  });

  it('submits only after an explicit action and renders the backend Track A response', async () => {
    mocks.getTrackAStatus.mockResolvedValueOnce(trackAReady);
    mocks.runTrackAInference.mockResolvedValueOnce(inferenceFixture);
    renderPage();
    const input = await screen.findByLabelText('Select a local X-ray image');
    fireEvent.change(input, { target: { files: [new File(['synthetic image fixture'], 'synthetic-xray.png', { type: 'image/png' })] } });
    const button = await screen.findByRole('button', { name: 'Run X-ray scan' });
    expect(button).toBeEnabled();
    expect(mocks.runTrackAInference).not.toHaveBeenCalled();
    fireEvent.click(button);
    expect(await screen.findByText('Actual Track A backend response')).toBeInTheDocument();
    expect(screen.getByText('Acute bone fracture class')).toBeInTheDocument();
    expect(mocks.runTrackAInference).toHaveBeenCalledTimes(1);
    expect(mocks.runTrackAInference.mock.calls[0][1]).toBe('high_sensitivity_screening');
    expect(screen.queryByText(/^accuracy:?$/i)).not.toBeInTheDocument();
  });

  it('handles unavailable track and readiness endpoints without displaying raw exceptions', async () => {
    mocks.getResearchTracks.mockRejectedValueOnce(new Error('private gateway path'));
    renderPage();
    expect(await screen.findByRole('alert')).toHaveTextContent(/Research tracks are unavailable/);
    expect(screen.queryByText(/private gateway path/)).not.toBeInTheDocument();
    mocks.getResearchTracks.mockResolvedValueOnce(tracks);
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await screen.findByRole('heading', { name: 'FracAtlas fracture research' });
    mocks.getTrackDDatasetReadiness.mockRejectedValueOnce(new Error('private readiness path'));
    fireEvent.change(screen.getByRole('combobox', { name: 'Research track' }), { target: { value: 'track_d' } });
    expect(await screen.findByRole('alert')).toHaveTextContent(/Readiness is unavailable/);
    expect(screen.queryByText(/private readiness path/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry readiness check' }));
    await waitFor(() => expect(mocks.getTrackDDatasetReadiness).toHaveBeenCalledTimes(2));
  });
});
