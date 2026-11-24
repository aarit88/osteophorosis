import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ResearchWorkbenchPage } from '../src/pages/ResearchWorkbenchPage';
import type { ResearchTracksResponse, TrackDDatasetReadinessResponse } from '../src/api/types';

const mocks = vi.hoisted(() => ({ getResearchTracks: vi.fn(), getTrackDDatasetReadiness: vi.fn(), runSyntheticWorkbench: vi.fn() }));
vi.mock('../src/api/client', () => mocks);

const tracks: ResearchTracksResponse = {
  timestamp: '2026-11-24T00:00:00Z', active_tracks_count: 2,
  tracks: [
    { track_id: 'track_a', title: 'FracAtlas fracture research', modality: 'X-ray', task_type: 'Binary fracture research', target_classes: ['fracture', 'no fracture'], status: 'HISTORICAL_RESEARCH_FROZEN', historical_metrics: {}, limitations: ['Historical research only.'], governance_clearance: 'RESEARCH_ONLY', disclaimer: 'Not a clinical diagnostic.' },
    { track_id: 'track_d', title: 'Four-class development', modality: 'X-ray, bone scintigraphy, CT', task_type: 'Four-class research', target_classes: ['SECONDARY_BONE_METASTASIS', 'PAGETS_DISEASE', 'OSTEOGENESIS_IMPERFECTA', 'HEALTHY_SKELETAL_CONTROL'], status: 'BLOCKED_BY_CLINICAL_DATA', historical_metrics: {}, limitations: ['No validated four-class model.'], governance_clearance: 'BLOCKED', disclaimer: 'Research use only.' },
  ], disclaimer: 'Research metadata only.',
};
const readiness = { readiness_status: 'NOT_AVAILABLE', blockers: ['Genuine authorized clinical data are unavailable.'] } as TrackDDatasetReadinessResponse;
const renderPage = () => render(<MemoryRouter><ResearchWorkbenchPage /></MemoryRouter>);

describe('research workbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getResearchTracks.mockResolvedValue(tracks);
    mocks.getTrackDDatasetReadiness.mockResolvedValue(readiness);
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
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(await screen.findByText('Genuine authorized clinical data are unavailable.')).toBeInTheDocument();
    expect(mocks.getTrackDDatasetReadiness).toHaveBeenCalledTimes(1);
    expect(mocks.runSyntheticWorkbench).not.toHaveBeenCalled();
    expect(screen.getByRole('link', { name: /Review governance/ })).toHaveAttribute('href', '/governance');
  });

  it('shows module availability and never invokes the synthetic workbench endpoint', async () => {
    renderPage();
    expect(await screen.findByRole('heading', { name: 'Research module availability' })).toBeInTheDocument();
    expect(screen.getByText(/No synthetic image-derived outputs/)).toBeInTheDocument();
    for (const [name, href] of [['Radiomics', '/radiomics'], ['Biomechanics', '/biomechanics'], ['Explainability', '/explainability'], ['Uncertainty', '/uncertainty']]) {
      const article = screen.getByRole('heading', { name }).closest('article');
      expect(article && within(article).getByRole('link', { name: 'Open module' })).toHaveAttribute('href', href);
    }
    expect(mocks.runSyntheticWorkbench).not.toHaveBeenCalled();
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
