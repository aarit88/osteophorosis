import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ResearchTracksPage } from '../src/pages/ResearchTracksPage';
import { RadiomicsPage } from '../src/pages/RadiomicsPage';
import { BiomechanicsPage } from '../src/pages/BiomechanicsPage';

const mocks = vi.hoisted(() => ({ getResearchTracks: vi.fn(), runRadiomicsDemo: vi.fn(), runBiomechanicsGeometry: vi.fn(), runBiomechanicsMirels: vi.fn() }));
vi.mock('../src/api/client', () => mocks);

const tracks = {
  timestamp: '2026-11-24T00:00:00Z', active_tracks_count: 2, disclaimer: 'Research registry only.', tracks: [
    { track_id: 'track_a', title: 'FracAtlas fracture research', modality: 'Plain radiography', task_type: 'Binary fracture classification', target_classes: ['fracture', 'non-fracture'], status: 'HISTORICAL_RESEARCH_FROZEN', historical_metrics: {}, limitations: ['Burned holdout is preserved.'], governance_clearance: 'Academic research', disclaimer: 'Not clinical diagnosis.' },
    { track_id: 'track_d', title: 'Four-class research', modality: 'X-ray, scintigraphy, CT', task_type: 'Four-class differential', target_classes: ['SECONDARY_BONE_METASTASIS', 'PAGETS_DISEASE', 'OSTEOGENESIS_IMPERFECTA', 'HEALTHY_SKELETAL_CONTROL'], status: 'BLOCKED_BY_CLINICAL_DATA', historical_metrics: {}, limitations: ['Clinical cohorts unavailable.'], governance_clearance: 'Pending governance', disclaimer: 'FOUR_CLASS_MODEL_NOT_READY' },
  ],
};
const renderPage = (page: React.ReactNode) => render(<MemoryRouter>{page}</MemoryRouter>);

describe('Stage 44 research modules', () => {
  beforeEach(() => { vi.clearAllMocks(); mocks.getResearchTracks.mockResolvedValue(tracks); });

  it('renders real track boundaries without exposing metrics or checkpoint material', async () => {
    renderPage(<ResearchTracksPage />);
    expect(await screen.findByRole('heading', { name: 'FracAtlas fracture research' })).toBeInTheDocument();
    expect(screen.getByText(/Historical evaluation evidence is recorded/)).toBeInTheDocument();
    expect(screen.getByText(/Burned holdout partitions are preserved/)).toBeInTheDocument();
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(screen.queryByText(/SHA-256/)).not.toBeInTheDocument();
  });

  it('documents actual radiomics families while keeping synthetic feature values out of the runtime UI', () => {
    renderPage(<RadiomicsPage />);
    expect(screen.getByRole('heading', { name: 'Implemented feature families' })).toBeInTheDocument();
    expect(screen.getByText('GLCM')).toBeInTheDocument();
    expect(screen.getByText(/81 feature dimensions as analytically verified/)).toBeInTheDocument();
    expect(screen.getByText(/21 dimensions were described as engineering heuristics/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Run phantom/i })).not.toBeInTheDocument();
    expect(mocks.runRadiomicsDemo).not.toHaveBeenCalled();
  });

  it('keeps geometry outputs unavailable without a governed image segmentation input', () => {
    renderPage(<BiomechanicsPage />);
    expect(screen.getByRole('heading', { name: 'Cross-section geometry' })).toBeInTheDocument();
    expect(screen.getByText(/no governed image-segmentation input/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Calculate geometry' })).not.toBeInTheDocument();
    expect(mocks.runBiomechanicsGeometry).not.toHaveBeenCalled();
  });

  it('validates Mirels factor inputs and shows only the arithmetic score', async () => {
    mocks.runBiomechanicsMirels.mockResolvedValue({ total_score: 9, fracture_risk_percent: 33.3, recommendation_tier: 'HIGH_RISK', prophylactic_fixation_indicated: true, clinical_guidance: 'Not a patient evaluation.', disclaimer: 'Conventional rule only.' });
    renderPage(<BiomechanicsPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Calculate hypothetical score' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Choose all four scoring categories');
    expect(mocks.runBiomechanicsMirels).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole('combobox', { name: 'Anatomic site' }), { target: { value: '2' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Pain category' }), { target: { value: '2' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Radiographic nature' }), { target: { value: '3' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Lesion size' }), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Calculate hypothetical score' }));
    expect(await screen.findByText('Total score: 9 / 12')).toBeInTheDocument();
    expect(screen.queryByText('33.3%')).not.toBeInTheDocument();
    expect(screen.queryByText('HIGH RISK')).not.toBeInTheDocument();
    expect(mocks.runBiomechanicsMirels).toHaveBeenCalledWith({ site: 2, pain: 2, radiographic_nature: 3, lesion_size: 2 });
  });
});
