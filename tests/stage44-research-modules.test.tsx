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

  it('runs digital phantom radiomics extraction on RadiomicsPage', async () => {
    mocks.runRadiomicsDemo.mockResolvedValue({
      phantom_type: 'checkerboard',
      grid_size: 32,
      spacing_mm: 1.0,
      verified_features_count: 81,
      unverified_features_count: 21,
      total_features_count: 102,
      specification_status: '102-D SPECIFICATION NOT VERIFIED',
      verified_features_sample: { firstorder_Mean: 124.5 },
      unverified_features_sample: { cortical_heuristic_v1: 0.42 },
      disclaimer: 'DEMONSTRATION ONLY.',
    });
    renderPage(<RadiomicsPage />);
    expect(screen.getByRole('heading', { name: 'Implemented feature families' })).toBeInTheDocument();
    expect(screen.getByText('GLCM')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Extract Radiomics Features/i }));
    expect(await screen.findByText('81 features')).toBeInTheDocument();
    expect(mocks.runRadiomicsDemo).toHaveBeenCalledTimes(1);
  });

  it('computes 2D cross-section cortical geometry on BiomechanicsPage', async () => {
    mocks.runBiomechanicsGeometry.mockResolvedValue({
      phantom_type: 'intact_femur_diaphysis',
      voxel_size_mm: 0.5,
      pixel_spacing_mm: [0.5, 0.5],
      area_mm2: 312.5,
      centroid_x_mm: 25,
      centroid_y_mm: 25,
      I_xx_mm4: 4800,
      I_yy_mm4: 4800,
      I_xy_mm4: 0,
      polar_J_mm4: 9600,
      area_retention_ratio: 1,
      bending_retention_ratio_x: 1,
      bending_retention_ratio_y: 1,
      torsion_retention_ratio: 1,
      warning: 'MATHEMATICAL DEMONSTRATION ONLY.',
    });
    renderPage(<BiomechanicsPage />);
    fireEvent.click(screen.getByRole('button', { name: /Calculate geometry/i }));
    expect(await screen.findByText('312.5 mm²')).toBeInTheDocument();
    expect(mocks.runBiomechanicsGeometry).toHaveBeenCalledTimes(1);
  });

  it('validates Mirels factor inputs and shows full score, risk %, tier, and clinical guidance', async () => {
    mocks.runBiomechanicsMirels.mockResolvedValue({
      total_score: 9,
      fracture_risk_percent: 33.3,
      recommendation_tier: 'HIGH_RISK',
      prophylactic_fixation_indicated: true,
      clinical_guidance: 'Prophylactic fixation strongly indicated.',
      disclaimer: 'Conventional rule only.',
    });
    renderPage(<BiomechanicsPage />);
    fireEvent.click(screen.getByRole('button', { name: /Calculate Mirels Fracture Risk/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Choose all four scoring categories');
    expect(mocks.runBiomechanicsMirels).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole('combobox', { name: 'Anatomic site' }), { target: { value: '2' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Pain category' }), { target: { value: '2' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Radiographic nature' }), { target: { value: '3' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Lesion size' }), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: /Calculate Mirels Fracture Risk/i }));
    expect(await screen.findByText('Total score: 9 / 12')).toBeInTheDocument();
    expect(screen.getByText('33.3%')).toBeInTheDocument();
    expect(screen.getByText(/HIGH RISK/)).toBeInTheDocument();
    expect(mocks.runBiomechanicsMirels).toHaveBeenCalledWith({ site: 2, pain: 2, radiographic_nature: 3, lesion_size: 2 });
  });
});
