import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ResearchTracksPage } from '../src/pages/ResearchTracksPage';
import { RadiomicsPage } from '../src/pages/RadiomicsPage';
import { BiomechanicsPage } from '../src/pages/BiomechanicsPage';
import type { ResearchTracksResponse, RadiomicsDemoResponse, BiomechanicsGeometryResponse, BiomechanicsMirelsResponse } from '../src/api/types';

const mocks = vi.hoisted(() => ({
  getResearchTracks: vi.fn(),
  runRadiomicsDemo: vi.fn(),
  runBiomechanicsGeometry: vi.fn(),
  runBiomechanicsMirels: vi.fn(),
}));
vi.mock('../src/api/client', () => mocks);

const tracks: ResearchTracksResponse = {
  timestamp: '2026-11-24T00:00:00Z', active_tracks_count: 2, disclaimer: 'Research registry only.', tracks: [
    { track_id: 'track_a', title: 'FracAtlas fracture research', modality: 'Plain radiography', task_type: 'Binary fracture classification', target_classes: ['fracture', 'non-fracture'], status: 'HISTORICAL_RESEARCH_FROZEN', historical_metrics: { roc_auc: 0.9 }, limitations: ['Burned holdout is preserved.'], governance_clearance: 'Academic research', disclaimer: 'Not clinical diagnosis.' },
    { track_id: 'track_d', title: 'Four-class research', modality: 'X-ray, scintigraphy, CT', task_type: 'Four-class differential', target_classes: ['SECONDARY_BONE_METASTASIS', 'PAGETS_DISEASE', 'OSTEOGENESIS_IMPERFECTA', 'HEALTHY_SKELETAL_CONTROL'], status: 'BLOCKED_BY_CLINICAL_DATA', historical_metrics: { model_readiness: 'ZERO MODELS TRAINED' }, limitations: ['Clinical cohorts unavailable.'], governance_clearance: 'Pending governance', disclaimer: 'FOUR_CLASS_MODEL_NOT_READY' },
  ],
};

const radiomicsResult: RadiomicsDemoResponse = {
  phantom_type: 'checkerboard', grid_size: 64, spacing_mm: 1, verified_features_count: 81, unverified_features_count: 21,
  total_features_count: 102, specification_status: '102-D NOT FULLY VERIFIED',
  verified_features_sample: { first_order_mean: 0.75, glcm_contrast: 2.5 },
  unverified_features_sample: { gldm_heuristic_feature: 3.1 },
  disclaimer: 'Synthetic phantom only; 21 features remain unverified.',
};

const geometryResult: BiomechanicsGeometryResponse = {
  phantom_type: 'intact_circular_ring', voxel_size_mm: 0.5, pixel_spacing_mm: [0.5, 0.5],
  area_mm2: 10, centroid_x_mm: 2, centroid_y_mm: 3, I_xx_mm4: 4, I_yy_mm4: 5, I_xy_mm4: 0,
  polar_J_mm4: 9, area_retention_ratio: 1, bending_retention_ratio_x: 1, bending_retention_ratio_y: 1,
  torsion_retention_ratio: 1, warning: 'MATHEMATICAL DEMONSTRATION ONLY: Does NOT compute patient-specific fracture risk or clinical bone strength.',
};

const mirelsResult: BiomechanicsMirelsResponse = {
  total_score: 9, fracture_risk_percent: 33.3, recommendation_tier: 'HIGH_RISK', prophylactic_fixation_indicated: true,
  clinical_guidance: 'Historical rule output; not a patient evaluation.', disclaimer: 'Conventional scoring rule only.',
};

const renderPage = (page: React.ReactNode) => render(<MemoryRouter>{page}</MemoryRouter>);

describe('Stage 44 research modules', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getResearchTracks.mockResolvedValue(tracks);
    mocks.runRadiomicsDemo.mockResolvedValue(radiomicsResult);
    mocks.runBiomechanicsGeometry.mockImplementation(async ({ phantom_type, voxel_size_mm }) => ({
      ...geometryResult,
      phantom_type,
      voxel_size_mm,
      pixel_spacing_mm: phantom_type === 'anisotropic_spacing_annulus' ? [0.4, 0.8] : [voxel_size_mm, voxel_size_mm],
    }));
    mocks.runBiomechanicsMirels.mockResolvedValue(mirelsResult);
  });

  it('renders existing tracks and avoids exposing training commands, metrics, or checkpoint material', async () => {
    renderPage(<ResearchTracksPage />);
    expect(await screen.findByRole('heading', { name: 'FracAtlas fracture research' })).toBeInTheDocument();
    expect(screen.getByText(/Historical evaluation evidence is recorded/)).toBeInTheDocument();
    expect(screen.getByText(/Burned holdout partitions are preserved/)).toBeInTheDocument();
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(screen.getByText(/No trained four-class model is available/)).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: /Open research methods workbench/ })[0]).toHaveAttribute('href', '/workbench');
    expect(screen.queryByRole('button', { name: /copy.*command/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/SHA-256|0\.9/)).not.toBeInTheDocument();
  });

  it('does not fabricate radiomics output on page load and renders actual response values after explicit request', async () => {
    renderPage(<RadiomicsPage />);
    expect(await screen.findByText(/No run yet/)).toBeInTheDocument();
    expect(mocks.runRadiomicsDemo).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Run phantom extraction' }));
    expect(await screen.findByText('first_order_mean')).toBeInTheDocument();
    expect(screen.getByText('0.750000')).toBeInTheDocument();
    expect(screen.getByText(/Unverified \/ heuristic sample/)).toBeInTheDocument();
    expect(screen.getByText(/API response does not include server timestamp or software build\/version/)).toBeInTheDocument();
    expect(mocks.runRadiomicsDemo).toHaveBeenCalledWith({ phantom_type: 'checkerboard', grid_size: 64, spacing_mm: 1 });
  });

  it('handles radiomics API errors without displaying previous or invented feature values', async () => {
    mocks.runRadiomicsDemo.mockRejectedValueOnce(new Error('Service unavailable'));
    renderPage(<RadiomicsPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Run phantom extraction' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Service unavailable');
    expect(screen.queryByText('first_order_mean')).not.toBeInTheDocument();
  });

  it('uses actual supported synthetic geometry controls and displays API units and spacing', async () => {
    renderPage(<BiomechanicsPage />);
    expect(mocks.runBiomechanicsGeometry).not.toHaveBeenCalled();
    expect(screen.getByRole('combobox', { name: 'Synthetic bone phantom' })).toHaveValue('intact_circular_ring');
    fireEvent.change(screen.getByRole('combobox', { name: 'Synthetic bone phantom' }), { target: { value: 'anisotropic_spacing_annulus' } });
    expect(screen.getByRole('combobox', { name: 'Pixel spacing' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Calculate geometry' }));
    expect(await screen.findByLabelText('Geometry result')).toHaveTextContent('10 mm²');
    expect(screen.getByText(/effective pixel spacing dy 0.4 \/ dx 0.8 mm/)).toBeInTheDocument();
    expect(mocks.runBiomechanicsGeometry).toHaveBeenCalledWith({ phantom_type: 'anisotropic_spacing_annulus', voxel_size_mm: 0.5 });
    expect(screen.queryByText(/patient-specific fracture risk or clinical bone strength/i)).toBeInTheDocument();
  });

  it('requires all hypothetical Mirels categories and suppresses risk percentages and treatment recommendations', async () => {
    renderPage(<BiomechanicsPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Calculate hypothetical score' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Choose all four hypothetical score inputs');
    expect(mocks.runBiomechanicsMirels).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole('combobox', { name: 'Anatomic site' }), { target: { value: '2' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Pain category' }), { target: { value: '2' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Radiographic nature' }), { target: { value: '3' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Lesion size' }), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Calculate hypothetical score' }));
    expect(await screen.findByLabelText('Mirels score result')).toHaveTextContent('Total score: 9 / 12');
    expect(screen.queryByText('33.3%')).not.toBeInTheDocument();
    expect(screen.queryByText(/prophylactic fixation|fixation indicated/i)).not.toBeInTheDocument();
    expect(mocks.runBiomechanicsMirels).toHaveBeenCalledWith({ site: 2, pain: 2, radiographic_nature: 3, lesion_size: 2 });
  });
});
