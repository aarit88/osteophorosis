import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { ExplainabilityPage } from '../src/pages/ExplainabilityPage';
import { UncertaintyPage } from '../src/pages/UncertaintyPage';
import { Results } from '../src/pages/Results';
import * as api from '../src/api/client';
import type { AnalyzeResponse } from '../src/api/types';

const analysisMocks = vi.hoisted(() => ({ lastResult: null as unknown }));
vi.mock('../src/context/AnalysisContext', () => ({
  useAnalysis: () => ({ lastResult: analysisMocks.lastResult }),
}));

const blocked = {
  status: 'BLOCKED',
  error_code: 'FOUR_CLASS_MODEL_NOT_READY',
  message: 'Four-class inference remains unavailable until its gates pass.',
  target_classes: [],
  blockers: ['No verified Track D model is available.'],
  mandatory_safeguard: 'Abstention requires human review.',
};

describe('Stage 45 explainability, uncertainty and results capabilities', () => {
  it('renders Grad-CAM synthetic localization demo alongside blocked clinical Track D readiness', async () => {
    const readiness = vi.spyOn(api, 'requestFourClassInference').mockResolvedValue(blocked);
    const explain = vi.spyOn(api, 'runExplainabilityDemo').mockResolvedValue({
      synthetic_input_shape: [1, 1, 32, 32],
      target_class: 1,
      active_region_ratio: 0.25,
      peak_coordinates: [16, 16],
      heatmap_grid_sample: Array.from({ length: 8 }, () => Array(8).fill(0.5)),
      method: 'Grad-CAM',
      shap_status: 'PARTIAL',
      warning: 'SYNTHETIC DEMO ONLY.',
    });
    render(<MemoryRouter><ExplainabilityPage /></MemoryRouter>);
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Generate Grad-CAM Activation Grid/i }));
    expect(await screen.findByText('25.0%')).toBeInTheDocument();
    expect(explain).toHaveBeenCalledTimes(1);
    expect(readiness).toHaveBeenCalledTimes(1);
  });

  it('evaluates entropy and OOD scenarios interactively on UncertaintyPage', async () => {
    vi.spyOn(api, 'requestFourClassInference').mockResolvedValue(blocked);
    const uncertainty = vi.spyOn(api, 'runUncertaintyDemo').mockResolvedValue({
      probabilities: [0.48, 0.48, 0.02, 0.02],
      predicted_class_idx: 0,
      predicted_class_name: 'Secondary Bone Metastasis',
      max_probability: 0.48,
      top_two_margin: 0.0,
      shannon_entropy_nats: 0.87,
      normalized_entropy: 0.628,
      candidate_threshold: 0.6,
      decision: 'ABSTAIN_HUMAN_REVIEW_REQUIRED',
      human_review_required: true,
      abstention_reason: 'ELEVATED_ENTROPY',
      reason_description: 'ABSTENTION: Prediction entropy exceeds threshold.',
      ood_detected: false,
      clinical_guidance: 'Human specialist review mandated.',
      disclaimer: 'RESEARCH ONLY.',
    });
    render(<MemoryRouter><UncertaintyPage /></MemoryRouter>);
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Evaluate Uncertainty Gate/i }));
    expect(await screen.findByText('0.628')).toBeInTheDocument();
    expect(uncertainty).toHaveBeenCalledTimes(1);
  });

  it('shows a helpful empty result state linking back to Home Upload and Workbench', () => {
    analysisMocks.lastResult = null;
    render(<MemoryRouter><Results /></MemoryRouter>);
    expect(screen.getByRole('heading', { name: 'No current analysis result' })).toBeInTheDocument();
    expect(screen.getByText(/FOUR_CLASS_MODEL_NOT_READY/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Upload Scan or Run Demo/ })).toHaveAttribute('href', '/');
  });

  it('shows AbstentionBanner advisory while keeping diagnostic XAI inspection panels accessible when entropy exceeds threshold', () => {
    analysisMocks.lastResult = {
      requestId: 'fixture-request',
      timestamp: new Date(0).toISOString(),
      modality: 'ct',
      imageUrl: '/api/v1/ephemeral/orig.png',
      gradcamUrl: '/api/v1/ephemeral/grad.png',
      prediction: {
        topClass: 'paget',
        confidence: 0.8,
        probabilities: { metastasis: 0.1, paget: 0.8, oi: 0.05, healthy: 0.05 },
        uncertainty: 'high',
      },
      uncertaintyDetail: {
        entropy: 0.8,
        confidenceInterval: [0.6, 0.9],
        oodFlag: false,
        calibrationBin: { predicted: 0.8, observed: 0.75 },
      },
      shap: [{ name: 'GLCM Contrast', value: 0.42 }],
      counterfactuals: [
        { feature: 'Cortical Thickness', deltaDirection: 'decrease', percentChange: 20, flipTo: 'healthy', probabilityShift: 0.2 },
      ],
      fractureRisk: { level: 'medium', percent: 45 },
      disclaimer: 'Synthetic test fixture only.',
    } satisfies AnalyzeResponse;
    render(<MemoryRouter><Results /></MemoryRouter>);
    expect(screen.getByText(/Model Recommends Human Review/i)).toBeInTheDocument();
    expect(screen.getByText("Paget's Disease of Bone")).toBeInTheDocument();
    expect(screen.getByText('4-Class Differential Probabilities')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Generate Traceable Report/i })).toBeInTheDocument();
    analysisMocks.lastResult = null;
  });

  it('offers retry when model readiness is unavailable and never shows raw exception internals', async () => {
    const request = vi.spyOn(api, 'requestFourClassInference')
      .mockRejectedValueOnce(new Error('private service path'))
      .mockResolvedValueOnce(blocked);
    render(<MemoryRouter><ExplainabilityPage /></MemoryRouter>);
    expect(await screen.findByRole('alert')).toHaveTextContent(/Current model readiness could not be retrieved/i);
    expect(screen.queryByText(/private service path/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry readiness check' }));
    await waitFor(() => expect(screen.getByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument());
    expect(request).toHaveBeenCalledTimes(2);
  });
});
