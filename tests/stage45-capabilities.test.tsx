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

describe('Stage 45 explainability, uncertainty and results safety', () => {
  it('reports actual blocked readiness and never invokes synthetic explanation endpoints', async () => {
    const readiness = vi.spyOn(api, 'requestFourClassInference').mockResolvedValue(blocked);
    const explain = vi.spyOn(api, 'runExplainabilityDemo');
    render(<MemoryRouter><ExplainabilityPage /></MemoryRouter>);
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(screen.getAllByText('Clinical generation unavailable').length).toBeGreaterThan(0);
    expect(explain).not.toHaveBeenCalled();
    expect(readiness).toHaveBeenCalledTimes(1);
  });

  it('describes unsupported calibration measures without generating probability scenarios', async () => {
    vi.spyOn(api, 'requestFourClassInference').mockResolvedValue(blocked);
    const uncertainty = vi.spyOn(api, 'runUncertaintyDemo');
    render(<MemoryRouter><UncertaintyPage /></MemoryRouter>);
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(screen.getByText(/not clinically validated/i)).toBeInTheDocument();
    expect(uncertainty).not.toHaveBeenCalled();
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
  });

  it('shows a useful empty result state rather than redirecting or restoring cached output', () => {
    analysisMocks.lastResult = null;
    render(<MemoryRouter><Results /></MemoryRouter>);
    expect(screen.getByRole('heading', { name: 'No current analysis result' })).toBeInTheDocument();
    expect(screen.getByText(/FOUR_CLASS_MODEL_NOT_READY/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Open research workbench/ })).toHaveAttribute('href', '/workbench');
  });

  it('withholds class probabilities and derived panels when uncertainty requires abstention', () => {
    analysisMocks.lastResult = {
      requestId: 'fixture-request', timestamp: new Date(0).toISOString(), modality: 'xray', imageUrl: '', gradcamUrl: '',
      prediction: { topClass: 'metastasis', confidence: 0.8, probabilities: { metastasis: 0.8, paget: 0.1, oi: 0.05, healthy: 0.05 }, uncertainty: 'high' },
      uncertaintyDetail: { entropy: 0.8, confidenceInterval: [0.6, 0.9], oodFlag: false, calibrationBin: { predicted: 0.8, observed: 0.75 } },
      shap: [], counterfactuals: [], fractureRisk: { level: 'medium', percent: 40 }, disclaimer: 'Synthetic test fixture only.',
    } satisfies AnalyzeResponse;
    render(<MemoryRouter><Results /></MemoryRouter>);
    expect(screen.getByRole('heading', { name: 'Result withheld' })).toBeInTheDocument();
    expect(screen.queryByText('Backend-reported probability values')).not.toBeInTheDocument();
    expect(screen.queryByText('Secondary bone metastasis')).not.toBeInTheDocument();
    analysisMocks.lastResult = null;
  });

  it('offers retry when model readiness is unavailable and never shows fake result data', async () => {
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
