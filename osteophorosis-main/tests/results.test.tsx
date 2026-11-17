import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { GradCamCompare } from '../src/components/GradCamCompare';
import { UncertaintyCard } from '../src/components/UncertaintyCard';
import { AbstentionBanner } from '../src/components/AbstentionBanner';
import { AnalyzeResponse } from '../src/api/types';

describe('Results Subcomponents Rendering', () => {
  it('renders GradCamCompare with required caption', () => {
    render(
      <GradCamCompare
        originalUrl="/uploads/orig.png"
        gradcamUrl="/uploads/grad.png"
      />
    );

    expect(
      screen.getByText(/Simulated Grad-CAM — placeholder attention map/i)
    ).toBeInTheDocument();
  });

  it('renders UncertaintyCard with normalized entropy value', () => {
    const mockUncertainty = {
      entropy: 0.285,
      confidenceInterval: [0.72, 0.86] as [number, number],
      oodFlag: false,
      calibrationBin: { predicted: 0.8, observed: 0.78 },
    };

    const mockPrediction = {
      topClass: 'metastasis' as const,
      confidence: 0.82,
      probabilities: { metastasis: 0.82, paget: 0.08, oi: 0.05, healthy: 0.05 },
      uncertainty: 'low' as const,
    };

    render(
      <UncertaintyCard
        uncertaintyDetail={mockUncertainty}
        prediction={mockPrediction}
      />
    );

    expect(screen.getByText('0.285')).toBeInTheDocument();
    expect(screen.getByText(/Confident Prediction/i)).toBeInTheDocument();
  });

  it('renders AbstentionBanner when model uncertainty exceeds threshold', () => {
    render(<AbstentionBanner entropy={0.68} oodFlag={true} />);

    expect(
      screen.getByText(/Model Recommends Human Review — Primary Prediction Withheld/i)
    ).toBeInTheDocument();
  });
});
