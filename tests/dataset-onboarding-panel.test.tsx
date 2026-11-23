import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import DatasetOnboardingPanel from '../src/components/DatasetOnboardingPanel';

const getSources = vi.fn();
vi.mock('../src/api/client', () => ({
  getOnboardingSources: (...args: unknown[]) => getSources(...args),
  registerOnboardingSource: vi.fn(),
  transitionOnboardingSource: vi.fn(),
  stageOnboardingSource: vi.fn(),
  verifyOnboardingSource: vi.fn(),
}));

describe('Stage 38 governance interface', () => {
  beforeEach(() => getSources.mockReset());

  it('displays acquisition, verification, integrity, governance and staging separately', async () => {
    getSources.mockResolvedValue({ sources: [{
      source_id: 'SRC_FIXTURE', name: 'Fixture source', version: 'v1', acquisition_status: 'ACQUIRED',
      verification_status: 'NOT_VERIFIED', integrity_status: 'NOT_CHECKED', staging_status: 'NOT_STAGED',
      governance_approved: false, package: null,
    }] });
    render(<DatasetOnboardingPanel />);
    fireEvent.change(screen.getByLabelText('Reviewer bearer token'), { target: { value: 'in-memory-test-token' } });
    fireEvent.click(screen.getByText('Load registry'));
    await waitFor(() => expect(screen.getByText('ACQUIRED')).toBeInTheDocument());
    expect(screen.getByText('NOT_VERIFIED')).toBeInTheDocument();
    expect(screen.getByText('NOT_CHECKED')).toBeInTheDocument();
    expect(screen.getByText('NOT_STAGED')).toBeInTheDocument();
    expect(screen.getByText('No')).toBeInTheDocument();
    expect(getSources).toHaveBeenCalledWith('in-memory-test-token');
  });

});
