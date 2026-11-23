import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import DatasetPreparationPanel from '../src/components/DatasetPreparationPanel';

const getSources = vi.fn();
const getRuns = vi.fn();
vi.mock('../src/api/client', () => ({
  getOnboardingSources: (...args: unknown[]) => getSources(...args),
  getDatasetPreparationRuns: (...args: unknown[]) => getRuns(...args),
  createDatasetPreparationRun: vi.fn(),
  advanceDatasetPreparationRun: vi.fn(),
}));

describe('Stage 39 preparation interface', () => {
  beforeEach(() => { getSources.mockReset(); getRuns.mockReset(); });

  it('shows staged source and preparation gates without claiming readiness', async () => {
    getSources.mockResolvedValue({ sources: [{ source_id: 'SRC_FIXTURE', name: 'Synthetic test source', version: 'v1', verification_status: 'VERIFIED', staging_status: 'STAGED_FOR_STAGE33_INTAKE', package: { valid: true } }] });
    getRuns.mockResolvedValue({ runs: [{ run_id: 'prep_fixture', source_id: 'SRC_FIXTURE', source_version: 'v1', state: 'QC_COMPLETED', validation_status: 'PASSED', config_version: '1.0.0', inventory: { file_count: 2, inventory_sha256: 'a'.repeat(64), missing_metadata: [] }, qc: { status: 'PASSED', finding_count: 2, blocker_count: 0 }, blockers: [] }] });
    render(<DatasetPreparationPanel />);
    fireEvent.change(screen.getByLabelText('Reviewer bearer token'), { target: { value: 'memory-only-token' } });
    fireEvent.click(screen.getByText('Load preparation registry'));
    await waitFor(() => expect(screen.getByText('QC_COMPLETED')).toBeInTheDocument());
    expect(screen.getByText(/does not establish clinical validity/)).toBeInTheDocument();
    expect(screen.getByText(/Inventory: 2 files/)).toBeInTheDocument();
    expect(getRuns).toHaveBeenCalledWith('memory-only-token');
    expect(screen.getByText('Harmonize images')).toBeInTheDocument();
  });
});
