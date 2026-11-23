import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import DatasetSourceQualificationPanel from '../src/components/DatasetSourceQualificationPanel';

const getQualifications = vi.fn();
const getReports = vi.fn();
const getOnboarding = vi.fn();
const createSource = vi.fn();
vi.mock('../src/api/client', () => ({
  addSourceEvidence: vi.fn(), createAcquisitionPlan: vi.fn(), createSourceQualification: (...args: unknown[]) => createSource(...args),
  getOnboardingSources: (...args: unknown[]) => getOnboarding(...args), getSourceQualificationReports: (...args: unknown[]) => getReports(...args),
  getSourceQualifications: (...args: unknown[]) => getQualifications(...args), linkQualificationStage38: vi.fn(),
  reviewSourceQualification: vi.fn(), transitionAcquisitionPlan: vi.fn(),
}));

describe('Stage 40 source qualification UI', () => {
  beforeEach(() => { getQualifications.mockReset(); getReports.mockReset(); getOnboarding.mockReset(); createSource.mockReset(); });
  it('distinguishes candidate coverage and only registers a DISCOVERED source', async () => {
    getQualifications.mockResolvedValue({ sources: [] });
    getReports.mockResolvedValue({ coverage: { basis: 'CANDIDATE_SOURCE_RECORDS; NOT A PATIENT COHORT', actual_patient_counts: null }, feasibility: { actual_patient_count: null }, plans: [] });
    getOnboarding.mockResolvedValue({ sources: [] });
    createSource.mockResolvedValue({ source_id: 'candidate_fixture', qualification_status: 'DISCOVERED' });
    render(<DatasetSourceQualificationPanel />);
    fireEvent.change(screen.getByLabelText('Reviewer bearer token'), { target: { value: 'memory-token' } });
    fireEvent.click(screen.getByText('Load qualification records'));
    await waitFor(() => expect(screen.getByText('Candidate registry loaded. Candidate coverage is not an acquired cohort.')).toBeInTheDocument());
    expect(screen.getByText(/not an acquired cohort/i)).toBeInTheDocument();
    fireEvent.click(screen.getByText('Register a candidate source as DISCOVERED'));
    const field = screen.getByLabelText('Candidate source JSON');
    fireEvent.change(field, { target: { value: JSON.stringify({ source_id: 'candidate_fixture', source_name: 'Synthetic fixture', original_url: 'https://example.invalid', repository_or_institution: 'Fixture', qualification_status: 'DISCOVERED' }) } });
    fireEvent.click(screen.getByText('Register candidate'));
    await waitFor(() => expect(createSource).toHaveBeenCalledWith('memory-token', expect.objectContaining({ source_id: 'candidate_fixture', qualification_status: 'DISCOVERED' })));
  });
});
