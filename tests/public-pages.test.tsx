import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Home } from '../src/pages/Home';
import { About } from '../src/pages/About';
import { ArchitecturePage } from '../src/pages/ArchitecturePage';
import { APPLICATION_ROUTES } from '../src/app/routes';
import type { ArchitectureReadinessResponse } from '../src/api/types';

const mocks = vi.hoisted(() => ({
  getArchitectureReadiness: vi.fn(),
  getScalabilityStatus: vi.fn(),
  enqueueAnalyzeJob: vi.fn(),
  getJobsList: vi.fn(),
  pollAnalyzeJob: vi.fn(),
}));

vi.mock('../src/api/client', () => ({
  getArchitectureReadiness: mocks.getArchitectureReadiness,
  getScalabilityStatus: mocks.getScalabilityStatus,
  enqueueAnalyzeJob: mocks.enqueueAnalyzeJob,
  getJobsList: mocks.getJobsList,
  pollAnalyzeJob: mocks.pollAnalyzeJob,
}));

const readinessFixture: ArchitectureReadinessResponse = {
  timestamp: '2026-11-23T00:00:00Z',
  system_readiness: 'ENGINEERING_COMPONENTS_VERIFIED__CLINICAL_MODEL_BLOCKED',
  components: [{
    component_name: 'Track D classifier head',
    layer_group: 'Inference and output',
    status: 'BLOCKED_BY_DATA',
    implemented_features: 'The output ontology is defined.',
    unresolved_gaps: 'No validated four-class model is available.',
    governance_dependency: 'Authorized clinical cohorts and validation.',
  }],
  fusion_contract_status: 'Specified, not verified',
  inference_eligibility: 'STRICTLY_BLOCKED',
  disclaimer: 'Research engineering status only.',
};

const renderAt = (path: string, element: React.ReactNode) => render(<MemoryRouter initialEntries={[path]}>{element}</MemoryRouter>);

describe('public project pages', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getArchitectureReadiness.mockResolvedValue(readinessFixture);
  });

  it('renders the homepage diagnostic workbench with modality selector, upload zone, sample presets, and honest clinical boundaries', () => {
    renderAt('/', <Home />);
    expect(screen.getByRole('heading', { name: /Bone Disorder Differential/ })).toBeInTheDocument();
    const modalityGroup = screen.getByRole('group', { name: /Select imaging modality/i });
    expect(within(modalityGroup).getByRole('button', { name: /2D Digital X-ray|X-ray/i })).toBeInTheDocument();
    expect(within(modalityGroup).getByRole('button', { name: /CT Scan/i })).toBeInTheDocument();
    expect(within(modalityGroup).getByRole('button', { name: /MRI/i })).toBeInTheDocument();
    expect(within(modalityGroup).getByRole('button', { name: /Bone Scan/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run Bone Diagnosis/i })).toBeDisabled();
    expect(screen.getAllByText('FOUR_CLASS_MODEL_NOT_READY').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/SECONDARY_BONE_METASTASIS/)).toBeInTheDocument();
    expect(screen.getByText(/PAGETS_DISEASE/)).toBeInTheDocument();
    expect(screen.getByText(/OSTEOGENESIS_IMPERFECTA/)).toBeInTheDocument();
    expect(screen.queryByText(/patients analyzed|FDA approval/i)).not.toBeInTheDocument();
  });

  it('explains motivation, ontology, modality scope, current implementation, and limits on About', () => {
    renderAt('/about', <About />);
    expect(screen.getByRole('heading', { name: 'About OsteoGuard AI' })).toBeInTheDocument();
    for (const identifier of ['SECONDARY_BONE_METASTASIS', 'PAGETS_DISEASE', 'OSTEOGENESIS_IMPERFECTA', 'HEALTHY_SKELETAL_CONTROL']) {
      expect(screen.getByText(identifier)).toBeInTheDocument();
    }
    expect(screen.getByText('Bone scintigraphy')).toBeInTheDocument();
    expect(screen.getByText(/not clinically ready/i)).toBeInTheDocument();
    expect(screen.queryByText(/clinical accuracy|FDA approved|hospital partner/i)).not.toBeInTheDocument();
  });

  it('renders the actual app layers, API boundary, and live readiness data on Architecture', async () => {
    renderAt('/architecture', <ArchitecturePage />);
    expect(screen.getByRole('heading', { name: /layered research platform/i })).toBeInTheDocument();
    expect(screen.getByText('React · TypeScript · Vite')).toBeInTheDocument();
    expect(screen.getByText('Node.js · Express 5')).toBeInTheDocument();
    expect(screen.getByText('FastAPI · Python')).toBeInTheDocument();
    expect(screen.getByText(/FastAPI is not the public web entry point/i)).toBeInTheDocument();
    expect(await screen.findByText('Track D classifier head')).toBeInTheDocument();
    expect(screen.getByText('Blocked by data')).toBeInTheDocument();
  });

  it('shows unavailable readiness and retries without inventing a status', async () => {
    mocks.getArchitectureReadiness.mockRejectedValueOnce(new Error('private connection detail'))
      .mockResolvedValueOnce(readinessFixture);
    renderAt('/architecture', <ArchitecturePage />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Current readiness is unavailable');
    expect(screen.queryByText(/private connection detail/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.getByText('Track D classifier head')).toBeInTheDocument());
  });

  it('explains an empty readiness response without fabricating component rows', async () => {
    mocks.getArchitectureReadiness.mockResolvedValueOnce({ ...readinessFixture, components: [] });
    renderAt('/architecture', <ArchitecturePage />);
    expect(await screen.findByText('The readiness service returned no component records. Refresh to check again; no component status is being inferred.')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('navigates from About to Architecture using the existing internal route', async () => {
    render(<MemoryRouter initialEntries={['/about']}><Routes>
      <Route path="/about" element={<About />} />
      <Route path="/architecture" element={<ArchitecturePage />} />
    </Routes></MemoryRouter>);
    fireEvent.click(screen.getByRole('link', { name: 'View architecture' }));
    expect(await screen.findByRole('heading', { name: /layered research platform/i })).toBeInTheDocument();
  });

  it('keeps every public-page internal link on a registered application route', () => {
    const registered = new Set(APPLICATION_ROUTES.map((route) => route.path));
    const pages = [<Home key="home" />, <About key="about" />, <ArchitecturePage key="architecture" />];
    for (const page of pages) {
      const view = render(<MemoryRouter>{page}</MemoryRouter>);
      const links = within(view.container).getAllByRole('link');
      for (const link of links) expect(registered.has(link.getAttribute('href') || '')).toBe(true);
      view.unmount();
    }
  });
});
