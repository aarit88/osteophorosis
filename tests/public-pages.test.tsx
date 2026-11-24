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
  requestFourClassInference: vi.fn(),
  executeAnalysisWithQueue: vi.fn(),
  setError: vi.fn(),
}));

vi.mock('../src/api/client', () => ({
  getArchitectureReadiness: mocks.getArchitectureReadiness,
  requestFourClassInference: mocks.requestFourClassInference,
}));

vi.mock('../src/context/AnalysisContext', () => ({
  useAnalysis: () => ({ executeAnalysisWithQueue: mocks.executeAnalysisWithQueue, isAnalyzing: false, error: null, setError: mocks.setError }),
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
    mocks.requestFourClassInference.mockResolvedValue({
      status: 'BLOCKED',
      error_code: 'FOUR_CLASS_MODEL_NOT_READY',
      message: 'Four-class inference remains unavailable.',
      target_classes: [],
      blockers: ['No validated clinical model.'],
      mandatory_safeguard: 'Inference is disabled.',
    });
  });

  it('renders the homepage with accurate scope, direct Workbench and Tracks links, and no invented metrics', async () => {
    renderAt('/', <Home />);
    await screen.findByRole('button', { name: 'View status' });
    expect(screen.getByRole('heading', { name: /A research framework for bone imaging/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Open research workbench/ })).toHaveAttribute('href', '/workbench');
    expect(screen.getByRole('link', { name: /Explore research tracks/ })).toHaveAttribute('href', '/tracks');
    expect(screen.getByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(screen.getByText(/SECONDARY_BONE_METASTASIS/)).toBeInTheDocument();
    expect(screen.getByText(/PAGETS_DISEASE/)).toBeInTheDocument();
    expect(screen.queryByText(/accuracy|patients analyzed|clinical approval/i)).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Four target categories, three distinct modalities' }).closest('section')).toHaveClass('lg:grid-cols-[1fr_.8fr]');
    expect(screen.getByRole('heading', { name: 'Image analysis availability' }).closest('section')).toHaveClass('scroll-mt-24');
  });

  it('keeps image submission disabled when the backend readiness gate is blocked', async () => {
    renderAt('/', <Home />);
    fireEvent.click(screen.getByRole('button', { name: 'View status' }));
    expect(await screen.findByText('FOUR_CLASS_MODEL_NOT_READY')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run authorized analysis' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Review governance readiness' })).toHaveAttribute('href', '/governance');
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
