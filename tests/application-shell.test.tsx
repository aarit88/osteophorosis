import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { ApplicationRoutes } from '../src/App';
import { ApplicationShell } from '../src/components/ApplicationShell';
import { NavBar } from '../src/components/NavBar';
import { APPLICATION_ROUTES, ROUTE_BY_PATH } from '../src/app/routes';

vi.mock('../src/context/AnalysisContext', () => ({
  useAnalysis: () => ({
    engineMode: 'mock',
    activeQueueCount: 0,
    queuedJobsCount: 0,
    recentAnalyses: [],
    setLastResult: vi.fn(),
    clearRecentAnalyses: vi.fn(),
  }),
}));

vi.mock('../src/components/EngineStatusWidget', () => ({ EngineStatusWidget: () => <div>Engine status</div> }));
vi.mock('../src/components/QueueIndicator', () => ({ QueueIndicator: () => <div>Queue</div> }));
vi.mock('../src/components/RecentAnalysesDrawer', () => ({ RecentAnalysesDrawer: () => null }));

describe('application route registry and shell', () => {
  it('keeps a unique registry for every implemented route', () => {
    const paths = APPLICATION_ROUTES.map(({ path }) => path);
    expect(new Set(paths).size).toBe(paths.length);
    expect(ROUTE_BY_PATH.get('/governance')?.title).toBe('Governance');
    expect(ROUTE_BY_PATH.has('/results')).toBe(true);
  });

  it('renders a real not-found page for unknown routes', async () => {
    render(<MemoryRouter initialEntries={['/this-page-does-not-exist']}><ApplicationRoutes /></MemoryRouter>);
    expect(await screen.findByRole('heading', { name: 'This route is not part of OsteoGuard' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Go to analysis' })).toHaveAttribute('href', '/');
  });

  it('provides page context and a keyboard skip link in the shared shell', () => {
    render(<MemoryRouter initialEntries={['/governance']}><ApplicationShell><h1>Governance content</h1></ApplicationShell></MemoryRouter>);
    expect(screen.getByRole('link', { name: 'Skip to main content' })).toHaveAttribute('href', '#main-content');
    expect(screen.getByRole('navigation', { name: 'Breadcrumb' })).toHaveTextContent('Home/Governance');
    expect(screen.getByRole('main')).toContainElement(screen.getByRole('heading', { name: 'Governance content' }));
    const footerLinks = within(screen.getByRole('navigation', { name: 'Footer navigation' })).getAllByRole('link');
    for (const link of footerLinks) expect(ROUTE_BY_PATH.has(link.getAttribute('href') || '')).toBe(true);
  });

  it('opens the compact mobile navigation and exposes implemented page routes', async () => {
    render(<MemoryRouter initialEntries={['/tracks']}><NavBar /></MemoryRouter>);
    const menu = screen.getByRole('button', { name: 'Open navigation menu' });
    fireEvent.click(menu);
    expect(screen.getByRole('button', { name: 'Close navigation menu' })).toHaveAttribute('aria-expanded', 'true');
    const mobileNavigation = screen.getByRole('navigation', { name: 'Primary navigation' }).querySelector('#mobile-primary-navigation');
    expect(mobileNavigation).not.toBeNull();
    const mobile = within(mobileNavigation as HTMLElement);
    expect(mobile.getByRole('link', { name: 'Governance' })).toHaveAttribute('href', '/governance');
    expect(mobile.getByRole('link', { name: 'Research tracks' })).toHaveAttribute('href', '/tracks');
    expect(mobile.getByRole('link', { name: 'Research tracks' })).toHaveAttribute('aria-current', 'page');
    expect(mobile.queryByRole('link', { name: 'Results' })).not.toBeInTheDocument();
  });
});
