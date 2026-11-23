import React, { useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { DisclaimerBanner } from './DisclaimerBanner';
import { NavBar } from './NavBar';
import { ROUTE_BY_PATH } from '../app/routes';

interface ApplicationShellProps {
  children: React.ReactNode;
}

export const ApplicationShell: React.FC<ApplicationShellProps> = ({ children }) => {
  const { pathname } = useLocation();
  const route = ROUTE_BY_PATH.get(pathname);

  useEffect(() => {
    document.getElementById('main-content')?.focus({ preventScroll: true });
  }, [pathname]);

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 font-sans text-slate-100 selection:bg-cyan-500/25 selection:text-cyan-200">
      <a href="#main-content" className="sr-only z-[60] rounded bg-cyan-700 px-4 py-3 font-semibold text-white focus:not-sr-only focus:fixed focus:left-4 focus:top-4">Skip to main content</a>
      <div className="sticky top-0 z-40"><DisclaimerBanner /><NavBar /></div>
      <main id="main-content" tabIndex={-1} className="mx-auto w-full max-w-7xl flex-1 px-4 py-4 focus:outline-none sm:px-6 sm:py-6 lg:px-8">
        {route && route.path !== '/' && <nav aria-label="Breadcrumb" className="mb-4 text-xs text-slate-400">
          <ol className="flex flex-wrap items-center gap-2">
            <li><Link to="/" className="rounded hover:text-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500">Home</Link></li>
            <li aria-hidden="true" className="text-slate-600">/</li>
            <li aria-current="page" className="font-medium text-slate-300">{route.title}</li>
          </ol>
        </nav>}
        {children}
      </main>
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-xs text-slate-500 sm:py-8">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 text-center sm:flex-row sm:px-6 sm:text-left lg:px-8">
          <div className="flex flex-wrap items-center justify-center gap-2 sm:justify-start"><span className="font-bold text-slate-300">OsteoGuard AI</span><span aria-hidden="true">•</span><span>Bone imaging research prototype</span></div>
          <nav aria-label="Footer navigation" className="flex flex-wrap justify-center gap-x-4 gap-y-2 sm:order-3 sm:justify-end">{[
            ['/about', 'About'], ['/workbench', 'Workbench'], ['/tracks', 'Tracks'], ['/governance', 'Governance'], ['/architecture', 'Architecture'], ['/evidence', 'Evidence'], ['/audit', 'Audit'],
          ].map(([path, label]) => <Link key={path} to={path} className="rounded text-slate-400 underline-offset-4 hover:text-cyan-300 hover:underline focus:outline-none focus:ring-2 focus:ring-cyan-400">{label}</Link>)}</nav>
          <p className="max-w-2xl text-center sm:order-2 sm:text-right">Research use only. Human review is required; this prototype does not provide a clinical diagnosis.</p>
        </div>
      </footer>
    </div>
  );
};

export default ApplicationShell;
