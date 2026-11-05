import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { NavBar } from './components/NavBar';
import { Home } from './pages/Home';
import { AnalysisProvider } from './context/AnalysisContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { SkeletonCard } from './components/Loader';
import './i18n/config';

// Code-split pages for performance budget optimization
const Results = lazy(() => import('./pages/Results').then((m) => ({ default: m.Results })));
const About = lazy(() => import('./pages/About').then((m) => ({ default: m.About })));
const AuditPage = lazy(() => import('./pages/AuditPage').then((m) => ({ default: m.AuditPage })));

const AnimatedRoutes: React.FC = () => {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Suspense fallback={<div className="max-w-4xl mx-auto py-12"><SkeletonCard /></div>}>
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<Home />} />
          <Route path="/results" element={<Results />} />
          <Route path="/about" element={<About />} />
          <Route path="/audit" element={<AuditPage />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </Suspense>
    </AnimatePresence>
  );
};

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <AnalysisProvider>
        <BrowserRouter>
          <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/25 selection:text-cyan-200">
            {/* 1. Persistent, non-dismissible red warning banner */}
            <DisclaimerBanner />

            {/* 2. Sticky top navigation bar */}
            <NavBar />

            {/* 3. Main routed content */}
            <main id="main-content" className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
              <AnimatedRoutes />
            </main>

            {/* 4. Global Footer */}
            <footer className="border-t border-slate-900 bg-slate-950 py-8 text-xs text-slate-500">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-4">
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-slate-300">OsteoGuard AI</span>
                  <span>•</span>
                  <span>Explainable Bone Pathology Classification Prototype</span>
                </div>
                <div className="text-slate-500 text-center sm:text-right">
                  <span>EU AI Act Art. 14 Audit Trail • FDA GMLP Principles Supported</span>
                </div>
              </div>
            </footer>
          </div>
        </BrowserRouter>
      </AnalysisProvider>
    </ErrorBoundary>
  );
};

export default App;
