import React, { Suspense } from 'react';
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { AnalysisProvider } from './context/AnalysisContext';
import { ApplicationShell } from './components/ApplicationShell';
import { ErrorBoundary } from './components/ErrorBoundary';
import { RouteLoadingState } from './components/RouteLoadingState';
import { APPLICATION_ROUTES } from './app/routes';
import { NotFoundPage } from './pages/NotFoundPage';
import './i18n/config';

export const ApplicationRoutes: React.FC = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Suspense key={location.pathname} fallback={<RouteLoadingState />}>
        <Routes location={location} key={location.pathname}>
          {APPLICATION_ROUTES.map(({ path, component: Page }) => <Route key={path} path={path} element={<Page />} />)}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </AnimatePresence>
  );
};

export const App: React.FC = () => (
  <ErrorBoundary>
    <AnalysisProvider>
      <BrowserRouter>
        <ApplicationShell>
          <ApplicationRoutes />
        </ApplicationShell>
      </BrowserRouter>
    </AnalysisProvider>
  </ErrorBoundary>
);

export default App;
