import React, { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Bone, ChevronDown, Globe, History, Menu, X } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';
import { EngineStatusWidget } from './EngineStatusWidget';
import { QueueIndicator } from './QueueIndicator';
import { RecentAnalysesDrawer } from './RecentAnalysesDrawer';
import { useAnalysis } from '../context/AnalysisContext';
import { useTranslation } from 'react-i18next';
import { APPLICATION_ROUTES } from '../app/routes';

const visibleRoutes = APPLICATION_ROUTES.filter((route) => route.navigation !== 'hidden');
const primaryRoutes = visibleRoutes.filter((route) => route.navigation === 'primary');
const researchRoutes = visibleRoutes.filter((route) => route.navigation === 'research');
const supportRoutes = visibleRoutes.filter((route) => route.navigation === 'support');

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `inline-flex items-center gap-1.5 rounded-lg px-2.5 py-2 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
    isActive ? 'border border-cyan-800/60 bg-cyan-950/50 text-cyan-300' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
  }`;

export const NavBar: React.FC = () => {
  const location = useLocation();
  const { i18n } = useTranslation();
  const {
    engineMode,
    activeQueueCount,
    queuedJobsCount,
    recentAnalyses,
    setLastResult,
    clearRecentAnalyses,
  } = useAnalysis();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isResearchOpen, setIsResearchOpen] = useState(false);
  const [isSupportOpen, setIsSupportOpen] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  useEffect(() => {
    setIsResearchOpen(false);
    setIsSupportOpen(false);
    setIsMobileOpen(false);
  }, [location.pathname]);

  const toggleLanguage = () => {
    void i18n.changeLanguage(i18n.language === 'en' ? 'hi' : 'en');
  };
  const hasActiveResearch = researchRoutes.some((route) => route.path === location.pathname);
  const hasActiveSupport = supportRoutes.some((route) => route.path === location.pathname);

  return (
    <>
      <nav aria-label="Primary navigation" className="border-b border-slate-800/80 bg-slate-900/95 backdrop-blur-md transition-colors dark:bg-slate-950/95 light:bg-white/95">
        <div className="mx-auto max-w-7xl px-3 sm:px-6 lg:px-8">
          <div className="flex min-h-14 items-center justify-between gap-2 sm:min-h-16">
            <NavLink to="/" end aria-label="OsteoGuard AI home" className="group flex min-w-0 items-center gap-2 rounded-lg p-1 focus:outline-none focus:ring-2 focus:ring-cyan-500 sm:gap-2.5">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-700 to-teal-400 text-white shadow-lg shadow-cyan-500/20 transition-transform group-hover:scale-105 sm:h-9 sm:w-9">
                <Bone className="h-4 w-4 rotate-45 sm:h-5 sm:w-5" aria-hidden="true" />
              </span>
              <span className="min-w-0">
                <span className="flex items-center gap-1 text-base font-black tracking-tight text-white dark:text-white light:text-slate-900 sm:text-lg">
                  <span className="truncate">Osteo<span className="text-cyan-400">Guard</span></span>
                  <span className="shrink-0 rounded-full border border-cyan-800 bg-cyan-950/80 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-cyan-300">AI</span>
                </span>
                <span className="hidden text-[10px] text-slate-400 xl:block">Bone pathology research prototype</span>
              </span>
            </NavLink>

            <div className="hidden items-center gap-1 lg:flex" aria-label="Application pages">
              {primaryRoutes.map((route) => {
                const Icon = route.icon;
                return <NavLink key={route.path} to={route.path} end={route.path === '/'} className={linkClass} title={route.description}><Icon className="h-3.5 w-3.5" aria-hidden="true" /><span>{route.title}</span></NavLink>;
              })}
              <div className="relative">
                <button type="button" aria-haspopup="true" aria-expanded={isResearchOpen} onClick={() => { setIsResearchOpen((open) => !open); setIsSupportOpen(false); }} className={`${linkClass({ isActive: hasActiveResearch })}`}>
                  Research tools <ChevronDown className={`h-3 w-3 transition-transform ${isResearchOpen ? 'rotate-180' : ''}`} aria-hidden="true" />
                </button>
                <AnimatePresence>
                  {isResearchOpen && <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} className="absolute right-0 z-50 mt-2 w-64 rounded-xl border border-slate-800 bg-slate-900 p-1.5 shadow-xl" role="menu" aria-label="Research tools">
                    {researchRoutes.map((route) => { const Icon = route.icon; return <NavLink key={route.path} role="menuitem" to={route.path} className="flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"><Icon className="h-4 w-4 text-cyan-400" aria-hidden="true" />{route.title}</NavLink>; })}
                  </motion.div>}
                </AnimatePresence>
              </div>
              <div className="relative">
                <button type="button" aria-haspopup="true" aria-expanded={isSupportOpen} onClick={() => { setIsSupportOpen((open) => !open); setIsResearchOpen(false); }} className={linkClass({ isActive: hasActiveSupport })}>
                  More <ChevronDown className={`h-3 w-3 transition-transform ${isSupportOpen ? 'rotate-180' : ''}`} aria-hidden="true" />
                </button>
                <AnimatePresence>
                  {isSupportOpen && <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} className="absolute right-0 z-50 mt-2 w-52 rounded-xl border border-slate-800 bg-slate-900 p-1.5 shadow-xl" role="menu" aria-label="Project information">
                    {supportRoutes.map((route) => { const Icon = route.icon; return <NavLink key={route.path} role="menuitem" to={route.path} className="flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"><Icon className="h-4 w-4 text-cyan-400" aria-hidden="true" />{route.title}</NavLink>; })}
                  </motion.div>}
                </AnimatePresence>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-1 sm:gap-2">
              <QueueIndicator activeCount={activeQueueCount} queuedCount={queuedJobsCount} />
              <button type="button" onClick={() => setIsDrawerOpen(true)} className="relative rounded-xl border border-slate-800 p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-white sm:p-2" title="Recent analyses cache" aria-label="Open recent analyses history drawer">
                <History className="h-4 w-4" aria-hidden="true" />
                {recentAnalyses.length > 0 && <span className="absolute -right-1 -top-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-cyan-500 px-0.5 text-[9px] font-bold text-slate-950">{recentAnalyses.length}</span>}
              </button>
              <button type="button" onClick={toggleLanguage} className="hidden items-center gap-1 rounded-xl border border-slate-800 bg-slate-900 px-2 py-1.5 text-xs font-mono text-slate-300 transition hover:bg-slate-800 sm:flex" title="Change interface language" aria-label={`Change language, current language ${i18n.language}`}>
                <Globe className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" /><span className="font-bold uppercase">{i18n.language}</span>
              </button>
              <div className="hidden xl:block"><EngineStatusWidget mode={engineMode} /></div>
              <div className="hidden sm:block"><ThemeToggle /></div>
              <button type="button" className="rounded-lg border border-slate-700 p-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500 lg:hidden" aria-expanded={isMobileOpen} aria-controls="mobile-primary-navigation" aria-label={isMobileOpen ? 'Close navigation menu' : 'Open navigation menu'} onClick={() => setIsMobileOpen((open) => !open)}>
                {isMobileOpen ? <X className="h-4 w-4" aria-hidden="true" /> : <Menu className="h-4 w-4" aria-hidden="true" />}
              </button>
            </div>
          </div>

          <AnimatePresence>
            {isMobileOpen && <motion.div id="mobile-primary-navigation" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden border-t border-slate-800/70 py-3 lg:hidden">
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3" aria-label="Application pages">
                {visibleRoutes.map((route) => {
                  const Icon = route.icon;
                  return <NavLink key={route.path} to={route.path} end={route.path === '/'} className={({ isActive }) => `flex min-h-11 items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs font-semibold transition focus:outline-none focus:ring-2 focus:ring-cyan-500 ${isActive ? 'border-cyan-700 bg-cyan-950/60 text-cyan-200' : 'border-slate-800 bg-slate-900/70 text-slate-300 hover:bg-slate-800'}`}><Icon className="h-4 w-4 shrink-0 text-cyan-400" aria-hidden="true" /><span>{route.title}</span></NavLink>;
                })}
              </div>
              <div className="mt-3 flex items-center justify-between border-t border-slate-800 pt-3 sm:hidden">
                <button type="button" onClick={toggleLanguage} className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-slate-800 px-3 text-xs text-slate-300"><Globe className="h-4 w-4 text-cyan-400" aria-hidden="true" />Language: {i18n.language.toUpperCase()}</button>
                <ThemeToggle />
              </div>
              <div className="mt-3 xl:hidden"><EngineStatusWidget mode={engineMode} /></div>
            </motion.div>}
          </AnimatePresence>
        </div>
      </nav>
      <RecentAnalysesDrawer isOpen={isDrawerOpen} onClose={() => setIsDrawerOpen(false)} recentList={recentAnalyses} onSelectResult={setLastResult} onClearHistory={clearRecentAnalyses} />
    </>
  );
};

export default NavBar;
