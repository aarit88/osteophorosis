import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Bone, Info, Activity, Sparkles, Scale, History, Globe } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';
import { EngineStatusWidget } from './EngineStatusWidget';
import { QueueIndicator } from './QueueIndicator';
import { RecentAnalysesDrawer } from './RecentAnalysesDrawer';
import { useAnalysis } from '../context/AnalysisContext';
import { useTranslation } from 'react-i18next';

export const NavBar: React.FC = () => {
  const location = useLocation();
  const { t, i18n } = useTranslation();
  const {
    lastResult,
    engineMode,
    activeQueueCount,
    queuedJobsCount,
    recentAnalyses,
    setLastResult,
    clearRecentAnalyses,
  } = useAnalysis();

  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);

  const toggleLanguage = () => {
    const nextLang = i18n.language === 'en' ? 'hi' : 'en';
    i18n.changeLanguage(nextLang);
  };

  const navLinks = [
    { to: '/', label: t('nav.analysis', 'Analysis & Upload'), icon: Activity },
    ...(lastResult ? [{ to: '/results', label: t('nav.results', 'Latest Results'), icon: Sparkles }] : []),
    { to: '/audit', label: t('nav.audit', 'Audit Log (Art. 14)'), icon: Scale },
    { to: '/about', label: t('nav.about', 'Architecture & Model Info'), icon: Info },
  ];

  return (
    <>
      <nav className="sticky top-[37px] z-40 border-b border-slate-800/80 bg-slate-900/90 dark:bg-slate-950/90 light:bg-white/90 backdrop-blur-md transition-colors duration-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Brand Logo */}
            <NavLink
              to="/"
              className="flex items-center space-x-3 group focus:outline-none focus:ring-2 focus:ring-cyan-500 rounded-lg p-1"
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-teal-400 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-bold group-hover:scale-105 transition-transform duration-200">
                <Bone className="w-6 h-6 transform rotate-45" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="text-xl font-extrabold tracking-tight text-white dark:text-white light:text-slate-900">
                    Osteo<span className="text-cyan-400">Guard</span>
                  </span>
                  <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800 tracking-wider">
                    AI
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 hidden sm:block">Explainable Bone Pathology Classification</p>
              </div>
            </NavLink>

            {/* Nav Links with animated active indicator */}
            <div className="hidden md:flex items-center space-x-1 lg:space-x-2">
              {navLinks.map((link) => {
                const Icon = link.icon;
                const isActive = location.pathname === link.to;

                return (
                  <NavLink
                    key={link.to}
                    to={link.to}
                    className={`relative px-3.5 py-2 rounded-lg text-sm font-semibold transition-colors duration-150 flex items-center space-x-1.5 ${
                      isActive
                        ? 'text-cyan-400 dark:text-cyan-300 light:text-cyan-600'
                        : 'text-slate-400 hover:text-slate-200 dark:text-slate-400 dark:hover:text-slate-200 light:text-slate-600 light:hover:text-slate-900'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{link.label}</span>

                    {isActive && (
                      <motion.div
                        layoutId="navbar-underline"
                        className="absolute bottom-0 left-2 right-2 h-0.5 bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full"
                        transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                      />
                    )}
                  </NavLink>
                );
              })}
            </div>

            {/* Right utilities: Queue, History Drawer, Engine Widget, Theme, Language */}
            <div className="flex items-center space-x-2 sm:space-x-3">
              <QueueIndicator activeCount={activeQueueCount} queuedCount={queuedJobsCount} />

              {/* Recent Analyses Drawer Toggle */}
              <button
                type="button"
                onClick={() => setIsDrawerOpen(true)}
                className="relative p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800/80 border border-slate-800 transition"
                title="Recent Analyses Cache"
                aria-label="Open recent analyses history drawer"
              >
                <History className="w-4 h-4" />
                {recentAnalyses.length > 0 && (
                  <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-cyan-500 text-[9px] font-bold text-slate-950 flex items-center justify-center">
                    {recentAnalyses.length}
                  </span>
                )}
              </button>

              {/* Language Scaffold Toggle */}
              <button
                type="button"
                onClick={toggleLanguage}
                className="flex items-center space-x-1 text-xs font-mono px-2.5 py-1.5 rounded-xl border border-slate-800 bg-slate-900 hover:bg-slate-800 text-slate-300 transition"
                title="Toggle Language Locale (i18n scaffold)"
              >
                <Globe className="w-3.5 h-3.5 text-cyan-400" />
                <span className="uppercase font-bold">{i18n.language}</span>
              </button>

              <EngineStatusWidget mode={engineMode} />

              <ThemeToggle />
            </div>
          </div>

          {/* Mobile Navigation bar */}
          <div className="md:hidden flex items-center justify-around py-2 border-t border-slate-800/60">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = location.pathname === link.to;

              return (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={`flex items-center space-x-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold ${
                    isActive
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{link.label}</span>
                </NavLink>
              );
            })}
          </div>
        </div>
      </nav>

      <RecentAnalysesDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        recentList={recentAnalyses}
        onSelectResult={setLastResult}
        onClearHistory={clearRecentAnalyses}
      />
    </>
  );
};
