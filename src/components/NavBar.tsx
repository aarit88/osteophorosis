import React, { useState, useRef, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bone,
  Info,
  Activity,
  Sparkles,
  Scale,
  History,
  Globe,
  Cpu,
  Layers,
  Sliders,
  Compass,
  Eye,
  Database,
  FileCheck2,
  Lock,
  ChevronDown,
} from 'lucide-react';
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
  const [isEnginesOpen, setIsEnginesOpen] = useState<boolean>(false);
  const enginesDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        enginesDropdownRef.current &&
        !enginesDropdownRef.current.contains(event.target as Node)
      ) {
        setIsEnginesOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close dropdown on navigation
  useEffect(() => {
    setIsEnginesOpen(false);
  }, [location.pathname]);

  const toggleLanguage = () => {
    const nextLang = i18n.language === 'en' ? 'hi' : 'en';
    i18n.changeLanguage(nextLang);
  };

  const engineLinks = [
    { to: '/radiomics', label: 'Radiomics Engine (Stage 27)', icon: Sliders },
    { to: '/biomechanics', label: 'Biomechanics & Mirels (Stage 28)', icon: Compass },
    { to: '/explainability', label: 'Explainability & SHAP (Stage 29)', icon: Eye },
    { to: '/uncertainty', label: 'Uncertainty Triage (Stage 30)', icon: Activity },
  ];

  const isEngineActive = engineLinks.some((l) => location.pathname === l.to);

  return (
    <>
      <nav className="border-b border-slate-800/80 bg-slate-900/90 dark:bg-slate-950/90 light:bg-white/90 backdrop-blur-md transition-colors duration-200">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-2 h-14 sm:h-16">
            {/* Brand Logo */}
            <NavLink
              to="/"
              className="flex items-center space-x-2 sm:space-x-2.5 group focus:outline-none focus:ring-2 focus:ring-cyan-500 rounded-lg p-1 min-w-0"
            >
              <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-tr from-cyan-600 to-teal-400 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-bold group-hover:scale-105 transition-transform duration-200 shrink-0">
                <Bone className="w-4 h-4 sm:w-5 sm:h-5 transform rotate-45" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center space-x-1">
                  <span className="text-base sm:text-lg font-black tracking-tight text-white dark:text-white light:text-slate-900 truncate">
                    Osteo<span className="text-cyan-400">Guard</span>
                  </span>
                  <span className="text-[9px] uppercase font-bold px-1.5 py-0.2 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800 tracking-wider shrink-0">
                    AI
                  </span>
                </div>
                <p className="text-[10px] text-slate-400 hidden xl:block">Explainable Bone Pathology Classification</p>
              </div>
            </NavLink>

            {/* Desktop Nav Links */}
            <div className="hidden lg:flex items-center space-x-1">
              {/* Analysis & Upload */}
              <NavLink
                to="/"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`
                }
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Analysis</span>
              </NavLink>

              {/* Research Workbench */}
              <NavLink
                to="/workbench"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`
                }
              >
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                <span>Workbench</span>
              </NavLink>

              {/* Research Tracks */}
              <NavLink
                to="/tracks"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`
                }
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Tracks</span>
              </NavLink>

              {/* Engines Dropdown */}
              <div className="relative" ref={enginesDropdownRef}>
                <button
                  type="button"
                  onClick={() => setIsEnginesOpen(!isEnginesOpen)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isEngineActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`}
                >
                  <Sliders className="w-3.5 h-3.5" />
                  <span>Engines</span>
                  <ChevronDown className={`w-3 h-3 transition-transform ${isEnginesOpen ? 'rotate-180' : ''}`} />
                </button>

                <AnimatePresence>
                  {isEnginesOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 8 }}
                      transition={{ duration: 0.15 }}
                      className="absolute left-0 mt-2 w-64 rounded-xl bg-slate-900 border border-slate-800 shadow-xl shadow-black/50 py-1.5 z-50 space-y-0.5"
                    >
                      {engineLinks.map((eng) => {
                        const Icon = eng.icon;
                        const isSelected = location.pathname === eng.to;
                        return (
                          <NavLink
                            key={eng.to}
                            to={eng.to}
                            className={`flex items-center space-x-2.5 px-3 py-2 text-xs font-medium transition ${
                              isSelected
                                ? 'bg-cyan-950 text-cyan-300 border-l-2 border-cyan-400'
                                : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                            }`}
                          >
                            <Icon className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                            <span>{eng.label}</span>
                          </NavLink>
                        );
                      })}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Governance */}
              <NavLink
                to="/governance"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`
                }
              >
                <Database className="w-3.5 h-3.5" />
                <span>Governance</span>
              </NavLink>

              {/* Evidence Registry */}
              <NavLink
                to="/evidence"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`
                }
              >
                <FileCheck2 className="w-3.5 h-3.5" />
                <span>Evidence</span>
              </NavLink>

              {/* Safety Gate */}
              <NavLink
                to="/architecture"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-rose-300 bg-rose-950/50 border border-rose-800/60'
                      : 'text-slate-400 hover:text-rose-300 hover:bg-slate-850/50'
                  }`
                }
              >
                <Lock className="w-3.5 h-3.5 text-rose-400" />
                <span>Safety Gate</span>
              </NavLink>

              {/* Audit */}
              <NavLink
                to="/audit"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`
                }
              >
                <Scale className="w-3.5 h-3.5" />
                <span>Audit</span>
              </NavLink>

              {/* About */}
              <NavLink
                to="/about"
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors ${
                    isActive
                      ? 'text-cyan-300 bg-cyan-950/50 border border-cyan-800/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50'
                  }`
                }
              >
                <Info className="w-3.5 h-3.5" />
                <span>About</span>
              </NavLink>
            </div>

            {/* Right utilities */}
            <div className="flex items-center space-x-1.5 sm:space-x-3 shrink-0">
              <QueueIndicator activeCount={activeQueueCount} queuedCount={queuedJobsCount} />

              {/* Recent Analyses Drawer Toggle */}
              <button
                type="button"
                onClick={() => setIsDrawerOpen(true)}
                className="relative p-1.5 sm:p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800/80 border border-slate-800 transition"
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
                className="flex items-center space-x-1 text-xs font-mono px-2 py-1.5 rounded-xl border border-slate-800 bg-slate-900 hover:bg-slate-800 text-slate-300 transition"
                title="Toggle Language Locale (i18n scaffold)"
              >
                <Globe className="w-3.5 h-3.5 text-cyan-400" />
                <span className="uppercase font-bold">{i18n.language}</span>
              </button>

              <EngineStatusWidget mode={engineMode} />

              <ThemeToggle />
            </div>
          </div>

          {/* Mobile Secondary Navigation bar */}
          <div className="lg:hidden flex items-center overflow-x-auto space-x-2 py-2 border-t border-slate-800/60 text-xs no-scrollbar">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Activity className="w-3 h-3" />
              <span>Analysis</span>
            </NavLink>
            <NavLink
              to="/workbench"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Cpu className="w-3 h-3" />
              <span>Workbench</span>
            </NavLink>
            <NavLink
              to="/tracks"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Layers className="w-3 h-3" />
              <span>Tracks</span>
            </NavLink>
            <NavLink
              to="/radiomics"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Sliders className="w-3 h-3" />
              <span>Radiomics</span>
            </NavLink>
            <NavLink
              to="/biomechanics"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Compass className="w-3 h-3" />
              <span>Biomechanics</span>
            </NavLink>
            <NavLink
              to="/explainability"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Eye className="w-3 h-3" />
              <span>Explainability</span>
            </NavLink>
            <NavLink
              to="/uncertainty"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Activity className="w-3 h-3" />
              <span>Uncertainty</span>
            </NavLink>
            <NavLink
              to="/governance"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Database className="w-3 h-3" />
              <span>Governance</span>
            </NavLink>
            <NavLink
              to="/evidence"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <FileCheck2 className="w-3 h-3" />
              <span>Evidence</span>
            </NavLink>
            <NavLink
              to="/architecture"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' : 'text-slate-400'
                }`
              }
            >
              <Lock className="w-3 h-3" />
              <span>Safety Gate</span>
            </NavLink>
            <NavLink
              to="/audit"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Scale className="w-3 h-3" />
              <span>Audit</span>
            </NavLink>
            <NavLink
              to="/about"
              className={({ isActive }) =>
                `px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex items-center space-x-1 ${
                  isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400'
                }`
              }
            >
              <Info className="w-3 h-3" />
              <span>About</span>
            </NavLink>
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

export default NavBar;
