import React from 'react';
import { Activity, ShieldAlert, Bone, Pill, Dumbbell, FileText, ChevronDown, RefreshCw } from 'lucide-react';
import { CaseStudy } from '../types/osteoporosis';
import { CLINICAL_CASE_STUDIES } from '../data/mockData';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onLoadCase: (caseStudy: CaseStudy) => void;
  onOpenReport: () => void;
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  onLoadCase,
  onOpenReport,
  onReset,
}) => {
  const [showCaseMenu, setShowCaseMenu] = React.useState(false);

  const tabs = [
    { id: 'frax', label: 'FRAX Calculator', icon: ShieldAlert },
    { id: 'dexa', label: 'DEXA Analyzer', icon: Activity },
    { id: 'imaging', label: 'Imaging & Trabeculae', icon: Bone },
    { id: 'pharma', label: 'Pharmacotherapy', icon: Pill },
    { id: 'remodeling', label: 'Bone Remodeling', icon: RefreshCw },
    { id: 'nutrition', label: 'Nutrition & Fall Risk', icon: Dumbbell },
  ];

  return (
    <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-teal-400 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-bold text-xl">
              <Bone className="w-6 h-6 transform rotate-45" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-white">Osteo<span className="text-cyan-400">Shield</span></span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">Clinical v2.4</span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">WHO Osteoporosis & Fracture Risk Decision Support</p>
            </div>
          </div>

          {/* Right Action buttons: Case selector & Report button */}
          <div className="flex items-center space-x-2 sm:space-x-3">
            {/* Case Studies Dropdown */}
            <div className="relative">
              <button
                onClick={() => setShowCaseMenu(!showCaseMenu)}
                className="flex items-center space-x-2 text-xs sm:text-sm font-medium px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
              >
                <span>Sample Patient Cases</span>
                <ChevronDown className="w-4 h-4 text-slate-400" />
              </button>

              {showCaseMenu && (
                <div
                  className="absolute right-0 mt-2 w-80 rounded-xl bg-slate-800/95 backdrop-blur-xl border border-slate-700 shadow-2xl py-2 z-50"
                  onClick={() => setShowCaseMenu(false)}
                >
                  <div className="px-3 py-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-700/60">
                    Clinical Scenarios
                  </div>
                  {CLINICAL_CASE_STUDIES.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => onLoadCase(c)}
                      className="w-full text-left px-3 py-2.5 hover:bg-cyan-950/40 hover:text-cyan-300 text-slate-200 text-xs transition border-b border-slate-700/30 last:border-0"
                    >
                      <div className="font-semibold text-white">{c.title}</div>
                      <div className="text-slate-400 truncate">{c.subtitle}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Reset Defaults */}
            <button
              onClick={onReset}
              title="Reset to default patient values"
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 border border-slate-700 transition"
            >
              <RefreshCw className="w-4 h-4" />
            </button>

            {/* Print/Export Clinical Summary */}
            <button
              onClick={onOpenReport}
              className="flex items-center space-x-1.5 text-xs sm:text-sm font-semibold px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-500 hover:from-cyan-500 hover:to-teal-400 text-white shadow-md shadow-cyan-600/20 transition"
            >
              <FileText className="w-4 h-4" />
              <span className="hidden sm:inline">Clinical Report</span>
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex overflow-x-auto no-scrollbar space-x-1 sm:space-x-2 py-2 border-t border-slate-800/60">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-xs sm:text-sm font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
