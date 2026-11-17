import React, { useState } from 'react';
import { EngineMode } from '../engine';
import { Cpu, CheckCircle2, AlertTriangle, Info, X } from 'lucide-react';

interface EngineStatusWidgetProps {
  mode: EngineMode;
  onRefresh?: () => void;
}

export const EngineStatusWidget: React.FC<EngineStatusWidgetProps> = ({ mode }) => {
  const [showInfo, setShowInfo] = useState<boolean>(false);

  const config = {
    live: {
      label: 'Live PyTorch',
      dotColor: 'bg-emerald-400',
      badgeClass: 'bg-emerald-950/70 border-emerald-800 text-emerald-300',
      desc: 'Connected to active production ML inference server via VITE_API_BASE_URL_LIVE.',
    },
    mock: {
      label: 'Simulation Mock',
      dotColor: 'bg-cyan-400',
      badgeClass: 'bg-cyan-950/70 border-cyan-800 text-cyan-300',
      desc: 'Running deterministic SHA-256 seeded simulation engine on port 3000 / 8787.',
    },
    fallback: {
      label: 'Fallback: Simulation',
      dotColor: 'bg-amber-400',
      badgeClass: 'bg-amber-950/70 border-amber-800 text-amber-300',
      desc: 'Live backend was unconfigured or unreachable. Auto-switched to simulation mock.',
    },
  }[mode] || {
    label: 'Engine: Auto',
    dotColor: 'bg-slate-400',
    badgeClass: 'bg-slate-900 border-slate-800 text-slate-300',
    desc: 'Automatic engine resolution.',
  };

  return (
    <>
      <button
        onClick={() => setShowInfo(true)}
        type="button"
        className={`flex items-center space-x-1.5 text-[11px] font-mono px-2.5 py-1 rounded-full border shadow-sm transition hover:scale-105 ${config.badgeClass}`}
        title="Prediction Engine Status"
        aria-label={`Inference engine status: ${config.label}`}
      >
        <span className={`w-2 h-2 rounded-full ${config.dotColor} animate-pulse`} />
        <span>{config.label}</span>
      </button>

      {showInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="p-6 rounded-3xl bg-slate-900 border border-slate-700 max-w-sm w-full shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Cpu className="w-5 h-5 text-cyan-400" />
                <h4 className="text-sm font-bold text-white">Prediction Engine Architecture</h4>
              </div>
              <button
                onClick={() => setShowInfo(false)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-slate-400">Current Mode:</span>
                <span className="font-bold text-white uppercase font-mono">{mode}</span>
              </div>
              <p className="text-slate-300 leading-relaxed text-[11px]">{config.desc}</p>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-[10px] text-slate-400 font-mono space-y-1">
              <div>VITE_ENGINE: &quot;mock&quot; | &quot;live&quot; | &quot;auto&quot;</div>
              <div>VITE_API_BASE_URL_LIVE: Real backend URL</div>
            </div>

            <button
              onClick={() => setShowInfo(false)}
              className="w-full py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs transition"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
};
