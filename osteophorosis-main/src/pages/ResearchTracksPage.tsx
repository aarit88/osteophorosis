import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Layers,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Lock,
  Activity,
  FileText,
  BarChart2,
  Database,
  ArrowRight,
} from 'lucide-react';
import { getResearchTracks } from '../api/client';
import { ResearchTracksResponse, ResearchTrackDetail } from '../api/types';

export const ResearchTracksPage: React.FC = () => {
  const [data, setData] = useState<ResearchTracksResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getResearchTracks()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load research tracks');
        setLoading(false);
      });
  }, []);

  const getStatusBadge = (status: string) => {
    if (status.includes('BLOCKED')) {
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-rose-950/80 text-rose-300 border border-rose-800">
          <Lock className="w-3 h-3" />
          <span>BLOCKED BY CLINICAL DATA</span>
        </span>
      );
    }
    if (status.includes('LIMITED')) {
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-950/80 text-amber-300 border border-amber-800">
          <AlertTriangle className="w-3 h-3" />
          <span>GENERALIZATION LIMITED</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-800">
        <CheckCircle2 className="w-3 h-3" />
        <span>HISTORICAL FROZEN RESEARCH</span>
      </span>
    );
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="space-y-3 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-xs font-semibold">
          <Layers className="w-3.5 h-3.5" />
          <span>Research Governance & Track Separation</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Discrete Research Tracks
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          OsteoGuard AI maintains four strictly segregated research tracks. Sub-track models (FracAtlas, Bone Scintigraphy, BTXRD) must <strong className="text-rose-400">never</strong> be conflated with or mapped into the final four-class differential diagnosis.
        </p>
      </div>

      {loading && (
        <div className="p-8 text-center text-slate-400 animate-pulse">
          Loading research track registry...
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 gap-6">
          {data.tracks.map((track: ResearchTrackDetail, idx: number) => {
            const isTrackD = track.track_id === 'track_d';
            const isTrackC = track.track_id === 'track_c';

            return (
              <motion.div
                key={track.track_id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className={`p-6 rounded-2xl border transition-all ${
                  isTrackD
                    ? 'bg-rose-950/20 border-rose-900/60 shadow-lg shadow-rose-950/30'
                    : isTrackC
                    ? 'bg-amber-950/15 border-amber-900/50'
                    : 'bg-slate-900/60 border-slate-800'
                }`}
              >
                <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                        {track.track_id.toUpperCase().replace('_', ' ')}
                      </span>
                      <span>•</span>
                      <span className="text-xs text-slate-400">{track.modality}</span>
                    </div>
                    <h2 className="text-xl font-extrabold text-white">{track.title}</h2>
                    <p className="text-xs text-slate-400 mt-0.5">{track.task_type}</p>
                  </div>
                  <div>{getStatusBadge(track.status)}</div>
                </div>

                {/* Target Classes */}
                <div className="mb-4">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1.5">
                    Target Output Classes:
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {track.target_classes.map((cls, cIdx) => (
                      <span
                        key={cIdx}
                        className="px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-800 text-slate-200 border border-slate-700/60"
                      >
                        {cls}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Historical Metrics Grid */}
                {track.historical_metrics && (
                  <div className="mb-4 p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
                    <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider block mb-2">
                      Historical Validation & Evaluation Evidence:
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                      {Object.entries(track.historical_metrics).map(([k, v]) => (
                        <div key={k} className="p-2 rounded-lg bg-slate-900/80 border border-slate-800">
                          <span className="text-slate-400 block text-[11px] capitalize">
                            {k.replace(/_/g, ' ')}
                          </span>
                          <span className="font-bold text-white text-sm">
                            {typeof v === 'number'
                              ? v < 1
                                ? `${(v * 100).toFixed(1)}%`
                                : v.toLocaleString()
                              : typeof v === 'object'
                              ? JSON.stringify(v)
                              : String(v)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Limitations & Disclosures */}
                <div className="space-y-2 mb-4">
                  <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider block">
                    Verified Limitations & Scientific Disclosures:
                  </span>
                  <ul className="space-y-1 text-xs text-slate-300 list-disc list-inside">
                    {track.limitations.map((lim, lIdx) => (
                      <li key={lIdx} className="leading-relaxed">
                        {lim}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Disclaimer banner */}
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-400 italic">
                  <strong>Governance Clearance:</strong> {track.governance_clearance}
                  <br />
                  <strong>Notice:</strong> {track.disclaimer}
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
};
export default ResearchTracksPage;

