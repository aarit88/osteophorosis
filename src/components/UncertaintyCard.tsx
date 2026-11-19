import React from 'react';
import { motion } from 'framer-motion';
import { UncertaintyDetail, PredictionResult } from '../api/types';
import { AlertTriangle, CheckCircle2, HelpCircle, ShieldAlert, Activity } from 'lucide-react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts';

interface UncertaintyCardProps {
  uncertaintyDetail: UncertaintyDetail;
  prediction: PredictionResult;
}

export const UncertaintyCard: React.FC<UncertaintyCardProps> = ({ uncertaintyDetail, prediction }) => {
  const { entropy, confidenceInterval, oodFlag, calibrationBin } = uncertaintyDetail;

  const isLowConfidence = entropy > 0.6 || oodFlag;
  const isCautious = !isLowConfidence && entropy > 0.3;
  const isConfident = !isLowConfidence && !isCautious;

  const statusConfig = isLowConfidence
    ? {
        label: 'Low Confidence — Refer for Review',
        colorText: 'text-amber-400',
        badgeBg: 'bg-amber-500/20 text-amber-300 border-amber-500/50',
        icon: AlertTriangle,
        desc: 'Elevated entropy or out-of-distribution feature detected. Abstention policy activated.',
      }
    : isCautious
    ? {
        label: 'Cautious (Intermediate Entropy)',
        colorText: 'text-sky-400',
        badgeBg: 'bg-sky-500/20 text-sky-300 border-sky-500/50',
        icon: HelpCircle,
        desc: 'Model prediction has moderate dispersion across differential classes.',
      }
    : {
        label: 'Confident Prediction',
        colorText: 'text-emerald-400',
        badgeBg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50',
        icon: CheckCircle2,
        desc: 'Sharp Softmax peak with low entropy and tight confidence interval.',
      };

  const StatusIcon = statusConfig.icon;

  // Mini reliability calibration scatter points
  const scatterData = [
    { predicted: 0.2, observed: 0.19 },
    { predicted: 0.4, observed: 0.42 },
    { predicted: 0.6, observed: 0.58 },
    { predicted: 0.8, observed: 0.79 },
    { predicted: calibrationBin.predicted, observed: calibrationBin.observed, isCurrent: true },
  ];

  return (
    <div className="p-6 rounded-3xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-white">Model Confidence & Entropy Analysis</h3>
        </div>
        <div className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold border ${statusConfig.badgeBg}`}>
          <StatusIcon className="w-3.5 h-3.5 shrink-0" />
          <span>{statusConfig.label}</span>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Normalized Entropy */}
        <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div className="flex justify-between items-start">
            <span className="text-[11px] text-slate-400">Shannon Entropy:</span>
            <span className="text-[10px] font-mono text-slate-500">0.0–1.0</span>
          </div>
          <div className="flex items-baseline space-x-1.5">
            <span className={`text-2xl font-extrabold font-mono ${statusConfig.colorText}`}>
              {entropy.toFixed(3)}
            </span>
          </div>
          {/* Progress bar */}
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                entropy > 0.6 ? 'bg-amber-400' : entropy > 0.3 ? 'bg-sky-400' : 'bg-emerald-400'
              }`}
              style={{ width: `${entropy * 100}%` }}
            />
          </div>
          <span className="text-[10px] text-slate-400 block">
            {entropy < 0.3 ? 'Low dispersion' : entropy <= 0.6 ? 'Moderate dispersion' : 'High dispersion'}
          </span>
        </div>

        {/* 95% Confidence Interval */}
        <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <span className="text-[11px] text-slate-400 block">95% Confidence Interval:</span>
          <div className="text-xl font-extrabold font-mono text-white mt-1">
            [{Math.round(confidenceInterval[0] * 100)}%, {Math.round(confidenceInterval[1] * 100)}%]
          </div>
          <p className="text-[10px] text-slate-400 leading-tight">
            Bootstrap CI on top class prob (±{(Math.round((confidenceInterval[1] - confidenceInterval[0]) * 50))}%)
          </p>
        </div>

        {/* Out of Distribution (OOD) */}
        <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <span className="text-[11px] text-slate-400 block">OOD Heuristic Detector:</span>
          <div className="flex items-center space-x-2 mt-1">
            <span
              className={`px-2 py-0.5 rounded text-xs font-bold font-mono ${
                oodFlag
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
              }`}
            >
              {oodFlag ? 'OOD DETECTED' : 'IN-DISTRIBUTION'}
            </span>
          </div>
          <p className="text-[10px] text-slate-400 leading-tight">
            {oodFlag ? 'Atypical radiograph geometry / artifact' : 'Consistent with training feature domain'}
          </p>
        </div>
      </div>

      {/* Mini Reliability Diagram (Predicted vs Observed) */}
      <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-300">Calibration Reliability Bin (Predicted vs. Observed)</span>
          <span className="text-[10px] text-slate-500 font-mono">Ideal: Diagonal y = x</span>
        </div>
        <div className="w-full h-28">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <XAxis type="number" dataKey="predicted" domain={[0, 1]} tick={{ fontSize: 9, fill: '#64748b' }} unit="" />
              <YAxis type="number" dataKey="observed" domain={[0, 1]} tick={{ fontSize: 9, fill: '#64748b' }} width={24} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="3,3" />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-slate-900 border border-slate-700 p-2 rounded text-[10px] text-white">
                        <div>Predicted: {(d.predicted * 100).toFixed(0)}%</div>
                        <div>Observed: {(d.observed * 100).toFixed(0)}%</div>
                        {d.isCurrent && <div className="text-cyan-400 font-bold">Current Image</div>}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter data={scatterData} fill="#38bdf8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
