import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  RotateCw,
  CheckCircle2,
  ShieldAlert,
  Sliders,
  HelpCircle,
  Lock,
} from 'lucide-react';
import { runUncertaintyDemo } from '../api/client';
import { UncertaintyDemoResponse } from '../api/types';

export const UncertaintyPage: React.FC = () => {
  const [preset, setPreset] = useState<string>('highly_confident');
  const [oodFlag, setOodFlag] = useState<boolean>(false);
  const [data, setData] = useState<UncertaintyDemoResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUncertainty = async (p = preset, ood = oodFlag) => {
    setLoading(true);
    setError(null);
    try {
      const res = await runUncertaintyDemo({
        distribution_preset: p,
        ood_flag: ood,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to compute uncertainty triage.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUncertainty();
  }, []);

  const getDecisionBadge = (decision: string) => {
    if (decision.includes('ACCEPT')) {
      return (
        <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>ACCEPTED (H_norm &lt; 0.35)</span>
        </span>
      );
    }
    if (decision.includes('WARN')) {
      return (
        <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-950 text-amber-300 border border-amber-800">
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>WARNING: BORDERLINE (0.35 ≤ H_norm &lt; 0.60)</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-950 text-rose-300 border border-rose-800">
        <Lock className="w-3.5 h-3.5" />
        <span>ABSTAINED: HUMAN REVIEW REQUIRED (H_norm ≥ 0.60)</span>
      </span>
    );
  };

  const classNames = [
    'Secondary Bone Metastasis',
    "Paget's Disease of Bone",
    'Osteogenesis Imperfecta',
    'Healthy Skeletal Control',
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-purple-950/70 border border-purple-800 text-purple-300 text-xs font-semibold">
          <Activity className="w-3.5 h-3.5" />
          <span>Stage 30 Engine: Selective Prediction & Abstention</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Uncertainty Quantification & Abstention Gate
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Evaluates normalized Shannon entropy (H_norm = H / ln(4)), margin confidence, and out-of-distribution indicators to prevent silent model failure via a deterministic tri-state triage mechanism.
        </p>
      </div>

      {/* Mandatory Safeguard Banner */}
      <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800 text-rose-200 text-xs flex items-start space-x-3">
        <ShieldAlert className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
        <div className="space-y-1">
          <strong className="text-rose-300 font-bold">Mandatory Clinical Safeguard:</strong>
          <p>
            An abstention response represents <strong>epistemic or aleatoric uncertainty</strong> and must <strong>NEVER</strong> be treated as a negative diagnosis or a healthy finding. When confidence drops below threshold (H_norm ≥ 0.60), the system strictly refuses to output a speculative diagnosis and mandates human expert review.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">
          Simulated Softmax Probability Distribution Presets
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Diagnostic Scenario Preset
            </label>
            <select
              value={preset}
              onChange={(e) => {
                setPreset(e.target.value);
                fetchUncertainty(e.target.value, oodFlag);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              <option value="highly_confident">Confident Metastasis (p = [0.94, 0.02, 0.02, 0.02])</option>
              <option value="moderately_confident">Moderate Confidence (p = [0.65, 0.15, 0.10, 0.10])</option>
              <option value="ambiguous_two_class">Ambiguous Metastasis vs Paget (p = [0.48, 0.48, 0.02, 0.02])</option>
              <option value="uniform_four_class">Uniform Dispersion / High Entropy (p = [0.25, 0.25, 0.25, 0.25])</option>
              <option value="near_equal">Near-Equal Spread (p = [0.28, 0.26, 0.24, 0.22])</option>
            </select>
          </div>

          <div className="flex items-center space-x-3 pt-6">
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={oodFlag}
                onChange={(e) => {
                  setOodFlag(e.target.checked);
                  fetchUncertainty(preset, e.target.checked);
                }}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
              <span className="ml-3 text-xs font-medium text-slate-300">
                Trigger Out-of-Distribution (OOD) Flag
              </span>
            </label>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Results Dashboard */}
      {data && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          {/* Triage Decision Banner */}
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs text-slate-400 block mb-1">Selective Prediction Triage Status</span>
              <div>{getDecisionBadge(data.decision)}</div>
            </div>
            <div className="text-right">
              <span className="text-xs text-slate-400 block mb-1">Human Radiologist Review</span>
              <span className={`text-sm font-bold ${
                data.human_review_required ? 'text-rose-400' : 'text-emerald-400'
              }`}>
                {data.human_review_required ? 'MANDATORY REQUIRED' : 'Standard Double-Reading'}
              </span>
            </div>
          </div>

          {/* Probabilities and Metrics Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Probability Bars */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <h3 className="font-bold text-white text-base">Class Probability Distribution</h3>
              <div className="space-y-3">
                {data.probabilities.map((prob, idx) => {
                  const isMax = idx === data.predicted_class_idx;
                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className={`font-medium ${isMax ? 'text-white font-bold' : 'text-slate-400'}`}>
                          {classNames[idx]}
                        </span>
                        <span className={`font-mono ${isMax ? 'text-purple-300 font-bold' : 'text-slate-400'}`}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            isMax ? 'bg-gradient-to-r from-purple-500 to-cyan-400' : 'bg-slate-700'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Metrics Panel */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <h3 className="font-bold text-white text-base">Information-Theoretic Uncertainty Metrics</h3>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Normalized Entropy (H_norm)</span>
                  <span className="text-xl font-mono font-bold text-purple-300 mt-1 block">
                    {data.normalized_entropy.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Scale: [0.00, 1.00]</span>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Abstention Cutoff (H*)</span>
                  <span className="text-xl font-mono font-bold text-slate-300 mt-1 block">
                    {data.candidate_threshold.toFixed(2)}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Candidate threshold</span>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Shannon Entropy (nats)</span>
                  <span className="text-xl font-mono font-bold text-white mt-1 block">
                    {data.shannon_entropy_nats.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Max: ln(4) ≈ 1.386</span>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Top-Two Margin</span>
                  <span className="text-xl font-mono font-bold text-cyan-300 mt-1 block">
                    {(data.top_two_margin * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Confidence gap</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-1">
                <span className="font-bold text-slate-200">Clinical Action & Guidance:</span>
                <p className="text-slate-300 text-[11px] leading-relaxed">{data.clinical_guidance}</p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default UncertaintyPage;
