import React, { useState } from 'react';
import { Activity, HelpCircle, RotateCw, ShieldAlert } from 'lucide-react';
import { FourClassGatePanel } from '../components/FourClassGatePanel';
import { runUncertaintyDemo } from '../api/client';
import type { UncertaintyDemoResponse } from '../api/types';

const PRESETS = [
  { id: 'highly_confident', label: 'Highly Confident Peak (0.94 / 0.02 / 0.02 / 0.02)' },
  { id: 'moderately_confident', label: 'Moderately Confident (0.65 / 0.15 / 0.10 / 0.10)' },
  { id: 'ambiguous_two_class', label: "Ambiguous Borderline (0.48 Paget's vs 0.48 Metastasis)" },
  { id: 'uniform_four_class', label: 'Maximum Dispersion (0.25 / 0.25 / 0.25 / 0.25)' },
  { id: 'near_equal', label: 'Near-Uniform Low Margin (0.28 / 0.26 / 0.24 / 0.22)' },
];

export const UncertaintyPage: React.FC = () => {
  const [preset, setPreset] = useState('highly_confident');
  const [oodFlag, setOodFlag] = useState(false);
  const [result, setResult] = useState<UncertaintyDemoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runUncertaintyDemo({
        distribution_preset: preset,
        ood_flag: oodFlag,
      });
      setResult(res);
    } catch {
      setError('Unable to run uncertainty evaluation. Check backend API connectivity.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 py-4 sm:py-6">
      <header className="space-y-2 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-purple-800 bg-purple-950/60 px-3 py-1 text-xs font-semibold text-purple-200">
          <Activity className="h-3.5 w-3.5" aria-hidden="true" /> Selective Prediction &amp; Entropy Gate
        </div>
        <h1 className="text-3xl font-black tracking-tight text-white">Uncertainty &amp; Abstention Triage</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-300">
          Test Shannon predictive entropy (H_norm &gt; 0.60 threshold), top-two probability margin, and out-of-distribution (OOD) abstention behavior interactively.
        </p>
      </header>

      <section className="rounded-2xl border border-purple-900/60 bg-slate-900/75 p-5 sm:p-6 space-y-4" aria-labelledby="uncertainty-sim-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 id="uncertainty-sim-title" className="text-xl font-bold text-white">Entropy &amp; OOD Selective Prediction Simulator</h2>
            <p className="mt-1 text-xs text-slate-400">Calls <code className="text-purple-300">POST /api/v1/uncertainty/demo</code>.</p>
          </div>
          <span className="rounded-full border border-purple-800 bg-purple-950/40 px-2.5 py-1 font-mono text-[10px] text-purple-200">
            H_norm &gt; 0.60 Threshold
          </span>
        </div>

        <div className="grid gap-4 sm:grid-cols-[1.4fr_.6fr] sm:items-end">
          <label className="text-xs font-semibold text-slate-300">
            Softmax Probability Distribution Preset
            <select
              aria-label="Probability Distribution Preset"
              value={preset}
              onChange={(e) => setPreset(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100"
            >
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </label>

          <label className="flex min-h-11 items-center gap-2.5 rounded-lg border border-slate-700 bg-slate-950 px-3.5 py-2 text-xs font-semibold text-slate-200 cursor-pointer">
            <input
              type="checkbox"
              checked={oodFlag}
              onChange={(e) => setOodFlag(e.target.checked)}
              className="h-4 w-4 accent-purple-500"
            />
            Simulate Out-of-Distribution (OOD) Flag
          </label>
        </div>

        <button
          type="button"
          onClick={() => void handleEvaluate()}
          disabled={loading}
          className="inline-flex min-h-10 items-center gap-2 rounded-xl bg-purple-600 px-5 py-2.5 text-sm font-bold text-white hover:bg-purple-500 disabled:opacity-50"
        >
          {loading ? <RotateCw className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Activity className="h-4 w-4" aria-hidden="true" />}
          {loading ? 'Evaluating Entropy…' : 'Evaluate Uncertainty Gate'}
        </button>

        {error && <p role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</p>}

        {result && (
          <div className="space-y-4 border-t border-slate-800 pt-4">
            <div className="grid gap-3 sm:grid-cols-4 font-mono">
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Normalized Entropy</span>
                <p className={`mt-1 text-xl font-black ${result.normalized_entropy > 0.6 ? 'text-amber-300' : 'text-emerald-300'}`}>
                  {result.normalized_entropy.toFixed(3)}
                </p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Top-Two Margin</span>
                <p className="mt-1 text-xl font-black text-cyan-300">{result.top_two_margin.toFixed(3)}</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Top Class</span>
                <p className="mt-1 text-sm font-bold text-white">{result.predicted_class_name}</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Gate Decision</span>
                <p className={`mt-1 text-xs font-bold ${result.human_review_required ? 'text-amber-300' : 'text-emerald-300'}`}>
                  {result.decision.replace(/_/g, ' ')}
                </p>
              </div>
            </div>

            <div className={`rounded-xl border p-4 text-xs leading-5 ${result.human_review_required ? 'border-amber-800 bg-amber-950/30 text-amber-100' : 'border-emerald-800 bg-emerald-950/20 text-emerald-100'}`}>
              <p className="font-bold">{result.reason_description}</p>
              <p className="mt-1 opacity-90">{result.clinical_guidance}</p>
            </div>
            <p className="text-xs text-slate-400">{result.disclaimer}</p>
          </div>
        )}
      </section>

      <FourClassGatePanel />

      <aside className="flex items-start gap-3 rounded-xl border border-amber-900/70 bg-amber-950/20 p-4 text-sm leading-6 text-amber-100">
        <HelpCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <p>The normalized entropy cutoff of 0.60 is an engineering threshold. Abstention always triggers human specialist review and must never be interpreted as a healthy finding.</p>
      </aside>
      <div className="flex items-start gap-3 rounded-xl border border-rose-900/70 bg-rose-950/20 p-4 text-sm text-rose-100">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <p>Clinical Track D weights remain gated under <code>FOUR_CLASS_MODEL_NOT_READY</code> until cohort governance passes.</p>
      </div>
    </div>
  );
};

export default UncertaintyPage;
