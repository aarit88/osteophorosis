import React, { useState } from 'react';
import { Eye, Layers, RotateCw, ShieldAlert } from 'lucide-react';
import { FourClassGatePanel } from '../components/FourClassGatePanel';
import { runExplainabilityDemo } from '../api/client';
import type { ExplainabilityDemoResponse } from '../api/types';

const TARGET_CLASSES = [
  { idx: 0, label: '0 · Secondary Bone Metastasis' },
  { idx: 1, label: "1 · Paget's Disease of Bone" },
  { idx: 2, label: '2 · Osteogenesis Imperfecta (OI)' },
  { idx: 3, label: '3 · Healthy Skeletal Control' },
];

export const ExplainabilityPage: React.FC = () => {
  const [targetClass, setTargetClass] = useState(0);
  const [shiftHorizontal, setShiftHorizontal] = useState(0);
  const [demoResult, setDemoResult] = useState<ExplainabilityDemoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRunDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runExplainabilityDemo({
        target_class: targetClass,
        shift_horizontal: shiftHorizontal,
      });
      setDemoResult(res);
    } catch {
      setError('Unable to generate Grad-CAM synthetic map. Check gateway API connectivity.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 py-4 sm:py-6">
      <header className="space-y-2 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-indigo-800 bg-indigo-950/60 px-3 py-1 text-xs font-semibold text-indigo-200">
          <Eye className="h-3.5 w-3.5" aria-hidden="true" /> Visual &amp; Radiomic Explainability Explorer
        </div>
        <h1 className="text-3xl font-black tracking-tight text-white">Explainability (Grad-CAM &amp; SHAP)</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-300">
          Explore synthetic Grad-CAM spatial activation localization across the four target skeletal classes, alongside the Track D clinical checkpoint gate.
        </p>
      </header>

      {/* Interactive Synthetic Grad-CAM Explorer */}
      <section className="rounded-2xl border border-indigo-900/60 bg-slate-900/75 p-5 sm:p-6 space-y-4" aria-labelledby="gradcam-demo-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 id="gradcam-demo-title" className="text-xl font-bold text-white">Interactive Grad-CAM Spatial Salience Demo</h2>
            <p className="mt-1 text-xs text-slate-400">
              Calls <code className="text-indigo-300">POST /api/v1/explainability/demo</code> on a 32×32 controlled skeletal lesion fixture.
            </p>
          </div>
          <span className="rounded-full border border-indigo-800 bg-indigo-950/50 px-2.5 py-1 font-mono text-[10px] text-indigo-200">
            SYNTHETIC CNN FIXTURE
          </span>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="text-xs font-semibold text-slate-300">
            Target Pathology Activation Class
            <select
              aria-label="Target Pathology Class"
              value={targetClass}
              onChange={(e) => setTargetClass(Number(e.target.value))}
              className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100"
            >
              {TARGET_CLASSES.map((item) => (
                <option key={item.idx} value={item.idx}>{item.label}</option>
              ))}
            </select>
          </label>

          <label className="text-xs font-semibold text-slate-300">
            Lesion Horizontal Shift ({shiftHorizontal >= 0 ? `+${shiftHorizontal}` : shiftHorizontal} px)
            <input
              type="range"
              min={-8}
              max={8}
              step={1}
              value={shiftHorizontal}
              onChange={(e) => setShiftHorizontal(Number(e.target.value))}
              className="mt-3 block w-full accent-indigo-400 cursor-pointer"
              aria-label="Lesion Horizontal Shift"
            />
          </label>
        </div>

        <button
          type="button"
          onClick={() => void handleRunDemo()}
          disabled={loading}
          className="inline-flex min-h-10 items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-bold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {loading ? <RotateCw className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Layers className="h-4 w-4" aria-hidden="true" />}
          {loading ? 'Generating Activation Grid…' : 'Generate Grad-CAM Activation Grid'}
        </button>

        {error && <p role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</p>}

        {demoResult && (
          <div className="grid gap-5 border-t border-slate-800 pt-4 md:grid-cols-[240px_1fr]">
            <div>
              <p className="mb-2 text-xs font-semibold text-slate-300">8×8 Normalized Salience Grid</p>
              <div className="grid grid-cols-8 gap-1 rounded-xl border border-slate-800 bg-slate-950 p-2" role="img" aria-label="Synthetic Grad-CAM heatmap grid">
                {demoResult.heatmap_grid_sample.flatMap((row, rIdx) =>
                  row.map((val, cIdx) => (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      title={`[${rIdx},${cIdx}] = ${val.toFixed(2)}`}
                      className="h-6 w-6 rounded-sm"
                      style={{
                        backgroundColor: `rgba(244, 63, 94, ${Math.max(0.08, Math.min(1, val))})`,
                      }}
                    />
                  )),
                )}
              </div>
            </div>

            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-3 font-mono">
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <span className="text-[11px] text-slate-400">Active Region Ratio</span>
                  <p className="mt-1 text-lg font-black text-indigo-300">{(demoResult.active_region_ratio * 100).toFixed(1)}%</p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <span className="text-[11px] text-slate-400">Peak Coordinates</span>
                  <p className="mt-1 text-lg font-black text-cyan-300">[{demoResult.peak_coordinates.join(', ')}]</p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <span className="text-[11px] text-slate-400">Method</span>
                  <p className="mt-1 text-xs font-bold text-white">{demoResult.method}</p>
                </div>
              </div>
              <p className="text-xs text-slate-300"><strong>SHAP Status:</strong> {demoResult.shap_status}</p>
              <p className="text-xs text-amber-200/90">{demoResult.warning}</p>
            </div>
          </div>
        )}
      </section>

      <FourClassGatePanel />

      <div className="flex items-start gap-3 rounded-xl border border-rose-900/70 bg-rose-950/20 p-4 text-sm text-rose-100">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <p>Clinical Track D inference remains gated with <code>FOUR_CLASS_MODEL_NOT_READY</code> until authorized clinical datasets pass validation.</p>
      </div>
    </div>
  );
};

export default ExplainabilityPage;
