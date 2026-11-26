import React, { useState } from 'react';
import { AlertTriangle, FlaskConical, RotateCw, SlidersHorizontal } from 'lucide-react';
import { Link } from 'react-router-dom';
import { runRadiomicsDemo } from '../api/client';
import type { RadiomicsDemoResponse } from '../api/types';

const families = [
  ['First-order intensity', 'Intensity distribution statistics computed within bone trabecular ROI (mean, energy, entropy, skewness).'],
  ['GLCM', 'Gray-level co-occurrence matrix texture descriptors capturing local trabecular coarseness and contrast.'],
  ['GLRLM', 'Gray-level run-length matrix descriptors capturing structural trabecular continuity in Paget’s and OI.'],
  ['GLSZM', 'Gray-level size-zone matrix descriptors quantifying focal lytic vs sclerotic cluster homogeneity.'],
  ['GLDM / NGTDM-related', 'Neighborhood-dependence and tone-difference metrics modeling cortical/marrow heterogeneity.'],
];

const PHANTOMS = [
  { id: 'checkerboard', label: 'Checkerboard Reference Phantom' },
  { id: 'trabecular_paget', label: "Coarsened Trabecular (Paget's Pattern)" },
  { id: 'lytic_metastasis', label: 'Focal Lytic Rim (Metastasis Pattern)' },
  { id: 'osteopenic_oi', label: 'Hyperlucent Thin Cortex (OI Pattern)' },
];

export const RadiomicsPage: React.FC = () => {
  const [phantomType, setPhantomType] = useState('checkerboard');
  const [gridSize, setGridSize] = useState(32);
  const [spacingMm, setSpacingMm] = useState(1.0);
  const [result, setResult] = useState<RadiomicsDemoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRunDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runRadiomicsDemo({
        phantom_type: phantomType,
        grid_size: gridSize,
        spacing_mm: spacingMm,
      });
      setResult(res);
    } catch {
      setError('Unable to reach the radiomics extraction service. Check API connection and retry.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <header className="space-y-3 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200">
          <SlidersHorizontal size={14} aria-hidden="true" /> Interactive IBSI Radiomics Lab
        </div>
        <h1 className="text-3xl font-black tracking-tight text-white">Radiomics Feature Engine</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-300">
          Run deterministic 102-dimensional radiomics extraction on digital skeletal phantoms. Separately tracks the 81 analytically verified IBSI dimensions from 21 engineering heuristic dimensions.
        </p>
      </header>

      {/* Interactive Feature Extractor */}
      <section className="rounded-2xl border border-cyan-900/60 bg-slate-900/75 p-5 sm:p-6 space-y-4" aria-labelledby="radiomics-runner-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 id="radiomics-runner-title" className="text-xl font-bold text-white">Run Digital Phantom Feature Extraction</h2>
            <p className="mt-1 text-xs text-slate-400">Calls <code className="text-cyan-300">POST /api/v1/radiomics/demo</code> on deterministic skeletal phantoms.</p>
          </div>
          <span className="rounded-full border border-cyan-900 bg-cyan-950/50 px-2.5 py-1 font-mono text-[11px] text-cyan-200">
            81 Verified / 21 Heuristic
          </span>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <label className="text-xs font-semibold text-slate-300">
            Digital Phantom Pattern
            <select
              aria-label="Digital Phantom Pattern"
              value={phantomType}
              onChange={(e) => setPhantomType(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100"
            >
              {PHANTOMS.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </label>

          <label className="text-xs font-semibold text-slate-300">
            ROI Grid Size (px)
            <select
              aria-label="ROI Grid Size"
              value={gridSize}
              onChange={(e) => setGridSize(Number(e.target.value))}
              className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100"
            >
              <option value={16}>16 × 16</option>
              <option value={32}>32 × 32</option>
              <option value={64}>64 × 64</option>
            </select>
          </label>

          <label className="text-xs font-semibold text-slate-300">
            Voxel Spacing (mm)
            <select
              aria-label="Voxel Spacing"
              value={spacingMm}
              onChange={(e) => setSpacingMm(Number(e.target.value))}
              className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100"
            >
              <option value={0.5}>0.5 mm (High Res CT)</option>
              <option value={1.0}>1.0 mm (Standard)</option>
              <option value={2.0}>2.0 mm (Coarse Grid)</option>
            </select>
          </label>
        </div>

        <button
          type="button"
          onClick={() => void handleRunDemo()}
          disabled={loading}
          className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-cyan-600 px-5 py-2.5 text-sm font-bold text-white transition hover:bg-cyan-500 disabled:opacity-50"
        >
          {loading ? <RotateCw className="h-4 w-4 animate-spin" aria-hidden="true" /> : <FlaskConical className="h-4 w-4" aria-hidden="true" />}
          {loading ? 'Extracting Features…' : 'Extract Radiomics Features'}
        </button>

        {error && <p role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</p>}

        {result && (
          <div className="space-y-4 border-t border-slate-800 pt-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Analytically Verified</span>
                <p className="mt-1 font-mono text-xl font-black text-emerald-300">{result.verified_features_count} features</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Heuristic / Unverified</span>
                <p className="mt-1 font-mono text-xl font-black text-amber-300">{result.unverified_features_count} features</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Spec Contract Status</span>
                <p className="mt-1 text-xs font-semibold text-slate-200">{result.specification_status}</p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-emerald-300 mb-2">Verified Feature Sample (IBSI)</h3>
                <dl className="divide-y divide-slate-800/70 text-xs font-mono">
                  {Object.entries(result.verified_features_sample).map(([k, v]) => (
                    <div key={k} className="flex justify-between py-1.5">
                      <dt className="text-slate-300">{k}</dt>
                      <dd className="font-bold text-white">{typeof v === 'number' ? v.toFixed(4) : String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-amber-300 mb-2">Engineering Heuristic Sample</h3>
                <dl className="divide-y divide-slate-800/70 text-xs font-mono">
                  {Object.entries(result.unverified_features_sample).map(([k, v]) => (
                    <div key={k} className="flex justify-between py-1.5">
                      <dt className="text-slate-300">{k}</dt>
                      <dd className="font-bold text-amber-200">{typeof v === 'number' ? v.toFixed(4) : String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>

            <p className="text-xs text-slate-400">{result.disclaimer}</p>
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5" aria-labelledby="families-title">
        <h2 id="families-title" className="text-lg font-bold text-white">Implemented feature families</h2>
        <p className="mt-1 text-sm leading-6 text-slate-400">
          Five radiomics texture families feed the multimodal fusion specification.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {families.map(([name, description]) => (
            <article key={name} className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
              <h3 className="font-semibold text-slate-100">{name}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2" aria-label="Validation status">
        <div className="rounded-xl border border-teal-900/70 bg-teal-950/15 p-4">
          <h2 className="font-semibold text-teal-100">Reported analytic verification</h2>
          <p className="mt-2 text-sm leading-6 text-slate-300">81 feature dimensions are analytically verified on deterministic phantoms.</p>
        </div>
        <div className="rounded-xl border border-amber-900/70 bg-amber-950/20 p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-200" aria-hidden="true" />
            <div>
              <h2 className="font-semibold text-amber-100">Unresolved dimensions</h2>
              <p className="mt-2 text-sm leading-6 text-slate-300">21 heuristic dimensions remain subject to IBSI Benchmark 2 governance audit before clinical fusion.</p>
            </div>
          </div>
        </div>
      </section>

      <div className="flex flex-wrap gap-3">
        <Link to="/" className="inline-flex min-h-10 items-center rounded-lg bg-cyan-700 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-600">Upload Bone Scan →</Link>
        <Link to="/governance" className="inline-flex min-h-10 items-center rounded-lg border border-slate-700 px-3 py-2 text-sm font-semibold text-cyan-200 hover:bg-slate-900">Review dataset governance</Link>
      </div>
    </main>
  );
};

export default RadiomicsPage;
