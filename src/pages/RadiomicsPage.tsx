import React, { useRef, useState } from 'react';
import { AlertTriangle, RotateCw, SlidersHorizontal, Sparkles } from 'lucide-react';
import { runRadiomicsDemo } from '../api/client';
import type { RadiomicsDemoResponse } from '../api/types';

const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';
const categoryInfo = [
  { id: 'first_order', title: 'First-order intensity', count: 18, description: 'Intensity distribution statistics inside the phantom mask; histogram-derived.' },
  { id: 'glcm', title: 'GLCM texture', count: 26, description: 'Gray-level co-occurrence texture relationships.' },
  { id: 'glrlm', title: 'GLRLM texture', count: 16, description: 'Gray-level run-length texture descriptors.' },
  { id: 'glszm', title: 'GLSZM texture', count: 16, description: 'Gray-level size-zone texture descriptors.' },
  { id: 'gldm', title: 'GLDM / NGTDM-related texture', count: 26, description: 'Neighborhood dependence and tone-difference family; the implementation documents unresolved specification/heuristic items.' },
];

function familyFor(name: string) {
  if (name.startsWith('first_order_')) return categoryInfo[0];
  if (name.startsWith('glcm_')) return categoryInfo[1];
  if (name.startsWith('glrlm_')) return categoryInfo[2];
  if (name.startsWith('glszm_')) return categoryInfo[3];
  return categoryInfo[4];
}

export const RadiomicsPage: React.FC = () => {
  const [phantomType, setPhantomType] = useState('checkerboard');
  const [gridSize, setGridSize] = useState(64);
  const [spacingMm, setSpacingMm] = useState(1);
  const [data, setData] = useState<RadiomicsDemoResponse | null>(null);
  const [receivedAt, setReceivedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestLock = useRef(false);

  const compute = async () => {
    if (requestLock.current || loading) return;
    if (!['checkerboard', 'linear_gradient', 'annulus', 'uniform'].includes(phantomType)
      || ![32, 64, 128].includes(gridSize) || ![0.5, 1, 2].includes(spacingMm)) {
      setError('Select a supported phantom, grid size, and spacing before running extraction.');
      setData(null);
      return;
    }
    requestLock.current = true;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = await runRadiomicsDemo({ phantom_type: phantomType, grid_size: gridSize, spacing_mm: spacingMm });
      setData(result);
      setReceivedAt(new Date().toISOString());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Radiomics service could not complete extraction.');
    } finally {
      requestLock.current = false;
      setLoading(false);
    }
  };

  const features = data ? [
    ...Object.entries(data.verified_features_sample).map(([name, value]) => ({ name, value, validation: 'Reported in verified feature sample' })),
    ...Object.entries(data.unverified_features_sample).map(([name, value]) => ({ name, value, validation: 'Unverified / heuristic sample; not for validated fusion' })),
  ] : [];

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <header className="space-y-3 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200"><SlidersHorizontal size={14} /> Radiomics · synthetic engine</div>
        <h1 className="text-3xl font-black tracking-tight text-white">Radiomics feature explorer</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-300">The current API extracts features from generated 2D mathematical phantoms only. It does not accept clinical images or governed dataset references. The project records 81 features as mathematically verified and 21 as engineering heuristics or unresolved specification items; the 102-feature contract is not fully verified as IBSI-compliant.</p>
      </header>

      <section className={panelClass} aria-labelledby="radiomics-categories-title">
        <h2 id="radiomics-categories-title" className="text-lg font-bold text-white">Implemented feature families</h2>
        <p className="mt-1 text-xs leading-5 text-slate-400">Engine family inventory. Result values below are only the representative feature sample returned by the demo API.</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          {categoryInfo.map((category) => <article key={category.id} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
            <h3 className="text-sm font-bold text-slate-100">{category.title}</h3><p className="mt-1 text-xs text-cyan-200">{category.count} engine dimensions</p><p className="mt-2 text-xs leading-5 text-slate-400">{category.description}</p>
          </article>)}
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <p className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 text-xs text-slate-300"><strong>Shape features:</strong> unavailable in the current radiomics demo response.</p>
          <p className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 text-xs text-slate-300"><strong>Spatial/geometric measurements:</strong> handled by the separate synthetic 2D geometry module; they are not part of this radiomics endpoint.</p>
        </div>
      </section>

      <section className={panelClass} aria-labelledby="phantom-config-title">
        <h2 id="phantom-config-title" className="text-lg font-bold text-white">Configure a synthetic phantom run</h2>
        <p className="mt-1 text-xs leading-5 text-slate-400">Inputs accepted by <code>POST /api/v1/radiomics/demo</code>: phantom type, grid size (8–128 px), and pixel spacing (0.1–5 mm). The controls use supported API values. No image or file path is submitted.</p>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <label className="text-xs font-semibold text-slate-300">Phantom
            <select aria-label="Phantom type" value={phantomType} onChange={(event) => setPhantomType(event.target.value)} className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100">
              <option value="checkerboard">Checkerboard</option><option value="linear_gradient">Linear gradient</option><option value="annulus">Geometric annulus</option><option value="uniform">Uniform</option>
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300">Grid resolution
            <select aria-label="Grid resolution" value={gridSize} onChange={(event) => setGridSize(Number(event.target.value))} className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100">
              <option value={32}>32 × 32 px</option><option value={64}>64 × 64 px</option><option value={128}>128 × 128 px</option>
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300">Pixel spacing
            <select aria-label="Pixel spacing" value={spacingMm} onChange={(event) => setSpacingMm(Number(event.target.value))} className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100">
              <option value={0.5}>0.5 mm</option><option value={1}>1.0 mm</option><option value={2}>2.0 mm</option>
            </select>
          </label>
        </div>
        <button type="button" onClick={() => void compute()} disabled={loading} className="mt-4 inline-flex items-center gap-2 rounded-lg bg-cyan-700 px-4 py-2.5 text-sm font-bold text-white hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-50">{loading ? <RotateCw className="animate-spin" size={16} /> : <Sparkles size={16} />}{loading ? 'Extracting…' : 'Run phantom extraction'}</button>
      </section>

      {loading && <p role="status" className="rounded-xl border border-slate-800 p-4 text-sm text-cyan-200">Waiting for the backend. This endpoint does not report intermediate progress.</p>}
      {error && <div role="alert" className="rounded-xl border border-rose-800 bg-rose-950/30 p-4 text-sm text-rose-200"><p>Radiomics request failed: {error}</p><button className="mt-3 underline" type="button" onClick={() => void compute()} disabled={loading}>Retry</button></div>}
      {!data && !loading && !error && <div className="rounded-xl border border-dashed border-slate-700 p-6 text-sm text-slate-400">No run yet. Choose a phantom configuration and run the existing demo API to see actual output.</div>}

      {data && <section className="space-y-4" aria-label="Radiomics results">
        <div className="grid gap-3 sm:grid-cols-3">
          <div className={panelClass}><p className="text-xs text-slate-400">Verified count returned</p><p className="mt-1 text-2xl font-bold text-emerald-300">{data.verified_features_count}</p></div>
          <div className={panelClass}><p className="text-xs text-slate-400">Unverified / heuristic count</p><p className="mt-1 text-2xl font-bold text-amber-300">{data.unverified_features_count}</p></div>
          <div className={panelClass}><p className="text-xs text-slate-400">Specification status</p><p className="mt-1 text-sm font-bold text-cyan-200">{data.specification_status}</p></div>
        </div>
        <div className={panelClass}>
          <div className="flex flex-wrap items-start justify-between gap-2"><div><h2 className="text-lg font-bold text-white">Returned feature sample</h2><p className="text-xs text-slate-400">Configuration: {data.phantom_type}, {data.grid_size} × {data.grid_size}, {data.spacing_mm} mm/pixel</p></div><span className="text-xs text-slate-400">Browser received: {receivedAt ? new Date(receivedAt).toLocaleString() : 'Unavailable'}</span></div>
          <div className="mt-4 overflow-x-auto rounded-lg border border-slate-800"><table className="w-full min-w-[700px] text-left text-xs"><thead className="bg-slate-950 text-slate-400"><tr><th className="p-3">Feature</th><th className="p-3">Family / explanation</th><th className="p-3">Value</th><th className="p-3">Interpretation and status</th></tr></thead><tbody>{features.map(({ name, value, validation }) => {
            const family = familyFor(name);
            return <tr key={name} className="border-t border-slate-800"><td className="p-3 font-mono text-slate-100">{name}</td><td className="p-3 text-slate-300">{family.title}: {family.description}</td><td className="p-3 font-mono text-cyan-200">{value.toPrecision(6)}</td><td className="p-3 text-slate-300">Numeric output from this synthetic phantom only; no clinical interpretation. {validation}</td></tr>;
          })}</tbody></table></div>
          <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-900/70 bg-amber-950/20 p-3 text-xs leading-5 text-amber-100"><AlertTriangle size={15} className="mt-0.5 shrink-0" />{data.disclaimer}</div>
          <p className="mt-3 text-xs text-slate-500">Input reference: generated synthetic phantom (no dataset/study ID). API response does not include server timestamp or software build/version; those values are not inferred here.</p>
        </div>
      </section>}
    </main>
  );
};

export default RadiomicsPage;
