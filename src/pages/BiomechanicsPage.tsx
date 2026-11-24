import React, { useRef, useState } from 'react';
import { Activity, AlertTriangle, RotateCw, Ruler } from 'lucide-react';
import { runBiomechanicsMirels } from '../api/client';
import type { BiomechanicsMirelsResponse } from '../api/types';

const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';
const selectClass = 'mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100';
const actionClass = 'mt-4 inline-flex min-h-10 items-center gap-2 rounded-lg bg-amber-700 px-4 py-2.5 text-sm font-bold text-white hover:bg-amber-600 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-amber-300';
type MirelsFactors = { site: number | ''; pain: number | ''; nature: number | ''; size: number | '' };

export const BiomechanicsPage: React.FC = () => {
  const [factors, setFactors] = useState<MirelsFactors>({ site: '', pain: '', nature: '', size: '' });
  const [result, setResult] = useState<BiomechanicsMirelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestLock = useRef(false);

  const calculate = async () => {
    if (requestLock.current || loading) return;
    if ([factors.site, factors.pain, factors.nature, factors.size].some((value) => value === '')) {
      setError('Choose all four scoring categories before calculating.');
      setResult(null);
      return;
    }
    requestLock.current = true;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await runBiomechanicsMirels({ site: factors.site as number, pain: factors.pain as number, radiographic_nature: factors.nature as number, lesion_size: factors.size as number }));
    } catch {
      setError('The scoring service is unavailable. No result was created.');
    } finally { requestLock.current = false; setLoading(false); }
  };

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <header className="space-y-3 border-b border-slate-800 pb-5"><div className="inline-flex items-center gap-2 rounded-full border border-teal-800 bg-teal-950/60 px-3 py-1 text-xs font-semibold text-teal-200"><Ruler size={14} aria-hidden="true" />Research method status</div><h1 className="text-3xl font-black tracking-tight text-white">Biomechanics</h1><p className="max-w-3xl text-sm leading-6 text-slate-300">The repository contains 2D synthetic cross-section geometry routines and a four-factor Mirels scoring service. It does not compute validated mechanical properties or patient-specific fracture probability from imaging.</p></header>

      <section className={panelClass} aria-labelledby="geometry-title"><h2 id="geometry-title" className="text-xl font-bold text-white">Cross-section geometry</h2><p className="mt-2 text-sm leading-6 text-slate-300">The implemented routine calculates area (mm²), centroid (mm), second moments and polar moment (mm⁴), and geometric retention ratios from a 2D binary mask. The website has no governed image-segmentation input, so this operation is unavailable for study data.</p><div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-900/70 bg-amber-950/20 p-3 text-xs leading-5 text-amber-100"><AlertTriangle size={15} className="mt-0.5 shrink-0" aria-hidden="true" />Generated phantom outputs are retained for isolated tests and are not presented as user or clinical results. Geometry alone is not material strength or fracture risk.</div></section>

      <section className={panelClass} aria-labelledby="mirels-title">
        <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 id="mirels-title" className="text-xl font-bold text-white">Mirels score calculator</h2><p className="mt-1 text-sm leading-5 text-slate-300">The backend sums four manually selected 1–3 point categories (total 4–12). It is a rule calculation only; this interface does not analyze an image or create a patient-specific treatment recommendation.</p></div><span className="rounded-full border border-amber-900 bg-amber-950/40 px-2.5 py-1 text-xs text-amber-100">POST /api/v1/biomechanics/mirels</span></div>
        <p className="mt-3 rounded-lg border border-amber-900/70 bg-amber-950/20 p-3 text-xs leading-5 text-amber-100">Use hypothetical selections only. Do not enter patient identifiers, clinical notes, lesion descriptions, or protected data. The score's intended population and limitations must be reviewed by a qualified clinician; this page does not provide medical advice.</p>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <label className="text-xs font-semibold text-slate-300">Anatomic site<select className={selectClass} aria-label="Anatomic site" value={factors.site} onChange={(event) => setFactors({ ...factors, site: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>Upper extremity · 1</option><option value={2}>Lower extremity · 2</option><option value={3}>Peritrochanteric · 3</option></select></label>
          <label className="text-xs font-semibold text-slate-300">Pain category<select className={selectClass} aria-label="Pain category" value={factors.pain} onChange={(event) => setFactors({ ...factors, pain: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>Mild · 1</option><option value={2}>Moderate · 2</option><option value={3}>Functional · 3</option></select></label>
          <label className="text-xs font-semibold text-slate-300">Radiographic nature<select className={selectClass} aria-label="Radiographic nature" value={factors.nature} onChange={(event) => setFactors({ ...factors, nature: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>Blastic · 1</option><option value={2}>Mixed · 2</option><option value={3}>Lytic · 3</option></select></label>
          <label className="text-xs font-semibold text-slate-300">Lesion size relative to cortex<select className={selectClass} aria-label="Lesion size" value={factors.size} onChange={(event) => setFactors({ ...factors, size: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>&lt; 1/3 · 1</option><option value={2}>1/3–2/3 · 2</option><option value={3}>&gt; 2/3 · 3</option></select></label>
        </div>
        <button className={actionClass} type="button" onClick={() => void calculate()} disabled={loading}>{loading ? <RotateCw size={15} className="animate-spin" aria-hidden="true" /> : <Activity size={15} aria-hidden="true" />}{loading ? 'Calculating…' : 'Calculate hypothetical score'}</button>
        {loading && <p role="status" className="mt-2 text-xs text-amber-200">Waiting for the scoring service; no progress endpoint is available.</p>}
        {error && <div role="alert" className="mt-3 rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">{error}<button type="button" onClick={() => void calculate()} disabled={loading} className="ml-3 underline">Retry</button></div>}
        {!result && !loading && !error && <p className="mt-4 rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-400">Choose all four hypothetical categories. No result is pre-filled.</p>}
        {result && <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/70 p-4" role="status"><p className="text-xs font-semibold text-slate-400">Rule sum · not a patient risk estimate</p><p className="mt-1 text-2xl font-black text-amber-200">Total score: {result.total_score} / 12</p><p className="mt-2 text-xs leading-5 text-slate-400">Only the arithmetic total is shown. The API's risk percentage, tier, and action fields are not presented as individualized clinical guidance.</p></div>}
      </section>
      <p className="text-xs leading-5 text-slate-500">The calculator is not connected to a patient or staged dataset. Governed imaging workflows must pass through Stage 38 onboarding and Stage 39 preparation. Geometry analysis requires a validated segmentation workflow before study data can be used.</p>
    </main>
  );
};

export default BiomechanicsPage;
