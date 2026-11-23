import React, { useRef, useState } from 'react';
import { Activity, AlertTriangle, RotateCw, Ruler } from 'lucide-react';
import { runBiomechanicsGeometry, runBiomechanicsMirels } from '../api/client';
import type { BiomechanicsGeometryResponse, BiomechanicsMirelsResponse } from '../api/types';

const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';
const phantoms = [
  ['intact_circular_ring', 'Intact circular cortical ring'],
  ['uniform_cortical_thinning', 'Uniform cortical thinning'],
  ['symmetric_cortical_defect', 'Symmetric cortical defect'],
  ['eccentric_cortical_defect', 'Eccentric cortical defect'],
  ['lytic_cavity', 'Lytic cavity'],
  ['multiple_small_defects', 'Multiple small defects'],
  ['non_circular_elliptical_annulus', 'Elliptical annulus'],
  ['anisotropic_spacing_annulus', 'Anisotropic spacing annulus'],
] as const;
const selectClass = 'mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100';
const actionClass = 'mt-4 inline-flex items-center gap-2 rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-bold text-white hover:bg-teal-600 disabled:cursor-not-allowed disabled:opacity-50';

type MirelsFactors = { site: number | ''; pain: number | ''; nature: number | ''; size: number | '' };

export const BiomechanicsPage: React.FC = () => {
  const [phantomType, setPhantomType] = useState<string>('intact_circular_ring');
  const [spacing, setSpacing] = useState(0.5);
  const [geometry, setGeometry] = useState<BiomechanicsGeometryResponse | null>(null);
  const [geometryAt, setGeometryAt] = useState<string | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);
  const [geometryError, setGeometryError] = useState<string | null>(null);
  const geometryLock = useRef(false);

  const [factors, setFactors] = useState<MirelsFactors>({ site: '', pain: '', nature: '', size: '' });
  const [mirels, setMirels] = useState<BiomechanicsMirelsResponse | null>(null);
  const [mirelsAt, setMirelsAt] = useState<string | null>(null);
  const [mirelsLoading, setMirelsLoading] = useState(false);
  const [mirelsError, setMirelsError] = useState<string | null>(null);
  const mirelsLock = useRef(false);

  const computeGeometry = async () => {
    if (geometryLock.current || geometryLoading) return;
    if (!phantoms.some(([key]) => key === phantomType) || ![0.5, 1].includes(spacing)) {
      setGeometryError('Select a supported synthetic phantom and pixel spacing.');
      setGeometry(null);
      return;
    }
    geometryLock.current = true;
    setGeometryLoading(true);
    setGeometryError(null);
    setGeometry(null);
    try {
      setGeometry(await runBiomechanicsGeometry({ phantom_type: phantomType, voxel_size_mm: spacing }));
      setGeometryAt(new Date().toISOString());
    } catch (cause) {
      setGeometryError(cause instanceof Error ? cause.message : 'Geometry service could not complete this calculation.');
    } finally {
      geometryLock.current = false;
      setGeometryLoading(false);
    }
  };

  const computeMirels = async () => {
    if (mirelsLock.current || mirelsLoading) return;
    if ([factors.site, factors.pain, factors.nature, factors.size].some((value) => value === '')) {
      setMirelsError('Choose all four hypothetical score inputs before calculating.');
      setMirels(null);
      return;
    }
    mirelsLock.current = true;
    setMirelsLoading(true);
    setMirelsError(null);
    setMirels(null);
    try {
      setMirels(await runBiomechanicsMirels({ site: factors.site as number, pain: factors.pain as number, radiographic_nature: factors.nature as number, lesion_size: factors.size as number }));
      setMirelsAt(new Date().toISOString());
    } catch (cause) {
      setMirelsError(cause instanceof Error ? cause.message : 'Mirels scoring service could not complete this calculation.');
    } finally {
      mirelsLock.current = false;
      setMirelsLoading(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <header className="space-y-3 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-teal-800 bg-teal-950/60 px-3 py-1 text-xs font-semibold text-teal-200"><Ruler size={14} /> Biomechanics · research calculations</div>
        <h1 className="text-3xl font-black tracking-tight text-white">Biomechanics methods</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-300">Available operations are a 2D synthetic cross-section geometry calculation and a conventional four-factor Mirels score calculation. Neither operation accepts an image or estimates patient-specific mechanical properties from imaging.</p>
      </header>

      <section className={panelClass} aria-labelledby="geometry-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><h2 id="geometry-title" className="text-xl font-bold text-white">Synthetic cross-section geometry</h2><p className="mt-1 text-sm leading-5 text-slate-300">Backend method: validate a 2D binary phantom mask, then compute cortical area, centroid, second moments of area, and polar moment. This is geometric analysis, not a material-strength or fracture prediction.</p></div>
          <span className="rounded-full border border-teal-900 bg-teal-950/40 px-2.5 py-1 text-xs text-teal-100">POST /api/v1/biomechanics/geometry</span>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <label className="text-xs font-semibold text-slate-300">Synthetic mask
            <select aria-label="Synthetic bone phantom" className={selectClass} value={phantomType} onChange={(event) => setPhantomType(event.target.value)}>{phantoms.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
          </label>
          <label className="text-xs font-semibold text-slate-300">Pixel spacing (mm/pixel)
            <select aria-label="Pixel spacing" disabled={phantomType === 'anisotropic_spacing_annulus'} className={`${selectClass} disabled:opacity-50`} value={spacing} onChange={(event) => setSpacing(Number(event.target.value))}><option value={0.5}>0.5</option><option value={1}>1.0</option></select>
          </label>
        </div>
        {phantomType === 'anisotropic_spacing_annulus' && <p className="mt-3 text-xs text-amber-200">This fixture preserves its built-in anisotropic spacing (dy 0.4, dx 0.8 mm); the scalar spacing control is disabled.</p>}
        <p className="mt-3 text-xs leading-5 text-slate-400">Validation: backend requires a non-empty 2D binary mask containing at least one bone pixel; spacing must be positive and API-bounded (0.1–5 mm/pixel). The selected mask is a generated phantom, not a user study.</p>
        <button className={actionClass} type="button" onClick={() => void computeGeometry()} disabled={geometryLoading}>{geometryLoading ? <RotateCw size={15} className="animate-spin" /> : <Activity size={15} />}{geometryLoading ? 'Calculating…' : 'Calculate geometry'}</button>
        {geometryLoading && <p role="status" className="mt-2 text-xs text-teal-200">Waiting for backend calculation; no progress endpoint is available.</p>}
        {geometryError && <div role="alert" className="mt-3 rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">Geometry request failed: {geometryError}<button type="button" onClick={() => void computeGeometry()} disabled={geometryLoading} className="ml-3 underline">Retry</button></div>}
        {!geometry && !geometryLoading && !geometryError && <p className="mt-4 rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-400">No geometry calculation yet. Select phantom parameters and run the API calculation.</p>}
        {geometry && <div className="mt-4 space-y-3" aria-label="Geometry result">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[["Cortical area", geometry.area_mm2, 'mm²'], ['Centroid X', geometry.centroid_x_mm, 'mm'], ['Centroid Y', geometry.centroid_y_mm, 'mm'], ['Iₓₓ', geometry.I_xx_mm4, 'mm⁴'], ['Iᵧᵧ', geometry.I_yy_mm4, 'mm⁴'], ['Iₓᵧ', geometry.I_xy_mm4, 'mm⁴'], ['Polar J', geometry.polar_J_mm4, 'mm⁴'], ['Area retention', geometry.area_retention_ratio * 100, '%'], ['Bending retention X', geometry.bending_retention_ratio_x * 100, '%'], ['Bending retention Y', geometry.bending_retention_ratio_y * 100, '%'], ['Torsion retention', geometry.torsion_retention_ratio * 100, '%']].map(([label, value, unit]) => <div key={String(label)} className="rounded-lg border border-slate-800 bg-slate-950/70 p-3"><p className="text-xs text-slate-400">{label}</p><p className="mt-1 font-mono text-lg font-bold text-slate-100">{Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 })} <span className="text-xs font-normal text-slate-400">{unit}</span></p></div>)}
          </div>
          <div className="rounded-lg border border-amber-900/70 bg-amber-950/20 p-3 text-xs leading-5 text-amber-100"><AlertTriangle size={14} className="mr-1 inline" />{geometry.warning}</div>
          <p className="text-xs text-slate-400">Configuration: {geometry.phantom_type}; effective pixel spacing dy {geometry.pixel_spacing_mm[0]} / dx {geometry.pixel_spacing_mm[1]} mm. Input reference: synthetic phantom. Browser received: {geometryAt ? new Date(geometryAt).toLocaleString() : 'unavailable'}. Backend software version is not included in the response.</p>
        </div>}
      </section>

      <section className={panelClass} aria-labelledby="mirels-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><h2 id="mirels-title" className="text-xl font-bold text-white">Conventional Mirels score rule</h2><p className="mt-1 text-sm leading-5 text-slate-300">The service adds the four selected 1–3 point categories (total 4–12) and returns the rule's tier and historical fields. The score was developed for metastatic long-bone lesions; this page does not present its risk percentage or treatment recommendation as a patient-specific result.</p></div>
          <span className="rounded-full border border-amber-900 bg-amber-950/40 px-2.5 py-1 text-xs text-amber-100">POST /api/v1/biomechanics/mirels</span>
        </div>
        <p className="mt-3 rounded-lg border border-amber-900/70 bg-amber-950/20 p-3 text-xs leading-5 text-amber-100">Use only hypothetical category selections for this interface. No patient data, image, diagnosis, lesion report, or study reference is submitted. The score is not a clinical fracture-risk prediction or medical advice.</p>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <label className="text-xs font-semibold text-slate-300">Anatomic site<select className={selectClass} aria-label="Anatomic site" value={factors.site} onChange={(event) => setFactors({ ...factors, site: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>Upper extremity · 1</option><option value={2}>Lower extremity · 2</option><option value={3}>Peritrochanteric · 3</option></select></label>
          <label className="text-xs font-semibold text-slate-300">Pain category<select className={selectClass} aria-label="Pain category" value={factors.pain} onChange={(event) => setFactors({ ...factors, pain: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>Mild · 1</option><option value={2}>Moderate · 2</option><option value={3}>Functional · 3</option></select></label>
          <label className="text-xs font-semibold text-slate-300">Radiographic nature<select className={selectClass} aria-label="Radiographic nature" value={factors.nature} onChange={(event) => setFactors({ ...factors, nature: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>Blastic · 1</option><option value={2}>Mixed · 2</option><option value={3}>Lytic · 3</option></select></label>
          <label className="text-xs font-semibold text-slate-300">Lesion size relative to cortex<select className={selectClass} aria-label="Lesion size" value={factors.size} onChange={(event) => setFactors({ ...factors, size: event.target.value ? Number(event.target.value) : '' })}><option value="">Choose…</option><option value={1}>&lt; 1/3 · 1</option><option value={2}>1/3–2/3 · 2</option><option value={3}>&gt; 2/3 · 3</option></select></label>
        </div>
        <button className={`${actionClass} bg-amber-700 hover:bg-amber-600`} type="button" onClick={() => void computeMirels()} disabled={mirelsLoading}>{mirelsLoading ? <RotateCw size={15} className="animate-spin" /> : <Activity size={15} />}{mirelsLoading ? 'Calculating…' : 'Calculate hypothetical score'}</button>
        {mirelsError && <div role="alert" className="mt-3 rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">Score request failed: {mirelsError}<button type="button" onClick={() => void computeMirels()} disabled={mirelsLoading} className="ml-3 underline">Retry</button></div>}
        {!mirels && !mirelsLoading && !mirelsError && <p className="mt-4 rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-400">Choose all four hypothetical categories to enable a score calculation. No result is pre-filled.</p>}
        {mirels && <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/70 p-4" aria-label="Mirels score result">
          <p className="text-xs font-semibold text-slate-400">Rule output · not a patient risk estimate</p><p className="mt-1 text-2xl font-black text-amber-200">Total score: {mirels.total_score} / 12</p><p className="mt-1 text-sm text-slate-200">Rule tier: {mirels.recommendation_tier.replace(/_/g, ' ')}</p>
          <p className="mt-2 text-xs leading-5 text-slate-400">Selected factors: site {factors.site}, pain {factors.pain}, nature {factors.nature}, size {factors.size}. Browser received: {mirelsAt ? new Date(mirelsAt).toLocaleString() : 'unavailable'}. API disclaimer: {mirels.disclaimer} Backend software version is not included in the response.</p>
        </div>}
      </section>
      <p className="text-xs leading-5 text-slate-500">These APIs are stand-alone research calculations; they are not linked to a patient or staged dataset. Future governed data workflows must pass through Stage 38 onboarding and Stage 39 preparation before any permitted dataset analysis.</p>
    </main>
  );
};

export default BiomechanicsPage;
