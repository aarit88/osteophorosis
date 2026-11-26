import React, { useRef, useState } from 'react';
import { Activity, AlertTriangle, Box, RotateCw, Ruler } from 'lucide-react';
import { runBiomechanicsGeometry, runBiomechanicsMirels } from '../api/client';
import type { BiomechanicsGeometryResponse, BiomechanicsMirelsResponse } from '../api/types';

const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5 sm:p-6';
const selectClass = 'mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100';
const actionClass = 'mt-4 inline-flex min-h-10 items-center gap-2 rounded-lg bg-amber-700 px-4 py-2.5 text-sm font-bold text-white hover:bg-amber-600 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-amber-300';
type MirelsFactors = { site: number | ''; pain: number | ''; nature: number | ''; size: number | '' };

const GEOMETRY_PHANTOMS = [
  { id: 'intact_femur_diaphysis', label: 'Intact Femoral Diaphysis (Ring Baseline)' },
  { id: 'focal_anterior_defect', label: 'Focal Anterior Cortical Defect (Lytic)' },
  { id: 'eccentric_thinning', label: "Eccentric Cortical Remodeling (OI / Paget's)" },
];

export const BiomechanicsPage: React.FC = () => {
  const [phantomType, setPhantomType] = useState('intact_femur_diaphysis');
  const [voxelSize, setVoxelSize] = useState(0.5);
  const [geometryResult, setGeometryResult] = useState<BiomechanicsGeometryResponse | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);
  const [geometryError, setGeometryError] = useState<string | null>(null);

  const [factors, setFactors] = useState<MirelsFactors>({ site: '', pain: '', nature: '', size: '' });
  const [result, setResult] = useState<BiomechanicsMirelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestLock = useRef(false);

  const calculateGeometry = async () => {
    setGeometryLoading(true);
    setGeometryError(null);
    try {
      setGeometryResult(
        await runBiomechanicsGeometry({ phantom_type: phantomType, voxel_size_mm: voxelSize }),
      );
    } catch {
      setGeometryError('The geometry calculation service is unavailable.');
    } finally {
      setGeometryLoading(false);
    }
  };

  const calculateMirels = async () => {
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
      setResult(
        await runBiomechanicsMirels({
          site: factors.site as number,
          pain: factors.pain as number,
          radiographic_nature: factors.nature as number,
          lesion_size: factors.size as number,
        }),
      );
    } catch {
      setError('The scoring service is unavailable. No result was created.');
    } finally {
      requestLock.current = false;
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <header className="space-y-3 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-teal-800 bg-teal-950/60 px-3 py-1 text-xs font-semibold text-teal-200">
          <Ruler size={14} aria-hidden="true" /> Structural Mechanics &amp; Mirels Engine
        </div>
        <h1 className="text-3xl font-black tracking-tight text-white">Biomechanics &amp; Fracture Risk</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-300">
          Evaluate 2D cortical cross-sectional moments of inertia on synthetic skeletal phantoms and compute classical 4-factor Mirels (1989) pathological fracture triage scores.
        </p>
      </header>

      {/* Cross-Section Geometry Phantom Analyzer */}
      <section className={panelClass} aria-labelledby="geometry-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 id="geometry-title" className="text-xl font-bold text-white">Cross-section geometry (Synthetic Phantom)</h2>
            <p className="mt-1 text-sm leading-5 text-slate-300">
              Computes cross-sectional area (mm²), principal moments (I_xx, I_yy), polar torsion moment (J in mm⁴), and retention ratios relative to an intact diaphysis.
            </p>
          </div>
          <span className="rounded-full border border-teal-900 bg-teal-950/40 px-2.5 py-1 text-xs font-mono text-teal-200">
            POST /api/v1/biomechanics/geometry
          </span>
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <label className="text-xs font-semibold text-slate-300">
            Cortical Cross-Section Phantom
            <select
              aria-label="Cortical Cross-Section Phantom"
              value={phantomType}
              onChange={(e) => setPhantomType(e.target.value)}
              className={selectClass}
            >
              {GEOMETRY_PHANTOMS.map((g) => (
                <option key={g.id} value={g.id}>{g.label}</option>
              ))}
            </select>
          </label>

          <label className="text-xs font-semibold text-slate-300">
            Voxel Resolution (mm)
            <select
              aria-label="Voxel Resolution"
              value={voxelSize}
              onChange={(e) => setVoxelSize(Number(e.target.value))}
              className={selectClass}
            >
              <option value={0.5}>0.5 mm</option>
              <option value={1.0}>1.0 mm</option>
            </select>
          </label>
        </div>

        <button
          type="button"
          onClick={() => void calculateGeometry()}
          disabled={geometryLoading}
          className="mt-4 inline-flex min-h-10 items-center gap-2 rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-bold text-white hover:bg-teal-600 disabled:opacity-50"
        >
          {geometryLoading ? <RotateCw size={15} className="animate-spin" aria-hidden="true" /> : <Box size={15} aria-hidden="true" />}
          {geometryLoading ? 'Computing Moments…' : 'Calculate geometry'}
        </button>

        {geometryError && <p role="alert" className="mt-3 rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">{geometryError}</p>}

        {geometryResult && (
          <div className="mt-4 space-y-3 border-t border-slate-800 pt-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 font-mono">
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Cortical Area</span>
                <p className="mt-1 text-lg font-black text-white">{geometryResult.area_mm2} mm²</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Polar Moment J</span>
                <p className="mt-1 text-lg font-black text-teal-300">{geometryResult.polar_J_mm4} mm⁴</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Area Retention</span>
                <p className="mt-1 text-lg font-black text-cyan-300">{(geometryResult.area_retention_ratio * 100).toFixed(1)}%</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <span className="text-[11px] text-slate-400">Torsion Retention</span>
                <p className="mt-1 text-lg font-black text-amber-300">{(geometryResult.torsion_retention_ratio * 100).toFixed(1)}%</p>
              </div>
            </div>
            <p className="text-xs text-amber-200/90 flex items-center gap-2">
              <AlertTriangle size={14} className="shrink-0" aria-hidden="true" />
              {geometryResult.warning}
            </p>
          </div>
        )}
      </section>

      {/* Mirels 4-Factor Calculator */}
      <section className={panelClass} aria-labelledby="mirels-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 id="mirels-title" className="text-xl font-bold text-white">Mirels score calculator</h2>
            <p className="mt-1 text-sm leading-5 text-slate-300">
              Classical 4-factor scoring system (site, pain, radiographic nature, lesion size; total 4–12) estimating pathological fracture vulnerability.
            </p>
          </div>
          <span className="rounded-full border border-amber-900 bg-amber-950/40 px-2.5 py-1 text-xs font-mono text-amber-100">
            POST /api/v1/biomechanics/mirels
          </span>
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <label className="text-xs font-semibold text-slate-300">
            Anatomic site
            <select className={selectClass} aria-label="Anatomic site" value={factors.site} onChange={(event) => setFactors({ ...factors, site: event.target.value ? Number(event.target.value) : '' })}>
              <option value="">Choose…</option>
              <option value={1}>Upper extremity · 1</option>
              <option value={2}>Lower extremity · 2</option>
              <option value={3}>Peritrochanteric · 3</option>
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300">
            Pain category
            <select className={selectClass} aria-label="Pain category" value={factors.pain} onChange={(event) => setFactors({ ...factors, pain: event.target.value ? Number(event.target.value) : '' })}>
              <option value="">Choose…</option>
              <option value={1}>Mild · 1</option>
              <option value={2}>Moderate · 2</option>
              <option value={3}>Functional · 3</option>
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300">
            Radiographic nature
            <select className={selectClass} aria-label="Radiographic nature" value={factors.nature} onChange={(event) => setFactors({ ...factors, nature: event.target.value ? Number(event.target.value) : '' })}>
              <option value="">Choose…</option>
              <option value={1}>Blastic · 1</option>
              <option value={2}>Mixed · 2</option>
              <option value={3}>Lytic · 3</option>
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300">
            Lesion size relative to cortex
            <select className={selectClass} aria-label="Lesion size" value={factors.size} onChange={(event) => setFactors({ ...factors, size: event.target.value ? Number(event.target.value) : '' })}>
              <option value="">Choose…</option>
              <option value={1}>&lt; 1/3 · 1</option>
              <option value={2}>1/3–2/3 · 2</option>
              <option value={3}>&gt; 2/3 · 3</option>
            </select>
          </label>
        </div>

        <button className={actionClass} type="button" onClick={() => void calculateMirels()} disabled={loading}>
          {loading ? <RotateCw size={15} className="animate-spin" aria-hidden="true" /> : <Activity size={15} aria-hidden="true" />}
          {loading ? 'Calculating…' : 'Calculate Mirels Fracture Risk'}
        </button>

        {error && (
          <div role="alert" className="mt-3 rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200">
            {error}
            <button type="button" onClick={() => void calculateMirels()} disabled={loading} className="ml-3 underline">Retry</button>
          </div>
        )}

        {!result && !loading && !error && (
          <p className="mt-4 rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-400">
            Choose all four scoring categories above to compute the Mirels fracture score and orthopedic fixation threshold.
          </p>
        )}

        {result && (
          <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/75 p-5 space-y-3" role="status">
            <div className="grid gap-3 sm:grid-cols-3">
              <div>
                <p className="text-xs font-semibold text-slate-400">Mirels Composite Score</p>
                <p className="mt-1 text-2xl font-black text-amber-200 font-mono">Total score: {result.total_score} / 12</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-slate-400">1989 Cohort Fracture Risk</p>
                <p className="mt-1 text-2xl font-black text-cyan-300 font-mono">{result.fracture_risk_percent}%</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-slate-400">Prophylactic Fixation</p>
                <p className={`mt-1 text-base font-black font-mono ${result.prophylactic_fixation_indicated ? 'text-rose-300' : 'text-emerald-300'}`}>
                  {result.recommendation_tier.replace(/_/g, ' ')} {result.prophylactic_fixation_indicated ? '· INDICATED' : '· SURVEILLANCE'}
                </p>
              </div>
            </div>
            <p className="text-sm leading-6 text-slate-200">{result.clinical_guidance}</p>
            <p className="text-xs text-slate-400 border-t border-slate-800 pt-2">{result.disclaimer}</p>
          </div>
        )}
      </section>
    </main>
  );
};

export default BiomechanicsPage;
