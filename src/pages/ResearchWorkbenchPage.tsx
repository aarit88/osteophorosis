import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowRight, BookOpen, Database, FileCheck2, Layers, Lock, Play, RotateCw, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import { getResearchTracks, getTrackDDatasetReadiness, runSyntheticWorkbench } from '../api/client';
import type { ResearchTrackDetail, ResearchTracksResponse, SyntheticWorkbenchRunResponse, TrackDDatasetReadinessResponse } from '../api/types';

const linkClass = 'inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-200 hover:border-cyan-600 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400';
const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'The service could not complete this request.';
}

function humanize(value: string): string {
  return value.replace(/_/g, ' ');
}

export const ResearchWorkbenchPage: React.FC = () => {
  const [registry, setRegistry] = useState<ResearchTracksResponse | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<ResearchTrackDetail['track_id'] | ''>('');
  const [tracksLoading, setTracksLoading] = useState(true);
  const [tracksError, setTracksError] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<TrackDDatasetReadinessResponse | null>(null);
  const [readinessLoading, setReadinessLoading] = useState(false);
  const [readinessError, setReadinessError] = useState<string | null>(null);
  const [demo, setDemo] = useState<SyntheticWorkbenchRunResponse | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState<string | null>(null);
  const submitLock = useRef(false);

  const loadTracks = useCallback(async () => {
    setTracksLoading(true);
    setTracksError(null);
    try {
      const response = await getResearchTracks();
      setRegistry(response);
      setSelectedTrackId((current) => current && response.tracks.some((track) => track.track_id === current)
        ? current
        : response.tracks[0]?.track_id ?? '');
    } catch (error) {
      setTracksError(errorMessage(error));
    } finally {
      setTracksLoading(false);
    }
  }, []);

  useEffect(() => { void loadTracks(); }, [loadTracks]);

  const selectedTrack = registry?.tracks.find((track) => track.track_id === selectedTrackId) ?? null;
  const isTrackD = selectedTrack?.track_id === 'track_d';

  const loadReadiness = useCallback(async () => {
    setReadinessLoading(true);
    setReadinessError(null);
    try {
      setReadiness(await getTrackDDatasetReadiness());
    } catch (error) {
      setReadiness(null);
      setReadinessError(errorMessage(error));
    } finally {
      setReadinessLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isTrackD && !readiness && !readinessLoading && !readinessError) void loadReadiness();
  }, [isTrackD, readiness, readinessLoading, readinessError, loadReadiness]);

  const selectTrack = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const nextId = event.target.value as ResearchTrackDetail['track_id'];
    setSelectedTrackId(nextId);
    setDemo(null);
    setDemoError(null);
  };

  const runDemo = async () => {
    if (submitLock.current || demoLoading || isTrackD || !selectedTrack) return;
    submitLock.current = true;
    setDemoLoading(true);
    setDemoError(null);
    setDemo(null);
    try {
      setDemo(await runSyntheticWorkbench({}));
    } catch (error) {
      setDemoError(errorMessage(error));
    } finally {
      submitLock.current = false;
      setDemoLoading(false);
    }
  };

  const resetDemo = () => {
    if (demoLoading) return;
    setDemo(null);
    setDemoError(null);
  };

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6">
      <header className="flex flex-col gap-4 border-b border-slate-800 pb-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl space-y-2">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200"><Activity size={14} /> Research workspace</div>
          <h1 className="text-3xl font-black tracking-tight text-white sm:text-4xl">Workbench</h1>
          <p className="text-sm leading-6 text-slate-300">Inspect the available research tracks and their current constraints. This workspace exposes the existing synthetic methods demonstration; it does not provide track-specific clinical image inference.</p>
        </div>
        <nav aria-label="Related research pages" className="flex flex-wrap gap-2">
          <Link className={linkClass} to="/tracks"><Layers size={16} /> Tracks</Link>
          <Link className={linkClass} to="/governance"><ShieldCheck size={16} /> Governance</Link>
          <Link className={linkClass} to="/evidence"><FileCheck2 size={16} /> Evidence</Link>
          <Link className={linkClass} to="/results"><BookOpen size={16} /> Results</Link>
        </nav>
      </header>

      <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,.8fr)]" aria-label="Track selection and readiness">
        <div className={panelClass}>
          <div className="mb-4 flex items-center gap-2 text-sm font-bold text-white"><Layers size={17} className="text-cyan-300" /> Select a registered research track</div>
          {tracksLoading && <p role="status" className="text-sm text-slate-300">Loading track registry…</p>}
          {tracksError && <div className="space-y-3" role="alert"><p className="text-sm text-rose-300">Research tracks are unavailable: {tracksError}</p><button className={linkClass} onClick={() => void loadTracks()} type="button"><RotateCw size={15} /> Retry</button></div>}
          {!tracksLoading && !tracksError && registry?.tracks.length === 0 && <p className="text-sm text-slate-300">The backend returned no available research tracks.</p>}
          {!tracksLoading && !tracksError && registry && registry.tracks.length > 0 && <>
            <label htmlFor="research-track" className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-400">Research track</label>
            <select id="research-track" aria-label="Research track" value={selectedTrackId} onChange={selectTrack} className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30">
              {registry.tracks.map((track) => <option key={track.track_id} value={track.track_id}>{track.track_id.toUpperCase().replace('_', ' ')} · {track.title}</option>)}
            </select>
            {selectedTrack && <div className="mt-4 space-y-3" aria-live="polite">
              <h2 className="text-lg font-bold text-white">{selectedTrack.title}</h2>
              <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                <div><dt className="text-xs text-slate-400">Modality</dt><dd className="mt-1 text-slate-100">{selectedTrack.modality}</dd></div>
                <div><dt className="text-xs text-slate-400">Research task</dt><dd className="mt-1 text-slate-100">{selectedTrack.task_type}</dd></div>
                <div><dt className="text-xs text-slate-400">Track status</dt><dd className="mt-1 text-slate-100">{humanize(selectedTrack.status)}</dd></div>
                <div><dt className="text-xs text-slate-400">Governance</dt><dd className="mt-1 text-slate-100">{humanize(selectedTrack.governance_clearance)}</dd></div>
              </dl>
              <p className="text-xs leading-5 text-slate-400">{selectedTrack.disclaimer}</p>
              {selectedTrack.limitations.length > 0 && <ul className="list-inside list-disc space-y-1 text-xs text-amber-200">{selectedTrack.limitations.map((item) => <li key={item}>{item}</li>)}</ul>}
            </div>}
          </>}
        </div>

        <aside className={`${panelClass} ${isTrackD ? 'border-rose-900/80 bg-rose-950/20' : ''}`} aria-label="Model readiness">
          <div className="mb-3 flex items-center gap-2 text-sm font-bold text-white"><Lock size={17} className={isTrackD ? 'text-rose-300' : 'text-emerald-300'} /> Model and workflow readiness</div>
          {isTrackD ? <>
            <p className="text-sm font-bold text-rose-200">FOUR_CLASS_MODEL_NOT_READY</p>
            {readinessLoading && <p role="status" className="mt-2 text-sm text-slate-300">Checking Track D dataset readiness…</p>}
            {readinessError && <div role="alert" className="mt-3 space-y-3"><p className="text-sm text-rose-200">Readiness service unavailable: {readinessError}</p><button className={linkClass} type="button" onClick={() => void loadReadiness()}><RotateCw size={15} /> Retry readiness check</button></div>}
            {readiness && <>
              <p className="mt-2 text-sm text-slate-200">Backend dataset status: <strong>{humanize(readiness.readiness_status)}</strong></p>
              <p className="mt-2 text-xs leading-5 text-slate-300">Four-class inference is unavailable while genuine dataset, validation and governance requirements remain incomplete. The server-side 503 safety gate remains in force.</p>
              {readiness.blockers.length > 0 && <ul className="mt-3 list-inside list-disc space-y-1 text-xs text-rose-100">{readiness.blockers.map((item) => <li key={item}>{item}</li>)}</ul>}
            </>}
            <Link className={`${linkClass} mt-4`} to="/governance"><Database size={15} /> Review governance <ArrowRight size={14} /></Link>
          </> : selectedTrack ? <>
            <p className="text-sm font-semibold text-slate-100">Track registry status: {humanize(selectedTrack.status)}</p>
            <p className="mt-2 text-xs leading-5 text-slate-400">This page reports registry metadata only. It does not test checkpoint loading or authorize an inference operation.</p>
            <Link className={`${linkClass} mt-4`} to="/evidence"><FileCheck2 size={15} /> Inspect model evidence <ArrowRight size={14} /></Link>
          </> : <p className="text-sm text-slate-400">Select an available track to inspect its registry status.</p>}
        </aside>
      </section>

      <section className={panelClass} aria-labelledby="methods-demo-title">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-3xl">
            <div className="mb-2 inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-cyan-200"><Activity size={15} /> Existing backend capability</div>
            <h2 id="methods-demo-title" className="text-xl font-extrabold text-white">Synthetic methods demonstration</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">Runs the existing deterministic mathematical phantom endpoint for radiomics, geometry, explainability and uncertainty components. It is not an analysis of an uploaded image, is not associated with the selected track, and provides no clinical predictions.</p>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <button className="inline-flex items-center gap-2 rounded-lg bg-cyan-700 px-4 py-3 text-sm font-bold text-white hover:bg-cyan-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300 disabled:cursor-not-allowed disabled:opacity-50" type="button" onClick={() => void runDemo()} disabled={demoLoading || !selectedTrack || Boolean(tracksError) || isTrackD} aria-describedby="demo-safety-note">
              {demoLoading ? <><RotateCw className="animate-spin" size={16} /> Running demonstration…</> : <><Play size={16} /> Run synthetic demonstration</>}
            </button>
            {(demo || demoError) && <button className={linkClass} type="button" onClick={resetDemo} disabled={demoLoading}>Reset</button>}
          </div>
        </div>
        {isTrackD && <p className="mt-4 rounded-lg border border-rose-900 bg-rose-950/40 p-3 text-sm text-rose-200">The demonstration action is disabled while Track D is selected. No synthetic prediction or result is shown as a substitute for the gated four-class model.</p>}
        <p id="demo-safety-note" className="mt-3 text-xs text-slate-400">No image or patient data is accepted by this workbench flow. Track modalities above are informational; track-specific image upload and inference are not connected here.</p>
        {demoLoading && <p role="status" className="mt-4 text-sm text-cyan-200">Waiting for the synthetic demonstration service. The API does not provide progress updates.</p>}
        {demoError && <div className="mt-4 rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-200" role="alert">Demonstration failed: {demoError}<button type="button" className="ml-3 underline" onClick={() => void runDemo()} disabled={demoLoading || isTrackD}>Retry</button></div>}
        {demo && <div className="mt-4 rounded-lg border border-cyan-900 bg-cyan-950/20 p-4" role="status">
          <div className="flex items-center gap-2 text-sm font-bold text-cyan-100"><ShieldCheck size={17} /> Synthetic demonstration completed</div>
          <p className="mt-2 text-sm text-slate-200">Backend status: {humanize(demo.status)}</p>
          <p className="mt-1 text-xs leading-5 text-slate-400">{demo.disclaimer} Outputs are mathematical demonstration values only. They are not patient results, model predictions, validation metrics or evidence of clinical readiness.</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Synthetic methods outputs">
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3"><h3 className="text-xs font-bold text-cyan-200">Radiomics phantom</h3><p className="mt-2 text-sm text-slate-100">{demo.radiomics_panel.phantom_type} · {demo.radiomics_panel.grid_size} × {demo.radiomics_panel.grid_size}</p><p className="mt-1 text-xs text-slate-400">Verified features: {demo.radiomics_panel.verified_features_count} / {demo.radiomics_panel.total_features_count}</p></div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3"><h3 className="text-xs font-bold text-teal-200">Geometry phantom</h3><p className="mt-2 text-sm text-slate-100">{demo.biomechanics_geometry_panel.phantom_type}</p><p className="mt-1 text-xs text-slate-400">Area {demo.biomechanics_geometry_panel.area_mm2.toFixed(1)} mm² · polar J {demo.biomechanics_geometry_panel.polar_J_mm4.toFixed(1)} mm⁴</p></div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3"><h3 className="text-xs font-bold text-amber-200">Mirels calculation fixture</h3><p className="mt-2 text-sm text-slate-100">Score {demo.biomechanics_mirels_panel.total_score} · {humanize(demo.biomechanics_mirels_panel.recommendation_tier)}</p><p className="mt-1 text-xs text-slate-400">Synthetic inputs only; not patient risk assessment.</p></div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3"><h3 className="text-xs font-bold text-violet-200">Uncertainty / attribution methods</h3><p className="mt-2 text-sm text-slate-100">{demo.explainability_panel.method} · {humanize(demo.uncertainty_panel.decision)}</p><p className="mt-1 text-xs text-slate-400">Normalized entropy {demo.uncertainty_panel.normalized_entropy.toFixed(3)} (demonstration only)</p></div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Link className={linkClass} to="/radiomics">Radiomics <ArrowRight size={14} /></Link>
            <Link className={linkClass} to="/biomechanics">Biomechanics <ArrowRight size={14} /></Link>
            <Link className={linkClass} to="/explainability">Explainability <ArrowRight size={14} /></Link>
            <Link className={linkClass} to="/uncertainty">Uncertainty <ArrowRight size={14} /></Link>
          </div>
        </div>}
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 text-xs leading-5 text-slate-400">
        <strong className="text-slate-200">Input and activity availability:</strong> this route has no authorized clinical image/study selection workflow and no backend recent-analysis history endpoint. No upload, clinical analysis or fabricated recent activity is displayed. The Results route reflects the existing session result flow only.
      </section>
      <footer className="flex flex-wrap gap-2 border-t border-slate-800 pt-5" aria-label="Research navigation">
        <Link className={linkClass} to="/tracks">Tracks</Link><Link className={linkClass} to="/governance">Governance</Link><Link className={linkClass} to="/evidence">Evidence</Link><Link className={linkClass} to="/results">Results</Link>
      </footer>
    </main>
  );
};

export default ResearchWorkbenchPage;
