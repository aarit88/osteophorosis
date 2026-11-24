import React, { useCallback, useEffect, useState } from 'react';
import { Activity, ArrowRight, BookOpen, Database, FileCheck2, Layers, LockKeyhole, RotateCw, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import { getResearchTracks, getTrackDDatasetReadiness } from '../api/client';
import type { ResearchTrackDetail, ResearchTracksResponse, TrackDDatasetReadinessResponse } from '../api/types';

const linkClass = 'inline-flex min-h-10 items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-200 hover:border-cyan-600 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400';
const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';
const methodLinks = [
  { title: 'Radiomics', path: '/radiomics', description: 'Feature families are documented; clinical image input is not connected.' },
  { title: 'Biomechanics', path: '/biomechanics', description: 'Manual Mirels rule calculation is available; image-derived mechanics are not.' },
  { title: 'Explainability', path: '/explainability', description: 'Model-specific clinical attribution is unavailable.' },
  { title: 'Uncertainty', path: '/uncertainty', description: 'Patient-specific calibrated uncertainty is unavailable.' },
];

function safeErrorMessage(): string {
  return 'The service could not complete this request. Check the API connection and retry.';
}

function humanize(value: string): string { return value.replace(/_/g, ' '); }

export const ResearchWorkbenchPage: React.FC = () => {
  const [registry, setRegistry] = useState<ResearchTracksResponse | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<ResearchTrackDetail['track_id'] | ''>('');
  const [tracksLoading, setTracksLoading] = useState(true);
  const [tracksError, setTracksError] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<TrackDDatasetReadinessResponse | null>(null);
  const [readinessLoading, setReadinessLoading] = useState(false);
  const [readinessError, setReadinessError] = useState<string | null>(null);

  const loadTracks = useCallback(async () => {
    setTracksLoading(true);
    setTracksError(null);
    try {
      const response = await getResearchTracks();
      setRegistry(response);
      setSelectedTrackId((current) => current && response.tracks.some((track) => track.track_id === current) ? current : response.tracks[0]?.track_id ?? '');
    } catch {
      setTracksError(safeErrorMessage());
    } finally { setTracksLoading(false); }
  }, []);

  const loadReadiness = useCallback(async () => {
    setReadinessLoading(true);
    setReadinessError(null);
    try { setReadiness(await getTrackDDatasetReadiness()); }
    catch { setReadiness(null); setReadinessError(safeErrorMessage()); }
    finally { setReadinessLoading(false); }
  }, []);

  useEffect(() => { void loadTracks(); }, [loadTracks]);
  const selectedTrack = registry?.tracks.find((track) => track.track_id === selectedTrackId) ?? null;
  const isTrackD = selectedTrack?.track_id === 'track_d';
  useEffect(() => { if (isTrackD && !readiness && !readinessLoading && !readinessError) void loadReadiness(); }, [isTrackD, readiness, readinessLoading, readinessError, loadReadiness]);

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6">
      <header className="flex flex-col gap-4 border-b border-slate-800 pb-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl space-y-2"><div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200"><Activity size={14} aria-hidden="true" /> Research workspace</div><h1 className="text-3xl font-black tracking-tight text-white sm:text-4xl">Workbench</h1><p className="text-sm leading-6 text-slate-300">Inspect the live track registry, current Track D readiness, and links to supported research modules. No synthetic image-derived outputs or simulated clinical predictions are generated here.</p></div>
        <nav aria-label="Related research pages" className="flex flex-wrap gap-2"><Link className={linkClass} to="/tracks"><Layers size={16} aria-hidden="true" />Tracks</Link><Link className={linkClass} to="/governance"><ShieldCheck size={16} aria-hidden="true" />Governance</Link><Link className={linkClass} to="/evidence"><FileCheck2 size={16} aria-hidden="true" />Evidence</Link><Link className={linkClass} to="/results"><BookOpen size={16} aria-hidden="true" />Results</Link></nav>
      </header>

      <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,.8fr)]" aria-label="Track selection and readiness">
        <div className={panelClass}>
          <h2 className="mb-4 flex items-center gap-2 text-sm font-bold text-white"><Layers size={17} className="text-cyan-300" aria-hidden="true" />Select a registered research track</h2>
          {tracksLoading && <p role="status" className="text-sm text-slate-300">Loading track registry…</p>}
          {tracksError && <div className="space-y-3" role="alert"><p className="text-sm text-rose-200">Research tracks are unavailable. {tracksError}</p><button className={linkClass} onClick={() => void loadTracks()} type="button"><RotateCw size={15} aria-hidden="true" />Retry</button></div>}
          {!tracksLoading && !tracksError && registry?.tracks.length === 0 && <p role="status" className="text-sm text-slate-300">The backend returned no available research tracks.</p>}
          {!tracksLoading && !tracksError && registry && registry.tracks.length > 0 && <>
            <label htmlFor="research-track" className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-400">Research track</label>
            <select id="research-track" aria-label="Research track" value={selectedTrackId} onChange={(event) => setSelectedTrackId(event.target.value as ResearchTrackDetail['track_id'])} className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30">{registry.tracks.map((track) => <option key={track.track_id} value={track.track_id}>{track.track_id.toUpperCase().replace('_', ' ')} · {track.title}</option>)}</select>
            {selectedTrack && <div className="mt-4 space-y-3" aria-live="polite"><h3 className="text-lg font-bold text-white">{selectedTrack.title}</h3><dl className="grid gap-3 text-sm sm:grid-cols-2"><div><dt className="text-xs text-slate-400">Modality</dt><dd className="mt-1 text-slate-100">{selectedTrack.modality}</dd></div><div><dt className="text-xs text-slate-400">Research task</dt><dd className="mt-1 text-slate-100">{selectedTrack.task_type}</dd></div><div><dt className="text-xs text-slate-400">Track status</dt><dd className="mt-1 text-slate-100">{humanize(selectedTrack.status)}</dd></div><div><dt className="text-xs text-slate-400">Governance</dt><dd className="mt-1 text-slate-100">{humanize(selectedTrack.governance_clearance)}</dd></div></dl><p className="text-xs leading-5 text-slate-400">{selectedTrack.disclaimer}</p>{selectedTrack.limitations.length > 0 && <ul className="list-inside list-disc space-y-1 text-xs text-amber-200">{selectedTrack.limitations.map((item) => <li key={item}>{item}</li>)}</ul>}</div>}
          </>}
        </div>

        <aside className={`${panelClass} ${isTrackD ? 'border-rose-900/80 bg-rose-950/20' : ''}`} aria-label="Model readiness">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-bold text-white"><LockKeyhole size={17} className={isTrackD ? 'text-rose-300' : 'text-cyan-300'} aria-hidden="true" />Model and workflow readiness</h2>
          {isTrackD ? <><p className="text-sm font-bold text-rose-200">FOUR_CLASS_MODEL_NOT_READY</p>{readinessLoading && <p role="status" className="mt-2 text-sm text-slate-300">Checking Track D dataset readiness…</p>}{readinessError && <div role="alert" className="mt-3 space-y-3"><p className="text-sm text-rose-200">Readiness is unavailable. {readinessError}</p><button className={linkClass} type="button" onClick={() => void loadReadiness()}><RotateCw size={15} aria-hidden="true" />Retry readiness check</button></div>}{readiness && <><p className="mt-2 text-sm text-slate-200">Backend dataset status: <strong>{humanize(readiness.readiness_status)}</strong></p><p className="mt-2 text-xs leading-5 text-slate-300">Four-class inference is unavailable while genuine dataset, validation, and governance requirements remain incomplete. The server-side 503 guard remains in force.</p>{readiness.blockers.length > 0 && <ul className="mt-3 list-inside list-disc space-y-1 text-xs text-rose-100">{readiness.blockers.map((item) => <li key={item}>{item}</li>)}</ul>}</>}<Link className={`${linkClass} mt-4`} to="/governance"><Database size={15} aria-hidden="true" />Review governance<ArrowRight size={14} aria-hidden="true" /></Link></> : selectedTrack ? <><p className="text-sm font-semibold text-slate-100">Track registry status: {humanize(selectedTrack.status)}</p><p className="mt-2 text-xs leading-5 text-slate-400">Registry information does not verify checkpoint loading or authorize inference.</p><Link className={`${linkClass} mt-4`} to="/evidence"><FileCheck2 size={15} aria-hidden="true" />Inspect model evidence<ArrowRight size={14} aria-hidden="true" /></Link></> : <p className="text-sm text-slate-400">Select an available track to inspect its registry status.</p>}
        </aside>
      </section>

      <section className={panelClass} aria-labelledby="methods-title">
        <h2 id="methods-title" className="text-xl font-bold text-white">Research module availability</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">These routes describe implemented methods and their current input limitations. Generated phantom values are not produced from this user-facing workspace.</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{methodLinks.map((item) => <article key={item.path} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4"><h3 className="font-semibold text-slate-100">{item.title}</h3><p className="mt-2 min-h-10 text-xs leading-5 text-slate-400">{item.description}</p><Link className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-cyan-200 underline underline-offset-4" to={item.path}>Open module <ArrowRight size={13} aria-hidden="true" /></Link></article>)}</div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 text-xs leading-5 text-slate-400"><strong className="text-slate-200">Input and activity:</strong> this route has no authorized clinical image/study selection workflow and no backend recent-analysis history endpoint. No upload, clinical analysis, synthetic output, or fabricated recent activity is shown. The Results route displays only an in-memory response from an authorized backend request.</section>
      <footer className="flex flex-wrap gap-2 border-t border-slate-800 pt-5" aria-label="Research navigation"><Link className={linkClass} to="/tracks">Tracks</Link><Link className={linkClass} to="/governance">Governance</Link><Link className={linkClass} to="/evidence">Evidence</Link><Link className={linkClass} to="/results">Results</Link></footer>
    </main>
  );
};

export default ResearchWorkbenchPage;
