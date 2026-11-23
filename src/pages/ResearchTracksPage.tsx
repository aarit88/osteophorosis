import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, Database, FileCheck2, Layers, LockKeyhole, RotateCw } from 'lucide-react';
import { Link } from 'react-router-dom';
import { getResearchTracks } from '../api/client';
import type { ResearchTrackDetail, ResearchTracksResponse } from '../api/types';

const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';
const buttonClass = 'inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-200 hover:border-cyan-600 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400';

function description(track: ResearchTrackDetail) {
  switch (track.track_id) {
    case 'track_a': return 'Historical binary fracture research using the FracAtlas radiography cohort.';
    case 'track_b': return 'Historical binary skeletal metastasis assessment from a Tc-99m bone scintigraphy cohort.';
    case 'track_c': return 'Frozen three-class primary bone tumour research with documented cross-center generalization limits.';
    case 'track_d': return 'Target four-class differential research pathway; the model and clinical inference remain unavailable.';
  }
}

function sourceStatus(track: ResearchTrackDetail): string {
  if (track.track_id === 'track_d') return 'Clinical cohorts unavailable or pending governance; see Governance for acquisition status.';
  return `Registry source context: ${track.governance_clearance}. This page does not independently verify current dataset or checkpoint files.`;
}

function modelAvailability(track: ResearchTrackDetail): string {
  if (track.track_id === 'track_d') return 'No trained four-class model is available; inference remains gated.';
  return 'Registry records a historical frozen research track. This page does not load checkpoints or verify current artifact availability.';
}

function evaluationStatus(track: ResearchTrackDetail): string {
  if (track.track_id === 'track_d') return 'No clinical model evaluation is available for the four-class target.';
  if (track.track_id === 'track_c') return 'Historical multi-center evaluation recorded; severe generalization degradation is documented. External holdouts remain protected.';
  return 'Historical evaluation evidence is recorded. Burned holdout partitions are preserved and are not re-evaluated here.';
}

function moduleLink(track: ResearchTrackDetail): { href: string; label: string } {
  if (track.track_id === 'track_d') return { href: '/governance', label: 'Review governance status' };
  return { href: '/workbench', label: 'Open research methods workbench' };
}

export const ResearchTracksPage: React.FC = () => {
  const [data, setData] = useState<ResearchTracksResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadTracks = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getResearchTracks());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Research tracks could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadTracks(); }, [loadTracks]);

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <header className="space-y-3 border-b border-slate-800 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200"><Layers size={14} /> Research governance and track separation</div>
        <h1 className="text-3xl font-black tracking-tight text-white">Research tracks</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-300">Tracks are separate research efforts with different datasets, tasks and limitations. Their labels and results must not be combined into the proposed Track D diagnosis system.</p>
      </header>

      {loading && <p role="status" className="rounded-xl border border-slate-800 p-6 text-sm text-slate-300">Loading the backend track registry…</p>}
      {error && <div role="alert" className="rounded-xl border border-rose-800 bg-rose-950/30 p-4 text-sm text-rose-200"><p>Track registry unavailable: {error}</p><button className={`${buttonClass} mt-3`} type="button" onClick={() => void loadTracks()}><RotateCw size={15} /> Retry</button></div>}
      {!loading && !error && data?.tracks.length === 0 && <p className="rounded-xl border border-slate-800 p-6 text-sm text-slate-300">The backend returned no registered research tracks.</p>}

      {data && <section className="space-y-4" aria-label="Registered research tracks">
        {data.tracks.map((track) => {
          const link = moduleLink(track);
          const isD = track.track_id === 'track_d';
          const isC = track.track_id === 'track_c';
          return <article key={track.track_id} className={`${panelClass} ${isD ? 'border-rose-900/70' : isC ? 'border-amber-900/70' : ''}`}>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xs font-bold uppercase tracking-wide text-cyan-300">{track.track_id.toUpperCase().replace('_', ' ')} · {track.status.replace(/_/g, ' ')}</p>
                <h2 className="mt-1 text-xl font-extrabold text-white">{track.title}</h2>
                {isD && <p className="mt-1 font-mono text-xs font-semibold text-rose-200">FOUR_CLASS_MODEL_NOT_READY</p>}
              </div>
              <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${isD ? 'border-rose-800 bg-rose-950/50 text-rose-200' : isC ? 'border-amber-800 bg-amber-950/40 text-amber-100' : 'border-slate-700 bg-slate-950 text-slate-200'}`}>
                {isD || isC ? <LockKeyhole size={13} /> : <FileCheck2 size={13} />}{isD ? 'Inference blocked' : isC ? 'Frozen · limitations apply' : 'Historical research'}
              </span>
            </div>
            <p className="mt-3 text-sm leading-5 text-slate-300"><strong className="text-slate-100">Objective:</strong> {track.task_type}. {description(track)}</p>
            <dl className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"><dt className="flex items-center gap-1.5 text-xs font-semibold text-slate-400"><Layers size={13} /> Modality</dt><dd className="mt-1 text-sm text-slate-100">{track.modality}</dd></div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"><dt className="flex items-center gap-1.5 text-xs font-semibold text-slate-400"><Database size={13} /> Dataset/source status</dt><dd className="mt-1 text-sm leading-5 text-slate-100">{sourceStatus(track)}</dd></div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"><dt className="text-xs font-semibold text-slate-400">Model availability</dt><dd className="mt-1 text-sm leading-5 text-slate-100">{modelAvailability(track)}</dd></div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"><dt className="text-xs font-semibold text-slate-400">Evaluation status</dt><dd className="mt-1 text-sm leading-5 text-slate-100">{evaluationStatus(track)}</dd></div>
            </dl>
            {track.limitations.length > 0 && <div className="mt-4 rounded-lg border border-amber-900/70 bg-amber-950/20 p-3">
              <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-amber-200"><AlertTriangle size={14} /> Restrictions and limitations</h3>
              <ul className="mt-2 list-inside list-disc space-y-1 text-xs leading-5 text-slate-200">{track.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
            </div>}
            <div className="mt-4 flex flex-wrap gap-2">
              {track.target_classes.map((target) => <span key={target} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-300">{target}</span>)}
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <Link className={buttonClass} to={link.href}>{link.label} <ArrowRight size={14} /></Link>
              <Link className={buttonClass} to="/evidence">Evidence registry <ArrowRight size={14} /></Link>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-400">Governance: {track.governance_clearance}. {track.disclaimer}</p>
          </article>;
        })}
      </section>}
      {data && <p className="text-xs leading-5 text-slate-500">{data.disclaimer} This summary does not verify local checkpoint loadability or authorize evaluation. Protected Track A/B holdouts and Track C external holdouts must remain isolated.</p>}
    </main>
  );
};

export default ResearchTracksPage;
