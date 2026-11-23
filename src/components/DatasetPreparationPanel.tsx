import React, { useState } from 'react';
import {
  advanceDatasetPreparationRun, createDatasetPreparationRun, DatasetPreparationRun,
  getDatasetPreparationReport, getDatasetPreparationRuns, getOnboardingSources, OnboardingSource,
} from '../api/client';

const button = 'rounded bg-cyan-900 px-3 py-1.5 text-xs text-cyan-100 disabled:opacity-50';

export const DatasetPreparationPanel: React.FC = () => {
  const [token, setToken] = useState('');
  const [sources, setSources] = useState<OnboardingSource[]>([]);
  const [runs, setRuns] = useState<DatasetPreparationRun[]>([]);
  const [sourceId, setSourceId] = useState('');
  const [message, setMessage] = useState('');
  const [report, setReport] = useState('');
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    setBusy(true); setMessage('');
    try {
      const [sourceResult, runResult] = await Promise.all([getOnboardingSources(token), getDatasetPreparationRuns(token)]);
      setSources(sourceResult.sources); setRuns(runResult.runs); setMessage('Preparation registry loaded.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Preparation registry request failed.'); }
    finally { setBusy(false); }
  };

  const createRun = async () => {
    if (!sourceId) return;
    setBusy(true); setMessage('');
    try { await createDatasetPreparationRun(token, sourceId); await refresh(); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'Could not create preparation run.'); setBusy(false); }
  };

  const advance = async (run: DatasetPreparationRun, action: 'inventory' | 'qc' | 'harmonize' | 'splits') => {
    setBusy(true); setMessage('');
    try { await advanceDatasetPreparationRun(token, run.run_id, action); await refresh(); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'Preparation step failed.'); setBusy(false); }
  };

  const loadReport = async (run: DatasetPreparationRun) => {
    setBusy(true); setMessage('');
    try { setReport(String(await getDatasetPreparationReport(token, run.run_id, 'markdown'))); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'Report could not be loaded.'); }
    finally { setBusy(false); }
  };

  const staged = sources.filter((source) => source.verification_status === 'VERIFIED'
    && source.staging_status === 'STAGED_FOR_STAGE33_INTAKE' && source.package?.valid);

  return <section className="space-y-4 rounded-xl border border-cyan-900/70 bg-slate-900/60 p-5" aria-label="Stage 39 dataset preparation">
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-cyan-300">Stage 39 · Research data preparation</p>
      <h2 className="mt-1 text-lg font-bold text-white">Inventory, quality control and split audit</h2>
      <p className="mt-1 text-xs text-slate-400">Uses verified staged Track D packages. This workflow does not establish clinical validity, readiness or training authorization.</p>
    </div>
    <div className="flex flex-wrap gap-2">
      <label className="sr-only" htmlFor="preparation-reviewer-token">Reviewer bearer token</label>
      <input id="preparation-reviewer-token" type="password" autoComplete="off" value={token} onChange={(event) => setToken(event.target.value)} placeholder="Reviewer bearer token (kept in memory)" className="min-w-64 flex-1 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white" />
      <button className={button} disabled={busy || !token} onClick={refresh}>Load preparation registry</button>
    </div>
    <div className="flex flex-wrap gap-2">
      <label className="sr-only" htmlFor="preparation-source">Verified staged source</label>
      <select id="preparation-source" className="min-w-64 flex-1 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white" value={sourceId} onChange={(event) => setSourceId(event.target.value)}>
        <option value="">Choose a verified staged Track D source</option>
        {staged.map((source) => <option key={source.source_id} value={source.source_id}>{source.name} · {source.version}</option>)}
      </select>
      <button className={button} disabled={busy || !token || !sourceId} onClick={createRun}>Create preparation run</button>
    </div>
    {message && <p role="status" className="text-xs text-amber-300">{message}</p>}
    <div className="space-y-3">
      {runs.map((run) => <article key={run.run_id} className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
        <div className="flex flex-wrap justify-between gap-2"><div><h3 className="text-sm font-semibold text-white">{run.source_id} · {run.source_version}</h3><p className="text-[11px] text-slate-500">Run {run.run_id} · Config {run.config_version}</p></div><span className="rounded border border-cyan-900 px-2 py-1 text-xs text-cyan-200">{run.state}</span></div>
        <div className="mt-3 grid gap-2 text-xs text-slate-300 sm:grid-cols-2 lg:grid-cols-4">
          <span>Inventory: {run.inventory ? `${run.inventory.file_count} files` : run.state === 'INVENTORY_PENDING' ? 'Pending' : 'Not run'}</span>
          <span>QC: {run.qc?.status ?? 'Not run'}{run.qc ? ` · ${run.qc.blocker_count} blockers` : ''}</span>
          <span>Harmonization: {run.harmonization?.status ?? 'Not run'}</span>
          <span>Split audit: {run.splits?.leakage_audit.is_clean ? 'Clean' : run.splits ? 'Findings' : 'Not run'}</span>
          <span>Governance: {run.governance_state ?? 'Not reported'}</span>
          <span>Access: {run.access_restriction ?? 'Not reported'}</span>
        </div>
        {run.source_restrictions && run.source_restrictions.length > 0 && <p className="mt-2 text-xs text-amber-300">Source restrictions: {run.source_restrictions.join('; ')}</p>}
        {run.state === 'SOURCE_STAGED' && <button className={`${button} mt-3`} disabled={busy} onClick={() => advance(run, 'inventory')}>Generate inventory</button>}
        {run.state === 'INVENTORY_COMPLETED' && <button className={`${button} mt-3`} disabled={busy} onClick={() => advance(run, 'qc')}>Run modality QC</button>}
        {run.state === 'QC_COMPLETED' && <div className="mt-3 flex flex-wrap gap-2"><button className={button} disabled={busy} onClick={() => advance(run, 'harmonize')}>Harmonize images</button><button className={button} disabled={busy} onClick={() => advance(run, 'splits')}>Prepare patient-level splits</button></div>}
        {run.state === 'HARMONIZATION_COMPLETED' && <button className={`${button} mt-3`} disabled={busy} onClick={() => advance(run, 'splits')}>Run leakage and split audit</button>}
        {run.blockers?.length > 0 && <p className="mt-2 text-xs text-rose-300">Blocked: {run.blockers.join('; ')}</p>}
        {run.splits && <p className="mt-2 text-xs text-amber-300">Protected holdout records excluded: {run.splits.excluded_protected_holdout_records} · Training authorized: No</p>}
        <button className={`${button} mt-3`} disabled={busy || !token} onClick={() => loadReport(run)}>View preparation report</button>
      </article>)}
      {runs.length === 0 && <p className="text-xs text-slate-500">No preparation runs loaded.</p>}
    </div>
    {report && <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded border border-slate-800 bg-slate-950 p-4 text-xs text-slate-300" aria-label="Preparation report">{report}</pre>}
  </section>;
};

export default DatasetPreparationPanel;
