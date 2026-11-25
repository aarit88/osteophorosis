import React, { useCallback, useEffect, useState } from 'react';
import { Activity, ArrowRight, BookOpen, CheckCircle2, Database, FileCheck2, Layers, LockKeyhole, RotateCw, ScanLine, ShieldAlert, ShieldCheck, Trash2, UploadCloud } from 'lucide-react';
import { Link } from 'react-router-dom';
import { getResearchTracks, getTrackAStatus, getTrackDDatasetReadiness, runTrackAInference } from '../api/client';
import type { ResearchTrackDetail, ResearchTracksResponse, TrackAInferenceResponse, TrackAOperatingPoint, TrackAStatusResponse, TrackDDatasetReadinessResponse } from '../api/types';

const linkClass = 'inline-flex min-h-10 items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-200 hover:border-cyan-600 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400';
const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';
const methodLinks = [
  { title: 'Radiomics', path: '/radiomics', description: 'Feature families are documented; clinical image input is not connected.' },
  { title: 'Biomechanics', path: '/biomechanics', description: 'Manual Mirels rule calculation is available; image-derived mechanics are not.' },
  { title: 'Explainability', path: '/explainability', description: 'Model-specific clinical attribution is unavailable.' },
  { title: 'Uncertainty', path: '/uncertainty', description: 'Patient-specific calibrated uncertainty is unavailable.' },
];
const trackAOperatingPoints: Array<{ value: TrackAOperatingPoint; label: string; threshold: number }> = [
  { value: 'high_sensitivity_screening', label: 'High-sensitivity screening', threshold: 0.31 },
  { value: 'best_youden_j', label: 'Validation Youden J', threshold: 0.38 },
  { value: 'best_f1', label: 'Validation F1', threshold: 0.47 },
  { value: 'default_0_50', label: 'Default boundary', threshold: 0.5 },
];

function safeErrorMessage(): string {
  return 'The service could not complete this request. Check the API connection and retry.';
}

function safeTrackAErrorMessage(error: unknown): string {
  if (!(error instanceof Error)) return safeErrorMessage();
  if (error.message === 'TRACK_A_AUTHENTICATION_REQUIRED') return 'Gateway authentication is required for the protected scan service.';
  if (error.message === 'TRACK_A_ACCESS_FORBIDDEN') return 'Your account is not authorized to use the scan service.';
  return safeErrorMessage();
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
  const [trackAStatus, setTrackAStatus] = useState<TrackAStatusResponse | null>(null);
  const [trackAStatusLoading, setTrackAStatusLoading] = useState(true);
  const [trackAStatusError, setTrackAStatusError] = useState<string | null>(null);
  const [scanImage, setScanImage] = useState<File | null>(null);
  const [operatingPoint, setOperatingPoint] = useState<TrackAOperatingPoint>('high_sensitivity_screening');
  const [scanResult, setScanResult] = useState<TrackAInferenceResponse | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [scanRunning, setScanRunning] = useState(false);

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

  const loadTrackAStatus = useCallback(async () => {
    setTrackAStatusLoading(true);
    setTrackAStatusError(null);
    try { setTrackAStatus(await getTrackAStatus()); }
    catch (error) { setTrackAStatus(null); setTrackAStatusError(safeTrackAErrorMessage(error)); }
    finally { setTrackAStatusLoading(false); }
  }, []);

  const handleScanFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0] ?? null;
    event.currentTarget.value = '';
    if (!file) return;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
      setScanError('Choose a PNG, JPEG, or WebP X-ray image. DICOM and non-X-ray modalities are not supported by this Track A model.');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setScanError('Image is larger than the 15 MB limit.');
      return;
    }
    setScanImage(file);
    setScanResult(null);
    setScanError(null);
  };

  const runTrackAScan = async () => {
    if (!scanImage || !trackAStatus?.checkpoint_available || scanRunning) return;
    setScanRunning(true);
    setScanError(null);
    setScanResult(null);
    try { setScanResult(await runTrackAInference(scanImage, operatingPoint)); }
    catch (error) { setScanError(safeTrackAErrorMessage(error) === safeErrorMessage() ? 'The image was not analyzed. Check the Track A checkpoint and backend status, then retry.' : safeTrackAErrorMessage(error)); }
    finally { setScanRunning(false); }
  };

  const resetTrackAScan = () => {
    setScanImage(null);
    setScanResult(null);
    setScanError(null);
  };

  useEffect(() => { void loadTracks(); }, [loadTracks]);
  useEffect(() => { void loadTrackAStatus(); }, [loadTrackAStatus]);
  const selectedTrack = registry?.tracks.find((track) => track.track_id === selectedTrackId) ?? null;
  const isTrackD = selectedTrack?.track_id === 'track_d';
  useEffect(() => { if (isTrackD && !readiness && !readinessLoading && !readinessError) void loadReadiness(); }, [isTrackD, readiness, readinessLoading, readinessError, loadReadiness]);

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6">
      <header className="flex flex-col gap-4 border-b border-slate-800 pb-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl space-y-2"><div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200"><Activity size={14} aria-hidden="true" /> Research workspace</div><h1 className="text-3xl font-black tracking-tight text-white sm:text-4xl">Workbench</h1><p className="text-sm leading-6 text-slate-300">Inspect the live track registry, current Track D readiness, and links to supported research modules. No synthetic image-derived outputs or simulated clinical predictions are generated here.</p></div>
        <nav aria-label="Related research pages" className="flex flex-wrap gap-2"><Link className={linkClass} to="/tracks"><Layers size={16} aria-hidden="true" />Tracks</Link><Link className={linkClass} to="/governance"><ShieldCheck size={16} aria-hidden="true" />Governance</Link><Link className={linkClass} to="/evidence"><FileCheck2 size={16} aria-hidden="true" />Evidence</Link><Link className={linkClass} to="/results"><BookOpen size={16} aria-hidden="true" />Results</Link></nav>
      </header>

      <section className="rounded-2xl border border-cyan-900/70 bg-slate-900/70 p-5 sm:p-6" aria-labelledby="track-a-scanner-title">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-3xl">
            <p className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[.16em] text-cyan-300"><ScanLine size={15} aria-hidden="true" />Local X-ray scan portal</p>
            <h2 id="track-a-scanner-title" className="mt-2 text-2xl font-black text-white">Track A · FracAtlas fracture research</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">Upload one X-ray for the existing ResNet-101 binary fracture model. This is a preliminary Track A task; it is not the four-class diagnostic model and does not compute accuracy for a single image.</p>
          </div>
          <Link className={linkClass} to="/tracks"><Layers size={15} aria-hidden="true" />Track details</Link>
        </div>

        <div className={`mt-5 rounded-xl border p-4 ${trackAStatus?.checkpoint_available ? 'border-emerald-800 bg-emerald-950/20' : 'border-amber-800 bg-amber-950/20'}`}>
          {trackAStatusLoading && <p role="status" className="text-sm text-slate-200">Checking Track A checkpoint availability…</p>}
          {trackAStatusError && <div role="alert" className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-amber-100">Model status is unavailable. {trackAStatusError}</p><button type="button" onClick={() => void loadTrackAStatus()} className={linkClass}><RotateCw size={14} aria-hidden="true" />Retry status</button></div>}
          {!trackAStatusLoading && !trackAStatusError && trackAStatus && (trackAStatus.checkpoint_available
            ? <p role="status" className="flex items-center gap-2 text-sm font-semibold text-emerald-100"><CheckCircle2 size={17} aria-hidden="true" />Track A checkpoint file is present. It will be load-validated when a scan is run.</p>
            : trackAStatus.error_code === 'ML_SERVICE_NOT_CONFIGURED'
              ? <div role="status"><p className="flex items-center gap-2 text-sm font-semibold text-amber-100"><ShieldAlert size={17} aria-hidden="true" />Track A scan is unavailable: the FastAPI ML service is not connected to this gateway.</p><p className="mt-2 text-xs leading-5 text-amber-50/80">Start the ML service and configure <code>BACKEND_URL</code> for the Express gateway. No scan request will be sent until service status is available.</p></div>
              : <div role="status"><p className="flex items-center gap-2 text-sm font-semibold text-amber-100"><ShieldAlert size={17} aria-hidden="true" />Track A scan is unavailable: no checkpoint file is configured in this checkout.</p><p className="mt-2 text-xs leading-5 text-amber-50/80">The expected default is <code>ml_artifacts/fracatlas_full_training/best_checkpoint.pt</code>. Place your genuine trained Track A checkpoint in the artifact store or set <code>OSTEO_TRACK_A_CHECKPOINT_PATH</code> to its path under <code>ml_artifacts/</code>. Do not upload model weights through this image form.</p></div>
          )}
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,.75fr)]">
          <div className="space-y-4">
            <label htmlFor="track-a-xray" className="block text-sm font-semibold text-white">Select a local X-ray image</label>
            <input id="track-a-xray" type="file" accept="image/png,image/jpeg,image/webp" onChange={handleScanFile} disabled={trackAStatusLoading || scanRunning} className="block w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-200 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-800 file:px-3 file:py-2 file:text-xs file:font-bold file:text-white disabled:opacity-50" />
            <p className="text-xs leading-5 text-slate-400">Maximum 15 MB. The gateway strips image metadata and processes the selected image in memory. The filename is not shown or retained.</p>
            {scanImage && <p role="status" className="text-sm text-slate-200">X-ray selected ({(scanImage.size / 1024).toFixed(1)} KB); the local filename remains hidden.</p>}
            <label htmlFor="track-a-operating-point" className="block text-sm font-semibold text-white">Research operating point</label>
            <select id="track-a-operating-point" value={operatingPoint} onChange={(event) => setOperatingPoint(event.target.value as TrackAOperatingPoint)} disabled={scanRunning} className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30">
              {trackAOperatingPoints.map((item) => <option key={item.value} value={item.value}>{item.label} · threshold {item.threshold.toFixed(2)}</option>)}
            </select>
            <p className="text-xs leading-5 text-slate-500">These are recorded Track A validation thresholds. Choosing one here does not estimate accuracy or recalibrate the model.</p>
            <div className="flex flex-wrap gap-3">
              <button type="button" onClick={() => void runTrackAScan()} disabled={!scanImage || !trackAStatus?.checkpoint_available || scanRunning} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-cyan-700 px-5 py-3 text-sm font-bold text-white hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300">
                {scanRunning ? <><RotateCw size={16} className="animate-spin" aria-hidden="true" />Running Track A inference…</> : <><ScanLine size={16} aria-hidden="true" />Run X-ray scan</>}
              </button>
              {(scanImage || scanResult || scanError) && <button type="button" onClick={resetTrackAScan} disabled={scanRunning} className={linkClass}><Trash2 size={15} aria-hidden="true" />Clear scan</button>}
            </div>
            {scanError && <p role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-100">{scanError}</p>}
          </div>

          <aside className="rounded-xl border border-slate-800 bg-slate-950/70 p-4" aria-label="Track A model details">
            <h3 className="font-semibold text-white">Model expectations</h3>
            <dl className="mt-3 space-y-3 text-sm"><div><dt className="text-xs text-slate-400">Input</dt><dd className="mt-1 text-slate-100">X-ray · 224 × 224 · RGB ImageNet normalization</dd></div><div><dt className="text-xs text-slate-400">Output classes</dt><dd className="mt-1 text-slate-100">Candidate control radiograph / acute bone fracture</dd></div><div><dt className="text-xs text-slate-400">Scope</dt><dd className="mt-1 text-slate-100">Binary preliminary fracture research only</dd></div></dl>
            <p className="mt-4 text-xs leading-5 text-amber-100">Do not interpret a candidate-control output as a healthy diagnosis. This model does not identify Paget disease, OI, or bone metastasis.</p>
          </aside>
        </div>

        {scanRunning && <p role="status" className="mt-5 rounded-lg border border-cyan-900 bg-cyan-950/30 p-3 text-sm text-cyan-100">The configured Track A model is processing the image. No analysis progress estimate is available.</p>}
        {scanResult && <section className="mt-5 rounded-xl border border-cyan-800 bg-slate-950/80 p-5" aria-labelledby="track-a-result-title">
          <p className="text-xs font-bold uppercase tracking-[.14em] text-cyan-300">Actual Track A backend response</p>
          <h3 id="track-a-result-title" className="mt-2 text-xl font-bold text-white">{scanResult.predicted_class_name === 'acute_bone_fracture' ? 'Acute bone fracture class' : 'Candidate control radiograph class'}</h3>
          <div className="mt-4 grid gap-3 sm:grid-cols-2"><p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200">Fracture score: <strong>{(scanResult.prob_fractured * 100).toFixed(1)}%</strong></p><p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200">Candidate-control score: <strong>{(scanResult.prob_non_fractured * 100).toFixed(1)}%</strong></p><p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200">Applied threshold: <strong>{scanResult.active_threshold.toFixed(2)}</strong></p><p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200">Checkpoint SHA-256: <code className="break-all text-xs">{scanResult.checkpoint_sha256}</code></p></div>
          <p className="mt-4 text-xs leading-5 text-amber-100">{scanResult.disclaimer}</p>
        </section>}
      </section>

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

      <section className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 text-xs leading-5 text-slate-400"><strong className="text-slate-200">Analysis history:</strong> this application does not persist patient images or show server-side recent scan history. Track A results stay in this browser session and are cleared when you use “Clear scan.” The separate four-class Track D route remains unavailable with <code>FOUR_CLASS_MODEL_NOT_READY</code>.</section>
      <footer className="flex flex-wrap gap-2 border-t border-slate-800 pt-5" aria-label="Research navigation"><Link className={linkClass} to="/tracks">Tracks</Link><Link className={linkClass} to="/governance">Governance</Link><Link className={linkClass} to="/evidence">Evidence</Link><Link className={linkClass} to="/results">Results</Link></footer>
    </main>
  );
};

export default ResearchWorkbenchPage;
