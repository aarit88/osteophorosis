import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Database,
  FileCheck2,
  Layers,
  LockKeyhole,
  RotateCw,
  ScanLine,
  ShieldAlert,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import {
  analyzeImage,
  getResearchTracks,
  getTrackAStatus,
  getTrackDDatasetReadiness,
  requestFourClassInference,
  runSyntheticWorkbench,
  runTrackAInference,
} from '../api/client';
import type {
  AnalyzeResponse,
  BlockedInferenceResponse,
  ResearchTrackDetail,
  ResearchTracksResponse,
  SyntheticWorkbenchRunResponse,
  TrackAInferenceResponse,
  TrackAOperatingPoint,
  TrackAStatusResponse,
  TrackDDatasetReadinessResponse,
} from '../api/types';
import { useOptionalAnalysis } from '../context/AnalysisContext';
import { getModalitySpec, getPathologyProfile } from '../utils/pathologyCatalog';
import {
  buildPortalResearchPayload,
  DIAGNOSTIC_UPLOAD_PORTALS,
  type DiagnosticPortalModality,
  validatePortalScanFile,
} from '../utils/workbenchPortals';

const linkClass =
  'inline-flex min-h-10 items-center gap-2 whitespace-nowrap rounded-lg border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-200 hover:border-cyan-600 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400';
const panelClass = 'rounded-2xl border border-slate-800 bg-slate-900/60 p-5';
const methodLinks = [
  { title: 'Radiomics', path: '/radiomics', description: 'IBSI 102-D digital phantom extraction (81 verified / 21 heuristic) across trabecular patterns.' },
  { title: 'Biomechanics', path: '/biomechanics', description: 'Cross-sectional cortical moment of inertia (J) retention and Mirels (1989) fracture scoring.' },
  { title: 'Explainability', path: '/explainability', description: 'Controlled synthetic Grad-CAM spatial salience maps paired with Track D gate verification.' },
  { title: 'Uncertainty', path: '/uncertainty', description: 'Selective prediction triage under candidate entropy threshold H_norm > 0.60 and OOD checks.' },
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

function humanize(value: string): string {
  return value.replace(/_/g, ' ');
}

interface PortalUploadState {
  file: File | null;
  error: string | null;
  running: boolean;
}

const INITIAL_PORTAL_STATE: Record<'ct' | 'mri', PortalUploadState> = {
  ct: { file: null, error: null, running: false },
  mri: { file: null, error: null, running: false },
};

export const ResearchWorkbenchPage: React.FC = () => {
  const navigate = useNavigate();
  const analysisContext = useOptionalAnalysis();

  const [registry, setRegistry] = useState<ResearchTracksResponse | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<ResearchTrackDetail['track_id'] | ''>('');
  const [tracksLoading, setTracksLoading] = useState(true);
  const [tracksError, setTracksError] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<TrackDDatasetReadinessResponse | null>(null);
  const [readinessLoading, setReadinessLoading] = useState(false);
  const [readinessError, setReadinessError] = useState<string | null>(null);

  // Track A X-ray portal state
  const [trackAStatus, setTrackAStatus] = useState<TrackAStatusResponse | null>(null);
  const [trackAStatusLoading, setTrackAStatusLoading] = useState(true);
  const [trackAStatusError, setTrackAStatusError] = useState<string | null>(null);
  const [scanImage, setScanImage] = useState<File | null>(null);
  const [operatingPoint, setOperatingPoint] = useState<TrackAOperatingPoint>('high_sensitivity_screening');
  const [scanResult, setScanResult] = useState<TrackAInferenceResponse | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [scanRunning, setScanRunning] = useState(false);

  // Multimodal CT / MRI & X-ray FastAPI Research intake state
  const [portalStates, setPortalStates] = useState<Record<'ct' | 'mri', PortalUploadState>>(INITIAL_PORTAL_STATE);
  const [multimodalResult, setMultimodalResult] = useState<AnalyzeResponse | null>(null);
  const [multimodalGate, setMultimodalGate] = useState<BlockedInferenceResponse | null>(null);
  const [multimodalModality, setMultimodalModality] = useState<DiagnosticPortalModality | null>(null);
  const [xrayResearchRunning, setXrayResearchRunning] = useState(false);

  // Synchronized Stage 31 FastAPI Research Pipeline state
  const [workbenchRun, setWorkbenchRun] = useState<SyntheticWorkbenchRunResponse | null>(null);
  const [workbenchRunning, setWorkbenchRunning] = useState(false);
  const [workbenchError, setWorkbenchError] = useState<string | null>(null);
  const [activeResearchModality, setActiveResearchModality] = useState<DiagnosticPortalModality | null>(null);

  const loadTracks = useCallback(async () => {
    setTracksLoading(true);
    setTracksError(null);
    try {
      const response = await getResearchTracks();
      setRegistry(response);
      setSelectedTrackId((current) =>
        current && response.tracks.some((track) => track.track_id === current)
          ? current
          : response.tracks[0]?.track_id ?? '',
      );
    } catch {
      setTracksError(safeErrorMessage());
    } finally {
      setTracksLoading(false);
    }
  }, []);

  const loadReadiness = useCallback(async () => {
    setReadinessLoading(true);
    setReadinessError(null);
    try {
      setReadiness(await getTrackDDatasetReadiness());
    } catch {
      setReadiness(null);
      setReadinessError(safeErrorMessage());
    } finally {
      setReadinessLoading(false);
    }
  }, []);

  const loadTrackAStatus = useCallback(async () => {
    setTrackAStatusLoading(true);
    setTrackAStatusError(null);
    try {
      setTrackAStatus(await getTrackAStatus());
    } catch (error) {
      setTrackAStatus(null);
      setTrackAStatusError(safeTrackAErrorMessage(error));
    } finally {
      setTrackAStatusLoading(false);
    }
  }, []);

  const handleScanFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0] ?? null;
    event.currentTarget.value = '';
    if (!file) return;
    const validation = validatePortalScanFile(file, 'xray');
    if (!validation.valid) {
      setScanError(
        'Choose a PNG, JPEG, or WebP X-ray image. DICOM and non-X-ray modalities are not supported by this Track A model.',
      );
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
    try {
      setScanResult(await runTrackAInference(scanImage, operatingPoint));
    } catch (error) {
      setScanError(
        safeTrackAErrorMessage(error) === safeErrorMessage()
          ? 'The image was not analyzed. Check the Track A checkpoint and backend status, then retry.'
          : safeTrackAErrorMessage(error),
      );
    } finally {
      setScanRunning(false);
    }
  };

  const resetTrackAScan = () => {
    setScanImage(null);
    setScanResult(null);
    setScanError(null);
  };

  const handlePortalFileChange = (modality: 'ct' | 'mri', event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0] ?? null;
    event.currentTarget.value = '';
    if (!file) return;
    const validation = validatePortalScanFile(file, modality);
    if (!validation.valid) {
      setPortalStates((prev) => ({
        ...prev,
        [modality]: { ...prev[modality], error: validation.reason },
      }));
      return;
    }
    setPortalStates((prev) => ({
      ...prev,
      [modality]: { file, error: null, running: false },
    }));
  };

  const clearPortalFile = (modality: 'ct' | 'mri') => {
    setPortalStates((prev) => ({
      ...prev,
      [modality]: { file: null, error: null, running: false },
    }));
    if (multimodalModality === modality) {
      setMultimodalResult(null);
      setMultimodalGate(null);
      setMultimodalModality(null);
    }
  };

  const executeCombinedWorkbench = async (modality?: DiagnosticPortalModality) => {
    setWorkbenchRunning(true);
    setWorkbenchError(null);
    setActiveResearchModality(modality ?? null);
    try {
      const payload = modality ? buildPortalResearchPayload(modality) : {};
      setWorkbenchRun(await runSyntheticWorkbench(payload));
    } catch {
      setWorkbenchError('Combined research pipeline demonstration could not be executed. Check the API connection and retry.');
    } finally {
      setWorkbenchRunning(false);
    }
  };

  const runPortalResearchIntake = async (modality: DiagnosticPortalModality) => {
    const targetFile = modality === 'xray' ? scanImage : portalStates[modality].file;
    const validation = validatePortalScanFile(targetFile, modality);
    if (!validation.valid || !targetFile) {
      if (modality === 'xray') {
        setScanError(validation.reason);
      } else {
        setPortalStates((prev) => ({
          ...prev,
          [modality]: { ...prev[modality], error: validation.reason },
        }));
      }
      return;
    }

    if (modality === 'xray') {
      setXrayResearchRunning(true);
      setScanError(null);
    } else {
      setPortalStates((prev) => ({
        ...prev,
        [modality]: { ...prev[modality], running: true, error: null },
      }));
    }

    setMultimodalModality(modality);

    try {
      const [gateOutcome, researchOutcome, analysisOutcome] = await Promise.allSettled([
        requestFourClassInference(),
        runSyntheticWorkbench(buildPortalResearchPayload(modality)),
        analyzeImage({ image: targetFile, modality }),
      ]);

      if (gateOutcome.status === 'fulfilled') {
        setMultimodalGate(gateOutcome.value);
      }
      if (researchOutcome.status === 'fulfilled') {
        setWorkbenchRun(researchOutcome.value);
        setActiveResearchModality(modality);
        setWorkbenchError(null);
      }
      if (analysisOutcome.status === 'fulfilled') {
        setMultimodalResult(analysisOutcome.value);
        analysisContext?.setLastResult(analysisOutcome.value);
      } else {
        setMultimodalResult(null);
        const msg =
          analysisOutcome.reason instanceof Error && analysisOutcome.reason.message
            ? analysisOutcome.reason.message
            : 'Four-class clinical inference is gated until verified weights are mounted.';
        if (modality === 'xray') {
          setScanError(msg);
        } else {
          setPortalStates((prev) => ({
            ...prev,
            [modality]: { ...prev[modality], error: msg },
          }));
        }
      }
    } finally {
      if (modality === 'xray') {
        setXrayResearchRunning(false);
      } else {
        setPortalStates((prev) => ({
          ...prev,
          [modality]: { ...prev[modality], running: false },
        }));
      }
    }
  };

  useEffect(() => {
    void loadTracks();
  }, [loadTracks]);
  useEffect(() => {
    void loadTrackAStatus();
  }, [loadTrackAStatus]);

  const selectedTrack = registry?.tracks.find((track) => track.track_id === selectedTrackId) ?? null;
  const isTrackD = selectedTrack?.track_id === 'track_d';
  useEffect(() => {
    if (isTrackD && !readiness && !readinessLoading && !readinessError) {
      void loadReadiness();
    }
  }, [isTrackD, readiness, readinessLoading, readinessError, loadReadiness]);

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6">
      <header className="flex flex-col gap-4 border-b border-slate-800 pb-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl space-y-2">
          <p className="font-mono text-xs font-semibold uppercase tracking-[.18em] text-cyan-300">
            Diagnostic &amp; Research Workbench · X-ray · CT Scan · MRI
          </p>
          <h1 className="text-3xl font-black tracking-tight text-white sm:text-4xl">Workbench</h1>
          <p className="text-sm leading-6 text-slate-300">
            Dedicated upload portals for <strong>2D Digital X-ray</strong>, <strong>CT Scan</strong>, and <strong>MRI</strong> integrated with Track A fracture scoring, FastAPI Stage 31 research pipelines, and Track D clinical gate checks.
          </p>
        </div>
        <nav aria-label="Related research pages" className="flex flex-wrap gap-2 lg:shrink-0 lg:flex-nowrap">
          <Link className={linkClass} to="/tracks">
            <Layers size={16} aria-hidden="true" />
            Tracks
          </Link>
          <Link className={linkClass} to="/governance">
            <ShieldCheck size={16} aria-hidden="true" />
            Governance
          </Link>
          <Link className={linkClass} to="/evidence">
            <FileCheck2 size={16} aria-hidden="true" />
            Evidence
          </Link>
          <Link className={linkClass} to="/results">
            <BookOpen size={16} aria-hidden="true" />
            Results
          </Link>
        </nav>
      </header>

      {/* Primary Track A + Multimodal X-ray Upload Portal */}
      <section className="rounded-2xl border border-cyan-900/70 bg-slate-900/70 p-5 sm:p-6" aria-labelledby="track-a-scanner-title">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-3xl">
            <p className="inline-flex items-center gap-2 font-mono text-xs font-bold uppercase tracking-[.16em] text-cyan-300">
              <ScanLine size={15} aria-hidden="true" />
              Portal 01 · X-ray Upload Portal
            </p>
            <h2 id="track-a-scanner-title" className="mt-2 text-2xl font-black text-white">
              Track A · FracAtlas fracture research
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Upload one X-ray for the existing ResNet-101 binary fracture model, or dispatch it through the FastAPI multimodal research pipeline. This preliminary Track A task is separate from the gated four-class diagnostic model.
            </p>
          </div>
          <Link className={linkClass} to="/tracks">
            <Layers size={15} aria-hidden="true" />
            Track details
          </Link>
        </div>

        <div
          className={`mt-5 rounded-xl border p-4 ${
            trackAStatus?.checkpoint_available
              ? 'border-emerald-800 bg-emerald-950/20'
              : 'border-amber-800 bg-amber-950/20'
          }`}
        >
          {trackAStatusLoading && (
            <p role="status" className="text-sm text-slate-200">
              Checking Track A checkpoint availability…
            </p>
          )}
          {trackAStatusError && (
            <div role="alert" className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-amber-100">Model status is unavailable. {trackAStatusError}</p>
              <button type="button" onClick={() => void loadTrackAStatus()} className={linkClass}>
                <RotateCw size={14} aria-hidden="true" />
                Retry status
              </button>
            </div>
          )}
          {!trackAStatusLoading &&
            !trackAStatusError &&
            trackAStatus &&
            (trackAStatus.checkpoint_available ? (
              <p role="status" className="flex items-center gap-2 text-sm font-semibold text-emerald-100">
                <CheckCircle2 size={17} aria-hidden="true" />
                Track A checkpoint file is present. It will be load-validated when a scan is run.
              </p>
            ) : trackAStatus.error_code === 'ML_SERVICE_NOT_CONFIGURED' ? (
              <div role="status">
                <p className="flex items-center gap-2 text-sm font-semibold text-amber-100">
                  <ShieldAlert size={17} aria-hidden="true" />
                  Track A scan is unavailable: the FastAPI ML service is not connected to this gateway.
                </p>
                <p className="mt-2 text-xs leading-5 text-amber-50/80">
                  Start the ML service and configure <code>BACKEND_URL</code> for the Express gateway. Multimodal research endpoints remain accessible via the portal controls below.
                </p>
              </div>
            ) : (
              <div role="status">
                <p className="flex items-center gap-2 text-sm font-semibold text-amber-100">
                  <ShieldAlert size={17} aria-hidden="true" />
                  Track A scan is unavailable: no checkpoint file is configured in this checkout.
                </p>
                <p className="mt-2 text-xs leading-5 text-amber-50/80">
                  The expected default is <code>ml_artifacts/fracatlas_full_training/best_checkpoint.pt</code>. Place your genuine trained Track A checkpoint in the artifact store or set <code>OSTEO_TRACK_A_CHECKPOINT_PATH</code> to its path under <code>ml_artifacts/</code>. Do not upload model weights through this image form.
                </p>
              </div>
            ))}
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,.75fr)]">
          <div className="space-y-4">
            <label htmlFor="track-a-xray" className="block text-sm font-semibold text-white">
              Select a local X-ray image
            </label>
            <input
              id="track-a-xray"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={handleScanFile}
              disabled={trackAStatusLoading || scanRunning || xrayResearchRunning}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-200 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-800 file:px-3 file:py-2 file:text-xs file:font-bold file:text-white disabled:opacity-50"
            />
            <p className="text-xs leading-5 text-slate-400">
              Maximum 15 MB. The gateway strips image metadata in RAM and hides the local filename.
            </p>
            {scanImage && (
              <p role="status" className="text-sm text-slate-200">
                X-ray selected ({(scanImage.size / 1024).toFixed(1)} KB); the local filename remains hidden.
              </p>
            )}
            <label htmlFor="track-a-operating-point" className="block text-sm font-semibold text-white">
              Research operating point
            </label>
            <select
              id="track-a-operating-point"
              value={operatingPoint}
              onChange={(event) => setOperatingPoint(event.target.value as TrackAOperatingPoint)}
              disabled={scanRunning}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
            >
              {trackAOperatingPoints.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label} · threshold {item.threshold.toFixed(2)}
                </option>
              ))}
            </select>
            <p className="text-xs leading-5 text-slate-500">
              Recorded Track A validation thresholds; choosing one does not recalibrate the model.
            </p>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => void runTrackAScan()}
                disabled={!scanImage || !trackAStatus?.checkpoint_available || scanRunning}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-cyan-700 px-5 py-3 text-sm font-bold text-white hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
              >
                {scanRunning ? (
                  <>
                    <RotateCw size={16} className="animate-spin" aria-hidden="true" />
                    Running Track A inference…
                  </>
                ) : (
                  <>
                    <ScanLine size={16} aria-hidden="true" />
                    Run X-ray scan
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={() => void runPortalResearchIntake('xray')}
                disabled={!scanImage || xrayResearchRunning || scanRunning}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-cyan-700 bg-cyan-950/60 px-4 py-3 text-xs font-bold text-cyan-100 hover:bg-cyan-900/60 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
              >
                {xrayResearchRunning ? (
                  <>
                    <RotateCw size={15} className="animate-spin" aria-hidden="true" />
                    Running X-ray Research Pipeline…
                  </>
                ) : (
                  <>
                    <Activity size={15} aria-hidden="true" />
                    Run X-ray Research Analysis
                  </>
                )}
              </button>
              {(scanImage || scanResult || scanError) && (
                <button type="button" onClick={resetTrackAScan} disabled={scanRunning || xrayResearchRunning} className={linkClass}>
                  <Trash2 size={15} aria-hidden="true" />
                  Clear scan
                </button>
              )}
            </div>
            {scanError && (
              <p role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-sm text-rose-100">
                {scanError}
              </p>
            )}
          </div>

          <aside className="rounded-xl border border-slate-800 bg-slate-950/70 p-4" aria-label="Track A model details">
            <h3 className="font-semibold text-white">Model expectations</h3>
            <dl className="mt-3 space-y-3 text-sm">
              <div>
                <dt className="text-xs text-slate-400">Input</dt>
                <dd className="mt-1 text-slate-100">2D Digital X-ray · 224 × 224 · CLAHE &amp; ImageNet normalization</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">FastAPI routes</dt>
                <dd className="mt-1 font-mono text-xs text-cyan-300">
                  {DIAGNOSTIC_UPLOAD_PORTALS.xray.endpointLabel}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">Output classes</dt>
                <dd className="mt-1 text-slate-100">Candidate control radiograph / acute bone fracture</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">Scope</dt>
                <dd className="mt-1 text-slate-100">Binary preliminary fracture research + Stage 31 X-ray intake</dd>
              </div>
            </dl>
            <p className="mt-4 text-xs leading-5 text-amber-100">
              Do not interpret a candidate-control output as a healthy diagnosis. This model does not identify Paget disease, OI, or bone metastasis.
            </p>
          </aside>
        </div>

        {scanRunning && (
          <p role="status" className="mt-5 rounded-lg border border-cyan-900 bg-cyan-950/30 p-3 text-sm text-cyan-100">
            The configured Track A model is processing the image. No analysis progress estimate is available.
          </p>
        )}
        {scanResult && (
          <section className="mt-5 rounded-xl border border-cyan-800 bg-slate-950/80 p-5" aria-labelledby="track-a-result-title">
            <p className="text-xs font-bold uppercase tracking-[.14em] text-cyan-300">Actual Track A backend response</p>
            <h3 id="track-a-result-title" className="mt-2 text-xl font-bold text-white">
              {scanResult.predicted_class_name === 'acute_bone_fracture'
                ? 'Acute bone fracture class'
                : 'Candidate control radiograph class'}
            </h3>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200 font-mono tabular-nums">
                Fracture score: <strong>{(scanResult.prob_fractured * 100).toFixed(1)}%</strong>
              </p>
              <p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200 font-mono tabular-nums">
                Candidate-control score: <strong>{(scanResult.prob_non_fractured * 100).toFixed(1)}%</strong>
              </p>
              <p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200 font-mono tabular-nums">
                Applied threshold: <strong>{scanResult.active_threshold.toFixed(2)}</strong>
              </p>
              <p className="rounded-lg bg-slate-900 p-3 text-sm text-slate-200">
                Checkpoint SHA-256: <code className="break-all text-xs">{scanResult.checkpoint_sha256}</code>
              </p>
            </div>
            <p className="mt-4 text-xs leading-5 text-amber-100">{scanResult.disclaimer}</p>
          </section>
        )}
      </section>

      {/* Dedicated CT Scan & MRI Upload Portals integrated with FastAPI Research Endpoints */}
      <section className="space-y-4" aria-labelledby="multimodal-portals-title">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="font-mono text-xs font-bold uppercase tracking-[.16em] text-cyan-300">
              Cross-Sectional &amp; Marrow Intake · FastAPI Integration
            </p>
            <h2 id="multimodal-portals-title" className="mt-1 text-2xl font-black text-white">
              CT Scan &amp; MRI Upload Portals
            </h2>
          </div>
          <span className="font-mono text-xs text-slate-400">
            RAM-only EXIF strip · Track D 503 gate enforced
          </span>
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          {(['ct', 'mri'] as const).map((portalKey, idx) => {
            const portal = DIAGNOSTIC_UPLOAD_PORTALS[portalKey];
            const state = portalStates[portalKey];
            const inputId = `portal-${portalKey}-upload`;
            const portalNum = idx === 0 ? '02' : '03';
            return (
              <article
                key={portalKey}
                className="flex flex-col justify-between rounded-2xl border border-slate-800 bg-slate-900/70 p-5"
                aria-labelledby={`portal-heading-${portalKey}`}
              >
                <div className="space-y-4">
                  <div className="flex items-start justify-between gap-3 border-b border-slate-800 pb-3">
                    <div>
                      <p className="font-mono text-[11px] font-bold uppercase tracking-[.15em] text-cyan-300">
                        Portal {portalNum} · {portal.shortTitle}
                      </p>
                      <h3 id={`portal-heading-${portalKey}`} className="mt-1 text-xl font-black text-white">
                        {portal.title}
                      </h3>
                    </div>
                    <span className="font-mono text-[11px] font-semibold text-slate-400">
                      {portalKey.toUpperCase()} INTAKE
                    </span>
                  </div>

                  <p className="text-xs leading-5 text-slate-300 lg:min-h-[2.5rem]">{portal.calibrationProtocol}</p>
                  <p className="font-mono text-[11px] text-cyan-300/90 break-all">{portal.endpointLabel}</p>

                  <div className="space-y-2">
                    <label htmlFor={inputId} className="block text-sm font-semibold text-white">
                      Select a local {portal.shortTitle} image
                    </label>
                    <input
                      id={inputId}
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      onChange={(event) => handlePortalFileChange(portalKey, event)}
                      disabled={state.running}
                      className="block w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-200 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-800 file:px-3 file:py-2 file:text-xs file:font-bold file:text-white disabled:opacity-50"
                    />
                    <p className="text-[11px] text-slate-400">
                      {portal.acceptedFormatsLabel}. Local filename remains hidden.
                    </p>
                    {state.file && (
                      <p role="status" className="text-xs font-semibold text-emerald-200 font-mono tabular-nums">
                        {portal.shortTitle} selected ({(state.file.size / 1024).toFixed(1)} KB); ready for FastAPI research dispatch.
                      </p>
                    )}
                    {state.error && (
                      <p role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-2.5 text-xs text-rose-100">
                        {state.error}
                      </p>
                    )}
                  </div>
                </div>

                <div className="mt-5 flex flex-wrap gap-2.5 border-t border-slate-800 pt-4">
                  <button
                    type="button"
                    onClick={() => void runPortalResearchIntake(portalKey)}
                    disabled={!state.file || state.running}
                    className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-cyan-700 px-4 py-2 text-xs font-bold text-white hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
                  >
                    {state.running ? (
                      <>
                        <RotateCw size={14} className="animate-spin" aria-hidden="true" />
                        Analyzing {portal.shortTitle}…
                      </>
                    ) : (
                      <>
                        <ScanLine size={14} aria-hidden="true" />
                        Run {portal.shortTitle} Analysis
                      </>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => void executeCombinedWorkbench(portalKey)}
                    disabled={workbenchRunning}
                    className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-cyan-700/80 bg-cyan-950/50 px-3.5 py-2 text-xs font-semibold text-cyan-200 hover:bg-cyan-900/50 disabled:opacity-50"
                  >
                    <Activity size={14} aria-hidden="true" />
                    Run {portal.shortTitle} Research Pipeline
                  </button>
                  {(state.file || state.error) && (
                    <button
                      type="button"
                      onClick={() => clearPortalFile(portalKey)}
                      disabled={state.running}
                      className={linkClass}
                    >
                      <Trash2 size={14} aria-hidden="true" />
                      Clear
                    </button>
                  )}
                </div>
              </article>
            );
          })}
        </div>

        {/* Unified Multimodal FastAPI Intake & Safety Gate Readout */}
        {(multimodalResult || multimodalGate) && multimodalModality && (
          <section
            className="rounded-2xl border border-cyan-800/80 bg-slate-950/85 p-5 space-y-4"
            aria-label="Multimodal portal diagnostic output"
          >
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-800 pb-3">
              <div>
                <p className="font-mono text-xs font-bold uppercase tracking-[.15em] text-cyan-300">
                  {getModalitySpec(multimodalModality).label} · FastAPI Intake Readout
                </p>
                {multimodalResult && (
                  <h3 className="mt-1 text-xl font-black text-white">
                    {getPathologyProfile(multimodalResult.prediction.topClass).title}
                  </h3>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {multimodalGate && (
                  <span className="font-mono text-xs font-semibold text-amber-200">
                    Gate: {multimodalGate.error_code}
                  </span>
                )}
                {multimodalResult && (
                  <button
                    type="button"
                    onClick={() => {
                      analysisContext?.setLastResult(multimodalResult);
                      navigate('/results');
                    }}
                    className="inline-flex min-h-9 items-center gap-1.5 rounded-lg bg-cyan-700 px-3.5 py-1.5 text-xs font-bold text-white hover:bg-cyan-600"
                  >
                    Open Full XAI Report
                    <ArrowRight size={14} aria-hidden="true" />
                  </button>
                )}
              </div>
            </div>

            {multimodalResult && (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-xs font-mono tabular-nums">
                <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                  <p className="text-slate-400">Top Confidence</p>
                  <p className="mt-1 text-base font-bold text-white">
                    {(multimodalResult.prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                  <p className="text-slate-400">Fracture Risk Index</p>
                  <p className="mt-1 text-base font-bold text-white">
                    {multimodalResult.fractureRisk.percent}% · {multimodalResult.fractureRisk.level.toUpperCase()}
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                  <p className="text-slate-400">Shannon Entropy</p>
                  <p className="mt-1 text-base font-bold text-white">
                    {multimodalResult.uncertaintyDetail.entropy.toFixed(3)} ({multimodalResult.prediction.uncertainty})
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                  <p className="text-slate-400">Session Request ID</p>
                  <p className="mt-1 truncate text-sm font-bold text-cyan-200">{multimodalResult.requestId}</p>
                </div>
              </div>
            )}

            {multimodalGate && (
              <p className="text-xs leading-5 text-amber-100">
                Track D safety gate status: <code>{multimodalGate.error_code}</code> — {multimodalGate.message}
              </p>
            )}
          </section>
        )}
      </section>

      <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,.8fr)]" aria-label="Track selection and readiness">
        <div className={panelClass}>
          <h2 className="mb-4 flex items-center gap-2 text-sm font-bold text-white">
            <Layers size={17} className="text-cyan-300" aria-hidden="true" />
            Select a registered research track
          </h2>
          {tracksLoading && (
            <p role="status" className="text-sm text-slate-300">
              Loading track registry…
            </p>
          )}
          {tracksError && (
            <div className="space-y-3" role="alert">
              <p className="text-sm text-rose-200">Research tracks are unavailable. {tracksError}</p>
              <button className={linkClass} onClick={() => void loadTracks()} type="button">
                <RotateCw size={15} aria-hidden="true" />
                Retry
              </button>
            </div>
          )}
          {!tracksLoading && !tracksError && registry?.tracks.length === 0 && (
            <p role="status" className="text-sm text-slate-300">
              The backend returned no available research tracks.
            </p>
          )}
          {!tracksLoading && !tracksError && registry && registry.tracks.length > 0 && (
            <>
              <label htmlFor="research-track" className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-400">
                Research track
              </label>
              <select
                id="research-track"
                aria-label="Research track"
                value={selectedTrackId}
                onChange={(event) => setSelectedTrackId(event.target.value as ResearchTrackDetail['track_id'])}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
              >
                {registry.tracks.map((track) => (
                  <option key={track.track_id} value={track.track_id}>
                    {track.track_id.toUpperCase().replace('_', ' ')} · {track.title}
                  </option>
                ))}
              </select>
              {selectedTrack && (
                <div className="mt-4 space-y-3" aria-live="polite">
                  <h3 className="text-lg font-bold text-white">{selectedTrack.title}</h3>
                  <dl className="grid gap-3 text-sm sm:grid-cols-2">
                    <div>
                      <dt className="text-xs text-slate-400">Modality</dt>
                      <dd className="mt-1 text-slate-100">{selectedTrack.modality}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-slate-400">Research task</dt>
                      <dd className="mt-1 text-slate-100">{selectedTrack.task_type}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-slate-400">Track status</dt>
                      <dd className="mt-1 text-slate-100">{humanize(selectedTrack.status)}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-slate-400">Governance</dt>
                      <dd className="mt-1 text-slate-100">{humanize(selectedTrack.governance_clearance)}</dd>
                    </div>
                  </dl>
                  <p className="text-xs leading-5 text-slate-400">{selectedTrack.disclaimer}</p>
                  {selectedTrack.limitations.length > 0 && (
                    <ul className="list-inside list-disc space-y-1 text-xs text-amber-200">
                      {selectedTrack.limitations.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        <aside
          className={`${panelClass} ${isTrackD ? 'border-rose-900/80 bg-rose-950/20' : ''}`}
          aria-label="Model readiness"
        >
          <h2 className="mb-3 flex items-center gap-2 text-sm font-bold text-white">
            <LockKeyhole size={17} className={isTrackD ? 'text-rose-300' : 'text-cyan-300'} aria-hidden="true" />
            Model and workflow readiness
          </h2>
          {isTrackD ? (
            <>
              <p className="text-sm font-bold text-rose-200">FOUR_CLASS_MODEL_NOT_READY</p>
              {readinessLoading && (
                <p role="status" className="mt-2 text-sm text-slate-300">
                  Checking Track D dataset readiness…
                </p>
              )}
              {readinessError && (
                <div role="alert" className="mt-3 space-y-3">
                  <p className="text-sm text-rose-200">Readiness is unavailable. {readinessError}</p>
                  <button className={linkClass} type="button" onClick={() => void loadReadiness()}>
                    <RotateCw size={15} aria-hidden="true" />
                    Retry readiness check
                  </button>
                </div>
              )}
              {readiness && (
                <>
                  <p className="mt-2 text-sm text-slate-200">
                    Backend dataset status: <strong>{humanize(readiness.readiness_status)}</strong>
                  </p>
                  <p className="mt-2 text-xs leading-5 text-slate-300">
                    Four-class inference is unavailable while genuine dataset, validation, and governance requirements remain incomplete. The server-side 503 guard remains in force.
                  </p>
                  {readiness.blockers.length > 0 && (
                    <ul className="mt-3 list-inside list-disc space-y-1 text-xs text-rose-100">
                      {readiness.blockers.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  )}
                </>
              )}
              <Link className={`${linkClass} mt-4`} to="/governance">
                <Database size={15} aria-hidden="true" />
                Review governance
                <ArrowRight size={14} aria-hidden="true" />
              </Link>
            </>
          ) : selectedTrack ? (
            <>
              <p className="text-sm font-semibold text-slate-100">
                Track registry status: {humanize(selectedTrack.status)}
              </p>
              <p className="mt-2 text-xs leading-5 text-slate-400">
                Registry information does not verify checkpoint loading or authorize inference.
              </p>
              <Link className={`${linkClass} mt-4`} to="/evidence">
                <FileCheck2 size={15} aria-hidden="true" />
                Inspect model evidence
                <ArrowRight size={14} aria-hidden="true" />
              </Link>
            </>
          ) : (
            <p className="text-sm text-slate-400">Select an available track to inspect its registry status.</p>
          )}
        </aside>
      </section>

      <section className={panelClass} aria-labelledby="methods-title">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 id="methods-title" className="text-xl font-bold text-white">
              Interactive XAI &amp; Diagnostic Research Modules
            </h2>
            <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">
              Launch the 4-class X-ray/CT/MRI differential workbench on Home or run all 4 validated FastAPI engineering engines together on Stage 31 fixtures.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => void executeCombinedWorkbench()}
              disabled={workbenchRunning}
              className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-cyan-600 bg-cyan-950/60 px-4 py-2 text-xs font-bold text-cyan-200 hover:bg-cyan-900/60 disabled:opacity-50"
            >
              {workbenchRunning ? (
                <RotateCw size={14} className="animate-spin" aria-hidden="true" />
              ) : (
                <Activity size={14} aria-hidden="true" />
              )}
              Run Combined Stage 31 Pipeline
            </button>
            <Link
              to="/"
              className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-cyan-700 px-4 py-2 text-xs font-bold text-white hover:bg-cyan-600"
            >
              <ScanLine size={14} aria-hidden="true" />
              4-Class X-ray / CT / MRI Home Workbench
            </Link>
          </div>
        </div>
        {workbenchError && (
          <p role="alert" className="mt-4 rounded-lg border border-rose-800 bg-rose-950/30 p-3 text-xs text-rose-200">
            {workbenchError}
          </p>
        )}
        {workbenchRun && (
          <div
            className="mt-4 space-y-3 rounded-xl border border-cyan-800/70 bg-slate-950/80 p-4"
            aria-label="Combined synthetic pipeline output"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-300">
                Synchronized 4-Engine Stage 31 Output
                {activeResearchModality ? ` · ${DIAGNOSTIC_UPLOAD_PORTALS[activeResearchModality].shortTitle} Profile` : ''}
              </span>
              <span className="font-mono text-xs font-semibold text-amber-200">
                {workbenchRun.four_class_system_status}
              </span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-xs font-mono tabular-nums">
              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                <p className="font-sans font-semibold text-slate-400">IBSI Radiomics</p>
                <p className="mt-1 font-bold text-white">
                  {workbenchRun.radiomics_panel.verified_features_count}/
                  {workbenchRun.radiomics_panel.total_features_count} Verified
                </p>
                <p className="mt-1 font-sans text-[11px] text-slate-400">
                  Phantom: {workbenchRun.radiomics_panel.phantom_type}
                </p>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                <p className="font-sans font-semibold text-slate-400">Biomechanics &amp; Mirels</p>
                <p className="mt-1 font-bold text-white">
                  Score {workbenchRun.biomechanics_mirels_panel.total_score}/12 ·{' '}
                  {workbenchRun.biomechanics_mirels_panel.fracture_risk_percent}%
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  J: {workbenchRun.biomechanics_geometry_panel.polar_J_mm4.toFixed(1)} mm⁴
                </p>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                <p className="font-sans font-semibold text-slate-400">Grad-CAM Localization</p>
                <p className="mt-1 font-bold text-white">
                  {(workbenchRun.explainability_panel.active_region_ratio * 100).toFixed(1)}% Active Region
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  Peak: [{workbenchRun.explainability_panel.peak_coordinates.join(', ')}]
                </p>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
                <p className="font-sans font-semibold text-slate-400">Uncertainty Triage</p>
                <p className="mt-1 font-bold text-white">
                  {workbenchRun.uncertainty_panel.decision} (
                  {workbenchRun.uncertainty_panel.normalized_entropy.toFixed(3)})
                </p>
                <p className="mt-1 font-sans text-[11px] text-slate-400">
                  {workbenchRun.uncertainty_panel.predicted_class_name}
                </p>
              </div>
            </div>
          </div>
        )}
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {methodLinks.map((item) => (
            <article key={item.path} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <h3 className="font-semibold text-slate-100">{item.title}</h3>
              <p className="mt-2 min-h-10 text-xs leading-5 text-slate-400">{item.description}</p>
              <Link
                className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-cyan-200 underline underline-offset-4"
                to={item.path}
              >
                Open interactive module <ArrowRight size={13} aria-hidden="true" />
              </Link>
            </article>
          ))}
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 text-xs leading-5 text-slate-400">
        <strong className="text-slate-200">Session Boundary:</strong> Track A results and multimodal X-ray, CT scan, and MRI session runs stay in RAM for this browser session only. The clinical Track D diagnostic gate remains enforced with <code>FOUR_CLASS_MODEL_NOT_READY</code> until clinical cohorts are verified.
      </section>
      <footer className="flex flex-wrap gap-2 border-t border-slate-800 pt-5" aria-label="Research navigation">
        <Link className={linkClass} to="/tracks">
          Tracks
        </Link>
        <Link className={linkClass} to="/governance">
          Governance
        </Link>
        <Link className={linkClass} to="/evidence">
          Evidence
        </Link>
        <Link className={linkClass} to="/results">
          Results
        </Link>
      </footer>
    </main>
  );
};

export default ResearchWorkbenchPage;
