import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  Bone,
  CheckCircle2,
  ClipboardCheck,
  Database,
  FileCheck2,
  FlaskConical,
  GitBranch,
  Layers3,
  ListChecks,
  LockKeyhole,
  RotateCw,
  ScanLine,
  ShieldAlert,
  Sparkles,
  type LucideIcon,
} from 'lucide-react';
import type { BonePathologyClass, JobStatusResponse, ModalityType, TriagePriorityTier } from '../api/types';
import { analyzeImage, enqueueAnalyzeJob, getJobsList, pollAnalyzeJob } from '../api/client';
import { useOptionalAnalysis } from '../context/AnalysisContext';
import { UploadZone } from '../components/UploadZone';
import {
  MODALITY_SPECS,
  PATHOLOGY_PROFILES,
  REFERENCE_SCAN_PRESETS,
  type ReferenceScanPreset,
} from '../utils/pathologyCatalog';

const ORDERED_CLASSES: BonePathologyClass[] = ['metastasis', 'paget', 'oi', 'healthy'];
const MODALITIES: ModalityType[] = ['xray', 'ct', 'mri', 'bone_scan'];

const WORKFLOW_STEPS = [
  {
    icon: ScanLine,
    title: '1. Multimodal intake',
    body: 'Upload planar X-rays, CT bone-window slices, MRI marrow scans, or Tc-99m scintigraphy with RAM-only EXIF metadata removal.',
  },
  {
    icon: Layers3,
    title: '2. 4-class differential',
    body: 'Evaluate differential likelihoods across Secondary Bone Metastasis, Paget’s Disease of Bone, Osteogenesis Imperfecta, and Healthy Controls.',
  },
  {
    icon: Sparkles,
    title: '3. Visual & radiomic XAI',
    body: 'Inspect Grad-CAM lesion localization, SHAP radiomic attributions, Mirels fracture vulnerability, and counterfactual boundaries.',
  },
  {
    icon: ClipboardCheck,
    title: '4. Uncertainty & oversight',
    body: 'Selective prediction flags high Shannon entropy (> 0.60) or OOD inputs and logs AES-256-GCM clinician verification.',
  },
];

const GOVERNANCE_FEATURES: Array<{ icon: LucideIcon; title: string; body: string }> = [
  { icon: Database, title: 'Provenance', body: 'Source references and package integrity are tracked separately from clinical validity.' },
  { icon: GitBranch, title: 'Patient-level grouping', body: 'Split preparation is group-aware to reduce cross-patient data leakage.' },
  { icon: LockKeyhole, title: 'Protected tracks', body: 'Track A, B, and C protections and external holdout boundaries remain enforced.' },
  { icon: FileCheck2, title: 'Reviewable audit trail', body: 'Encrypted clinician agreement and disagreement logs inspectable under /audit.' },
];

export const Home: React.FC = () => {
  const navigate = useNavigate();
  const analysis = useOptionalAnalysis();

  const [modality, setModality] = useState<ModalityType>('xray');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<ReferenceScanPreset | null>(null);
  const [localLoading, setLocalLoading] = useState(false);
  const [batchQueueing, setBatchQueueing] = useState(false);
  const [worklistJobs, setWorklistJobs] = useState<JobStatusResponse[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const isRunning = Boolean(analysis?.isAnalyzing) || localLoading;

  const triageBadgeStyle = (tier?: TriagePriorityTier) => {
    switch (tier) {
      case 'STAT_URGENT':
        return 'border-rose-600/80 bg-rose-950/70 text-rose-200';
      case 'PRIORITY_REVIEW':
        return 'border-amber-600/80 bg-amber-950/60 text-amber-200';
      default:
        return 'border-emerald-700/70 bg-emerald-950/50 text-emerald-200';
    }
  };

  const sortWorklist = (jobs: JobStatusResponse[]) => {
    const rank: Record<string, number> = { STAT_URGENT: 3, PRIORITY_REVIEW: 2, ROUTINE: 1 };
    return [...jobs].sort((a, b) => {
      const diff = (rank[b.triagePriority || 'ROUTINE'] || 0) - (rank[a.triagePriority || 'ROUTINE'] || 0);
      if (diff !== 0) return diff;
      return (b.result?.fractureRisk.percent || 0) - (a.result?.fractureRisk.percent || 0);
    });
  };

  const refreshWorklist = async () => {
    try {
      const latest = await getJobsList();
      if (latest.length > 0) {
        setWorklistJobs(sortWorklist(latest));
      }
    } catch {
      // Non-fatal fallback; local polling already updates worklistJobs
    }
  };

  const dispatchBatchItems = async (
    items: Array<{ file: File; modality: ModalityType; sampleCase?: BonePathologyClass }>,
  ) => {
    if (items.length === 0 || batchQueueing) return;
    setUploadError(null);
    setBatchQueueing(true);
    try {
      const initialJobs = await Promise.all(
        items.map((item) =>
          enqueueAnalyzeJob({
            image: item.file,
            modality: item.modality,
            ...(item.sampleCase ? { sampleCase: item.sampleCase } : {}),
          }),
        ),
      );
      setWorklistJobs((prev) => {
        const merged = new Map(prev.map((j) => [j.jobId || j.id, j]));
        initialJobs.forEach((j) => merged.set(j.jobId || j.id, j));
        return sortWorklist(Array.from(merged.values()));
      });

      const finished = await Promise.all(initialJobs.map((job) => pollAnalyzeJob(job.jobId || job.id)));
      setWorklistJobs((prev) => {
        const merged = new Map(prev.map((j) => [j.jobId || j.id, j]));
        finished.forEach((j) => merged.set(j.jobId || j.id, j));
        return sortWorklist(Array.from(merged.values()));
      });
      await refreshWorklist();
    } catch (err) {
      const message =
        err instanceof Error && err.message
          ? err.message
          : 'Unable to enqueue batch triage worklist. Check server connectivity and retry.';
      setUploadError(message);
    } finally {
      setBatchQueueing(false);
    }
  };

  const handleMultipleFiles = async (files: File[]) => {
    await dispatchBatchItems(files.map((file) => ({ file, modality })));
  };

  const handleQueueAllPresets = async () => {
    if (batchQueueing || isRunning) return;
    setUploadError(null);
    setBatchQueueing(true);
    try {
      const presetFiles = await Promise.all(
        REFERENCE_SCAN_PRESETS.map(async (preset) => {
          const response = await fetch(preset.imagePath);
          if (!response.ok) throw new Error(`Could not load ${preset.title}`);
          const blob = await response.blob();
          const file = new File([blob], preset.filename, { type: blob.type || 'image/png' });
          return { file, modality: preset.modality, sampleCase: preset.sampleCase };
        }),
      );
      setBatchQueueing(false);
      await dispatchBatchItems(presetFiles);
    } catch {
      setUploadError('Unable to load all bundled reference cases for batch triage.');
      setBatchQueueing(false);
    }
  };

  const handleSelectFile = (file: File | null) => {
    setSelectedFile(file);
    setSelectedPreset(null);
    setUploadError(null);
  };

  const handleLoadPreset = async (preset: ReferenceScanPreset, autoRun = false) => {
    setUploadError(null);
    setModality(preset.modality);
    setSelectedPreset(preset);
    try {
      const response = await fetch(preset.imagePath);
      if (!response.ok) {
        throw new Error('Could not load bundled reference scan.');
      }
      const blob = await response.blob();
      const file = new File([blob], preset.filename, { type: blob.type || 'image/png' });
      setSelectedFile(file);
      if (autoRun) {
        await submitAnalysis(file, preset.modality, preset.sampleCase);
      }
    } catch {
      setUploadError('Unable to load the selected sample scan. Please upload a PNG, JPEG, or WebP bone image.');
    }
  };

  const submitAnalysis = async (
    fileToAnalyze: File,
    activeModality: ModalityType,
    sampleCase?: BonePathologyClass,
  ) => {
    setUploadError(null);
    setLocalLoading(true);
    try {
      const payload = {
        image: fileToAnalyze,
        modality: activeModality,
        ...(sampleCase ? { sampleCase } : {}),
      };
      if (analysis?.executeAnalysisWithQueue) {
        await analysis.executeAnalysisWithQueue(payload);
      } else {
        await analyzeImage(payload);
      }
      navigate('/results');
    } catch (err) {
      const message =
        err instanceof Error && err.message
          ? err.message
          : 'Unable to complete bone imaging analysis. Check image format and retry.';
      setUploadError(message);
    } finally {
      setLocalLoading(false);
    }
  };

  const handleRunDiagnosis = async () => {
    if (!selectedFile || isRunning) {
      if (!selectedFile) {
        setUploadError('Upload a bone X-ray, CT scan, MRI, or Bone Scintigraphy image — or pick a clinical sample case below.');
      }
      return;
    }
    await submitAnalysis(selectedFile, modality, selectedPreset?.sampleCase);
  };

  return (
    <div className="space-y-14 pb-10 sm:space-y-18">
      {/* ━━━ Primary Diagnostic Workbench Hero ━━━ */}
      <section
        className="relative isolate overflow-hidden rounded-[2rem] border border-slate-800 bg-slate-900/85 px-5 py-8 shadow-2xl shadow-cyan-950/20 sm:px-8 sm:py-10 lg:px-10"
        aria-labelledby="home-title"
      >
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,rgba(8,145,178,.18),transparent_48%),radial-gradient(ellipse_at_bottom_left,rgba(13,148,136,.12),transparent_44%)]" />

        <div className="grid items-start gap-8 lg:grid-cols-[1.05fr_1.15fr]">
          {/* Left column: Value proposition & Modality protocol selector */}
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-800/70 bg-cyan-950/60 px-3 py-1.5 text-xs font-semibold text-cyan-200">
              <Bone className="h-4 w-4" aria-hidden="true" />
              Multimodal Skeletal Diagnosis &amp; Explainable AI
            </div>

            <div>
              <h1 id="home-title" className="text-3xl font-black leading-tight tracking-tight text-white sm:text-4xl lg:text-5xl">
                Bone Disorder Differential <span className="text-cyan-400">Workbench</span>
              </h1>
              <p className="mt-4 text-sm leading-6 text-slate-300 sm:text-base">
                Analyze skeletal <strong className="text-white">X-ray</strong>, <strong className="text-white">CT scan</strong>, <strong className="text-white">MRI</strong>, and <strong className="text-white">Bone Scintigraphy</strong> images for <strong className="text-cyan-200">Secondary Bone Metastasis</strong>, <strong className="text-amber-200">Paget&apos;s Disease</strong>, <strong className="text-sky-200">Osteogenesis Imperfecta (OI)</strong>, and <strong className="text-emerald-200">Healthy Control</strong> evaluation with Grad-CAM, SHAP radiomics, Fracture Risk, and Traceable PDF reporting.
              </p>
            </div>

            {/* Modality Selector */}
            <div className="space-y-2.5" role="group" aria-label="Select imaging modality">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-[.16em] text-cyan-300">1. Select Imaging Modality</span>
                <span className="font-mono text-[11px] text-slate-400">RAM-only EXIF strip</span>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {MODALITIES.map((mod) => {
                  const spec = MODALITY_SPECS[mod];
                  const active = modality === mod;
                  return (
                    <button
                      key={mod}
                      type="button"
                      onClick={() => {
                        setModality(mod);
                        if (selectedPreset) setSelectedPreset(null);
                      }}
                      aria-pressed={active}
                      className={`rounded-xl border px-3 py-2.5 text-left transition focus:outline-none focus:ring-2 focus:ring-cyan-400 ${
                        active
                          ? 'border-cyan-400 bg-cyan-950/70 text-white shadow-lg shadow-cyan-950/50'
                          : 'border-slate-800 bg-slate-950/60 text-slate-300 hover:border-slate-600 hover:text-white'
                      }`}
                    >
                      <span className="block whitespace-nowrap text-xs font-bold sm:text-sm">{spec.shortLabel}</span>
                      <span className="mt-0.5 block truncate font-mono text-[10px] text-slate-400">{spec.id.toUpperCase()}</span>
                    </button>
                  );
                })}
              </div>
              <p className="rounded-xl border border-slate-800 bg-slate-950/70 px-3.5 py-2.5 text-xs leading-5 text-slate-300">
                <strong className="text-cyan-300">{MODALITY_SPECS[modality].label}:</strong> {MODALITY_SPECS[modality].protocolNote}
              </p>
            </div>

            {/* Quick Navigation Links */}
            <div className="flex flex-wrap gap-2.5 pt-1">
              <Link
                to="/radiomics"
                className="inline-flex min-h-10 items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-950/70 px-3.5 py-2 text-xs font-semibold text-slate-200 transition hover:border-cyan-600 hover:text-cyan-200"
              >
                <FlaskConical className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                IBSI Radiomics &amp; Biomechanics Labs
              </Link>
              <Link
                to="/workbench"
                className="inline-flex min-h-10 items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-950/70 px-3.5 py-2 text-xs font-semibold text-slate-200 transition hover:border-cyan-600 hover:text-cyan-200"
              >
                <ScanLine className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                X-ray, CT &amp; MRI Workbench Portals
              </Link>
              <Link
                to="/tracks"
                className="inline-flex min-h-10 items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-950/70 px-3.5 py-2 text-xs font-semibold text-slate-200 transition hover:border-slate-500"
              >
                Research tracks
              </Link>
            </div>
          </div>

          {/* Right column: Upload + Sample Presets + Run Action */}
          <div className="space-y-4 rounded-2xl border border-cyan-900/60 bg-slate-950/85 p-4 shadow-xl sm:p-6">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-bold uppercase tracking-[.16em] text-cyan-300">
                2. Upload Scan or Select Sample Case
              </span>
              <span className="rounded-md border border-teal-800/80 bg-teal-950/50 px-2 py-0.5 font-mono text-[10px] font-semibold text-teal-200">
                PROTOTYPE SIMULATOR READY
              </span>
            </div>

            <UploadZone
              selectedFile={selectedFile}
              onSelectFile={handleSelectFile}
              onSelectMultipleFiles={(files) => void handleMultipleFiles(files)}
              onError={(msg) => setUploadError(msg)}
              disabled={isRunning || batchQueueing}
            />

            {selectedPreset && (
              <div className="flex items-center justify-between gap-3 rounded-xl border border-cyan-800/70 bg-cyan-950/30 px-3.5 py-2 text-xs text-cyan-100">
                <span>
                  Loaded sample case: <strong>{selectedPreset.title}</strong> ({selectedPreset.anatomy})
                </span>
                <span className="font-mono text-[10px] uppercase text-cyan-300">{selectedPreset.sampleCase}</span>
              </div>
            )}

            {uploadError && (
              <div role="alert" className="rounded-xl border border-rose-800 bg-rose-950/40 px-4 py-3 text-xs font-medium text-rose-200">
                {uploadError}
              </div>
            )}

            <button
              type="button"
              onClick={() => void handleRunDiagnosis()}
              disabled={!selectedFile || isRunning}
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-600 to-teal-500 px-5 py-3 text-sm font-extrabold text-white shadow-lg shadow-cyan-900/40 transition hover:from-cyan-500 hover:to-teal-400 disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-cyan-300"
            >
              {isRunning ? (
                <>
                  <RotateCw className="h-4 w-4 animate-spin" aria-hidden="true" />
                  Running Differential &amp; XAI Pipeline…
                </>
              ) : (
                <>
                  <Activity className="h-4 w-4" aria-hidden="true" />
                  Run Bone Diagnosis ({MODALITY_SPECS[modality].shortLabel})
                  <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </>
              )}
            </button>

            {/* Quick 4-Class Clinical Reference Cases */}
            <div className="border-t border-slate-800 pt-4">
              <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
                <p className="text-[11px] font-bold uppercase tracking-[.14em] text-slate-400">
                  Quick-Test Clinical Reference Cases (Click to Run)
                </p>
                <button
                  type="button"
                  disabled={batchQueueing || isRunning}
                  onClick={() => void handleQueueAllPresets()}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-800/80 bg-cyan-950/60 px-2.5 py-1 text-[11px] font-bold text-cyan-200 transition hover:border-cyan-500 hover:bg-cyan-900/60 disabled:opacity-50 cursor-pointer"
                >
                  {batchQueueing ? (
                    <RotateCw className="h-3 w-3 animate-spin" aria-hidden="true" />
                  ) : (
                    <ListChecks className="h-3 w-3" aria-hidden="true" />
                  )}
                  <span>{batchQueueing ? 'Queueing Batch…' : 'Queue All 4 Cases (Triage)'}</span>
                </button>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {REFERENCE_SCAN_PRESETS.map((preset) => {
                  const isActivePreset = selectedPreset?.id === preset.id;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      disabled={isRunning || batchQueueing}
                      onClick={() => void handleLoadPreset(preset, true)}
                      className={`group flex items-start gap-3 rounded-xl border p-2.5 text-left transition disabled:opacity-50 ${
                        isActivePreset
                          ? 'border-cyan-500 bg-cyan-950/50'
                          : 'border-slate-800 bg-slate-900/70 hover:border-cyan-700 hover:bg-slate-900'
                      }`}
                    >
                      <img
                        src={preset.imagePath}
                        alt={preset.title}
                        className="h-11 w-11 shrink-0 rounded-lg border border-slate-700 object-cover bg-slate-950"
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-bold text-white group-hover:text-cyan-200">{preset.title}</p>
                        <p className="mt-0.5 truncate text-[10px] font-mono text-cyan-400">{preset.anatomy}</p>
                        <p className="mt-0.5 line-clamp-1 text-[10px] text-slate-400">{preset.clinicalNote}</p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Live Clinical Triage Reading Worklist (Multi-Scan Bounded Queue) */}
        {worklistJobs.length > 0 && (
          <div
            aria-label="Clinical reading worklist"
            className="mt-8 rounded-2xl border border-cyan-900/60 bg-slate-950/90 p-5 shadow-xl space-y-4"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2.5">
                <ListChecks className="h-5 w-5 text-cyan-400" aria-hidden="true" />
                <div>
                  <h2 className="text-sm font-bold text-white">Prioritized Clinical Reading Worklist</h2>
                  <p className="text-xs text-slate-400">
                    Bounded-concurrency async queue ordered by clinical urgency (STAT Urgent &gt; Priority Review &gt; Routine)
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setWorklistJobs([])}
                className="rounded-lg border border-slate-700 px-2.5 py-1 text-[11px] font-semibold text-slate-300 hover:bg-slate-800 cursor-pointer"
              >
                Clear Worklist
              </button>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              {worklistJobs.map((job) => {
                const topClass = job.result?.prediction.topClass;
                const confidencePct = job.result ? Math.round(job.result.prediction.confidence * 100) : null;
                const key = job.jobId || job.id;
                return (
                  <div
                    key={key}
                    className="flex flex-col justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/75 p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span
                            className={`rounded-md border px-2 py-0.5 font-mono text-[10px] font-bold uppercase ${triageBadgeStyle(
                              job.triagePriority ?? undefined,
                            )}`}
                          >
                            {job.triagePriority || job.status.toUpperCase()}
                          </span>
                          <span className="font-mono text-[11px] text-slate-400">
                            {job.modality ? job.modality.toUpperCase() : 'SCAN'} · {key}
                          </span>
                        </div>
                        <p className="text-sm font-bold text-white">
                          {job.result && topClass
                            ? `${PATHOLOGY_PROFILES[topClass].title} (${confidencePct}%)`
                            : job.status === 'failed'
                            ? 'Analysis Blocked / Failed'
                            : 'Queued in Worker Pool…'}
                        </p>
                        {job.triageReason && (
                          <p className="text-xs text-slate-400">{job.triageReason}</p>
                        )}
                      </div>

                      {job.result && (
                        <div className="text-right font-mono text-[11px] text-slate-300 shrink-0">
                          <div>Risk: <strong>{job.result.fractureRisk.percent}%</strong></div>
                          <div>H: <strong>{job.result.uncertaintyDetail.entropy.toFixed(2)}</strong></div>
                          {typeof job.latencyMs === 'number' && (
                            <div className="text-slate-500">{job.latencyMs}ms</div>
                          )}
                        </div>
                      )}
                    </div>

                    {job.status === 'done' && job.result && (
                      <div className="flex items-center justify-end border-t border-slate-800/80 pt-2.5">
                        <button
                          type="button"
                          onClick={() => {
                            analysis?.setLastResult(job.result!);
                            navigate('/results');
                          }}
                          className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-600/90 px-3 py-1.5 text-xs font-bold text-white transition hover:bg-cyan-500 cursor-pointer"
                        >
                          <span>Open in Results</span>
                          <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* ━━━ Four Target Bone Disorder Ontology & Radiographic Hallmarks ━━━ */}
      <section aria-labelledby="scope-title" className="space-y-5">
        <div className="max-w-3xl">
          <p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-400">Four-class diagnostic ontology</p>
          <h2 id="scope-title" className="mt-2 text-2xl font-bold text-white sm:text-3xl">
            Supported pathologies across X-ray, CT Scan, MRI, &amp; Bone Scintigraphy
          </h2>
          <p className="mt-2 text-sm leading-6 text-slate-400">
            Each uploaded scan is evaluated across four skeletal categories alongside IBSI radiomics features and Mirels biomechanical fracture scoring.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {ORDERED_CLASSES.map((cls, index) => {
            const profile = PATHOLOGY_PROFILES[cls];
            return (
              <article key={cls} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 transition hover:border-cyan-900/80">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <span className="font-mono text-[10px] font-semibold text-cyan-400">0{index + 1} · {profile.code}</span>
                    <h3 className="mt-1 text-lg font-bold text-white">{profile.title}</h3>
                    <p className="mt-0.5 text-xs text-slate-400">{profile.subtitle}</p>
                  </div>
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-teal-400" aria-hidden="true" />
                </div>
                <ul className="mt-3 space-y-1.5 text-xs leading-5 text-slate-300">
                  {profile.radiographicHallmarks.map((mark) => (
                    <li key={mark} className="flex items-start gap-2">
                      <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-400" aria-hidden="true" />
                      <span>{mark}</span>
                    </li>
                  ))}
                </ul>
                <div className="mt-4 flex flex-wrap gap-1.5 border-t border-slate-800/80 pt-3">
                  {profile.keyRadiomicsFeatures.map((feat) => (
                    <span key={feat} className="rounded-md border border-slate-700 bg-slate-950/70 px-2 py-0.5 font-mono text-[10px] text-slate-300">
                      {feat}
                    </span>
                  ))}
                </div>
              </article>
            );
          })}
        </div>
      </section>

      {/* ━━━ Diagnostic & XAI Workflow Overview ━━━ */}
      <section id="research-workflow" aria-labelledby="workflow-title" className="scroll-mt-24">
        <div className="mb-6 max-w-2xl">
          <p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-400">End-to-end architecture</p>
          <h2 id="workflow-title" className="mt-2 text-2xl font-bold text-white sm:text-3xl">Explainable pipeline &amp; governed validation</h2>
          <p className="mt-2 text-sm leading-6 text-slate-400">
            Interactive research prototype analysis runs deterministically in RAM while dataset qualification, patient-level splitting, and Track D clinical gates remain enforced.
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {WORKFLOW_STEPS.map(({ icon: Icon, title, body }) => (
            <article key={title} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 transition hover:border-cyan-900/80">
              <Icon className="h-5 w-5 text-cyan-400" aria-hidden="true" />
              <h3 className="mt-4 text-base font-bold text-white">{title}</h3>
              <p className="mt-2 text-xs leading-6 text-slate-400">{body}</p>
            </article>
          ))}
        </div>
      </section>

      {/* ━━━ Data Integrity & Track D Gate Transparency ━━━ */}
      <section aria-labelledby="integrity-title" className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5 sm:p-8">
        <div className="grid gap-6 lg:grid-cols-[.85fr_1.15fr] lg:items-center">
          <div>
            <p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-400">Data integrity &amp; security</p>
            <h2 id="integrity-title" className="mt-2 text-2xl font-bold text-white">Traceable preparation. Explicit safety boundaries.</h2>
            <p className="mt-3 text-sm leading-6 text-slate-400">
              All uploaded images are EXIF-stripped in RAM and held under AES-256-GCM ephemeral storage. Clinical deployment of the trained 4-class Track D checkpoint remains separate under <code className="rounded bg-slate-950 px-1.5 py-0.5 text-amber-200">FOUR_CLASS_MODEL_NOT_READY</code>.
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              <Link to="/governance" className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-cyan-900 px-3.5 py-2 text-sm font-semibold text-cyan-200 hover:bg-cyan-950/50">
                Review governance <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
              <Link to="/architecture" className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-slate-700 px-3.5 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800">
                View safety status
              </Link>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {GOVERNANCE_FEATURES.map(({ icon: Icon, title, body }) => (
              <div key={title} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                <Icon className="h-4 w-4 text-cyan-400" aria-hidden="true" />
                <h3 className="mt-3 text-sm font-bold text-slate-100">{title}</h3>
                <p className="mt-1 text-xs leading-5 text-slate-400">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section aria-labelledby="responsible-title" className="grid gap-4 rounded-2xl border border-amber-900/60 bg-amber-950/20 p-5 sm:p-7 lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="flex items-start gap-4">
          <ShieldAlert className="mt-0.5 h-6 w-6 shrink-0 text-amber-300" aria-hidden="true" />
          <div>
            <h2 id="responsible-title" className="text-lg font-bold text-white">Research prototype simulation active — not a certified clinical device</h2>
            <p className="mt-1.5 max-w-3xl text-sm leading-6 text-slate-300">
              Diagnostic results produced while trained four-class clinical weights remain gated (<code className="rounded bg-slate-950 px-1.5 py-0.5 text-amber-200">FOUR_CLASS_MODEL_NOT_READY</code>) are deterministic simulation outputs for architectural evaluation and research workflows.
            </p>
          </div>
        </div>
        <Link to="/evidence" className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-amber-800 px-4 py-2 text-sm font-semibold text-amber-100 hover:bg-amber-950/50">
          Model evidence <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </section>

      <nav aria-label="Explore OsteoGuard pages" className="border-t border-slate-800 pt-6">
        <p className="mb-3 text-xs font-semibold uppercase tracking-[.16em] text-slate-500">Explore platform</p>
        <div className="flex flex-wrap gap-x-5 gap-y-2 text-sm">
          {[
            ['/results', 'Latest Results'],
            ['/radiomics', 'Radiomics Lab'],
            ['/biomechanics', 'Biomechanics & Mirels'],
            ['/explainability', 'Grad-CAM Explorer'],
            ['/uncertainty', 'Uncertainty & Abstention'],
            ['/workbench', 'Track A Workbench'],
            ['/tracks', 'Research Tracks'],
            ['/governance', 'Governance'],
            ['/evidence', 'Evidence Registry'],
            ['/architecture', 'Architecture & Safety'],
            ['/audit', 'Clinician Audit'],
            ['/about', 'About'],
          ].map(([path, label]) => (
            <Link key={path} to={path} className="rounded text-slate-400 underline-offset-4 hover:text-cyan-300 hover:underline focus:outline-none focus:ring-2 focus:ring-cyan-400">
              {label}
            </Link>
          ))}
        </div>
      </nav>
    </div>
  );
};

export default Home;
