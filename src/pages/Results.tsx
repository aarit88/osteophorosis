import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  FileText,
  Layers,
  ShieldAlert,
  SlidersHorizontal,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { AbstentionBanner } from '../components/AbstentionBanner';
import { GradCamCompare } from '../components/GradCamCompare';
import { ProbabilityChart } from '../components/ProbabilityChart';
import { ShapChart } from '../components/ShapChart';
import { FractureGauge } from '../components/FractureGauge';
import { UncertaintyCard } from '../components/UncertaintyCard';
import { CounterfactualPanel } from '../components/CounterfactualPanel';
import { ClinicianFeedbackCard } from '../components/ClinicianFeedbackCard';
import { TraceableReportModal } from '../components/TraceableReportModal';
import {
  evaluateAbstentionAdvisory,
  getModalitySpec,
  getPathologyProfile,
} from '../utils/pathologyCatalog';

const TONE_BADGE: Record<string, string> = {
  rose: 'border-rose-700/80 bg-rose-950/60 text-rose-200',
  amber: 'border-amber-700/80 bg-amber-950/60 text-amber-200',
  sky: 'border-sky-700/80 bg-sky-950/60 text-sky-200',
  emerald: 'border-emerald-700/80 bg-emerald-950/60 text-emerald-200',
};

export const Results: React.FC = () => {
  const { lastResult, engineMode } = useAnalysis();
  const [reportOpen, setReportOpen] = useState(false);

  if (!lastResult) {
    return (
      <section className="mx-auto max-w-3xl space-y-5 rounded-3xl border border-slate-800 bg-slate-900/75 p-6 sm:p-8" aria-labelledby="results-empty-title">
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200">
          <Activity className="h-3.5 w-3.5" aria-hidden="true" /> Diagnostic session empty
        </div>
        <h1 id="results-empty-title" className="text-2xl font-black text-white sm:text-3xl">No current analysis result</h1>
        <p className="text-sm leading-6 text-slate-300">
          Upload a skeletal X-ray, CT scan, or Bone Scintigraphy image — or choose a 1-click reference case (Paget&apos;s Disease, Secondary Bone Metastasis, Osteogenesis Imperfecta, or Healthy Control) — to inspect predictions, Grad-CAM maps, SHAP radiomics, fracture index, and traceable reports.
        </p>
        <p role="status" className="rounded-xl border border-amber-800/80 bg-amber-950/30 p-3.5 text-xs leading-5 text-amber-100">
          Images and inference results stay in RAM for this browser session only. Track D clinical deployment gate remains separate with <code className="font-mono font-bold">FOUR_CLASS_MODEL_NOT_READY</code>.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link to="/" className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-cyan-600 px-5 py-2.5 text-sm font-bold text-white transition hover:bg-cyan-500">
            <ArrowLeft className="h-4 w-4" aria-hidden="true" /> Upload Scan or Run Demo
          </Link>
          <Link to="/workbench" className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-slate-700 px-4 py-2.5 text-sm font-semibold text-slate-200 hover:bg-slate-800">
            Open Research Workbench <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        </div>
      </section>
    );
  }

  const profile = getPathologyProfile(lastResult.prediction.topClass);
  const modalitySpec = getModalitySpec(lastResult.modality);
  const entropy = lastResult.uncertaintyDetail?.entropy ?? 0;
  const oodFlag = Boolean(lastResult.uncertaintyDetail?.oodFlag);
  const advisory = evaluateAbstentionAdvisory(entropy, oodFlag, lastResult.prediction.probabilities);
  const confidence = Number.isFinite(lastResult.prediction.confidence)
    ? lastResult.prediction.confidence
    : 0;

  return (
    <div className="mx-auto max-w-6xl space-y-6 py-4 sm:py-6">
      {/* Header + Actions */}
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-cyan-800 bg-cyan-950/70 px-2.5 py-0.5 font-mono text-[11px] font-bold uppercase text-cyan-200">
              {modalitySpec.shortLabel}
            </span>
            <span className="font-mono text-xs text-slate-400">
              {lastResult.requestId} · {new Date(lastResult.timestamp).toLocaleString()}
            </span>
          </div>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-white">
            Differential Diagnosis &amp; Explainability
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Research prototype output combining deep feature salience, IBSI radiomics attributions, Mirels structural risk, and uncertainty gating.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={() => setReportOpen(true)}
            className="inline-flex min-h-10 items-center gap-2 rounded-xl bg-cyan-600 px-4 py-2 text-xs font-bold text-white shadow-lg shadow-cyan-950/50 transition hover:bg-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-300 sm:text-sm"
          >
            <FileText className="h-4 w-4" aria-hidden="true" />
            Generate Traceable Report / PDF
          </button>
          <Link
            to="/"
            className="inline-flex min-h-10 items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-900 px-3.5 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-800 sm:text-sm"
          >
            <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
            New Scan
          </Link>
        </div>
      </header>

      {/* Advisory Abstention Banner when entropy > 0.6 or OOD */}
      {advisory.hasAdvisory && Number.isFinite(entropy) && (
        <AbstentionBanner entropy={entropy} oodFlag={oodFlag} />
      )}

      {/* Primary Classification & Biological Context Card */}
      <section
        className="rounded-3xl border border-cyan-900/70 bg-slate-900/80 p-5 shadow-xl sm:p-7"
        aria-labelledby="prediction-title"
      >
        <div className="grid gap-6 lg:grid-cols-[1.2fr_.8fr]">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2.5">
              <span id="prediction-title" className="text-xs font-bold uppercase tracking-[.16em] text-cyan-300">
                Primary Predicted Pathology
              </span>
              <span className={`rounded-full border px-2.5 py-0.5 font-mono text-[10px] font-bold ${TONE_BADGE[profile.accentTone]}`}>
                {profile.code}
              </span>
              {advisory.hasAdvisory && (
                <span className="rounded-full border border-amber-600/70 bg-amber-950/60 px-2.5 py-0.5 font-mono text-[10px] font-bold text-amber-200">
                  HIGH UNCERTAINTY ADVISORY
                </span>
              )}
            </div>

            <div className="flex flex-wrap items-baseline justify-between gap-4">
              <div>
                <h2 className="text-2xl font-black text-white sm:text-3xl">{profile.title}</h2>
                <p className="mt-1 text-sm text-slate-300">{profile.subtitle}</p>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-950/80 px-4 py-2.5 text-right">
                <span className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Model Score
                </span>
                <span className="font-mono text-2xl font-black tabular-nums text-cyan-300">
                  {(confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <p className="text-xs leading-6 text-slate-300 sm:text-sm">
              {profile.pathologySummary}
            </p>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/75 p-4 space-y-2.5">
            <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-200">
              <Layers className="h-4 w-4 text-cyan-400" aria-hidden="true" />
              Characteristic Imaging Hallmarks ({modalitySpec.shortLabel})
            </p>
            <ul className="space-y-1.5 text-xs leading-5 text-slate-300">
              {profile.radiographicHallmarks.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-teal-400" aria-hidden="true" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
            <p className="border-t border-slate-800 pt-2 text-[11px] leading-4 text-slate-400">
              {lastResult.disclaimer}
            </p>
          </div>
        </div>
      </section>

      {/* Grad-CAM Visual Comparison & Class Probabilities + Fracture Risk */}
      <div className="grid gap-6 lg:grid-cols-[1.15fr_.85fr]">
        {lastResult.imageUrl && lastResult.gradcamUrl ? (
          <GradCamCompare originalUrl={lastResult.imageUrl} gradcamUrl={lastResult.gradcamUrl} />
        ) : (
          <div className="flex items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 p-8 text-sm text-slate-400">
            Visual Grad-CAM image overlay unavailable for this fixture response.
          </div>
        )}

        <div className="space-y-6">
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-xl" aria-labelledby="probability-title">
            <h2 id="probability-title" className="text-base font-bold text-white">4-Class Differential Probabilities</h2>
            <p className="mb-3 mt-0.5 text-xs text-slate-400">
              Normalized softmax distribution across Metastasis, Paget&apos;s, OI, and Healthy Control.
            </p>
            {advisory.validProbabilities ? (
              <ProbabilityChart probabilities={lastResult.prediction.probabilities} />
            ) : (
              <div role="alert" className="flex items-center gap-2 rounded-xl border border-rose-800 bg-rose-950/30 p-3 text-xs text-rose-200">
                <ShieldAlert className="h-4 w-4 shrink-0" aria-hidden="true" />
                Probability vector failed integrity validation.
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-xl" aria-labelledby="fracture-title">
            <div className="mb-1 flex items-center justify-between">
              <h2 id="fracture-title" className="text-base font-bold text-white">Pathologic Fracture Risk Index</h2>
              <Link to="/biomechanics" className="text-xs font-semibold text-cyan-400 hover:underline">
                Mirels Calculator →
              </Link>
            </div>
            <p className="mb-2 text-xs text-slate-400">
              Estimated structural vulnerability from cortical breach geometry and pathology profile.
            </p>
            <FractureGauge fractureRisk={lastResult.fractureRisk} />
          </section>
        </div>
      </div>

      {/* Quantitative Radiomics (SHAP) & Uncertainty Quantification */}
      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-3xl border border-slate-800 bg-slate-900 p-6 shadow-xl" aria-labelledby="shap-title">
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-cyan-400" aria-hidden="true" />
              <div>
                <h2 id="shap-title" className="text-sm font-bold text-white">Radiomics Feature Attribution (SHAP)</h2>
                <p className="text-[11px] text-slate-400">Signed feature weights driving top-class differentiation</p>
              </div>
            </div>
            <Link to="/radiomics" className="text-xs font-semibold text-cyan-400 hover:underline">
              Radiomics Lab →
            </Link>
          </div>
          {lastResult.shap.length > 0 ? (
            <ShapChart shapFeatures={lastResult.shap} />
          ) : (
            <p className="py-12 text-center text-xs text-slate-500">No SHAP radiomics vectors returned.</p>
          )}
        </section>

        {lastResult.uncertaintyDetail && (
          <UncertaintyCard
            uncertaintyDetail={lastResult.uncertaintyDetail}
            prediction={lastResult.prediction}
          />
        )}
      </div>

      {/* Counterfactual Decision Boundary Panel */}
      {lastResult.counterfactuals.length > 0 && (
        <CounterfactualPanel
          counterfactuals={lastResult.counterfactuals}
          baseProbabilities={lastResult.prediction.probabilities}
          topClass={lastResult.prediction.topClass}
          requestId={lastResult.requestId}
        />
      )}

      {/* Clinician-in-the-Loop Verification & Audit Persistence */}
      <ClinicianFeedbackCard
        requestId={lastResult.requestId}
        predictedClass={lastResult.prediction.topClass}
      />

      {/* Traceable Clinical Decision Support Dossier Modal */}
      <TraceableReportModal
        isOpen={reportOpen}
        onClose={() => setReportOpen(false)}
        result={lastResult}
        engineMode={engineMode === 'simulation' ? 'Research Prototype Simulator' : engineMode}
      />
    </div>
  );
};

export default Results;
