import React from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, ArrowRight, ShieldAlert } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { ProbabilityChart } from '../components/ProbabilityChart';
import { AbstentionBanner } from '../components/AbstentionBanner';

const CLASS_DISPLAY = {
  metastasis: 'Secondary bone metastasis',
  paget: "Paget's disease of bone",
  oi: 'Osteogenesis imperfecta',
  healthy: 'Healthy skeletal control',
} as const;

export const Results: React.FC = () => {
  const { lastResult } = useAnalysis();

  if (!lastResult) {
    return (
      <section className="mx-auto max-w-3xl space-y-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-6" aria-labelledby="results-empty-title">
        <h1 id="results-empty-title" className="text-2xl font-bold text-white">No current analysis result</h1>
        <p className="text-sm leading-6 text-slate-300">Results are held in memory only and are cleared when this application session ends. No historical results are fabricated or restored from browser storage.</p>
        <p role="status" className="rounded-lg border border-amber-800 bg-amber-950/30 p-3 text-sm font-semibold text-amber-100">Track D diagnosis remains gated with FOUR_CLASS_MODEL_NOT_READY until its data, governance, model, and validation requirements pass.</p>
        <Link to="/workbench" className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-cyan-700 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300">Open research workbench <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link>
      </section>
    );
  }

  const classLabel = CLASS_DISPLAY[lastResult.prediction.topClass];
  const probabilities = Object.values(lastResult.prediction.probabilities);
  const validProbabilities = probabilities.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
    && Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 0.01;
  const entropy = lastResult.uncertaintyDetail?.entropy;
  const oodFlag = lastResult.uncertaintyDetail?.oodFlag ?? true;
  const shouldAbstain = !Number.isFinite(entropy) || (entropy as number) > 0.6 || oodFlag || !validProbabilities || !classLabel;
  const confidence = lastResult.prediction.confidence;
  const validConfidence = Number.isFinite(confidence) && confidence >= 0 && confidence <= 1;

  if (shouldAbstain) {
    return (
      <section className="mx-auto max-w-3xl space-y-5 rounded-2xl border border-amber-900/70 bg-slate-900/70 p-6" aria-labelledby="result-withheld-title">
        {Number.isFinite(entropy) ? <AbstentionBanner entropy={entropy as number} oodFlag={oodFlag} /> : <div role="alert" className="rounded-xl border border-rose-800 bg-rose-950/30 p-4 text-sm text-rose-100">Uncertainty information is missing or invalid. No prediction is displayed.</div>}
        <div className="flex items-start gap-3"><AlertTriangle className="mt-1 h-5 w-5 shrink-0 text-amber-300" aria-hidden="true" /><div><h1 id="result-withheld-title" className="text-2xl font-bold text-amber-100">Result withheld</h1><p className="mt-2 text-sm leading-6 text-slate-300">The response did not satisfy the uncertainty and output-integrity checks. The class, probabilities, confidence, and derived explanations are withheld; abstention is not a negative or healthy finding.</p></div></div>
        <p className="font-mono text-xs text-slate-500">Reference: {lastResult.requestId} · {lastResult.modality}</p>
        <div className="flex items-start gap-3 rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-xs leading-5 text-slate-400"><ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-rose-300" aria-hidden="true" /><p>Grad-CAM, SHAP, counterfactual, fracture-risk, and report-generation services are not available as verified model-specific outputs in this application.</p></div>
        <Link to="/governance" className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300">Review governance <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link>
      </section>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6 py-4 sm:py-6">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-800 pb-5">
        <div><p className="text-xs font-semibold uppercase tracking-[.16em] text-cyan-300">Research inference response</p><h1 className="mt-2 text-3xl font-black text-white">Analysis result</h1><p className="mt-2 text-sm text-slate-400">This output is not a clinical diagnosis. A model score is not calibrated confidence.</p></div>
        <p className="font-mono text-xs text-slate-500">{lastResult.requestId} · {lastResult.modality} · {new Date(lastResult.timestamp).toLocaleString()}</p>
      </header>

      <section className="rounded-2xl border border-cyan-900/70 bg-slate-900/70 p-6" aria-labelledby="prediction-title">
        <h2 id="prediction-title" className="text-xs font-bold uppercase tracking-wide text-slate-400">Backend-reported class</h2>
        <p className="mt-2 text-2xl font-extrabold text-cyan-100">{classLabel}</p>
        {validConfidence && <p className="mt-2 text-sm text-slate-300">Uncalibrated model score: {(confidence * 100).toFixed(1)}%</p>}
        <p className="mt-4 text-xs leading-5 text-slate-500">{lastResult.disclaimer}</p>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5" aria-labelledby="probability-title">
        <h2 id="probability-title" className="text-base font-bold text-white">Backend-reported probability values</h2>
        <p className="mb-4 mt-1 text-xs text-slate-400">Uncalibrated output values; not validated clinical evidence.</p>
        <ProbabilityChart probabilities={lastResult.prediction.probabilities} />
        <table className="sr-only"><caption>Backend-reported class probabilities</caption><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>{Object.entries(lastResult.prediction.probabilities).map(([key, value]) => <tr key={key}><td>{key}</td><td>{value}</td></tr>)}</tbody></table>
      </section>

      <section className="rounded-2xl border border-amber-900/70 bg-amber-950/15 p-5"><h2 className="font-semibold text-amber-100">Unavailable derived methods</h2><p className="mt-2 text-sm leading-6 text-slate-300">This response does not include verified, model-specific explainability, counterfactual, fracture-risk, or traceable report-generation evidence. These outputs are not synthesized in the browser.</p></section>
    </div>
  );
};

export default Results;
