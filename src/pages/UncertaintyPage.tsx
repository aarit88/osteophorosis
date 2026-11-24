import React from 'react';
import { Activity, HelpCircle, ShieldAlert } from 'lucide-react';
import { FourClassGatePanel } from '../components/FourClassGatePanel';

const measures = [
  ['Predictive entropy and top-two margin', 'The repository has deterministic calculations for supplied probability vectors. Current use is limited to demonstrations; no verified Track D model output is available here.'],
  ['Epistemic and aleatoric uncertainty', 'No active inference response currently exposes a validated decomposition of these uncertainty sources.'],
  ['Calibration and confidence intervals', 'No calibration dataset or validated calibration artifact is available for Track D. A softmax score must not be described as calibrated confidence.'],
  ['OOD detection and selective prediction', 'Engineering components and thresholds are not clinically validated. Abstention requires human review and is never a negative or healthy result.'],
];

export const UncertaintyPage: React.FC = () => (
  <div className="mx-auto max-w-5xl space-y-6 py-4 sm:py-6">
    <header className="space-y-2 border-b border-slate-800 pb-5">
      <div className="inline-flex items-center gap-2 rounded-full border border-purple-800 bg-purple-950/60 px-3 py-1 text-xs font-semibold text-purple-200"><Activity className="h-3.5 w-3.5" aria-hidden="true" /> Research method status</div>
      <h1 className="text-3xl font-black tracking-tight text-white">Uncertainty and abstention</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-300">This page describes supported engineering components and their limits. It does not simulate a calibration plot or present a synthetic probability vector as a patient prediction.</p>
    </header>

    <FourClassGatePanel />

    <section className="space-y-3" aria-labelledby="uncertainty-methods-title">
      <div><h2 id="uncertainty-methods-title" className="text-xl font-bold text-white">Measure availability</h2><p className="mt-1 text-sm text-slate-400">Availability is distinct from implementation of a mathematical component.</p></div>
      <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60">
        <dl className="divide-y divide-slate-800">
          {measures.map(([name, detail]) => <div key={name} className="grid gap-2 p-4 sm:grid-cols-[minmax(12rem,.6fr)_1.4fr] sm:gap-5">
            <dt className="font-semibold text-slate-100">{name}<span className="mt-1 block text-xs font-medium text-amber-200">Patient-specific result unavailable</span></dt>
            <dd className="text-sm leading-6 text-slate-400">{detail}</dd>
          </div>)}
        </dl>
      </div>
    </section>

    <aside className="flex items-start gap-3 rounded-xl border border-amber-900/70 bg-amber-950/20 p-4 text-sm leading-6 text-amber-100"><HelpCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" /><p>The proposed normalized entropy cutoff of 0.60 is an engineering candidate only. It has not been clinically validated and is not presented as an operational decision threshold.</p></aside>
    <div className="flex items-start gap-3 rounded-xl border border-rose-900/70 bg-rose-950/20 p-4 text-sm text-rose-100"><ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" /><p>When uncertainty is unavailable or a gate abstains, the result must be withheld for human review. It must never be interpreted as a healthy or negative finding.</p></div>
  </div>
);

export default UncertaintyPage;
