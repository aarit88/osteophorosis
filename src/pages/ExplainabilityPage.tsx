import React from 'react';
import { Eye, Info, ShieldAlert } from 'lucide-react';
import { FourClassGatePanel } from '../components/FourClassGatePanel';

const methods = [
  {
    name: 'Grad-CAM',
    status: 'Clinical generation unavailable',
    detail: 'The repository contains a synthetic CNN fixture demonstration. It is not connected to an available, verified OsteoGuard clinical model and must not be interpreted as an image explanation.',
  },
  {
    name: 'SHAP / feature attribution',
    status: 'Clinical generation unavailable',
    detail: 'Analytic fixture calculations exist, but a model-specific SHAP route for a verified clinical classifier is not available. No feature attribution is generated on this page.',
  },
  {
    name: 'Counterfactuals and feature importance',
    status: 'Unavailable for clinical inference',
    detail: 'No validated image-model counterfactual or feature-importance service is currently exposed. The interface does not synthesize a replacement result.',
  },
];

export const ExplainabilityPage: React.FC = () => (
  <div className="mx-auto max-w-5xl space-y-6 py-4 sm:py-6">
    <header className="space-y-2 border-b border-slate-800 pb-5">
      <div className="inline-flex items-center gap-2 rounded-full border border-indigo-800 bg-indigo-950/60 px-3 py-1 text-xs font-semibold text-indigo-200"><Eye className="h-3.5 w-3.5" aria-hidden="true" /> Research method status</div>
      <h1 className="text-3xl font-black tracking-tight text-white">Explainability</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-300">Attribution can describe model behavior; it does not establish medical correctness or causality. This route reports only the current method availability and does not generate a heatmap or attribution from a nonexistent prediction.</p>
    </header>

    <FourClassGatePanel />

    <section className="space-y-3" aria-labelledby="methods-title">
      <div><h2 id="methods-title" className="text-xl font-bold text-white">Implemented method boundaries</h2><p className="mt-1 text-sm text-slate-400">Synthetic engineering demonstrations are not clinical explanations.</p></div>
      <div className="grid gap-3 md:grid-cols-3">
        {methods.map((method) => <article key={method.name} className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-start justify-between gap-3"><h3 className="font-semibold text-slate-100">{method.name}</h3><span className="rounded-full border border-amber-800 bg-amber-950/40 px-2 py-1 text-[10px] font-semibold text-amber-200">Unavailable</span></div>
          <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-amber-200">{method.status}</p>
          <p className="mt-2 text-sm leading-6 text-slate-400">{method.detail}</p>
        </article>)}
      </div>
    </section>

    <aside className="flex items-start gap-3 rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-sm leading-6 text-slate-300"><Info className="mt-0.5 h-4 w-4 shrink-0 text-cyan-300" aria-hidden="true" /><p>Any future explanation must identify its input reference, model artifact, method, configuration, generation time, and limitations, and must use the same authorization and readiness gates as inference.</p></aside>
    <div className="flex items-start gap-3 rounded-xl border border-rose-900/70 bg-rose-950/20 p-4 text-sm text-rose-100"><ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" /><p>Track D diagnostic inference remains blocked with <code>FOUR_CLASS_MODEL_NOT_READY</code> until genuine dataset, governance, training, validation, and readiness requirements are met.</p></div>
  </div>
);

export default ExplainabilityPage;
