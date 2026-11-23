import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Bone, BookOpenCheck, Database, Layers3, ShieldAlert } from 'lucide-react';

const TARGETS = [
  ['SECONDARY_BONE_METASTASIS', 'Secondary bone metastasis'],
  ['PAGETS_DISEASE', "Paget's disease of bone"],
  ['OSTEOGENESIS_IMPERFECTA', 'Osteogenesis imperfecta'],
  ['HEALTHY_SKELETAL_CONTROL', 'Healthy skeletal control'],
] as const;

const TECH_STACK = [
  ['Frontend', 'React, TypeScript, Vite, and the existing responsive application shell.'],
  ['Gateway', 'Express provides the public application entry point and forwards supported API requests.'],
  ['Research service', 'FastAPI exposes research, governance, dataset, readiness, and protected inference APIs.'],
  ['Research components', 'Python imaging, radiomics, biomechanics, explainability, uncertainty, and dataset preparation modules.'],
] as const;

export const About: React.FC = () => (
  <article className="mx-auto max-w-5xl space-y-12 py-5 sm:py-9" aria-labelledby="about-title">
    <header className="max-w-3xl">
      <div className="inline-flex items-center gap-2 rounded-full border border-cyan-900 bg-cyan-950/40 px-3 py-1.5 text-xs font-semibold text-cyan-200"><Bone className="h-4 w-4" aria-hidden="true" /> Project overview</div>
      <h1 id="about-title" className="mt-5 text-3xl font-black tracking-tight text-white sm:text-4xl">About OsteoGuard AI</h1>
      <p className="mt-4 text-base leading-7 text-slate-300 sm:text-lg">OsteoGuard AI is a research software prototype exploring how skeletal imaging, quantitative methods, and explicit uncertainty and governance controls can be brought together for bone-disease research.</p>
    </header>

    <section className="grid gap-5 md:grid-cols-2" aria-label="Motivation and problem">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/65 p-5 sm:p-6"><BookOpenCheck className="h-5 w-5 text-cyan-400" aria-hidden="true" /><h2 className="mt-4 text-lg font-bold text-white">Why this project exists</h2><p className="mt-2 text-sm leading-6 text-slate-400">Bone disorders can require different imaging evidence and distinct clinical context. Research tools need to make data provenance, modality assumptions, patient-level separation, and uncertainty visible rather than hiding them behind a single prediction.</p></div>
      <div className="rounded-2xl border border-slate-800 bg-slate-900/65 p-5 sm:p-6"><Layers3 className="h-5 w-5 text-cyan-400" aria-hidden="true" /><h2 className="mt-4 text-lg font-bold text-white">Research problem and scope</h2><p className="mt-2 text-sm leading-6 text-slate-400">The proposal targets a future four-category differential research task across X-ray, bone scintigraphy, and CT. Those modalities have different acquisition properties and are treated as separate pathways; a unified clinical model has not been established.</p></div>
    </section>

    <section aria-labelledby="targets-title">
      <div className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Track D target ontology</p><h2 id="targets-title" className="mt-2 text-2xl font-bold text-white">The intended research categories</h2></div><span className="rounded-full border border-amber-800 bg-amber-950/40 px-3 py-1.5 text-xs font-bold text-amber-200">Model not clinically ready</span></div>
      <ul className="mt-5 grid gap-3 sm:grid-cols-2">
        {TARGETS.map(([identifier, label], index) => <li key={identifier} className="flex items-start gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4"><span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-800 font-mono text-xs text-cyan-300">0{index + 1}</span><div><h3 className="text-sm font-bold text-white">{label}</h3><code className="mt-1 block break-all text-[10px] leading-5 text-slate-500">{identifier}</code></div></li>)}
      </ul>
    </section>

    <section className="grid gap-8 lg:grid-cols-[.8fr_1.2fr]" aria-label="Modalities and technology">
      <div><p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Imaging scope</p><h2 className="mt-2 text-2xl font-bold text-white">Three distinct modalities</h2><ul className="mt-4 space-y-3">{[['X-ray', 'Planar projection images'], ['Bone scintigraphy', 'Nuclear medicine skeletal imaging'], ['CT', 'Cross-sectional computed tomography']].map(([name, detail]) => <li key={name} className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3"><p className="text-sm font-semibold text-slate-100">{name}</p><p className="mt-1 text-xs text-slate-500">{detail}</p></li>)}</ul></div>
      <div><p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Software at a glance</p><h2 className="mt-2 text-2xl font-bold text-white">Current technical architecture</h2><ul className="mt-4 divide-y divide-slate-800 rounded-2xl border border-slate-800 bg-slate-900/60 px-4">{TECH_STACK.map(([layer, detail]) => <li key={layer} className="py-4 sm:flex sm:gap-5"><h3 className="min-w-32 text-sm font-bold text-white">{layer}</h3><p className="mt-1 text-sm leading-6 text-slate-400 sm:mt-0">{detail}</p></li>)}</ul></div>
    </section>

    <section aria-labelledby="status-title" className="rounded-2xl border border-teal-900/60 bg-teal-950/15 p-5 sm:p-7">
      <div className="flex items-start gap-4"><Database className="mt-1 h-5 w-5 shrink-0 text-teal-300" aria-hidden="true" /><div><h2 id="status-title" className="text-xl font-bold text-white">What is implemented</h2><p className="mt-2 text-sm leading-6 text-slate-300">The repository includes a React research interface, an Express gateway, a FastAPI service, source qualification and onboarding, provenance and package verification, dataset quality-control and preparation workflows, and research modules for radiomics, biomechanics, explainability, and uncertainty. The governance and readiness routes expose current system status.</p><p className="mt-3 text-sm leading-6 text-slate-300">The intended model training, complete four-class clinical evaluation, calibration, and clinical validation are not represented as completed capabilities.</p></div></div>
    </section>

    <section aria-labelledby="limitations-title" className="rounded-2xl border border-amber-900/70 bg-amber-950/20 p-5 sm:p-7">
      <div className="flex items-start gap-4"><ShieldAlert className="mt-1 h-5 w-5 shrink-0 text-amber-300" aria-hidden="true" /><div><h2 id="limitations-title" className="text-xl font-bold text-white">Research-only limitations</h2><p className="mt-2 text-sm leading-6 text-slate-300">Track D does not have the genuine, authorized, sufficiently representative clinical datasets and validated model required for four-class diagnosis. Its diagnosis capability remains unavailable and the API is expected to return HTTP 503 <code className="rounded bg-slate-950 px-1.5 py-0.5 text-amber-200">FOUR_CLASS_MODEL_NOT_READY</code>. This prototype is not a medical device and must not be used for patient care.</p><p className="mt-3 text-sm leading-6 text-slate-400">The preliminary Tracks A, B, and C are separate research work. Their data, labels, checkpoints, and evaluation results are not evidence that the Track D task is ready.</p></div></div>
    </section>

    <nav aria-label="About page links" className="flex flex-wrap gap-3 border-t border-slate-800 pt-6"><Link to="/architecture" className="inline-flex min-h-11 items-center gap-2 rounded-lg bg-cyan-700 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-300">View architecture <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link><Link to="/tracks" className="inline-flex min-h-11 items-center rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-300">Explore research tracks</Link><Link to="/governance" className="inline-flex min-h-11 items-center rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-300">Dataset governance</Link></nav>
  </article>
);

export default About;
