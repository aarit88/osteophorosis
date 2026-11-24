import React from 'react';
import { AlertTriangle, SlidersHorizontal } from 'lucide-react';
import { Link } from 'react-router-dom';

const families = [
  ['First-order intensity', 'Intensity distribution statistics computed within a mask. Clinical image and approved ROI inputs are not connected to this page.'],
  ['GLCM', 'Gray-level co-occurrence matrix texture descriptors. Quantization and spacing assumptions must be controlled for reproducible use.'],
  ['GLRLM', 'Gray-level run-length matrix texture descriptors. Clinical values are not generated without a governed input image and mask.'],
  ['GLSZM', 'Gray-level size-zone matrix texture descriptors. The implementation requires further validation against the selected feature specification.'],
  ['GLDM / NGTDM-related', 'Neighborhood-dependence and tone-difference family. Some implemented dimensions are engineering heuristics and are not canonical IBSI claims.'],
];

export const RadiomicsPage: React.FC = () => (
  <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6">
    <header className="space-y-3 border-b border-slate-800 pb-5"><div className="inline-flex items-center gap-2 rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 text-xs font-semibold text-cyan-200"><SlidersHorizontal size={14} aria-hidden="true" />Research method inventory</div><h1 className="text-3xl font-black tracking-tight text-white">Radiomics</h1><p className="max-w-3xl text-sm leading-6 text-slate-300">The repository contains a radiomics engine and synthetic phantom validation, but the website does not connect this route to an authorized clinical image, mask, or staged dataset reference. No generated phantom feature values are presented as study results.</p></header>

    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5" aria-labelledby="families-title"><h2 id="families-title" className="text-lg font-bold text-white">Implemented feature families</h2><p className="mt-1 text-sm leading-6 text-slate-400">Feature-family inventory only. Availability in code does not establish feature validity, clinical meaning, or compatibility with the 102-dimensional fusion contract.</p><div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{families.map(([name, description]) => <article key={name} className="rounded-lg border border-slate-800 bg-slate-950/60 p-4"><h3 className="font-semibold text-slate-100">{name}</h3><p className="mt-2 text-sm leading-6 text-slate-400">{description}</p></article>)}</div></section>

    <section className="grid gap-4 md:grid-cols-2" aria-label="Validation status"><div className="rounded-xl border border-teal-900/70 bg-teal-950/15 p-4"><h2 className="font-semibold text-teal-100">Reported analytic verification</h2><p className="mt-2 text-sm leading-6 text-slate-300">Historical project records identify 81 feature dimensions as analytically verified on deterministic phantoms. This is not an independent clinical or IBSI certification.</p></div><div className="rounded-xl border border-amber-900/70 bg-amber-950/20 p-4"><div className="flex items-start gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-200" aria-hidden="true" /><div><h2 className="font-semibold text-amber-100">Unresolved dimensions</h2><p className="mt-2 text-sm leading-6 text-slate-300">21 dimensions were described as engineering heuristics or specification gaps. The full 102-dimensional contract must not be described as canonical IBSI-compliant or fully validated.</p></div></div></div></section>

    <section className="rounded-xl border border-slate-800 bg-slate-950/60 p-5"><h2 className="font-semibold text-white">Input and downstream use</h2><p className="mt-2 text-sm leading-6 text-slate-400">This page accepts no local paths or image uploads. Governed data must pass Stage 38 onboarding and Stage 39 preparation. No feature extraction endpoint for a staged clinical study is exposed here.</p><Link to="/governance" className="mt-4 inline-flex min-h-10 items-center rounded-lg border border-slate-700 px-3 py-2 text-sm font-semibold text-cyan-200 hover:bg-slate-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300">Review dataset governance</Link></section>
  </main>
);

export default RadiomicsPage;
