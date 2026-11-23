import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowDown,
  ArrowRight,
  Bone,
  CheckCircle2,
  ClipboardCheck,
  Database,
  FileCheck2,
  FlaskConical,
  GitBranch,
  Layers3,
  LockKeyhole,
  ScanLine,
  ShieldAlert,
  type LucideIcon,
} from 'lucide-react';
import { UploadZone } from '../components/UploadZone';
import { Toast } from '../components/Toast';
import { Loader } from '../components/Loader';
import type { ModalityType } from '../api/types';
import { useAnalysis } from '../context/AnalysisContext';

const CONDITIONS = [
  { id: 'SECONDARY_BONE_METASTASIS', label: 'Secondary bone metastasis', note: 'Metastatic disease involving the skeleton' },
  { id: 'PAGETS_DISEASE', label: "Paget's disease of bone", note: 'A distinct disorder of bone remodeling' },
  { id: 'OSTEOGENESIS_IMPERFECTA', label: 'Osteogenesis imperfecta', note: 'A genetic connective tissue and bone disorder' },
  { id: 'HEALTHY_SKELETAL_CONTROL', label: 'Healthy skeletal control', note: 'A carefully defined comparison cohort' },
] as const;

const MODALITIES: Array<{ value: ModalityType; label: string; detail: string }> = [
  { value: 'xray', label: 'X-ray', detail: 'Planar radiography' },
  { value: 'bone_scan', label: 'Bone scintigraphy', detail: 'Nuclear medicine imaging' },
  { value: 'ct', label: 'CT', detail: 'Cross-sectional imaging' },
];

const workflow = [
  { icon: Layers3, title: 'Explore research tracks', body: 'Review the scope and status of the existing fracture, scintigraphy, and frozen multi-center research tracks.' },
  { icon: ClipboardCheck, title: 'Qualify and govern sources', body: 'Record source evidence, access requirements, acquisition status, and provenance before a cohort is considered for research use.' },
  { icon: ScanLine, title: 'Prepare data reproducibly', body: 'Run technical inventory, imaging quality checks, modality-aware preparation, and patient-level split audits.' },
  { icon: FlaskConical, title: 'Inspect research methods', body: 'Use the workbench and research modules for engineering demonstrations and methods exploration.' },
];

const GOVERNANCE_FEATURES: Array<{ icon: LucideIcon; title: string; body: string }> = [
  { icon: Database, title: 'Provenance', body: 'Source references and package integrity are tracked separately from clinical validity.' },
  { icon: GitBranch, title: 'Patient-level grouping', body: 'Split preparation is group-aware to reduce data leakage.' },
  { icon: LockKeyhole, title: 'Protected tracks', body: 'Track A, B, and C protections and external holdout boundaries remain in force.' },
  { icon: FileCheck2, title: 'Reviewable reports', body: 'Preparation and governance states can be inspected without exposing patient identifiers.' },
];

const Home: React.FC = () => {
  const navigate = useNavigate();
  const { executeAnalysisWithQueue, isAnalyzing, error, setError } = useAnalysis();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModality, setSelectedModality] = useState<ModalityType>('xray');
  const [workspaceOpen, setWorkspaceOpen] = useState(false);

  const runPrototypeAnalysis = async () => {
    if (!selectedFile) return;
    try {
      await executeAnalysisWithQueue({ image: selectedFile, modality: selectedModality });
      navigate('/results');
    } catch {
      // The shared analysis context presents the current backend or simulation error.
    }
  };

  return (
    <div className="space-y-16 pb-8 sm:space-y-20">
      <Toast message={error} onClose={() => setError(null)} />

      <section className="relative isolate overflow-hidden rounded-[2rem] border border-slate-800 bg-slate-900/80 px-5 py-12 shadow-2xl shadow-cyan-950/20 sm:px-10 sm:py-16 lg:px-14" aria-labelledby="home-title">
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,rgba(8,145,178,.16),transparent_46%),radial-gradient(ellipse_at_bottom_left,rgba(13,148,136,.11),transparent_42%)]" />
        <div className="grid items-center gap-10 lg:grid-cols-[1.2fr_.8fr]">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-800/70 bg-cyan-950/50 px-3 py-1.5 text-xs font-semibold text-cyan-200">
              <Bone className="h-4 w-4" aria-hidden="true" /> Research software for skeletal imaging
            </div>
            <h1 id="home-title" className="max-w-3xl text-4xl font-black leading-tight tracking-tight text-white sm:text-5xl lg:text-6xl">
              OsteoGuard AI<span className="text-cyan-400">.</span><br />
              <span className="text-slate-300">A research framework for bone imaging.</span>
            </h1>
            <p className="mt-6 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
              Explore reproducible imaging methods, dataset governance, and safety controls for research into skeletal disease. The four-class Track D diagnostic capability is not clinically ready and remains gated.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/workbench" className="inline-flex min-h-12 items-center gap-2 rounded-xl bg-cyan-600 px-5 py-3 text-sm font-bold text-white transition hover:bg-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-300">
                Open research workbench <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
              <Link to="/tracks" className="inline-flex min-h-12 items-center gap-2 rounded-xl border border-slate-700 bg-slate-950/60 px-5 py-3 text-sm font-semibold text-slate-200 transition hover:border-slate-500 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-300">
                Explore research tracks
              </Link>
              <button type="button" onClick={() => { setWorkspaceOpen(true); document.getElementById('prototype-workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} className="inline-flex min-h-12 items-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold text-slate-300 transition hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300">
                Open image interface <ArrowDown className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
          </div>

          <div className="mx-auto w-full max-w-md" aria-label="Research workflow illustration">
            <div className="relative rounded-3xl border border-cyan-900/70 bg-slate-950/80 p-5 shadow-xl sm:p-7">
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div><p className="text-[10px] font-bold uppercase tracking-[.2em] text-cyan-400">Research pathway</p><p className="mt-1 text-sm font-semibold text-slate-100">Evidence before inference</p></div>
                <ScanLine className="h-6 w-6 text-cyan-300" aria-hidden="true" />
              </div>
              <div className="mt-5 space-y-3">
                {[
                  ['01', 'Source and governance', 'Evidence · access · provenance'],
                  ['02', 'Dataset preparation', 'Inventory · QC · patient groups'],
                  ['03', 'Research methods', 'Modality-aware analysis'],
                  ['04', 'Safety and review', 'Track D remains gated'],
                ].map(([number, title, subtitle], index) => (
                  <div key={number} className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/80 p-3">
                    <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border text-xs font-bold ${index === 3 ? 'border-amber-800 bg-amber-950/50 text-amber-300' : 'border-cyan-900 bg-cyan-950/40 text-cyan-300'}`}>{number}</span>
                    <div className="min-w-0"><p className="text-sm font-semibold text-slate-100">{title}</p><p className="mt-0.5 text-[11px] leading-4 text-slate-400">{subtitle}</p></div>
                    {index === 3 ? <LockKeyhole className="ml-auto h-4 w-4 shrink-0 text-amber-400" aria-label="Gated" /> : <CheckCircle2 className="ml-auto h-4 w-4 shrink-0 text-teal-400" aria-hidden="true" />}
                  </div>
                ))}
              </div>
              <p className="mt-4 text-[11px] leading-5 text-slate-500">Workflow states describe software controls; they do not establish clinical validity or dataset availability.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="research-workflow" aria-labelledby="workflow-title" className="scroll-mt-24">
        <div className="mb-6 max-w-2xl"><p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-400">How the platform is organized</p><h2 id="workflow-title" className="mt-2 text-2xl font-bold text-white sm:text-3xl">A governed research workflow</h2><p className="mt-3 text-sm leading-6 text-slate-400">Dataset qualification, verification, and preparation are separate steps. None automatically starts training or enables diagnosis.</p></div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {workflow.map(({ icon: Icon, title, body }, index) => <article key={title} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 transition hover:border-cyan-900/80">
            <div className="flex items-center justify-between"><span className="text-xs font-mono text-slate-500">0{index + 1}</span><Icon className="h-5 w-5 text-cyan-400" aria-hidden="true" /></div>
            <h3 className="mt-5 text-base font-bold text-white">{title}</h3><p className="mt-2 text-sm leading-6 text-slate-400">{body}</p>
          </article>)}
        </div>
      </section>

      <section aria-labelledby="scope-title" className="grid gap-8 lg:grid-cols-[1fr_.8fr]">
        <div>
          <p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-400">Research scope</p><h2 id="scope-title" className="mt-2 text-2xl font-bold text-white sm:text-3xl">Four target categories, three distinct modalities</h2>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">The Track D ontology is fixed. It is a research target definition, not a statement that a trained or validated four-class model is available.</p>
          <ul className="mt-5 grid gap-3 sm:grid-cols-2">
            {CONDITIONS.map(({ id, label, note }, index) => <li key={id} className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <span className="text-[10px] font-mono text-cyan-400">0{index + 1} · {id}</span><h3 className="mt-2 text-sm font-bold text-slate-100">{label}</h3><p className="mt-1 text-xs leading-5 text-slate-400">{note}</p>
            </li>)}
          </ul>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 sm:p-6">
          <div className="flex items-start gap-3"><div className="rounded-xl border border-teal-900 bg-teal-950/50 p-2.5 text-teal-300"><ScanLine className="h-5 w-5" aria-hidden="true" /></div><div><h3 className="text-lg font-bold text-white">Imaging modalities</h3><p className="mt-1 text-sm text-slate-400">Handled as distinct imaging pathways; they are not silently combined.</p></div></div>
          <ul className="mt-5 space-y-3">{MODALITIES.map(({ value, label, detail }) => <li key={value} className="flex items-center justify-between gap-4 border-t border-slate-800 pt-3"><div><p className="text-sm font-semibold text-slate-100">{label}</p><p className="text-xs text-slate-500">{detail}</p></div><span className="rounded-full border border-slate-700 px-2.5 py-1 text-[10px] font-mono uppercase text-slate-400">{value.replace('_', ' ')}</span></li>)}</ul>
          <p className="mt-5 rounded-lg bg-slate-950/70 p-3 text-xs leading-5 text-amber-200/90">Availability of a modality in the interface does not mean an authorized clinical cohort or validated model exists for it.</p>
        </div>
      </section>

      <section aria-labelledby="integrity-title" className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5 sm:p-8">
        <div className="grid gap-6 lg:grid-cols-[.8fr_1.2fr] lg:items-center">
          <div><p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-400">Data integrity and governance</p><h2 id="integrity-title" className="mt-2 text-2xl font-bold text-white">Traceable preparation. Explicit boundaries.</h2><p className="mt-3 text-sm leading-6 text-slate-400">The repository includes controlled source qualification, onboarding and provenance records, package verification, technical QC, and patient-level split audits.</p><Link to="/governance" className="mt-5 inline-flex min-h-10 items-center gap-2 rounded-lg border border-cyan-900 px-3 py-2 text-sm font-semibold text-cyan-200 hover:bg-cyan-950/50 focus:outline-none focus:ring-2 focus:ring-cyan-400">Review governance <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link></div>
          <div className="grid gap-3 sm:grid-cols-2">{GOVERNANCE_FEATURES.map(({ icon: Icon, title, body }) => <div key={title} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4"><Icon className="h-4 w-4 text-cyan-400" aria-hidden="true" /><h3 className="mt-3 text-sm font-bold text-slate-100">{title}</h3><p className="mt-1 text-xs leading-5 text-slate-400">{body}</p></div>)}</div>
        </div>
      </section>

      <section aria-labelledby="responsible-title" className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-center rounded-2xl border border-amber-900/60 bg-amber-950/20 p-5 sm:p-7">
        <div className="flex items-start gap-4"><ShieldAlert className="mt-0.5 h-6 w-6 shrink-0 text-amber-300" aria-hidden="true" /><div><h2 id="responsible-title" className="text-lg font-bold text-white">Research prototype, not a clinical decision system</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">Track D has no validated four-class clinical model. Its diagnostic API remains gated with HTTP 503 <code className="rounded bg-slate-950 px-1.5 py-0.5 text-amber-200">FOUR_CLASS_MODEL_NOT_READY</code>. Existing demos and preliminary track results do not establish clinical performance or readiness.</p></div></div>
        <Link to="/architecture" className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-amber-800 px-4 py-2 text-sm font-semibold text-amber-100 hover:bg-amber-950/50 focus:outline-none focus:ring-2 focus:ring-amber-300">View safety status <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link>
      </section>

      <section className="rounded-3xl bg-gradient-to-r from-cyan-950/60 to-slate-900 px-5 py-8 text-center ring-1 ring-cyan-900/60 sm:px-10 sm:py-10" aria-labelledby="next-step-title">
        <h2 id="next-step-title" className="text-2xl font-bold text-white sm:text-3xl">Start with the research, not a diagnosis.</h2><p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-slate-300">Explore implemented research demonstrations, inspect the track boundaries, or review the source governance workflow.</p>
        <div className="mt-6 flex flex-wrap justify-center gap-3"><Link to="/workbench" className="inline-flex min-h-11 items-center gap-2 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-bold text-white hover:bg-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-300">Open Workbench <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link><Link to="/tracks" className="inline-flex min-h-11 items-center rounded-lg border border-slate-600 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-300">Research tracks</Link><Link to="/about" className="inline-flex min-h-11 items-center rounded-lg px-4 py-2 text-sm font-semibold text-slate-300 hover:text-white focus:outline-none focus:ring-2 focus:ring-cyan-300">About the project</Link></div>
      </section>

      <section id="prototype-workspace" className="scroll-mt-24" aria-labelledby="workspace-title">
        <div className="mx-auto max-w-3xl rounded-2xl border border-slate-800 bg-slate-900/55 p-5 sm:p-7">
          <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="text-xs font-bold uppercase tracking-[.2em] text-slate-500">Existing interface</p><h2 id="workspace-title" className="mt-1 text-xl font-bold text-white">Prototype image workspace</h2><p className="mt-2 max-w-2xl text-xs leading-5 text-slate-400">This preserves the current upload flow. It is an engineering interface; do not interpret its output as a Track D clinical diagnosis.</p></div><button type="button" className="min-h-10 rounded-lg border border-slate-700 px-3 text-xs font-semibold text-slate-200 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-400" onClick={() => setWorkspaceOpen((open) => !open)} aria-expanded={workspaceOpen}>{workspaceOpen ? 'Close workspace' : 'Open workspace'}</button></div>
          {workspaceOpen && <div className="mt-6 space-y-5 border-t border-slate-800 pt-5">
            <p className="text-sm leading-6 text-amber-200/90">Use only images you are authorized to process. The prototype may use configured research or simulation paths; returned output is not clinical evidence.</p>
            <fieldset disabled={isAnalyzing} className="grid gap-2 sm:grid-cols-3"><legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-300">Select an input modality</legend>{MODALITIES.map(({ value, label, detail }) => <button type="button" key={value} aria-pressed={selectedModality === value} onClick={() => setSelectedModality(value)} className={`rounded-xl border p-3 text-left focus:outline-none focus:ring-2 focus:ring-cyan-400 ${selectedModality === value ? 'border-cyan-600 bg-cyan-950/40 text-cyan-100' : 'border-slate-800 bg-slate-950/60 text-slate-300'}`}><span className="block text-sm font-semibold">{label}</span><span className="mt-1 block text-xs text-slate-500">{detail}</span></button>)}</fieldset>
            <UploadZone selectedFile={selectedFile} onSelectFile={setSelectedFile} onError={setError} disabled={isAnalyzing} />
            {isAnalyzing ? <div role="status" aria-live="polite" className="rounded-xl border border-slate-800 bg-slate-950 p-4"><Loader message="Processing research prototype input…" /></div> : <button type="button" onClick={runPrototypeAnalysis} disabled={!selectedFile} className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-lg bg-cyan-700 px-4 py-3 text-sm font-bold text-white transition hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-cyan-300">Run prototype analysis <ArrowRight className="h-4 w-4" aria-hidden="true" /></button>}
          </div>}
        </div>
      </section>

      <nav aria-label="Explore OsteoGuard pages" className="border-t border-slate-800 pt-6">
        <p className="mb-3 text-xs font-semibold uppercase tracking-[.16em] text-slate-500">Explore</p>
        <div className="flex flex-wrap gap-x-5 gap-y-2 text-sm">{[
          ['/about', 'About'], ['/workbench', 'Research Workbench'], ['/tracks', 'Research Tracks'], ['/governance', 'Governance'], ['/evidence', 'Evidence Registry'], ['/architecture', 'Architecture and Safety'], ['/audit', 'Audit'],
        ].map(([path, label]) => <Link key={path} to={path} className="rounded text-slate-400 underline-offset-4 hover:text-cyan-300 hover:underline focus:outline-none focus:ring-2 focus:ring-cyan-400">{label}</Link>)}</div>
      </nav>
    </div>
  );
};

export { Home };
export default Home;
