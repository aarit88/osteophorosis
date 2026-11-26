import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, ArrowDown, ArrowRight, Boxes, Check, ChevronRight, Database, LockKeyhole, RotateCw, Server, ShieldAlert } from 'lucide-react';
import { getArchitectureReadiness, getScalabilityStatus, requestFourClassInference } from '../api/client';
import type { ArchitectureComponentStatus, ArchitectureReadinessResponse, BlockedInferenceResponse, ScalabilityStatusResponse } from '../api/types';

const stackLayers = [
  {
    title: 'Web application',
    technology: 'React · TypeScript · Vite',
    icon: Boxes,
    description: 'Shared shell, Home, Workbench, Tracks, Governance, research tools, and evidence pages.',
    status: 'Browser-facing UI',
  },
  {
    title: 'Express gateway',
    technology: 'Node.js · Express 5',
    icon: Server,
    description: 'Serves the built application and proxies supported API paths to the private ML service.',
    status: 'Public application entry',
  },
  {
    title: 'Research and governance API',
    technology: 'FastAPI · Python',
    icon: Database,
    description: 'Research endpoints, dataset onboarding and qualification, preparation, governance, readiness, and gated diagnosis routes.',
    status: 'Private service behind gateway',
  },
] as const;

const dataControls = [
  ['Stage 33–35', 'Manifest intake, clinical ontology/adjudication, and governance/security checks'],
  ['Stage 38', 'Source registration, acquisition tracking, provenance verification, and safe staging'],
  ['Stage 39–40', 'Dataset inventory, technical QC, harmonization/split audits, and source qualification'],
] as const;

function statusLabel(status: string): string {
  switch (status) {
    case 'IMPLEMENTED': return 'Implemented';
    case 'PARTIALLY_IMPLEMENTED': return 'Partially implemented';
    case 'ARCHITECTURALLY_SPECIFIED': return 'Specified, unverified';
    case 'BLOCKED_BY_DATA': return 'Blocked by data';
    default: return 'Not verified';
  }
}

export const ArchitecturePage: React.FC = () => {
  const [readiness, setReadiness] = useState<ArchitectureReadinessResponse | null>(null);
  const [scalability, setScalability] = useState<ScalabilityStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [gateResult, setGateResult] = useState<BlockedInferenceResponse | null>(null);
  const [gateError, setGateError] = useState(false);
  const [gateTesting, setGateTesting] = useState(false);

  const loadReadiness = useCallback(async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const [readinessRes, scalabilityRes] = await Promise.allSettled([
        getArchitectureReadiness(),
        getScalabilityStatus(),
      ]);
      if (readinessRes.status === 'fulfilled') {
        setReadiness(readinessRes.value);
      } else {
        setReadiness(null);
        setLoadError(true);
      }
      if (scalabilityRes.status === 'fulfilled') {
        setScalability(scalabilityRes.value);
      }
    } catch {
      setReadiness(null);
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadReadiness(); }, [loadReadiness]);

  const testReadinessGate = async () => {
    setGateTesting(true);
    setGateError(false);
    setGateResult(null);
    try {
      const result = await requestFourClassInference() as Partial<BlockedInferenceResponse>;
      if (result.error_code && result.message && result.status) {
        setGateResult({
          status: result.status,
          error_code: result.error_code,
          message: result.message,
          target_classes: result.target_classes ?? [],
          blockers: result.blockers ?? [],
          mandatory_safeguard: result.mandatory_safeguard ?? 'Track D inference is controlled by the backend readiness gate.',
        });
      } else {
        setGateError(true);
      }
    } catch {
      setGateError(true);
    } finally {
      setGateTesting(false);
    }
  };

  const componentBadge = (component: ArchitectureComponentStatus) => {
    const status = component.status;
    const color = status === 'IMPLEMENTED' ? 'border-teal-800 bg-teal-950/50 text-teal-200'
      : status === 'PARTIALLY_IMPLEMENTED' ? 'border-cyan-900 bg-cyan-950/50 text-cyan-200'
        : status === 'ARCHITECTURALLY_SPECIFIED' ? 'border-amber-800 bg-amber-950/40 text-amber-200'
          : 'border-rose-900 bg-rose-950/40 text-rose-200';
    return <span className={`inline-flex rounded-full border px-2.5 py-1 text-[10px] font-bold ${color}`}>{statusLabel(status)}</span>;
  };

  return (
    <article className="mx-auto max-w-6xl space-y-10 py-5 sm:py-9" aria-labelledby="architecture-title">
      <header className="max-w-3xl border-b border-slate-800 pb-6">
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-900 bg-cyan-950/40 px-3 py-1.5 text-xs font-semibold text-cyan-200"><Boxes className="h-4 w-4" aria-hidden="true" /> Current application architecture</div>
        <h1 id="architecture-title" className="mt-5 text-3xl font-black tracking-tight text-white sm:text-4xl">A layered research platform with explicit data and readiness gates</h1>
        <p className="mt-4 text-sm leading-6 text-slate-300 sm:text-base">This view describes the modules and API boundary present in this repository. The proposal’s future multimodal diagnostic model is not presented as an implemented or clinically ready system.</p>
      </header>

      <section aria-labelledby="request-path-title" className="space-y-4">
        <div><p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Runtime request path</p><h2 id="request-path-title" className="mt-1 text-2xl font-bold text-white">Web client → gateway → research API</h2></div>
        <ol className="grid gap-3 lg:grid-cols-3">
          {stackLayers.map(({ title, technology, icon: Icon, description, status }, index) => <React.Fragment key={title}>
            <li className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 sm:p-6">
              <div className="flex items-start justify-between gap-3"><span className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-900 bg-cyan-950/50 text-cyan-300"><Icon className="h-5 w-5" aria-hidden="true" /></span><span className="rounded-full border border-slate-700 px-2.5 py-1 text-[10px] text-slate-400">{status}</span></div>
              <h3 className="mt-5 text-lg font-bold text-white">{title}</h3><p className="mt-1 font-mono text-xs text-cyan-300">{technology}</p><p className="mt-3 text-sm leading-6 text-slate-400">{description}</p>
              {index < stackLayers.length - 1 && <div className="mt-5 hidden items-center gap-2 text-xs text-slate-500 lg:flex"><span className="h-px flex-1 bg-slate-700" /><ChevronRight className="h-4 w-4 text-cyan-500" aria-hidden="true" /></div>}
              {index < stackLayers.length - 1 && <div className="mt-4 flex items-center justify-center text-slate-600 lg:hidden"><ArrowDown className="h-4 w-4" aria-hidden="true" /></div>}
            </li>
          </React.Fragment>)}
        </ol>
        <p className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-xs leading-5 text-slate-400"><strong className="text-slate-200">API boundary:</strong> browser requests use supported gateway routes; FastAPI is not the public web entry point. Deployment configuration keeps the ML service private behind Express.</p>
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.05fr_.95fr]">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/65 p-5 sm:p-6">
          <p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Data governance path</p><h2 className="mt-2 text-xl font-bold text-white">From candidate source to prepared package</h2>
          <ol className="mt-5 space-y-3">{dataControls.map(([stage, description]) => <li key={stage} className="flex gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-4"><span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-teal-950/60 text-teal-300"><Check className="h-4 w-4" aria-hidden="true" /></span><div><p className="text-xs font-bold uppercase tracking-wide text-teal-300">{stage}</p><p className="mt-1 text-sm leading-5 text-slate-300">{description}</p></div></li>)}</ol>
          <p className="mt-4 text-xs leading-5 text-slate-500">Registration, qualification, integrity verification, preparation, and clinical validity are separate states. Staging does not trigger training.</p>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/65 p-5 sm:p-6">
          <p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Research modules</p><h2 className="mt-2 text-xl font-bold text-white">Methods available in the prototype</h2>
          <ul className="mt-5 space-y-3 text-sm">{[
            ['Radiomics', '/radiomics'], ['Biomechanics and Mirels-related calculations', '/biomechanics'], ['Explainability demonstrations', '/explainability'], ['Uncertainty and abstention methods', '/uncertainty'], ['Evidence registry', '/evidence'],
          ].map(([label, path]) => <li key={path}><Link to={path} className="flex min-h-10 items-center justify-between gap-3 rounded-lg px-3 py-2 text-slate-300 transition hover:bg-slate-800 hover:text-cyan-200 focus:outline-none focus:ring-2 focus:ring-cyan-400"><span>{label}</span><ArrowRight className="h-4 w-4 shrink-0 text-cyan-500" aria-hidden="true" /></Link></li>)}</ul>
          <p className="mt-4 text-xs leading-5 text-slate-500">A module being implemented does not imply that it is validated on clinical Track D data.</p>
        </div>
      </section>

      <section aria-labelledby="track-d-gate-title" className="rounded-2xl border border-rose-900/70 bg-rose-950/20 p-5 sm:p-7">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex gap-4"><LockKeyhole className="mt-1 h-6 w-6 shrink-0 text-rose-300" aria-hidden="true" /><div><h2 id="track-d-gate-title" className="text-xl font-bold text-white">Track D diagnosis remains gated</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">The four-class model is not clinically ready. The backend blocks diagnostic inference with HTTP 503 <code className="rounded bg-slate-950 px-1.5 py-0.5 text-rose-200">FOUR_CLASS_MODEL_NOT_READY</code> until genuine data, governance, training, validation, and readiness requirements are met.</p></div></div>
          <button type="button" onClick={testReadinessGate} disabled={gateTesting} className="inline-flex min-h-11 shrink-0 items-center justify-center gap-2 rounded-lg border border-rose-700 bg-rose-950/50 px-4 py-2 text-sm font-semibold text-rose-100 transition hover:bg-rose-900/50 disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-rose-300">{gateTesting ? <RotateCw className="h-4 w-4 animate-spin" aria-hidden="true" /> : <ShieldAlert className="h-4 w-4" aria-hidden="true" />}{gateTesting ? 'Checking gate…' : 'Verify backend gate'}</button>
        </div>
        {gateError && <p role="alert" className="mt-4 rounded-lg border border-amber-800 bg-amber-950/40 p-3 text-sm text-amber-200">The gate response could not be retrieved. This is a connection result, not a readiness result.</p>}
        {gateResult && <div role="status" aria-live="polite" className="mt-4 rounded-xl border border-rose-800 bg-slate-950/70 p-4"><div className="flex flex-wrap items-center gap-2"><span className="text-sm font-bold text-rose-200">Backend response: {gateResult.status}</span><span className="rounded border border-rose-900 px-2 py-0.5 font-mono text-[10px] text-rose-300">{gateResult.error_code}</span></div><p className="mt-2 text-sm text-slate-300">{gateResult.message}</p>{gateResult.blockers.length > 0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-slate-400">{gateResult.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul>}</div>}
      </section>

      <section aria-labelledby="readiness-title" className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Live API response</p><h2 id="readiness-title" className="mt-1 text-2xl font-bold text-white">Engineering component readiness</h2><p className="mt-2 text-sm text-slate-400">This status describes implementation and unresolved research gaps; it is not clinical validation.</p></div><button type="button" onClick={() => void loadReadiness()} disabled={loading} className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-800 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-cyan-400"><RotateCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />Refresh status</button></div>
        {loading && <div role="status" aria-live="polite" className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-sm text-slate-400">Loading current architecture readiness…</div>}
        {loadError && <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-rose-900 bg-rose-950/25 p-4"><p className="text-sm text-rose-200">Current readiness is unavailable. The architecture diagram above shows the configured application layers, not live service health.</p><button type="button" onClick={() => void loadReadiness()} className="min-h-10 rounded-lg border border-rose-800 px-3 py-2 text-xs font-semibold text-rose-100 hover:bg-rose-950 focus:outline-none focus:ring-2 focus:ring-rose-300">Retry</button></div>}
        {!loading && !loadError && readiness && <>
          <div className="grid gap-3 sm:grid-cols-3"><div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4"><p className="text-[10px] font-bold uppercase tracking-wide text-slate-500">System status</p><p className="mt-2 break-words text-sm font-semibold text-slate-200">{readiness.system_readiness.replace(/_/g, ' ')}</p></div><div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4"><p className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Fusion contract</p><p className="mt-2 text-sm text-slate-200">{readiness.fusion_contract_status}</p></div><div className="rounded-xl border border-rose-900/70 bg-rose-950/15 p-4"><p className="text-[10px] font-bold uppercase tracking-wide text-rose-300">Inference eligibility</p><p className="mt-2 text-sm text-rose-100">{readiness.inference_eligibility}</p></div></div>
          {readiness.components.length === 0
            ? <div role="status" className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-sm text-slate-400">The readiness service returned no component records. Refresh to check again; no component status is being inferred.</div>
            : <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50"><div className="overflow-x-auto"><table className="w-full min-w-[850px] text-left text-xs"><thead className="bg-slate-950/80 text-[10px] uppercase tracking-wide text-slate-400"><tr><th scope="col" className="px-4 py-3">Component</th><th scope="col" className="px-4 py-3">Layer</th><th scope="col" className="px-4 py-3">Implementation</th><th scope="col" className="px-4 py-3">Known gap and dependency</th></tr></thead><tbody className="divide-y divide-slate-800">{readiness.components.map((component) => <tr key={`${component.component_name}-${component.layer_group}`} className="align-top"><th scope="row" className="max-w-64 px-4 py-4 font-semibold text-slate-100">{component.component_name}</th><td className="px-4 py-4 text-slate-400">{component.layer_group}</td><td className="px-4 py-4"><div>{componentBadge(component)}</div><p className="mt-2 max-w-sm leading-5 text-slate-300">{component.implemented_features}</p></td><td className="max-w-md px-4 py-4 leading-5"><p className="text-amber-200/90">{component.unresolved_gaps}</p><p className="mt-2 text-slate-500">Dependency: {component.governance_dependency}</p></td></tr>)}</tbody></table></div></div>}
          <p className="text-xs leading-5 text-slate-500">Status as of {new Date(readiness.timestamp).toLocaleString()}. {readiness.disclaimer}</p>
        </>}
      </section>

      {scalability && (
        <section aria-labelledby="scalability-title" className="rounded-2xl border border-cyan-900/60 bg-slate-900/70 p-5 sm:p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <Activity className="h-5 w-5 text-cyan-400" aria-hidden="true" />
              <div>
                <p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-400">Clinical gateway scalability</p>
                <h2 id="scalability-title" className="text-lg font-bold text-white">Worker Pool, Rate Limiting &amp; Latency Telemetry</h2>
              </div>
            </div>
            <span className="rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1 font-mono text-[11px] text-slate-300">
              Prometheus: GET /api/metrics
            </span>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <p className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Worker Concurrency</p>
              <p className="mt-1.5 font-mono text-lg font-bold text-white">
                {scalability.workerPool.activeWorkers} / {scalability.workerPool.maxConcurrency} active
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Queue depth: {scalability.workerPool.queuedDepth} / {scalability.workerPool.maxQueueSize}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <p className="text-[10px] font-bold uppercase tracking-wide text-slate-400">API Latency (p50 / p95)</p>
              <p className="mt-1.5 font-mono text-lg font-bold text-cyan-300">
                {scalability.latency.p50Ms}ms / {scalability.latency.p95Ms}ms
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">
                p99: {scalability.latency.p99Ms}ms ({scalability.latency.totalRequests} reqs)
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <p className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Ingress Rate Guard</p>
              <p className="mt-1.5 font-mono text-lg font-bold text-emerald-300">
                {scalability.rateLimiter.maxRequestsPerMinute} RPM
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Active windows: {scalability.rateLimiter.trackedClients}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <p className="text-[10px] font-bold uppercase tracking-wide text-slate-400">RAM Asset Pool (AES-256-GCM)</p>
              <p className="mt-1.5 font-mono text-lg font-bold text-white">
                {scalability.ephemeralStore.activeEntries} / {scalability.ephemeralStore.maxEntries}
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">
                TTL eviction: {scalability.ephemeralStore.ttlSeconds}s
              </p>
            </div>
          </div>
        </section>
      )}

      <nav aria-label="Architecture links" className="flex flex-wrap gap-3 border-t border-slate-800 pt-6"><Link to="/governance" className="inline-flex min-h-11 items-center gap-2 rounded-lg bg-cyan-700 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-300">Dataset governance <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link><Link to="/about" className="inline-flex min-h-11 items-center rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-300">About the project</Link></nav>
    </article>
  );
};

export default ArchitecturePage;
