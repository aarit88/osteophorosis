import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { LockKeyhole, RotateCw, ShieldAlert } from 'lucide-react';
import { requestFourClassInference } from '../api/client';
import type { BlockedInferenceResponse } from '../api/types';

export const FourClassGatePanel: React.FC = () => {
  const [result, setResult] = useState<BlockedInferenceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setUnavailable(false);
    try {
      setResult(await requestFourClassInference());
    } catch {
      setResult(null);
      setUnavailable(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  return (
    <section className="rounded-2xl border border-amber-900/70 bg-amber-950/20 p-5" aria-labelledby="four-class-gate-title">
      <div className="flex items-start gap-3">
        {unavailable ? <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" aria-hidden="true" /> : <LockKeyhole className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" aria-hidden="true" />}
        <div className="min-w-0 flex-1">
          <h2 id="four-class-gate-title" className="text-base font-bold text-amber-100">Track D model readiness</h2>
          {loading && <p role="status" className="mt-2 text-sm text-slate-300">Checking the backend readiness gate…</p>}
          {!loading && unavailable && <div role="alert" className="mt-2 space-y-2 text-sm text-amber-100"><p>Current model readiness could not be retrieved. No prediction or explanation is available.</p><button type="button" onClick={() => void refresh()} className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-amber-800 px-3 py-2 text-xs font-semibold hover:bg-amber-900/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"><RotateCw className="h-3.5 w-3.5" aria-hidden="true" />Retry readiness check</button></div>}
          {!loading && result && <>
            <p className="mt-2 text-sm font-semibold text-amber-100">{result.error_code}</p>
            <p className="mt-1 text-sm leading-6 text-slate-300">{result.message}</p>
            {result.blockers.length > 0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-xs leading-5 text-slate-300">{result.blockers.map((blocker, index) => <li key={`${index}-${blocker}`}>{blocker}</li>)}</ul>}
            <p className="mt-3 text-xs leading-5 text-slate-400">A ready status would still not make a method available unless a model-specific explanation or uncertainty endpoint exists.</p>
          </>}
          <div className="mt-4 flex flex-wrap gap-3 text-sm">
            <Link to="/governance" className="rounded text-cyan-200 underline underline-offset-4 hover:text-cyan-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300">Open Governance</Link>
            <Link to="/workbench" className="rounded text-cyan-200 underline underline-offset-4 hover:text-cyan-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300">Open Workbench</Link>
          </div>
        </div>
      </div>
    </section>
  );
};
