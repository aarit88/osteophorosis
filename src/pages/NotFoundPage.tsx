import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ArrowLeft, Home, SearchX } from 'lucide-react';

export const NotFoundPage: React.FC = () => {
  const location = useLocation();

  return (
    <section className="mx-auto flex min-h-[52vh] max-w-2xl flex-col items-center justify-center px-4 py-12 text-center" aria-labelledby="not-found-title">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-amber-800/60 bg-amber-950/40 text-amber-300">
        <SearchX className="h-7 w-7" aria-hidden="true" />
      </div>
      <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-400">Page not found</p>
      <h1 id="not-found-title" className="mt-2 text-3xl font-bold text-white">This route is not part of OsteoGuard</h1>
      <p className="mt-3 max-w-lg text-sm leading-6 text-slate-400">
        The requested page <code className="break-all rounded bg-slate-900 px-1.5 py-0.5 text-slate-300">{location.pathname}</code> does not exist or may have moved. No analysis or governance data was changed.
      </p>
      <div className="mt-7 flex flex-wrap justify-center gap-3">
        <Link to="/" className="inline-flex min-h-11 items-center gap-2 rounded-lg bg-cyan-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-400">
          <Home className="h-4 w-4" aria-hidden="true" /> Go to analysis
        </Link>
        <button type="button" onClick={() => window.history.back()} className="inline-flex min-h-11 items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-400">
          <ArrowLeft className="h-4 w-4" aria-hidden="true" /> Go back
        </button>
      </div>
    </section>
  );
};

export default NotFoundPage;
