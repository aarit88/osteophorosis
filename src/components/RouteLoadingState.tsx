import React from 'react';

export const RouteLoadingState: React.FC = () => (
  <div className="mx-auto max-w-4xl py-8 sm:py-12" role="status" aria-live="polite" aria-busy="true">
    <span className="sr-only">Loading page</span>
    <div className="mb-5 h-4 w-28 animate-pulse rounded bg-slate-800" />
    <div className="mb-8 h-9 w-2/3 animate-pulse rounded bg-slate-800" />
    <div className="grid gap-4 md:grid-cols-2">
      <div className="h-36 animate-pulse rounded-2xl border border-slate-800 bg-slate-900/70" />
      <div className="h-36 animate-pulse rounded-2xl border border-slate-800 bg-slate-900/70" />
    </div>
    <div className="mt-4 h-52 animate-pulse rounded-2xl border border-slate-800 bg-slate-900/70" />
  </div>
);

export default RouteLoadingState;
