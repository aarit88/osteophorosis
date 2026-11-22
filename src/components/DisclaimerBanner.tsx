import React from 'react';
import { AlertTriangle } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div
      role="alert"
      className="w-full bg-rose-600 text-white px-3 sm:px-4 py-1.5 sm:py-2 text-center text-[11px] sm:text-sm font-bold tracking-wide shadow-md flex items-center justify-center space-x-2 select-none border-b border-rose-700"
    >
      <AlertTriangle className="w-4 h-4 shrink-0 animate-pulse" />
      <span>⚠️ RESEARCH PROTOTYPE — SIMULATED OUTPUT — NOT FOR CLINICAL USE</span>
      <span className="hidden md:inline text-rose-200 text-xs font-normal">
        (Investigational validation & engineering demo only)
      </span>
    </div>
  );
};
