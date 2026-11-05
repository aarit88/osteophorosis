import React from 'react';
import { AlertTriangle, ShieldCheck, UserCheck } from 'lucide-react';

interface AbstentionBannerProps {
  entropy: number;
  oodFlag: boolean;
}

export const AbstentionBanner: React.FC<AbstentionBannerProps> = ({ entropy, oodFlag }) => {
  return (
    <div
      role="alert"
      className="p-4 sm:p-5 rounded-2xl bg-gradient-to-r from-amber-950/80 to-amber-900/60 border-2 border-amber-500/80 shadow-2xl text-amber-100 flex items-start space-x-3 sm:space-x-4 animate-pulse"
    >
      <div className="w-10 h-10 rounded-xl bg-amber-500/30 text-amber-300 flex items-center justify-center shrink-0 border border-amber-500/60 mt-0.5">
        <AlertTriangle className="w-6 h-6" />
      </div>

      <div className="space-y-1 flex-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm sm:text-base font-extrabold text-amber-200">
            ⚠ Model Recommends Human Review — Primary Prediction Withheld
          </h3>
          <span className="text-[10px] uppercase font-mono font-bold px-2 py-0.5 rounded bg-amber-500/30 text-amber-200 border border-amber-400">
            Abstention Protocol Active
          </span>
        </div>

        <p className="text-xs sm:text-sm text-amber-100/90 leading-relaxed font-sans">
          The deep learning ensemble detected diagnostic uncertainty exceeding the autonomous clearance threshold
          {entropy > 0.6 ? ` (Softmax entropy: ${entropy.toFixed(3)} > 0.60)` : ''}
          {oodFlag ? ' and triggered an Out-of-Distribution (OOD) geometry flag' : ''}.
          To prevent anchoring bias and ensure patient safety, the prediction is demoted to unverified suggestion and must be verified by a board-certified radiologist.
        </p>

        <div className="pt-2 flex items-center space-x-3 text-[11px] text-amber-300 font-mono">
          <span className="flex items-center space-x-1">
            <UserCheck className="w-3.5 h-3.5" />
            <span>Refer to attending orthopedic oncology specialist</span>
          </span>
        </div>
      </div>
    </div>
  );
};
