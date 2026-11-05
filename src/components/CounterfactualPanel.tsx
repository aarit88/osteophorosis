import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Counterfactual, ClassProbabilities, BonePathologyClass } from '../api/types';
import { Sparkles, Sliders, ArrowRight, HelpCircle, Activity } from 'lucide-react';
import { logTelemetryEvent } from '../api/client';

interface CounterfactualPanelProps {
  counterfactuals: Counterfactual[];
  baseProbabilities: ClassProbabilities;
  topClass: BonePathologyClass;
  requestId: string;
}

export const CounterfactualPanel: React.FC<CounterfactualPanelProps> = ({
  counterfactuals,
  baseProbabilities,
  topClass,
  requestId,
}) => {
  const [selectedScenarioIdx, setSelectedScenarioIdx] = useState<number>(0);
  const [scrubberPercent, setScrubberPercent] = useState<number>(100); // 0% to 200% of suggested perturbation

  const activeScenario = counterfactuals[selectedScenarioIdx] || counterfactuals[0];

  // Client-side additive sensitivity simulation
  // Perturbation factor: 0.0 to 2.0x of the suggested shift
  const factor = scrubberPercent / 100;
  const appliedShift = activeScenario ? activeScenario.probabilityShift * factor : 0;

  // Approximate recalculated probability distribution
  const targetClass = activeScenario?.flipTo || 'healthy';
  const simulatedTargetProb = Math.min(Math.max((baseProbabilities[targetClass] || 0) + appliedShift, 0.02), 0.95);
  const remaining = 1.0 - simulatedTargetProb;

  const otherKeys = (Object.keys(baseProbabilities) as BonePathologyClass[]).filter((k) => k !== targetClass);
  const sumOtherBase = otherKeys.reduce((acc, k) => acc + (baseProbabilities[k] || 0.1), 0);

  const recomputedProbabilities: Record<BonePathologyClass, number> = {
    metastasis: 0,
    paget: 0,
    oi: 0,
    healthy: 0,
  };
  recomputedProbabilities[targetClass] = parseFloat(simulatedTargetProb.toFixed(2));

  otherKeys.forEach((k) => {
    const fraction = (baseProbabilities[k] || 0.1) / sumOtherBase;
    recomputedProbabilities[k] = parseFloat((fraction * remaining).toFixed(2));
  });

  const handleScrubberChange = (val: number) => {
    setScrubberPercent(val);
    logTelemetryEvent({
      type: 'counterfactual_scrubbed',
      requestId,
      metadata: { scenario: activeScenario?.feature, value: val },
    });
  };

  return (
    <div className="p-6 rounded-3xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-sm font-bold text-white">
              What Would Change the Diagnosis? (Counterfactual Sensitivity)
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Minimum feature boundary perturbations required to flip classification
            </p>
          </div>
        </div>

        <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">
          Illustrative Local Linear Model
        </span>
      </div>

      {/* Counterfactual Scenario Selector Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
        {counterfactuals.map((cf, idx) => {
          const isSelected = idx === selectedScenarioIdx;
          return (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setSelectedScenarioIdx(idx);
                setScrubberPercent(100);
              }}
              className={`p-3 rounded-2xl border text-left transition ${
                isSelected
                  ? 'bg-cyan-950/40 border-cyan-500/60 shadow-md shadow-cyan-950/50'
                  : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-bold text-white mb-1">
                <span className="truncate">{cf.feature}</span>
                <span className="text-cyan-400 font-mono text-[11px]">
                  {cf.deltaDirection === 'increase' ? `+${cf.percentChange}%` : `-${cf.percentChange}%`}
                </span>
              </div>
              <div className="text-[11px] text-slate-400 flex items-center space-x-1">
                <span>Flips to</span>
                <span className="font-semibold text-cyan-300 uppercase">{cf.flipTo}</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Interactive Sensitivity Scrubber */}
      {activeScenario && (
        <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <span className="text-xs font-semibold text-slate-300">
                Interactive Perturbation Multiplier:
              </span>
              <p className="text-[11px] text-slate-400">
                Simulating {activeScenario.deltaDirection === 'increase' ? 'elevation' : 'reduction'} of{' '}
                <strong className="text-white">{activeScenario.feature}</strong> by{' '}
                <strong className="text-cyan-300">
                  {Math.round(activeScenario.percentChange * (scrubberPercent / 100))}%
                </strong>
              </p>
            </div>

            <div className="font-mono text-sm font-bold text-cyan-400 self-end sm:self-auto">
              {(scrubberPercent / 100).toFixed(1)}x intensity
            </div>
          </div>

          <div className="space-y-1">
            <input
              type="range"
              min="0"
              max="200"
              step="5"
              value={scrubberPercent}
              onChange={(e) => handleScrubberChange(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
              aria-label="Counterfactual perturbation intensity slider"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0% (Baseline)</span>
              <span className="text-cyan-400">100% (Decision Boundary)</span>
              <span>200% (Strong Shift)</span>
            </div>
          </div>

          {/* Recomputed Probabilities Comparison Bar */}
          <div className="pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-semibold text-slate-300">Simulated Diagnostic Response:</span>
              <span className="text-[11px] font-mono text-cyan-300">
                Δ {activeScenario.flipTo}: +{(appliedShift * 100).toFixed(0)}%
              </span>
            </div>

            <div className="grid grid-cols-4 gap-2 text-center text-xs font-mono">
              {(['metastasis', 'paget', 'oi', 'healthy'] as BonePathologyClass[]).map((cls) => {
                const prob = recomputedProbabilities[cls];
                const isTarget = cls === activeScenario.flipTo;
                const isOriginal = cls === topClass;

                return (
                  <div
                    key={cls}
                    className={`p-2 rounded-xl border ${
                      isTarget
                        ? 'bg-cyan-500/15 border-cyan-500/50 text-cyan-200'
                        : isOriginal
                        ? 'bg-slate-900 border-slate-700 text-slate-300'
                        : 'bg-slate-900/50 border-slate-800 text-slate-400'
                    }`}
                  >
                    <span className="text-[10px] uppercase block truncate">{cls}</span>
                    <span className="text-sm font-bold block mt-0.5">{Math.round(prob * 100)}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
