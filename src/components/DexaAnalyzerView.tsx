import React, { useState } from 'react';
import { DexaAnalysis, DexaSite } from '../types/osteoporosis';
import { getWhoClassification } from '../utils/fraxCalculator';
import { Activity, AlertTriangle, Layers, TrendingDown, TrendingUp, Info } from 'lucide-react';

interface DexaAnalyzerViewProps {
  dexaData: DexaAnalysis;
  onUpdateDexa: (updated: DexaAnalysis) => void;
  onSyncWithFrax: (tScore: number) => void;
}

export const DexaAnalyzerView: React.FC<DexaAnalyzerViewProps> = ({
  dexaData,
  onUpdateDexa,
  onSyncWithFrax,
}) => {
  const [selectedSiteIndex, setSelectedSiteIndex] = useState<number>(0);
  const selectedSite = dexaData.sites[selectedSiteIndex] || dexaData.sites[0];

  const updateSiteTScore = (index: number, newTScore: number) => {
    const updatedSites = [...dexaData.sites];
    const currentSite = updatedSites[index];
    const newBmd = parseFloat((currentSite.referenceMean + newTScore * currentSite.referenceSd).toFixed(3));
    // Approximate zScore assuming age drop ~1.0 SD
    const newZScore = parseFloat((newTScore + 1.2).toFixed(1));

    updatedSites[index] = {
      ...currentSite,
      tScore: parseFloat(newTScore.toFixed(1)),
      bmd: newBmd,
      zScore: newZScore,
    };

    // Calculate lowest T-score
    let lowest = updatedSites[0].tScore;
    let lowestName = updatedSites[0].siteName;
    updatedSites.forEach((s) => {
      if (s.tScore < lowest) {
        lowest = s.tScore;
        lowestName = s.siteName;
      }
    });

    const whoClass = getWhoClassification(lowest, dexaData.fragilityFracturePresent);

    onUpdateDexa({
      ...dexaData,
      sites: updatedSites,
      lowestTScore: lowest,
      lowestSite: lowestName,
      whoCategory: whoClass.category,
    });
  };

  const toggleFragilityFracture = () => {
    const nextFragility = !dexaData.fragilityFracturePresent;
    const whoClass = getWhoClassification(dexaData.lowestTScore, nextFragility);
    onUpdateDexa({
      ...dexaData,
      fragilityFracturePresent: nextFragility,
      whoCategory: whoClass.category,
    });
  };

  const whoClassInfo = getWhoClassification(dexaData.lowestTScore, dexaData.fragilityFracturePresent);

  // Generate SVG bell curve points
  // X: -4.0 to +3.0 -> mapped to 20 to 380
  // Mean = 0 -> X = 200
  const getCurvePoint = (sd: number) => {
    const x = 200 + sd * 45;
    // Standard normal distribution: exp(-0.5 * sd^2)
    const y = 140 - Math.exp(-0.5 * sd * sd) * 110;
    return { x, y };
  };

  return (
    <div className="space-y-6">
      {/* Top Summary Banner */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* WHO Classification Card */}
        <div className="lg:col-span-2 rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl relative overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <div className="flex items-center space-x-2">
                <Activity className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-bold text-white">Dual-Energy X-Ray Absorptiometry (DXA) Analysis</h2>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">ISCD & WHO criteria applied to central skeletal sites</p>
            </div>

            <span className={`px-3 py-1 rounded-full text-xs font-bold border tracking-wide ${whoClassInfo.badgeClass}`}>
              {dexaData.whoCategory.toUpperCase()}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 mb-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <span className="text-xs text-slate-400">Diagnostic Skeletal Site (Lowest T-Score):</span>
                <div className="text-lg font-bold text-white flex items-center space-x-2 mt-0.5">
                  <span>{dexaData.lowestSite}</span>
                  <span className="font-mono text-cyan-400">({dexaData.lowestTScore > 0 ? `+${dexaData.lowestTScore}` : dexaData.lowestTScore} SD)</span>
                </div>
              </div>

              {/* Fragility Fracture Checkbox */}
              <div className="flex items-center space-x-3 bg-slate-900/90 px-3 py-2 rounded-lg border border-slate-700/60">
                <input
                  type="checkbox"
                  id="dexaFractureToggle"
                  checked={dexaData.fragilityFracturePresent}
                  onChange={toggleFragilityFracture}
                  className="w-4 h-4 rounded text-rose-500 bg-slate-950 border-slate-700 accent-rose-500 cursor-pointer"
                />
                <label htmlFor="dexaFractureToggle" className="cursor-pointer text-xs">
                  <span className="font-semibold text-slate-200 block">Fragility Fracture Present</span>
                  <span className="text-[10px] text-slate-400 block">Upgrades T &le; -2.5 to "Severe Osteoporosis"</span>
                </label>
              </div>
            </div>

            <p className="text-xs text-slate-300 mt-3 pt-3 border-t border-slate-800/80">
              {whoClassInfo.description}
            </p>
          </div>

          {/* Site Selector Tabs */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {dexaData.sites.map((site, idx) => {
              const isSelected = idx === selectedSiteIndex;
              const isLowest = site.tScore === dexaData.lowestTScore;
              return (
                <button
                  key={site.siteName}
                  onClick={() => setSelectedSiteIndex(idx)}
                  className={`p-3 rounded-xl border text-left transition relative ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/60 shadow-md shadow-cyan-950/50'
                      : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  {isLowest && (
                    <span className="absolute -top-2 right-2 text-[9px] bg-rose-500 text-white font-bold px-1.5 py-0.2 rounded-full">
                      LOWEST
                    </span>
                  )}
                  <span className="text-xs font-semibold text-slate-300 block truncate">{site.siteName}</span>
                  <div className="flex items-baseline space-x-1.5 mt-1">
                    <span className={`text-base font-bold font-mono ${
                      site.tScore <= -2.5 ? 'text-rose-400' : site.tScore <= -1.0 ? 'text-amber-400' : 'text-emerald-400'
                    }`}>
                      {site.tScore > 0 ? `+${site.tScore}` : site.tScore}
                    </span>
                    <span className="text-[10px] text-slate-500">T-score</span>
                  </div>
                  <span className="text-[11px] text-slate-400 block font-mono mt-0.5">{site.bmd.toFixed(3)} g/cm²</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Quick Sync & Diagnostic Criteria Guide */}
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <Layers className="w-5 h-5 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">WHO Diagnostic Criteria</h3>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40">
                <div className="font-bold text-emerald-400">Normal</div>
                <div className="text-slate-300 text-[11px]">T-score &ge; -1.0 SD</div>
              </div>

              <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-800/40">
                <div className="font-bold text-amber-400">Osteopenia (Low Bone Mass)</div>
                <div className="text-slate-300 text-[11px]">-1.0 SD &gt; T-score &gt; -2.5 SD</div>
              </div>

              <div className="p-2.5 rounded-lg bg-orange-950/30 border border-orange-800/40">
                <div className="font-bold text-orange-400">Osteoporosis</div>
                <div className="text-slate-300 text-[11px]">T-score &le; -2.5 SD at spine, hip, or radius</div>
              </div>

              <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-800/40">
                <div className="font-bold text-rose-400">Severe (Established) Osteoporosis</div>
                <div className="text-slate-300 text-[11px]">T-score &le; -2.5 SD with &ge;1 fragility fracture</div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800">
            <button
              onClick={() => onSyncWithFrax(dexaData.lowestTScore)}
              className="w-full py-2 px-3 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs shadow-md transition flex items-center justify-center space-x-1.5"
            >
              <Activity className="w-4 h-4" />
              <span>Sync {dexaData.lowestSite} T-Score into FRAX</span>
            </button>
          </div>
        </div>
      </div>

      {/* Interactive Bell Curve & Site Tuning */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-base font-bold text-white">
              T-Score Distribution & Bone Density Curve: {selectedSite.siteName}
            </h3>
            <p className="text-xs text-slate-400">
              Young Adult Peak Reference Mean (30-year-old cohort, SD = {selectedSite.referenceSd})
            </p>
          </div>

          <div className="flex items-center space-x-4">
            <div className="text-right">
              <span className="text-[11px] text-slate-400 block">Z-Score (Age-Matched):</span>
              <span className="font-mono text-sm font-bold text-cyan-300">
                {selectedSite.zScore > 0 ? `+${selectedSite.zScore}` : selectedSite.zScore} SD
              </span>
            </div>
            <div className="text-right">
              <span className="text-[11px] text-slate-400 block">BMD Absolute:</span>
              <span className="font-mono text-sm font-bold text-white">{selectedSite.bmd.toFixed(3)} g/cm²</span>
            </div>
          </div>
        </div>

        {/* SVG Normal Distribution Chart */}
        <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 mb-6">
          <svg viewBox="0 0 400 160" className="w-full h-auto">
            {/* Zones shading */}
            {/* Severe Osteoporosis zone (-4.0 to -2.5) */}
            <rect x="20" y="20" width="67.5" height="120" fill="rgba(244, 63, 94, 0.12)" />
            {/* Osteopenia zone (-2.5 to -1.0) */}
            <rect x="87.5" y="20" width="67.5" height="120" fill="rgba(245, 158, 11, 0.10)" />
            {/* Normal zone (-1.0 to +3.0) */}
            <rect x="155" y="20" width="225" height="120" fill="rgba(16, 185, 129, 0.08)" />

            {/* Threshold dividing lines */}
            <line x1="87.5" y1="20" x2="87.5" y2="140" stroke="#f43f5e" strokeDasharray="3,3" strokeWidth="1" />
            <line x1="155" y1="20" x2="155" y2="140" stroke="#f59e0b" strokeDasharray="3,3" strokeWidth="1" />
            <line x1="200" y1="20" x2="200" y2="140" stroke="#475569" strokeWidth="1" />

            {/* Curve path */}
            {(() => {
              const points = [];
              for (let sd = -3.8; sd <= 3.8; sd += 0.2) {
                const pt = getCurvePoint(sd);
                points.push(`${pt.x.toFixed(1)},${pt.y.toFixed(1)}`);
              }
              return (
                <path
                  d={`M ${points.join(' L ')}`}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="2.5"
                />
              );
            })()}

            {/* Base axis */}
            <line x1="20" y1="140" x2="380" y2="140" stroke="#334155" strokeWidth="1.5" />

            {/* Axis labels */}
            <text x="87.5" y="152" fill="#f43f5e" fontSize="9" textAnchor="middle">-2.5 SD (Osteoporosis)</text>
            <text x="155" y="152" fill="#f59e0b" fontSize="9" textAnchor="middle">-1.0 SD (Osteopenia)</text>
            <text x="200" y="152" fill="#94a3b8" fontSize="9" textAnchor="middle">0 (Peak Mean)</text>
            <text x="290" y="152" fill="#10b981" fontSize="9" textAnchor="middle">+2.0 SD (Normal)</text>

            {/* Patient pointer on curve */}
            {(() => {
              const clampedT = Math.min(Math.max(selectedSite.tScore, -3.8), 3.8);
              const pt = getCurvePoint(clampedT);
              return (
                <g>
                  {/* Vertical drop line to axis */}
                  <line x1={pt.x} y1={pt.y} x2={pt.x} y2="140" stroke="#ffffff" strokeWidth="1.5" strokeDasharray="2,2" />
                  <circle cx={pt.x} cy={pt.y} r="6" fill="#38bdf8" stroke="#ffffff" strokeWidth="2" />
                  <rect x={pt.x - 38} y="15" width="76" height="20" rx="4" fill="#0f172a" stroke="#38bdf8" strokeWidth="1" />
                  <text x={pt.x} y="29" fill="#ffffff" fontSize="9" fontWeight="bold" textAnchor="middle">
                    T = {selectedSite.tScore > 0 ? `+${selectedSite.tScore}` : selectedSite.tScore}
                  </text>
                </g>
              );
            })()}
          </svg>
        </div>

        {/* T-Score Slider for Selected Site */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <label className="text-xs font-semibold text-slate-300">
              Adjust T-Score for {selectedSite.siteName}:
            </label>
            <span className="font-mono text-base font-bold text-cyan-400">
              {selectedSite.tScore > 0 ? `+${selectedSite.tScore}` : selectedSite.tScore} SD
            </span>
          </div>

          <input
            type="range"
            min="-4.5"
            max="1.5"
            step="0.1"
            value={selectedSite.tScore}
            onChange={(e) => updateSiteTScore(selectedSiteIndex, parseFloat(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer"
          />

          <div className="flex justify-between text-xs text-slate-400">
            <span className="text-rose-400">-4.5 (Severe Loss)</span>
            <span className="text-amber-400">-2.5</span>
            <span className="text-yellow-400">-1.0</span>
            <span className="text-emerald-400">+1.5 (High BMD)</span>
          </div>
        </div>

        {/* Longitudinal Monitoring & Least Significant Change (LSC) */}
        <div className="mt-8 pt-6 border-t border-slate-800">
          <div className="flex items-center space-x-2 mb-3">
            <TrendingDown className="w-4 h-4 text-cyan-400" />
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Longitudinal DXA Precision & Least Significant Change (LSC)
            </h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Baseline Scan (2 Years Ago)</span>
              <span className="text-sm font-mono font-bold text-white mt-1 block">0.785 g/cm²</span>
              <span className="text-[10px] text-slate-500">T-Score: -2.3 SD</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Current Follow-up Scan</span>
              <span className="text-sm font-mono font-bold text-white mt-1 block">{selectedSite.bmd.toFixed(3)} g/cm²</span>
              <span className="text-[10px] text-slate-500">T-Score: {selectedSite.tScore} SD</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-[11px] text-slate-400 block">Change vs. LSC Threshold (3.5%)</span>
                {(() => {
                  const baselineBmd = 0.785;
                  const deltaPercent = ((selectedSite.bmd - baselineBmd) / baselineBmd) * 100;
                  const isSignificant = Math.abs(deltaPercent) >= 3.5;
                  return (
                    <div className="flex items-center space-x-1.5 mt-1">
                      {deltaPercent >= 0 ? (
                        <TrendingUp className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-rose-400" />
                      )}
                      <span className={`text-sm font-mono font-bold ${
                        deltaPercent >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}>
                        {deltaPercent > 0 ? `+${deltaPercent.toFixed(1)}%` : `${deltaPercent.toFixed(1)}%`}
                      </span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                        isSignificant ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-800 text-slate-400'
                      }`}>
                        {isSignificant ? 'Statistically Real' : 'Within Margin'}
                      </span>
                    </div>
                  );
                })()}
              </div>
              <span className="text-[10px] text-slate-500 mt-1">ISCD 95% confidence precision limit</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
