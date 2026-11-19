import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Cpu,
  Layers,
  Activity,
  ShieldAlert,
  Play,
  RotateCw,
  CheckCircle2,
  AlertTriangle,
  Lock,
  Compass,
  Eye,
  Sliders,
  ChevronRight,
} from 'lucide-react';
import { runSyntheticWorkbench } from '../api/client';
import { SyntheticWorkbenchRunResponse } from '../api/types';

export const ResearchWorkbenchPage: React.FC = () => {
  const [data, setData] = useState<SyntheticWorkbenchRunResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleRunWorkbench = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runSyntheticWorkbench({});
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to execute synthetic workbench demonstration.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-6 px-4">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-xs font-semibold mb-2">
            <Cpu className="w-3.5 h-3.5" />
            <span>KSCST Proposal Integrated Synthetic Demonstrator</span>
          </div>
          <h1 className="text-3xl font-black text-white tracking-tight">
            Synthetic Research Workbench
          </h1>
          <p className="text-slate-400 text-sm max-w-3xl mt-1">
            End-to-end integration demonstrator across Radiomics (Stage 27), Biomechanics & Mirels (Stage 28), Explainability (Stage 29), and Uncertainty (Stage 30) using deterministic synthetic phantoms.
          </p>
        </div>

        <div>
          <button
            type="button"
            onClick={handleRunWorkbench}
            disabled={loading}
            className="flex items-center space-x-2 px-5 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-teal-500 hover:from-cyan-500 hover:to-teal-400 text-white font-bold text-sm shadow-lg shadow-cyan-500/20 disabled:opacity-50 transition cursor-pointer"
          >
            {loading ? (
              <>
                <RotateCw className="w-4 h-4 animate-spin" />
                <span>Executing Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Run Full Synthetic Pipeline</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Prominent Disclaimer Notice */}
      <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/80 text-amber-200 text-xs flex items-start space-x-3">
        <ShieldAlert className="w-5 h-5 flex-shrink-0 text-amber-400 mt-0.5" />
        <div className="space-y-1">
          <strong className="text-amber-300 font-bold">Research Prototype Notice:</strong>
          <p>
            This workbench demonstrates computational feature extraction, geometric section calculations, gradient attribution, and selective prediction triage using <strong>mathematical synthetic phantoms</strong>. It strictly operates without patient data and does not provide clinical diagnostic inferences.
          </p>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Main Panels Grid */}
      {data ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-8"
        >
          {/* Status Overview Card */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <span className="text-slate-400 text-xs block">Execution Status</span>
              <span className="text-emerald-400 font-bold text-sm flex items-start space-x-1.5 mt-1.5 break-words">
                <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                <span className="min-w-0 break-words">{data.status.replace(/_/g, ' ')}</span>
              </span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <span className="text-slate-400 text-xs block">Verified Radiomic Features</span>
              <span className="text-cyan-400 font-bold text-lg mt-1 block font-mono">
                {data.radiomics_panel.verified_features_count} / {data.radiomics_panel.total_features_count}
              </span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <span className="text-slate-400 text-xs block">Mirels Score (1989)</span>
              <span className="text-amber-400 font-bold text-base mt-1 block break-words">
                Score: {data.biomechanics_mirels_panel.total_score} ({data.biomechanics_mirels_panel.recommendation_tier.replace(/_/g, ' ')})
              </span>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <span className="text-slate-400 text-xs block">Uncertainty Triage</span>
              <span className={`font-bold text-sm mt-1.5 block break-words ${
                data.uncertainty_panel.decision === 'ACCEPTED' || data.uncertainty_panel.decision === 'ACCEPT_PREDICTION'
                  ? 'text-emerald-400'
                  : data.uncertainty_panel.decision.includes('WARN')
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }`}>
                {data.uncertainty_panel.decision.replace(/_/g, ' ')}
              </span>
            </div>
          </div>

          {/* 4 Multi-Engine Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Panel 1: Radiomics Feature Engine */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Sliders className="w-5 h-5 text-cyan-400" />
                  <h3 className="font-extrabold text-white text-lg">1. Radiomics Feature Extraction</h3>
                </div>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-950 border border-cyan-800 text-cyan-300 font-mono">
                  Stage 27 Engine
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Extracted on synthetic <strong>{data.radiomics_panel.phantom_type}</strong> phantom ({data.radiomics_panel.grid_size}x{data.radiomics_panel.grid_size}).
              </p>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Verified IBSI Features</span>
                  <span className="font-mono text-cyan-300 font-bold text-base">
                    {data.radiomics_panel.verified_features_count}
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Unverified / Non-Standard</span>
                  <span className="font-mono text-amber-300 font-bold text-base">
                    {data.radiomics_panel.unverified_features_count}
                  </span>
                </div>
              </div>
              <div className="space-y-1.5">
                <span className="text-xs font-semibold text-slate-300 block">Sample Extracted Radiomics:</span>
                <div className="max-h-36 overflow-y-auto space-y-1 font-mono text-[11px] bg-slate-950 p-2.5 rounded-lg border border-slate-850">
                  {Object.entries(data.radiomics_panel.verified_features_sample).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-slate-300">
                      <span className="text-slate-400 truncate max-w-[200px]">{k}</span>
                      <span className="text-cyan-300 font-semibold">{v.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Panel 2: Biomechanical Geometry & Mirels */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Compass className="w-5 h-5 text-teal-400" />
                  <h3 className="font-extrabold text-white text-lg">2. Biomechanical Cross-Section & Mirels</h3>
                </div>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-teal-950 border border-teal-800 text-teal-300 font-mono">
                  Stage 28 Engine
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Evaluated on <strong>{data.biomechanics_geometry_panel.phantom_type}</strong> cross-section phantom.
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Area (mm²)</span>
                  <span className="font-mono text-white font-bold text-xs">{data.biomechanics_geometry_panel.area_mm2.toFixed(1)}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">I_xx (mm⁴)</span>
                  <span className="font-mono text-white font-bold text-xs">{data.biomechanics_geometry_panel.I_xx_mm4.toFixed(0)}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">I_yy (mm⁴)</span>
                  <span className="font-mono text-white font-bold text-xs">{data.biomechanics_geometry_panel.I_yy_mm4.toFixed(0)}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Polar J (mm⁴)</span>
                  <span className="font-mono text-teal-300 font-bold text-xs">{data.biomechanics_geometry_panel.polar_J_mm4.toFixed(0)}</span>
                </div>
              </div>
              {/* Mirels scoring card */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-slate-200">Conventional Mirels (1989):</span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-950 text-amber-300 border border-amber-800">
                    Risk Est: {data.biomechanics_mirels_panel.fracture_risk_percent}%
                  </span>
                </div>
                <p className="text-slate-400 text-[11px]">{data.biomechanics_mirels_panel.clinical_guidance}</p>
              </div>
            </div>

            {/* Panel 3: Explainability Engine */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Eye className="w-5 h-5 text-indigo-400" />
                  <h3 className="font-extrabold text-white text-lg">3. Explainability (Grad-CAM & SHAP)</h3>
                </div>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-950 border border-indigo-800 text-indigo-300 font-mono">
                  Stage 29 Engine
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Grad-CAM gradient attribution generated on controlled synthetic CNN fixture.
              </p>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Attribution Method</span>
                  <span className="font-mono text-indigo-300 font-bold text-xs">{data.explainability_panel.method}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">SHAP Implementation</span>
                  <span className="font-mono text-amber-300 font-bold text-xs">{data.explainability_panel.shap_status}</span>
                </div>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-400">
                <strong>Peak Saliency Coordinates:</strong> [{data.explainability_panel.peak_coordinates.join(', ')}]
                <br />
                <strong>Active Ratio:</strong> {(data.explainability_panel.active_region_ratio * 100).toFixed(1)}%
                <br />
                <span className="text-amber-400/90 italic">{data.explainability_panel.warning}</span>
              </div>
            </div>

            {/* Panel 4: Uncertainty Quantification & Abstention */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-purple-400" />
                  <h3 className="font-extrabold text-white text-lg">4. Uncertainty & Selective Triage</h3>
                </div>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-purple-950 border border-purple-800 text-purple-300 font-mono">
                  Stage 30 Engine
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Tri-state triage gate based on Shannon entropy threshold (H_norm ≥ 0.60).
              </p>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Predicted Class</span>
                  <span className="font-mono text-white font-bold text-xs truncate block">{data.uncertainty_panel.predicted_class_name}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Norm Entropy (H_norm)</span>
                  <span className="font-mono text-purple-300 font-bold text-xs">{data.uncertainty_panel.normalized_entropy.toFixed(3)}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Candidate Cutoff</span>
                  <span className="font-mono text-slate-300 font-bold text-xs">{data.uncertainty_panel.candidate_threshold.toFixed(2)}</span>
                </div>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-300 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-bold">Decision: {data.uncertainty_panel.decision}</span>
                  {data.uncertainty_panel.human_review_required && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-300 border border-rose-800">
                      Mandatory Clinical Review
                    </span>
                  )}
                </div>
                <p className="text-slate-400 text-[11px]">{data.uncertainty_panel.reason_description}</p>
              </div>
            </div>
          </div>

          {/* Outstanding Clinical Blockers Card */}
          <div className="p-6 rounded-2xl bg-rose-950/20 border border-rose-900/60 space-y-3">
            <div className="flex items-center space-x-2 text-rose-300 font-bold text-sm">
              <Lock className="w-4 h-4" />
              <span>Four-Class Clinical Deployment Status: {data.four_class_system_status}</span>
            </div>
            <p className="text-xs text-slate-300">
              The end-to-end mathematical pipeline is fully verified on synthetic fixtures. However, clinical inference remains locked pending resolution of the following requirements:
            </p>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-rose-200 list-disc list-inside">
              {data.outstanding_clinical_requirements.map((req, rIdx) => (
                <li key={rIdx}>{req}</li>
              ))}
            </ul>
          </div>
        </motion.div>
      ) : (
        <div className="p-12 text-center rounded-2xl bg-slate-900/40 border border-slate-800/80 space-y-4">
          <Cpu className="w-12 h-12 text-slate-600 mx-auto" />
          <h3 className="text-lg font-bold text-slate-200">Demonstration Pipeline Ready</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Click "Run Full Synthetic Pipeline" above to trigger a synchronized execution across all 4 core research engines using controlled mathematical digital phantoms.
          </p>
        </div>
      )}
    </div>
  );
};

export default ResearchWorkbenchPage;
