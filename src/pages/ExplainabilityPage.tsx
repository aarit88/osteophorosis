import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Eye,
  AlertTriangle,
  RotateCw,
  Sparkles,
  Layers,
  ShieldAlert,
  Info,
  CheckCircle2,
} from 'lucide-react';
import { runExplainabilityDemo } from '../api/client';
import { ExplainabilityDemoResponse } from '../api/types';

export const ExplainabilityPage: React.FC = () => {
  const [targetClass, setTargetClass] = useState<number>(0);
  const [shiftHorizontal, setShiftHorizontal] = useState<number>(0);
  const [data, setData] = useState<ExplainabilityDemoResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchExplainability = async (cls = targetClass, shift = shiftHorizontal) => {
    setLoading(true);
    setError(null);
    try {
      const res = await runExplainabilityDemo({
        target_class: cls,
        shift_horizontal: shift,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to generate explainability demo.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExplainability();
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-950/70 border border-indigo-800 text-indigo-300 text-xs font-semibold">
          <Eye className="w-3.5 h-3.5" />
          <span>Stage 29 Engine: Grad-CAM & Architectural SHAP Audit</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Explainability Engine
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Visualizes Gradient-weighted Class Activation Mapping (Grad-CAM) localization on controlled spatial CNN fixtures and details the architectural readiness of SHAP feature attribution.
        </p>
      </div>

      {/* Disclosures Alert */}
      <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/80 text-amber-200 text-xs flex items-start space-x-3">
        <ShieldAlert className="w-5 h-5 flex-shrink-0 text-amber-400 mt-0.5" />
        <div className="space-y-1">
          <strong className="text-amber-300 font-bold">Scientific Attribution Notice:</strong>
          <p>
            Saliency and attribution maps in this prototype are generated exclusively on synthetic test fixtures. Grad-CAM visualizes where feature maps responded most strongly in synthetic CNN layers, but does <strong>not</strong> constitute validated clinical diagnostic proof or ground-truth lesion boundaries.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">
          Controlled Synthetic CNN Target Selection
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Target Output Class Index
            </label>
            <select
              value={targetClass}
              onChange={(e) => {
                const val = Number(e.target.value);
                setTargetClass(val);
                fetchExplainability(val, shiftHorizontal);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value={0}>Class 0 (Metastasis Proxy Target)</option>
              <option value={1}>Class 1 (Paget's Disease Proxy Target)</option>
              <option value={2}>Class 2 (Osteogenesis Imperfecta Proxy Target)</option>
              <option value={3}>Class 3 (Healthy Control Proxy Target)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Synthetic Activation Center Shift
            </label>
            <select
              value={shiftHorizontal}
              onChange={(e) => {
                const val = Number(e.target.value);
                setShiftHorizontal(val);
                fetchExplainability(targetClass, val);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value={0}>Centered Feature (Offset 0)</option>
              <option value={-8}>Left-Shifted Feature (Offset -8 px)</option>
              <option value={8}>Right-Shifted Feature (Offset +8 px)</option>
            </select>
          </div>
        </div>

        <div className="flex justify-end pt-1">
          <button
            type="button"
            onClick={() => fetchExplainability(targetClass, shiftHorizontal)}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs disabled:opacity-50 transition cursor-pointer"
          >
            {loading ? <RotateCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            <span>Regenerate Grad-CAM</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Main Results Grid */}
      {data && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Visual Heatmap Grid Panel */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="font-bold text-white text-base">Grad-CAM Heatmap Matrix Sample</h3>
                <span className="text-xs font-mono text-indigo-400">Layer: conv1</span>
              </div>

              {/* Render 8x8 matrix sample */}
              <div className="space-y-2">
                <span className="text-xs text-slate-400 block">
                  Normalized Activation Grid (8x8 Sample of 64x64):
                </span>
                <div className="grid grid-cols-8 gap-1.5 p-3 rounded-xl bg-slate-950 border border-slate-850">
                  {data.heatmap_grid_sample.map((row, rIdx) =>
                    row.map((val, cIdx) => {
                      const intensity = Math.min(Math.max(val, 0), 1);
                      // Calculate heat color (from dark navy to cyan to bright amber)
                      const bgStyle =
                        intensity > 0.7
                          ? `rgba(245, 158, 11, ${intensity})`
                          : intensity > 0.3
                          ? `rgba(6, 182, 212, ${intensity})`
                          : `rgba(30, 41, 59, ${intensity + 0.2})`;

                      return (
                        <div
                          key={`${rIdx}-${cIdx}`}
                          className="aspect-square rounded flex items-center justify-center text-[9px] font-mono text-white/90 select-none transition-colors"
                          style={{ backgroundColor: bgStyle }}
                          title={`[${rIdx}, ${cIdx}]: ${val.toFixed(3)}`}
                        >
                          {val > 0.1 ? val.toFixed(1) : ''}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Peak Coordinate (Y, X)</span>
                  <span className="font-mono text-indigo-300 font-bold text-sm">
                    [{data.peak_coordinates.join(', ')}]
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Active Region Ratio</span>
                  <span className="font-mono text-cyan-300 font-bold text-sm">
                    {(data.active_region_ratio * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Architecture Audit & SHAP Status Panel */}
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="font-bold text-white text-base">Attribution Architecture Audit</h3>
                <span className="text-xs font-mono text-amber-400">Dual-Engine Plan</span>
              </div>

              <div className="space-y-3">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="font-bold text-slate-200">Grad-CAM Method Status:</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                      VERIFIED ON SYNTHETIC CNN
                    </span>
                  </div>
                  <p className="text-slate-400 text-xs">
                    Gradient extraction with positive rectification $ReLU(\sum \alpha_k A^k)$ verified across spatial coordinate shifts.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="font-bold text-slate-200">SHAP Attribution Status:</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800">
                      {data.shap_status}
                    </span>
                  </div>
                  <p className="text-slate-400 text-xs">
                    Analytical Shapley value calculation verified on synthetic low-dimensional fixtures (Stage 29). High-dimensional DeepSHAP/KernelSHAP for clinical CNN feature maps is blocked pending approved clinical models.
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-1.5">
                <span className="font-bold text-slate-300 block">European AI Act Art. 13 Disclosure:</span>
                <p className="text-[11px] text-slate-400">
                  Visual attribution maps must always be paired with quantitative confidence intervals and second-reader radiologist oversight. Saliency maps alone do not constitute diagnostic causality.
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ExplainabilityPage;

