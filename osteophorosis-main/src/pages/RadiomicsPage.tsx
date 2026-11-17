import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Sliders,
  CheckCircle2,
  AlertTriangle,
  RotateCw,
  Info,
  Layers,
  Sparkles,
  BarChart,
} from 'lucide-react';
import { runRadiomicsDemo } from '../api/client';
import { RadiomicsDemoResponse } from '../api/types';

export const RadiomicsPage: React.FC = () => {
  const [phantomType, setPhantomType] = useState<string>('checkerboard');
  const [gridSize, setGridSize] = useState<number>(64);
  const [spacingMm, setSpacingMm] = useState<number>(1.0);
  const [data, setData] = useState<RadiomicsDemoResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchFeatures = async (type = phantomType, size = gridSize, spacing = spacingMm) => {
    setLoading(true);
    setError(null);
    try {
      const res = await runRadiomicsDemo({
        phantom_type: type,
        grid_size: size,
        spacing_mm: spacing,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to extract radiomic features.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFeatures();
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-xs font-semibold">
          <Sliders className="w-3.5 h-3.5" />
          <span>Stage 27 Engine: Radiomics & Digital Phantoms</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Radiomics Feature Explorer
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Engineered extraction of quantitative radiomic textures and morphological descriptors adhering to Image Biomarker Standardisation Initiative (IBSI) benchmarks on deterministic digital phantoms.
        </p>
      </div>

      {/* Configuration & Controls */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">
          Synthetic Digital Phantom Configuration
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Digital Phantom Type
            </label>
            <select
              value={phantomType}
              onChange={(e) => setPhantomType(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value="checkerboard">Checkerboard Phantom (High contrast GLCM/GLRLM)</option>
              <option value="linear_gradient">Linear Gradient Phantom (Directional GLCM)</option>
              <option value="annulus">Geometric Annulus Phantom (Curvature & Shape)</option>
              <option value="uniform">Uniform Phantom (Zero variance baseline)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Grid Resolution (Pixels)
            </label>
            <select
              value={gridSize}
              onChange={(e) => setGridSize(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value={32}>32 x 32</option>
              <option value={64}>64 x 64 (Standard)</option>
              <option value={128}>128 x 128 (Fine)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Pixel Spacing (mm)
            </label>
            <select
              value={spacingMm}
              onChange={(e) => setSpacingMm(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value={0.5}>0.5 mm / pixel</option>
              <option value={1.0}>1.0 mm / pixel</option>
              <option value={2.0}>2.0 mm / pixel</option>
            </select>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            type="button"
            onClick={() => fetchFeatures(phantomType, gridSize, spacingMm)}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs disabled:opacity-50 transition cursor-pointer"
          >
            {loading ? <RotateCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            <span>Re-compute Radiomics</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Feature Breakdown Summary */}
      {data && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Verified Features</span>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              </div>
              <span className="text-2xl font-black text-emerald-400 font-mono">
                {data.verified_features_count}
              </span>
              <span className="text-[11px] text-slate-400 block mt-1">
                Adhering to verified IBSI mathematical definitions
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Unverified / Non-Standard</span>
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              </div>
              <span className="text-2xl font-black text-amber-400 font-mono">
                {data.unverified_features_count}
              </span>
              <span className="text-[11px] text-slate-400 block mt-1">
                Excluded from fusion due to undefined 2D/3D standards
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>102-D Proposal Spec Status</span>
                <Info className="w-4 h-4 text-cyan-400" />
              </div>
              <span className="text-sm font-extrabold text-cyan-300 block truncate">
                {data.specification_status}
              </span>
              <span className="text-[11px] text-slate-400 block mt-1">
                Total Proposal Vector: {data.total_features_count} Dimensions
              </span>
            </div>
          </div>

          {/* Sample Features List */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <h3 className="text-sm font-bold text-white">Verified Feature Sample (IBSI-Compliant)</h3>
                <span className="text-[11px] font-mono text-emerald-400">81 Verified Total</span>
              </div>
              <div className="max-h-72 overflow-y-auto space-y-1 font-mono text-xs pr-2">
                {Object.entries(data.verified_features_sample).map(([name, val]) => (
                  <div
                    key={name}
                    className="flex justify-between items-center py-1.5 px-2.5 rounded bg-slate-950/70 border border-slate-850 text-slate-300"
                  >
                    <span className="text-slate-400 truncate max-w-[240px]">{name}</span>
                    <span className="text-cyan-300 font-bold">{val.toFixed(5)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <h3 className="text-sm font-bold text-white">Unverified / Excluded Feature Sample</h3>
                <span className="text-[11px] font-mono text-amber-400">21 Excluded</span>
              </div>
              <div className="max-h-72 overflow-y-auto space-y-1 font-mono text-xs pr-2">
                {Object.entries(data.unverified_features_sample).map(([name, val]) => (
                  <div
                    key={name}
                    className="flex justify-between items-center py-1.5 px-2.5 rounded bg-slate-950/70 border border-slate-850 text-slate-300"
                  >
                    <span className="text-slate-400 truncate max-w-[240px]">{name}</span>
                    <span className="text-amber-300 font-bold">{val.toFixed(5)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Verification Notice */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-1">
            <strong>Scientific Notice:</strong> {data.disclaimer}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default RadiomicsPage;

