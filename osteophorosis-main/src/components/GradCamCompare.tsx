import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers, Eye, Sliders, Maximize2, Sparkles, Image as ImageIcon } from 'lucide-react';
import { resolveApiAssetUrl } from '../api/client';

interface GradCamCompareProps {
  originalUrl: string;
  gradcamUrl: string;
}

export const GradCamCompare: React.FC<GradCamCompareProps> = ({ originalUrl, gradcamUrl }) => {
  const [viewMode, setViewMode] = useState<'slider' | 'sideBySide' | 'toggle'>('toggle');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.85); // 0 to 1
  const [splitPosition, setSplitPosition] = useState<number>(50); // 0 to 100%
  const [showGradCamInToggle, setShowGradCamInToggle] = useState<boolean>(true);

  const fullOriginalUrl = resolveApiAssetUrl(originalUrl);
  const fullGradcamUrl = resolveApiAssetUrl(gradcamUrl);

  return (
    <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-xl flex flex-col justify-between">
      {/* Header & Mode Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white">Visual Explainability (Grad-CAM)</h3>
          </div>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Class Activation Mapping highlighting spatial feature salience
          </p>
        </div>

        {/* View mode buttons */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setViewMode('toggle')}
            className={`px-2.5 py-1 rounded-lg font-semibold transition ${
              viewMode === 'toggle'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Blend
          </button>
          <button
            onClick={() => setViewMode('slider')}
            className={`px-2.5 py-1 rounded-lg font-semibold transition ${
              viewMode === 'slider'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Split Slider
          </button>
          <button
            onClick={() => setViewMode('sideBySide')}
            className={`px-2.5 py-1 rounded-lg font-semibold transition ${
              viewMode === 'sideBySide'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Side-by-Side
          </button>
        </div>
      </div>

      {/* Main Image Stage */}
      <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center min-h-[340px]">
        {viewMode === 'toggle' && (
          <div className="relative w-full h-[380px] overflow-hidden flex items-center justify-center">
            {/* Base original image */}
            <img
              src={fullOriginalUrl}
              alt="Original bone radiograph"
              className="absolute inset-0 w-full h-full object-contain p-2"
            />

            {/* Grad-CAM overlay with animated opacity */}
            <motion.img
              src={fullGradcamUrl}
              alt="Grad-CAM heat attention map"
              initial={false}
              animate={{ opacity: showGradCamInToggle ? overlayOpacity : 0 }}
              transition={{ duration: 0.25 }}
              className="absolute inset-0 w-full h-full object-contain p-2 mix-blend-screen"
            />
          </div>
        )}

        {viewMode === 'slider' && (
          <div className="relative w-full h-[380px] overflow-hidden select-none">
            {/* Underneath: Grad-CAM */}
            <img
              src={fullGradcamUrl}
              alt="Grad-CAM full map"
              className="absolute inset-0 w-full h-full object-contain p-2"
            />

            {/* Over top: Original image clipped by split percentage */}
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ clipPath: `inset(0 ${100 - splitPosition}% 0 0)` }}
            >
              <img
                src={fullOriginalUrl}
                alt="Original clipped"
                className="absolute inset-0 w-full h-full object-contain p-2"
              />
            </div>

            {/* Vertical dividing slider bar */}
            <div
              className="absolute top-0 bottom-0 w-1 bg-cyan-400 cursor-ew-resize flex items-center justify-center shadow-lg"
              style={{ left: `${splitPosition}%` }}
            >
              <div className="w-6 h-6 rounded-full bg-slate-900 border-2 border-cyan-400 text-cyan-300 flex items-center justify-center shadow-md">
                <Sliders className="w-3 h-3 transform rotate-90" />
              </div>
            </div>

            {/* Slider touch/click listener input */}
            <input
              type="range"
              min="0"
              max="100"
              value={splitPosition}
              onChange={(e) => setSplitPosition(Number(e.target.value))}
              className="absolute inset-0 opacity-0 cursor-ew-resize w-full h-full"
              aria-label="Before and after split slider"
            />
          </div>
        )}

        {viewMode === 'sideBySide' && (
          <div className="grid grid-cols-2 gap-2 w-full p-2 h-[380px]">
            <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-900/50 flex flex-col items-center justify-center">
              <img
                src={fullOriginalUrl}
                alt="Original"
                className="w-full h-full object-contain p-1"
              />
              <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded text-[10px] font-mono bg-black/70 text-slate-300">
                Original Input
              </span>
            </div>
            <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-900/50 flex flex-col items-center justify-center">
              <img
                src={fullGradcamUrl}
                alt="Grad-CAM"
                className="w-full h-full object-contain p-1"
              />
              <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded text-[10px] font-mono bg-black/70 text-rose-300">
                Grad-CAM Heatmap
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Interactive Controls Bar */}
      <div className="mt-4 pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
        {viewMode === 'toggle' && (
          <div className="flex items-center space-x-3 w-full sm:w-auto">
            <button
              onClick={() => setShowGradCamInToggle(!showGradCamInToggle)}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-semibold transition ${
                showGradCamInToggle
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>{showGradCamInToggle ? 'Hide Heatmap' : 'Show Heatmap'}</span>
            </button>

            {showGradCamInToggle && (
              <div className="flex items-center space-x-2 flex-1 sm:w-48">
                <span className="text-[11px] text-slate-400">Intensity:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                  aria-label="Grad-CAM overlay intensity"
                />
              </div>
            )}
          </div>
        )}

        {viewMode === 'slider' && (
          <div className="text-[11px] text-slate-400">
            Drag the vertical divider left or right to inspect lesion boundaries.
          </div>
        )}

        {/* Required Caption */}
        <div className="text-[11px] font-mono text-cyan-300/80 flex items-center space-x-1">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Simulated Grad-CAM — placeholder attention map</span>
        </div>
      </div>
    </div>
  );
};
