import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Bone } from 'lucide-react';

interface LoaderProps {
  message?: string;
}

export const Loader: React.FC<LoaderProps> = ({ message = 'Running Dual-Stream ViT & Radiomics Extraction…' }) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-4 text-center">
      <div className="relative w-20 h-20 flex items-center justify-center">
        {/* Outer pulsing ring */}
        <motion.div
          animate={{ scale: [1, 1.25, 1], opacity: [0.3, 0.7, 0.3] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute inset-0 rounded-2xl bg-cyan-500/20 border border-cyan-500/40"
        />

        {/* Rotating scanner */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
          className="w-16 h-16 rounded-xl border-2 border-dashed border-cyan-400 flex items-center justify-center"
        >
          <Bone className="w-8 h-8 text-cyan-300 transform rotate-45" />
        </motion.div>

        {/* Scanner radar beam line */}
        <motion.div
          animate={{ y: [-24, 24, -24] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute w-14 h-0.5 bg-gradient-to-r from-transparent via-cyan-300 to-transparent shadow-lg shadow-cyan-400"
        />
      </div>

      <div className="space-y-1">
        <div className="text-sm font-bold text-white tracking-wide">{message}</div>
        <p className="text-xs text-slate-400 font-mono">
          Computing SHA-256 seed • Grad-CAM attention • SHAP feature weights
        </p>
      </div>

      {/* Shimmer progress bar */}
      <div className="w-64 h-1.5 bg-slate-800 rounded-full overflow-hidden relative mt-2">
        <motion.div
          animate={{ x: ['-100%', '100%'] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
          className="w-1/2 h-full bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
        />
      </div>
    </div>
  );
};

export const SkeletonCard: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`p-6 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse ${className}`}>
      <div className="h-4 bg-slate-800 rounded w-1/3 mb-4" />
      <div className="h-8 bg-slate-800 rounded w-1/2 mb-3" />
      <div className="h-20 bg-slate-800/60 rounded w-full mb-3" />
      <div className="h-4 bg-slate-800 rounded w-2/3" />
    </div>
  );
};
