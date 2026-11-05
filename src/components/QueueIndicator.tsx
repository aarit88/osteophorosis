import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers, Clock, Zap } from 'lucide-react';

interface QueueIndicatorProps {
  activeCount: number;
  queuedCount: number;
}

export const QueueIndicator: React.FC<QueueIndicatorProps> = ({ activeCount, queuedCount }) => {
  if (activeCount === 0 && queuedCount === 0) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-xl bg-cyan-950/80 border border-cyan-800 text-cyan-300 shadow-md backdrop-blur-md"
      >
        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
        <span>Queue: {activeCount} active (Max 2)</span>
        {queuedCount > 0 && (
          <span className="text-amber-400 font-bold">+{queuedCount} waiting</span>
        )}
      </motion.div>
    </AnimatePresence>
  );
};
