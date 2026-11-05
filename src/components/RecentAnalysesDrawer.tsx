import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, History, Sparkles, Clock, ArrowRight, Trash2 } from 'lucide-react';
import { AnalyzeResponse } from '../api/types';
import { resolveApiAssetUrl } from '../api/client';
import { useNavigate } from 'react-router-dom';

interface RecentAnalysesDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  recentList: AnalyzeResponse[];
  onSelectResult: (result: AnalyzeResponse) => void;
  onClearHistory: () => void;
}

export const RecentAnalysesDrawer: React.FC<RecentAnalysesDrawerProps> = ({
  isOpen,
  onClose,
  recentList,
  onSelectResult,
  onClearHistory,
}) => {
  const navigate = useNavigate();

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />

          {/* Drawer Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 300 }}
            className="relative w-full max-w-md bg-slate-900 border-l border-slate-800 shadow-2xl h-full flex flex-col justify-between z-10"
          >
            {/* Header */}
            <div className="p-5 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <History className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-bold text-white">Recent Analyses (Cached N={recentList.length}/20)</h3>
              </div>
              <button
                onClick={onClose}
                className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition"
                aria-label="Close recent analyses drawer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {recentList.length === 0 ? (
                <div className="text-center py-12 text-slate-500 text-xs">
                  <Clock className="w-8 h-8 mx-auto text-slate-600 mb-2 opacity-50" />
                  <span>No recent analyses in session cache. Upload a scan to populate history.</span>
                </div>
              ) : (
                recentList.map((item) => {
                  const conf = Math.round(item.prediction.confidence * 100);
                  return (
                    <div
                      key={item.requestId}
                      onClick={() => {
                        onSelectResult(item);
                        onClose();
                        navigate('/results');
                      }}
                      className="p-3.5 rounded-2xl bg-slate-950/80 border border-slate-800 hover:border-cyan-500/50 cursor-pointer transition flex items-center space-x-3.5 group"
                    >
                      {/* Thumbnail */}
                      <div className="w-14 h-14 rounded-xl overflow-hidden bg-slate-900 border border-slate-800 shrink-0">
                        <img
                          src={resolveApiAssetUrl(item.imageUrl)}
                          alt="Thumbnail"
                          className="w-full h-full object-cover"
                        />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-bold text-white uppercase truncate font-mono">
                            {item.prediction.topClass}
                          </span>
                          <span className="font-mono text-cyan-400 font-semibold">{conf}%</span>
                        </div>
                        <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                          <span>{item.requestId}</span>
                          <span>{new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        </div>
                      </div>

                      <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 transition shrink-0" />
                    </div>
                  );
                })
              )}
            </div>

            {/* Footer */}
            {recentList.length > 0 && (
              <div className="p-4 border-t border-slate-800 flex justify-between items-center text-xs">
                <span className="text-slate-500 font-mono">Fast client-side reload</span>
                <button
                  onClick={onClearHistory}
                  className="flex items-center space-x-1.5 text-rose-400 hover:text-rose-300 font-semibold"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear Cache</span>
                </button>
              </div>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
