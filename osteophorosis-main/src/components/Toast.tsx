import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, X } from 'lucide-react';

interface ToastProps {
  message: string | null;
  onClose: () => void;
  type?: 'error' | 'warning' | 'info';
}

export const Toast: React.FC<ToastProps> = ({ message, onClose, type = 'error' }) => {
  return (
    <AnimatePresence>
      {message && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          transition={{ duration: 0.2 }}
          className="fixed bottom-6 right-6 z-50 max-w-md w-full shadow-2xl rounded-2xl p-4 border backdrop-blur-xl flex items-start space-x-3 bg-rose-950/90 border-rose-600/60 text-rose-100"
          role="status"
          aria-live="polite"
        >
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1 text-xs sm:text-sm font-medium leading-relaxed">
            {message}
          </div>
          <button
            onClick={onClose}
            aria-label="Close notification"
            className="p-1 rounded-lg text-rose-300 hover:text-white hover:bg-rose-900/50 transition shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
