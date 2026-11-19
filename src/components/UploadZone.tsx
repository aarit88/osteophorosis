import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, FileImage, X, CheckCircle2, AlertTriangle, Sparkles } from 'lucide-react';

interface UploadZoneProps {
  selectedFile: File | null;
  onSelectFile: (file: File | null) => void;
  onError: (errorMessage: string) => void;
  disabled?: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  selectedFile,
  onSelectFile,
  onError,
  disabled = false,
}) => {
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Update preview when selectedFile changes
  React.useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const validateAndHandle = useCallback(
    (file: File) => {
      // Check file size (max 15MB)
      if (file.size > 15 * 1024 * 1024) {
        onError('File exceeds 15 MB limit. Please upload a smaller radiographic image.');
        return;
      }

      // Check mime type (images only)
      if (!file.type.startsWith('image/')) {
        onError('Unsupported file type. Please upload a standard medical radiograph image (PNG, JPEG, DICOM-converted).');
        return;
      }

      onSelectFile(file);
    },
    [onSelectFile, onError]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndHandle(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => {
    if (disabled) return;
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSelectFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/jpg"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            validateAndHandle(e.target.files[0]);
          }
        }}
        className="hidden"
        disabled={disabled}
      />

      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        aria-label="Upload medical bone scan image"
        className={`relative w-full rounded-2xl border-2 border-dashed transition-all duration-300 p-8 sm:p-10 flex flex-col items-center justify-center cursor-pointer select-none outline-none focus:ring-4 focus:ring-cyan-500/20 ${
          isDragOver
            ? 'border-cyan-400 bg-cyan-950/40 shadow-xl shadow-cyan-500/10 scale-[1.01]'
            : selectedFile
            ? 'border-cyan-500/50 bg-slate-900/90 dark:bg-slate-900/80 light:bg-slate-50'
            : 'border-slate-700 hover:border-cyan-500/70 bg-slate-900/50 dark:bg-slate-950/40 light:bg-slate-100/70 hover:bg-slate-900/80'
        } ${disabled ? 'opacity-60 cursor-not-allowed' : ''}`}
      >
        <AnimatePresence mode="wait">
          {selectedFile && previewUrl ? (
            /* Selected File Preview */
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col sm:flex-row items-center space-y-4 sm:space-y-0 sm:space-x-6 w-full max-w-lg"
            >
              <div className="relative w-28 h-28 sm:w-32 sm:h-32 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 shrink-0 shadow-lg">
                <img
                  src={previewUrl}
                  alt="Radiograph upload preview"
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent pointer-events-none" />
                <span className="absolute bottom-1 left-2 text-[9px] font-mono text-cyan-300">PREVIEW</span>
              </div>

              <div className="flex-1 min-w-0 text-center sm:text-left">
                <div className="flex items-center justify-center sm:justify-start space-x-2 text-emerald-400 mb-1">
                  <CheckCircle2 className="w-4 h-4" />
                  <span className="text-xs font-bold uppercase tracking-wider">Ready For Analysis</span>
                </div>
                <h4 className="text-sm font-bold text-white truncate max-w-xs">{selectedFile.name}</h4>
                <p className="text-xs text-slate-400 font-mono mt-0.5">{formatFileSize(selectedFile.size)}</p>

                <div className="mt-3 flex items-center justify-center sm:justify-start space-x-2">
                  <button
                    type="button"
                    onClick={handleRemove}
                    className="flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-semibold bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 border border-rose-500/30 transition"
                  >
                    <X className="w-3.5 h-3.5" />
                    <span>Change File</span>
                  </button>
                </div>
              </div>
            </motion.div>
          ) : (
            /* Empty Drop Zone */
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center text-center space-y-3"
            >
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-transform duration-300 ${
                isDragOver ? 'scale-110 bg-cyan-500/20 text-cyan-300' : 'bg-slate-800 text-slate-400'
              }`}>
                <UploadCloud className="w-8 h-8" />
              </div>

              <div>
                <p className="text-base font-bold text-white">
                  Drop your bone radiograph here, or <span className="text-cyan-400 hover:underline">browse files</span>
                </p>
                <p className="text-xs text-slate-400 mt-1 max-w-md">
                  Supports X-ray, CT slice, or Bone Scintigraphy (PNG, JPEG up to 15MB).
                </p>
              </div>

              <div className="flex items-center space-x-2 text-[11px] font-mono text-slate-500 pt-2">
                <span>Deterministic SHA-256 seeding</span>
                <span>•</span>
                <span>Zero model latency demo</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
