import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { UploadZone } from '../components/UploadZone';
import { ModalityType, TrainedModelsOverviewResponse } from '../api/types';
import { getTrainedModelsOverview } from '../api/client';
import { useAnalysis } from '../context/AnalysisContext';
import { Toast } from '../components/Toast';
import { Loader } from '../components/Loader';
import { useTranslation } from 'react-i18next';
import {
  Sparkles,
  Zap,
  ArrowRight,
  Database,
  WifiOff,
  RotateCcw,
  FileCheck2,
  FlaskConical,
  Play,
} from 'lucide-react';

const MODALITIES: Array<{ id: ModalityType; label: string; desc: string }> = [
  { id: 'xray', label: 'Plain Radiograph (X-Ray)', desc: 'Frontal or lateral planar beam' },
  { id: 'ct', label: 'Computed Tomography (CT)', desc: 'Axial or coronal bone window slice' },
  { id: 'bone_scan', label: 'Bone Scintigraphy (Tc-99m)', desc: 'Radionuclide whole-body scan' },
];

const DEMO_ASSETS = [
  {
    path: '/demo/femur-xray-1.png',
    name: 'demo-femur-lesion.png',
    modality: 'xray' as ModalityType,
    label: 'Femur X-Ray (Osteolytic Lesion)',
  },
  {
    path: '/demo/pelvis-xray-2.png',
    name: 'demo-pelvis-paget.png',
    modality: 'xray' as ModalityType,
    label: 'Pelvis AP (Sclerotic Expansion)',
  },
  {
    path: '/demo/tibia-xray-3.png',
    name: 'demo-tibia-healthy.png',
    modality: 'xray' as ModalityType,
    label: 'Tibia/Fibula (Normal Adult Cortical)',
  },
];

export const Home: React.FC = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const {
    executeAnalysisWithQueue,
    isAnalyzing,
    error,
    setError,
    isServerOffline,
    retryConnectivity,
  } = useAnalysis();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModality, setSelectedModality] = useState<ModalityType>('xray');
  const [demoIndex, setDemoIndex] = useState<number>(0);
  const [isDemoLoading, setIsDemoLoading] = useState<boolean>(false);
  const [trainedOverview, setTrainedOverview] = useState<TrainedModelsOverviewResponse | null>(null);

  useEffect(() => {
    getTrainedModelsOverview()
      .then((overview) => setTrainedOverview(overview))
      .catch((err) => console.debug('Trained models summary unavailable:', err));
  }, []);

  const fetchDemoFile = async (targetDemo: (typeof DEMO_ASSETS)[number]): Promise<File> => {
    const response = await fetch(targetDemo.path);
    if (!response.ok) {
      throw new Error(`Could not fetch demo file from ${targetDemo.path}`);
    }
    const blob = await response.blob();
    return new File([blob], targetDemo.name, { type: 'image/png' });
  };

  // Load bundled demo image from /public/demo/
  const handleLoadDemoImage = async (specificIdx?: number) => {
    setIsDemoLoading(true);
    const chosenIdx = typeof specificIdx === 'number' ? specificIdx : demoIndex % DEMO_ASSETS.length;
    const targetDemo = DEMO_ASSETS[chosenIdx];
    if (typeof specificIdx !== 'number') {
      setDemoIndex((prev) => prev + 1);
    }

    try {
      const file = await fetchDemoFile(targetDemo);
      setSelectedFile(file);
      setSelectedModality(targetDemo.modality);
      setError(null);
    } catch (err) {
      console.error('Demo asset load error:', err);
      setError('Unable to load demo radiograph asset.');
    } finally {
      setIsDemoLoading(false);
    }
  };

  // One-click demo load + immediate analysis
  const handleQuickAnalyzeDemo = async (idx: number) => {
    setIsDemoLoading(true);
    const targetDemo = DEMO_ASSETS[idx];
    try {
      const file = await fetchDemoFile(targetDemo);
      setSelectedFile(file);
      setSelectedModality(targetDemo.modality);
      setError(null);
      setIsDemoLoading(false);
      await executeAnalysisWithQueue({
        image: file,
        modality: targetDemo.modality,
      });
      navigate('/results');
    } catch (err) {
      console.error('Quick demo analysis failed:', err);
      setIsDemoLoading(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    try {
      await executeAnalysisWithQueue({
        image: selectedFile,
        modality: selectedModality,
      });
      navigate('/results');
    } catch (err) {
      // Error is set in context
      console.error('Analysis execution caught in Home:', err);
    }
  };

  return (
    <div className="space-y-16 py-6 sm:py-10">
      {/* Toast Notification */}
      <Toast message={error} onClose={() => setError(null)} />

      {/* Resilience Offline Banner */}
      {isServerOffline && (
        <div className="max-w-3xl mx-auto px-4">
          <div className="p-4 rounded-2xl bg-rose-950/80 border border-rose-500/80 text-rose-100 flex items-center justify-between shadow-2xl">
            <div className="flex items-center space-x-3">
              <WifiOff className="w-5 h-5 text-rose-400 shrink-0" />
              <div>
                <span className="font-bold text-sm block">Simulation Server Offline</span>
                <span className="text-xs text-rose-300">
                  Unable to reach OsteoGuard API endpoint. Dev server is starting up or restarting.
                </span>
              </div>
            </div>
            <button
              onClick={retryConnectivity}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      )}

      {/* Hero Section */}
      <section className="relative text-center max-w-4xl mx-auto px-4 overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-[500px] h-[300px] bg-gradient-to-r from-cyan-500/10 via-teal-500/10 to-indigo-500/10 blur-3xl pointer-events-none rounded-full" />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-800/60 text-cyan-300 text-xs font-semibold mb-6 shadow-sm"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>{t('hero.badge', 'Explainable Deep Learning + Radiomics Inference')}</span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-tight sm:leading-none"
        >
          Instant Bone Pathology Classification & <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-300 to-sky-400">XAI Attention</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-6 text-base sm:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed"
        >
          {t('hero.subtitle', "OsteoGuard AI fuses a dual-stream CNN + Vision Transformer with quantitative radiomics features to differentiate Metastasis, Paget's Disease, Osteogenesis Imperfecta, and Healthy Bone in seconds.")}
        </motion.p>
      </section>

      {/* Upload & Configuration Card */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="max-w-3xl mx-auto px-4"
      >
        <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 sm:p-8 shadow-2xl backdrop-blur-xl relative overflow-hidden">
          {/* Top Row: Modality Selector */}
          <div className="mb-6">
            <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2.5">
              {t('upload.selectModality', 'Select Radiographic Modality')}
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {MODALITIES.map((mod) => {
                const isSelected = selectedModality === mod.id;
                return (
                  <button
                    key={mod.id}
                    type="button"
                    onClick={() => setSelectedModality(mod.id)}
                    className={`p-3 rounded-xl border text-left transition relative ${
                      isSelected
                        ? 'bg-cyan-950/50 border-cyan-500/70 text-cyan-200 shadow-md shadow-cyan-950/60'
                        : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                    }`}
                  >
                    <div className="text-xs font-bold">{mod.label}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5 truncate">{mod.desc}</div>
                    {isSelected && (
                      <motion.div
                        layoutId="modality-pill"
                        className="absolute inset-0 rounded-xl border-2 border-cyan-400 pointer-events-none"
                        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Upload Drop Zone */}
          <div className="mb-6">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <label className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Upload Skeletal Radiograph
              </label>

              {/* Load Demo Image Button */}
              <button
                type="button"
                onClick={() => handleLoadDemoImage()}
                disabled={isDemoLoading || isAnalyzing}
                className="flex items-center space-x-1.5 text-xs font-semibold px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 hover:border-cyan-500/40 transition disabled:opacity-50"
              >
                <Database className="w-3.5 h-3.5" />
                <span>{isDemoLoading ? 'Loading…' : t('hero.loadDemo', 'Load Demo Image')}</span>
              </button>
            </div>

            <UploadZone
              selectedFile={selectedFile}
              onSelectFile={setSelectedFile}
              onError={setError}
              disabled={isAnalyzing}
            />

            {/* Quick 1-Click Demo Presets */}
            <div className="mt-3 p-3 rounded-2xl bg-slate-950/70 border border-slate-800/80 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] text-slate-400">
                <span className="font-semibold text-slate-300">Quick Sample Radiographs (1-Click End-to-End Demo)</span>
                <span>Generates Grad-CAM, SHAP & Uncertainty Triage</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {DEMO_ASSETS.map((demo, idx) => (
                  <button
                    key={demo.name}
                    type="button"
                    disabled={isDemoLoading || isAnalyzing}
                    onClick={() => handleQuickAnalyzeDemo(idx)}
                    className="flex items-center justify-between px-3 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/50 text-left transition disabled:opacity-50 group"
                  >
                    <div className="min-w-0 pr-2">
                      <div className="text-xs font-semibold text-slate-200 group-hover:text-cyan-300 truncate">
                        {demo.label}
                      </div>
                      <div className="text-[10px] font-mono text-slate-500 truncate">{demo.name}</div>
                    </div>
                    <Play className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Action Trigger Button or Loading State */}
          <div className="pt-2">
            {isAnalyzing ? (
              <div className="rounded-2xl bg-slate-950 border border-slate-800 p-4">
                <Loader message="Running Dual-Stream ViT & Radiomics Extraction…" />
              </div>
            ) : (
              <button
                type="button"
                onClick={handleRunAnalysis}
                disabled={!selectedFile || isAnalyzing}
                className={`w-full py-4 px-6 rounded-2xl font-bold text-sm sm:text-base flex items-center justify-center space-x-2 transition-all duration-200 shadow-xl ${
                  selectedFile
                    ? 'bg-gradient-to-r from-cyan-500 to-teal-400 hover:from-cyan-400 hover:to-teal-300 text-slate-950 shadow-cyan-500/25 cursor-pointer transform hover:-translate-y-0.5 active:translate-y-0'
                    : 'bg-slate-800/80 text-slate-500 border border-slate-800 cursor-not-allowed'
                }`}
              >
                <Zap className="w-5 h-5" />
                <span>{t('hero.runButton', 'Run OsteoGuard Analysis')}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </motion.section>

      {/* Target Diagnostic Classes */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-50px' }}
        transition={{ duration: 0.6 }}
        className="max-w-6xl mx-auto px-4"
      >
        <div className="text-center mb-8">
          <h2 className="text-xl sm:text-2xl font-bold text-white">4 Target Diagnostic Classes</h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Trained to capture osteolytic erosion, cortical thickening, and trabecular brittleness
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-rose-500/30 shadow-lg">
            <div className="w-8 h-8 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold text-xs mb-3 border border-rose-500/40">
              01
            </div>
            <h3 className="text-base font-bold text-white">Bone Metastasis</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Malignant osteolytic/osteoblastic secondary seeding with cortical thinning and periosteal disruption.
            </p>
            <span className="inline-block mt-3 text-[10px] font-mono text-rose-400">Class Color: #EF4444</span>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/70 border border-amber-500/30 shadow-lg">
            <div className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-xs mb-3 border border-amber-500/40">
              02
            </div>
            <h3 className="text-base font-bold text-white">Paget's Disease</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Disorganized osteoclastic/osteoblastic remodeling leading to cortical expansion and sclerosis.
            </p>
            <span className="inline-block mt-3 text-[10px] font-mono text-amber-400">Class Color: #F59E0B</span>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/70 border border-purple-500/30 shadow-lg">
            <div className="w-8 h-8 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold text-xs mb-3 border border-purple-500/40">
              03
            </div>
            <h3 className="text-base font-bold text-white">Osteogenesis Imp. (OI)</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Type I collagen structural mutation causing marked osteopenia, slender gracile bones, and bowing.
            </p>
            <span className="inline-block mt-3 text-[10px] font-mono text-purple-400">Class Color: #8B5CF6</span>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/70 border border-emerald-500/30 shadow-lg">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs mb-3 border border-emerald-500/40">
              04
            </div>
            <h3 className="text-base font-bold text-white">Healthy Bone</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Intact trabecular architecture, uniform cortical thickness, and normal physiological mineral attenuation.
            </p>
            <span className="inline-block mt-3 text-[10px] font-mono text-emerald-400">Class Color: #10B981</span>
          </div>
        </div>
      </motion.section>

      {/* Trained Sub-Track Artifacts Summary */}
      {trainedOverview && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5 }}
          className="max-w-6xl mx-auto px-4"
        >
          <div className="rounded-3xl bg-slate-900/80 border border-slate-800 p-6 sm:p-8 space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
              <div>
                <div className="inline-flex items-center space-x-2 text-xs font-bold text-cyan-400 uppercase tracking-wider mb-1">
                  <FileCheck2 className="w-4 h-4" />
                  <span>Verified Trained Model Artifacts</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-bold text-white">
                  {trainedOverview.trained_models_count} Sub-Track Models Evaluated on{' '}
                  {trainedOverview.total_holdout_samples_evaluated.toLocaleString()} Burned Holdout Samples
                </h2>
                <p className="text-xs text-slate-400 mt-1 max-w-2xl">
                  Explore real holdout ROC/PR curves, confusion matrices, and CSV test predictions from your trained ResNet-101, 5-Fold ResNet-18, and BTXRD ResNet-34 pipelines.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2.5 shrink-0">
                <Link
                  to="/evidence"
                  className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition"
                >
                  <FileCheck2 className="w-4 h-4" />
                  <span>Inspect Curves & Predictions</span>
                </Link>
                <Link
                  to="/workbench"
                  className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 font-semibold text-xs border border-slate-700 transition"
                >
                  <FlaskConical className="w-4 h-4 text-cyan-400" />
                  <span>Research Workbench</span>
                </Link>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {trainedOverview.models.map((m) => (
                <div
                  key={m.model_id}
                  className="p-4 rounded-2xl bg-slate-950/80 border border-slate-800 flex flex-col justify-between space-y-3"
                >
                  <div>
                    <div className="text-[11px] font-mono text-cyan-400 uppercase">{m.modality}</div>
                    <h3 className="text-sm font-bold text-white mt-0.5">{m.title}</h3>
                    <div className="text-[11px] text-slate-400 mt-1">{m.architecture}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-xs">
                    <div>
                      <span className="text-[10px] text-slate-500 block">Holdout Samples</span>
                      <span className="font-mono font-bold text-white">{m.total_holdout_evaluated.toLocaleString()}</span>
                    </div>
                    {m.headline_metrics.roc_auc !== undefined ? (
                      <div>
                        <span className="text-[10px] text-slate-500 block">ROC-AUC</span>
                        <span className="font-mono font-bold text-emerald-400">
                          {(m.headline_metrics.roc_auc * 100).toFixed(1)}%
                        </span>
                      </div>
                    ) : (
                      <div>
                        <span className="text-[10px] text-slate-500 block">Center 2 Acc</span>
                        <span className="font-mono font-bold text-amber-400">
                          {((m.headline_metrics.center2_accuracy || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.section>
      )}
    </div>
  );
};
