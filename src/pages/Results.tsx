import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, type Variants } from 'framer-motion';
import { useAnalysis } from '../context/AnalysisContext';
import { GradCamCompare } from '../components/GradCamCompare';
import { ProbabilityChart } from '../components/ProbabilityChart';
import { ShapChart } from '../components/ShapChart';
import { FractureGauge } from '../components/FractureGauge';
import { UncertaintyCard } from '../components/UncertaintyCard';
import { AbstentionBanner } from '../components/AbstentionBanner';
import { ClinicianFeedbackCard } from '../components/ClinicianFeedbackCard';
import { CounterfactualPanel } from '../components/CounterfactualPanel';
import { TraceableReportModal } from '../components/TraceableReportModal';
import {
  Sparkles,
  ArrowLeft,
  Printer,
  ShieldAlert,
  AlertTriangle,
  Info,
  CheckCircle2,
  Calendar,
  Layers,
  FileText,
  UserCheck,
} from 'lucide-react';

const CLASS_DISPLAY = {
  metastasis: {
    label: 'Bone Metastasis',
    colorText: 'text-rose-400',
    colorBorder: 'border-rose-500/50',
    badge: 'Malignancy Alert',
    description: 'High-confidence lytic or blastic focal lesion with cortical bone erosion.',
  },
  paget: {
    label: "Paget's Disease of Bone",
    colorText: 'text-amber-400',
    colorBorder: 'border-amber-500/50',
    badge: 'Remodeling Pathology',
    description: 'Excessive and disordered bone turnover with expansion and coarse trabeculae.',
  },
  oi: {
    label: 'Osteogenesis Imperfecta (OI)',
    colorText: 'text-purple-400',
    colorBorder: 'border-purple-500/50',
    badge: 'Genetic Collagen Dysplasia',
    description: 'Severe structural osteopenia, thin delicate cortex, and intrinsic bone fragility.',
  },
  healthy: {
    label: 'Healthy Normal Bone',
    colorText: 'text-emerald-400',
    colorBorder: 'border-emerald-500/50',
    badge: 'Physiological Baseline',
    description: 'Intact trabecular structure without evidence of lytic or blastic destruction.',
  },
};

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 15 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4 },
  },
};

export const Results: React.FC = () => {
  const navigate = useNavigate();
  const { lastResult, engineMode } = useAnalysis();
  const [isReportOpen, setIsReportOpen] = useState<boolean>(false);

  useEffect(() => {
    if (!lastResult) {
      navigate('/');
    }
  }, [lastResult, navigate]);

  if (!lastResult) return null;

  const topClassInfo = CLASS_DISPLAY[lastResult.prediction.topClass] || CLASS_DISPLAY.healthy;
  const confidencePercent = Math.round(lastResult.prediction.confidence * 1000) / 10;
  const entropy = lastResult.uncertaintyDetail?.entropy ?? 0.25;
  const oodFlag = lastResult.uncertaintyDetail?.oodFlag ?? false;
  const shouldAbstain = entropy > 0.6 || oodFlag;

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8 py-4 sm:py-6"
    >
      {/* Top Header Actions Bar */}
      <motion.div
        variants={itemVariants}
        className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4"
      >
        <div className="flex items-center space-x-3">
          <button
            onClick={() => navigate('/')}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 text-xs font-semibold transition"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Run Another Image</span>
          </button>

          <div className="text-xs text-slate-400 font-mono hidden sm:block">
            <span>Request ID: </span>
            <span className="text-cyan-400 font-semibold">{lastResult.requestId}</span>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400 font-mono bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
            <Calendar className="w-3.5 h-3.5 text-cyan-400" />
            <span>{new Date(lastResult.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          </div>

          <button
            onClick={() => setIsReportOpen(true)}
            className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-600 to-teal-500 hover:from-cyan-500 hover:to-teal-400 text-white font-semibold text-xs shadow-md shadow-cyan-600/20 transition cursor-pointer"
          >
            <FileText className="w-4 h-4" />
            <span>Download Traceable Report (PDF)</span>
          </button>
        </div>
      </motion.div>

      {/* Novelty 1: Abstention Protocol Banner */}
      {shouldAbstain && (
        <motion.div variants={itemVariants}>
          <AbstentionBanner entropy={entropy} oodFlag={oodFlag} />
        </motion.div>
      )}

      {/* Main Analysis Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Visual Grad-CAM & Radiomics */}
        <motion.div variants={itemVariants} className="lg:col-span-6 space-y-6">
          <GradCamCompare
            originalUrl={lastResult.imageUrl}
            gradcamUrl={lastResult.gradcamUrl}
          />

          {/* Uncertainty Card */}
          {lastResult.uncertaintyDetail && (
            <UncertaintyCard
              uncertaintyDetail={lastResult.uncertaintyDetail}
              prediction={lastResult.prediction}
            />
          )}

          {/* Clinician-in-the-Loop Feedback (Novelty 2) */}
          <ClinicianFeedbackCard
            requestId={lastResult.requestId}
            predictedClass={lastResult.prediction.topClass}
          />
        </motion.div>

        {/* Right Column: Prediction, Abstention UI, Charts & Counterfactuals */}
        <div className="lg:col-span-6 space-y-6">
          {/* Primary Diagnosis Card (or Demoted Suggestion if Abstained) */}
          <motion.div
            variants={itemVariants}
            className={`p-6 rounded-3xl bg-slate-900 border ${
              shouldAbstain ? 'border-amber-500/60 bg-amber-950/20' : topClassInfo.colorBorder
            } shadow-2xl relative overflow-hidden`}
          >
            <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
                {shouldAbstain ? (
                  <>
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <span className="text-amber-300">Model Suggestion (Unverified / Refer for Review)</span>
                  </>
                ) : (
                  <span>Predicted Primary Diagnosis</span>
                )}
              </span>

              {/* Uncertainty Badge */}
              <div
                className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold border ${
                  shouldAbstain
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/50 animate-pulse'
                    : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50'
                }`}
              >
                <span>Uncertainty: {shouldAbstain ? 'HIGH / ABSTAIN' : lastResult.prediction.uncertainty.toUpperCase()}</span>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
              <h2 className={`text-2xl sm:text-3xl font-extrabold ${topClassInfo.colorText}`}>
                {topClassInfo.label}
              </h2>

              <div className="flex items-baseline space-x-1">
                <span className="text-4xl font-mono font-extrabold text-white">
                  {confidencePercent}
                </span>
                <span className="text-sm font-bold text-slate-400">% Confidence</span>
              </div>
            </div>

            <p className="text-xs sm:text-sm text-slate-300 mt-3 leading-relaxed">
              {topClassInfo.description}
            </p>
          </motion.div>

          {/* 4-Class Probability Bar Chart Card */}
          <motion.div variants={itemVariants} className="p-6 rounded-3xl bg-slate-900 border border-slate-800 shadow-xl">
            <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">4-Class Probability Distribution</h3>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">Softmax Normalized</span>
            </div>

            <ProbabilityChart probabilities={lastResult.prediction.probabilities} />

            {/* Accessibility table for screen readers */}
            <table className="sr-only">
              <caption>Pathology probability distribution</caption>
              <thead><tr><th>Class</th><th>Probability</th></tr></thead>
              <tbody>
                {Object.entries(lastResult.prediction.probabilities).map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td>{Math.round(v * 100)}%</td></tr>
                ))}
              </tbody>
            </table>
          </motion.div>

          {/* SHAP Feature Importance Card */}
          <motion.div variants={itemVariants} className="p-6 rounded-3xl bg-slate-900 border border-slate-800 shadow-xl">
            <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">SHAP Radiomics Feature Attributions</h3>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">Signed (+ green, - red)</span>
            </div>

            <ShapChart shapFeatures={lastResult.shap} />

            {/* Accessibility table for screen readers */}
            <table className="sr-only">
              <caption>Radiomics SHAP feature attributions</caption>
              <thead><tr><th>Feature</th><th>Weight</th></tr></thead>
              <tbody>
                {lastResult.shap.map((s) => (
                  <tr key={s.name}><td>{s.name}</td><td>{s.value}</td></tr>
                ))}
              </tbody>
            </table>
          </motion.div>

          {/* Novelty 3: Counterfactual Diagnostic Sensitivity Panel */}
          {lastResult.counterfactuals && lastResult.counterfactuals.length > 0 && (
            <motion.div variants={itemVariants}>
              <CounterfactualPanel
                counterfactuals={lastResult.counterfactuals}
                baseProbabilities={lastResult.prediction.probabilities}
                topClass={lastResult.prediction.topClass}
                requestId={lastResult.requestId}
              />
            </motion.div>
          )}

          {/* Fracture Risk Card */}
          <motion.div variants={itemVariants} className="p-6 rounded-3xl bg-slate-900 border border-slate-800 shadow-xl">
            <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <ShieldAlert className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">Fracture Vulnerability Index</h3>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">Biomechanical Model</span>
            </div>

            <FractureGauge fractureRisk={lastResult.fractureRisk} />
          </motion.div>
        </div>
      </div>

      {/* Mandatory Disclaimer Footer */}
      <motion.div
        variants={itemVariants}
        className="p-5 rounded-2xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-2"
      >
        <div className="flex items-center space-x-2 text-rose-400 font-bold">
          <ShieldAlert className="w-4 h-4" />
          <span>Clinical Prototype Advisory</span>
        </div>
        <p className="leading-relaxed">
          {lastResult.disclaimer}
        </p>
      </motion.div>

      {/* Traceable 3-Page PDF Report Modal */}
      <TraceableReportModal
        isOpen={isReportOpen}
        onClose={() => setIsReportOpen(false)}
        result={lastResult}
        engineMode={engineMode}
      />
    </motion.div>
  );
};
