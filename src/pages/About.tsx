import React from 'react';
import { motion } from 'framer-motion';
import {
  UploadCloud,
  Cpu,
  Layers,
  Sparkles,
  ShieldAlert,
  GitBranch,
  Bone,
  CheckCircle2,
  Terminal,
  Activity,
  Code2,
} from 'lucide-react';

const PIPELINE_STEPS = [
  {
    step: '01',
    title: 'Image Ingestion & Preprocessing',
    icon: UploadCloud,
    description:
      'Standardization of input DICOM / planar radiographs (CLAHE histogram equalization, bone window normalization, spatial resizing to 512×512, and intensity scaling).',
  },
  {
    step: '02',
    title: 'Dual-Stream CNN + Vision Transformer',
    icon: Cpu,
    description:
      'Parallel deep learning feature extraction: a convolutional backbone (ResNet-50) extracts local cortical edge and microtexture representations, while a Vision Transformer (ViT-B/16) captures long-range anatomical self-attention.',
  },
  {
    step: '03',
    title: 'Quantitative Radiomics Extraction',
    icon: Activity,
    description:
      'Extraction of 6 standardized clinical radiomics markers: GLCM Contrast, Cortical Shell Thickness, Texture Entropy, First-order Mean Intensity, GLRLM Long-Run Emphasis, and Trabecular Bone Volume Fraction (BV/TV).',
  },
  {
    step: '04',
    title: 'Multi-Modal Feature Fusion',
    icon: Layers,
    description:
      'Cross-attention fusion layer aligning spatial ViT patch embeddings, CNN hierarchical feature maps, and normalized radiomics vectors into a consolidated diagnostic latent space.',
  },
  {
    step: '05',
    title: '4-Class Pathology Classification',
    icon: Bone,
    description:
      'Dense classification head with temperature-scaled Softmax output predicting probability distributions across Bone Metastasis, Paget’s Disease, Osteogenesis Imperfecta (OI), and Healthy Bone, coupled with Monte Carlo Dropout uncertainty estimation.',
  },
  {
    step: '06',
    title: 'Explainable AI: Grad-CAM & SHAP',
    icon: Sparkles,
    description:
      'Gradient-weighted Class Activation Mapping (Grad-CAM) projects pixel-level attention heatmaps back onto the original radiograph. Kernel SHAP assigns signed directional attributions to each extracted radiomics feature.',
  },
  {
    step: '07',
    title: 'Biomechanical Fracture Risk Module',
    icon: ShieldAlert,
    description:
      'Integrates cortical thinning parameters, osteolytic lesion footprint, and baseline bone density to compute a calibrated 0–100% fracture vulnerability score.',
  },
];

export const About: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-12 py-6 sm:py-10 px-4">
      {/* Intro Header */}
      <section className="text-center space-y-4">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-xs font-semibold">
          <Layers className="w-3.5 h-3.5" />
          <span>Technical Architecture & ML Specifications</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          About OsteoGuard AI Pipeline
        </h1>
        <p className="text-sm sm:text-base text-slate-300 max-w-2xl mx-auto leading-relaxed">
          OsteoGuard AI combines deep spatial representation learning with explainable clinical radiomics to empower orthopedic oncologists and radiologists with transparent decision support.
        </p>
      </section>

      {/* Team Handoff & Architecture Seam Callout Box */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-3xl bg-slate-900 border-2 border-cyan-500/50 p-6 sm:p-8 shadow-2xl relative overflow-hidden"
      >
        <div className="flex items-start space-x-4">
          <div className="w-10 h-10 rounded-2xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0 border border-cyan-500/40">
            <GitBranch className="w-5 h-5" />
          </div>
          <div className="space-y-3">
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <span>Frontend + Simulation Layer Architecture Seam</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
                Team Handoff Contract
              </span>
            </h2>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
              <strong>This build represents the production-quality frontend and simulation layer.</strong> Real model weights, GPU training pipelines, and backend inference are implemented by our teammates in subsequent phases.
            </p>
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-cyan-300 space-y-1">
              <div className="text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                Integration Contract Location:
              </div>
              <div className="text-white">File: <span className="text-cyan-400">/src/api/client.ts</span> & <span className="text-cyan-400">/src/api/types.ts</span></div>
              <div className="text-slate-400">Swap Mock → Real: Set <code className="text-amber-400">VITE_API_BASE_URL=https://api.yourbackend.com</code></div>
            </div>
          </div>
        </div>
      </motion.section>

      {/* 7-Step Animated Architecture Timeline / Stepper */}
      <section className="space-y-6">
        <div className="border-b border-slate-800 pb-3">
          <h2 className="text-xl font-bold text-white">End-to-End Deep Learning Pipeline</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Step-by-step workflow from raw input radiograph to visual explainability
          </p>
        </div>

        <div className="space-y-4">
          {PIPELINE_STEPS.map((item, idx) => {
            const Icon = item.icon;
            return (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ duration: 0.5, delay: idx * 0.08 }}
                className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition flex items-start space-x-4 shadow-lg"
              >
                <div className="w-10 h-10 rounded-xl bg-slate-800 text-cyan-400 flex items-center justify-center shrink-0 border border-slate-700 font-bold text-xs">
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-mono text-cyan-400 font-bold">{item.step}</span>
                    <h3 className="text-sm font-bold text-white">{item.title}</h3>
                  </div>
                  <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Mandatory Disclaimer Box */}
      <section className="p-6 rounded-2xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-2">
        <div className="flex items-center space-x-2 text-rose-400 font-bold">
          <ShieldAlert className="w-4 h-4" />
          <span>Research Prototype Disclaimer</span>
        </div>
        <p className="leading-relaxed">
          OsteoGuard AI is an investigational research prototype for algorithm demonstration and computational validation. It is not approved by regulatory bodies (FDA, CE-IVD) and must never be utilized as the sole foundation for patient care or surgical management.
        </p>
      </section>
    </div>
  );
};
