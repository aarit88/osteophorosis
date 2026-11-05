import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

export const resources = {
  en: {
    translation: {
      appName: 'OsteoGuard AI',
      prototypeWarning: 'RESEARCH PROTOTYPE — SIMULATED OUTPUT — NOT FOR CLINICAL USE',
      prototypeSubtext: 'Investigational validation & engineering demo only',
      nav: {
        analysis: 'Analysis & Upload',
        results: 'Latest Results',
        about: 'Architecture & Model Info',
        audit: 'Clinical Audit (Art. 14)',
        engine: 'Engine',
        recentAnalyses: 'Recent Analyses',
      },
      hero: {
        badge: 'Explainable Deep Learning + Radiomics Inference',
        title: 'Instant Bone Pathology Classification & XAI Attention',
        subtitle: 'OsteoGuard AI fuses a dual-stream CNN + Vision Transformer with quantitative radiomics features to differentiate Metastasis, Paget\'s Disease, Osteogenesis Imperfecta, and Healthy Bone in seconds.',
        runButton: 'Run OsteoGuard Analysis',
        loadDemo: 'Load Demo Image',
      },
      upload: {
        selectModality: 'Select Radiographic Modality',
        dropzoneTitle: 'Drop your bone radiograph here, or browse files',
        dropzoneSubtitle: 'Supports X-ray, CT slice, or Bone Scintigraphy (PNG, JPEG up to 15MB).',
        ready: 'Ready For Analysis',
        changeFile: 'Change File',
      },
      results: {
        predictedDiagnosis: 'Predicted Primary Diagnosis',
        confidence: 'Confidence',
        runAnother: 'Run Another Image',
        downloadReport: 'Download Traceable Report (PDF)',
        modelConfidence: 'Model Confidence & Entropy',
        abstentionNotice: 'Model recommends human review — prediction withheld from primary display',
        suggestionOnly: 'Model Suggestion (Unverified / Review Required)',
        counterfactualTitle: 'Counterfactual Sensitivity: What would change the diagnosis?',
        clinicianReview: 'Clinician Verification & Feedback (EU AI Act Art. 14)',
      },
    },
  },
  hi: {
    translation: {
      appName: 'OsteoGuard AI',
      prototypeWarning: 'RESEARCH PROTOTYPE — SIMULATED OUTPUT — NOT FOR CLINICAL USE',
      prototypeSubtext: 'Investigational validation & engineering demo only',
      nav: {
        analysis: 'Analysis & Upload',
        results: 'Latest Results',
        about: 'Architecture & Model Info',
        audit: 'Clinical Audit (Art. 14)',
        engine: 'Engine',
        recentAnalyses: 'Recent Analyses',
      },
      hero: {
        badge: 'Explainable Deep Learning + Radiomics Inference',
        title: 'Instant Bone Pathology Classification & XAI Attention',
        subtitle: 'OsteoGuard AI fuses a dual-stream CNN + Vision Transformer with quantitative radiomics features to differentiate Metastasis, Paget\'s Disease, Osteogenesis Imperfecta, and Healthy Bone in seconds.',
        runButton: 'Run OsteoGuard Analysis',
        loadDemo: 'Load Demo Image',
      },
      upload: {
        selectModality: 'Select Radiographic Modality',
        dropzoneTitle: 'Drop your bone radiograph here, or browse files',
        dropzoneSubtitle: 'Supports X-ray, CT slice, or Bone Scintigraphy (PNG, JPEG up to 15MB).',
        ready: 'Ready For Analysis',
        changeFile: 'Change File',
      },
      results: {
        predictedDiagnosis: 'Predicted Primary Diagnosis',
        confidence: 'Confidence',
        runAnother: 'Run Another Image',
        downloadReport: 'Download Traceable Report (PDF)',
        modelConfidence: 'Model Confidence & Entropy',
        abstentionNotice: 'Model recommends human review — prediction withheld from primary display',
        suggestionOnly: 'Model Suggestion (Unverified / Review Required)',
        counterfactualTitle: 'Counterfactual Sensitivity: What would change the diagnosis?',
        clinicianReview: 'Clinician Verification & Feedback (EU AI Act Art. 14)',
      },
    },
  },
};

i18n.use(initReactI18next).init({
  resources,
  lng: 'en',
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
});

export default i18n;
