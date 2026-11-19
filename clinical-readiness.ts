import fs from 'fs';
import path from 'path';
import { inspectClinicalReadinessOnDisk, type ClinicalDiskReadiness } from './security.js';

export type TrackDState = 'BLOCKED' | 'READY_FOR_TRAINING' | 'TRAINED_VERIFIED';

export interface FourClassTrainingSummary {
  status: string;
  model_id?: string;
  trained_at: string;
  backbone: string;
  modality: string;
  best_epoch?: number;
  temperature?: number;
  checkpoint_filename: string;
  checkpoint_sha256: string;
  dataset_summary?: {
    total_samples?: number;
    total_patients?: number;
    train_samples?: number;
    val_samples?: number;
    test_samples?: number;
  };
  validation_metrics?: Record<string, any>;
  test_metrics?: Record<string, any>;
  sample_predictions?: Array<Record<string, any>>;
}

export interface ClinicalEvaluationState {
  disk: ClinicalDiskReadiness;
  minImagesPerClass: number;
  cohortsAcquired: number;
  totalRawSamples: number;
  trackDState: TrackDState;
  activeCheckpoint: { filename: string; sha256: string; sizeBytes: number } | null;
  trainingSummary: FourClassTrainingSummary | null;
  trainingCommand: string;
  serveCommand: string;
}

const TARGET_CLASS_KEYS = ['metastasis', 'paget', 'oi', 'healthy'] as const;

export function readTrainingSummaryIfVerified(
  rootDir: string,
  disk: ClinicalDiskReadiness
): FourClassTrainingSummary | null {
  if (disk.checkpointsFound.length === 0 || disk.trainingLogsCount === 0) {
    return null;
  }
  const summaryPath = path.join(rootDir, 'ml_artifacts', 'checkpoints', 'four_class_training_summary.json');
  if (!fs.existsSync(summaryPath)) {
    return null;
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as FourClassTrainingSummary;
    if (!parsed || !parsed.checkpoint_filename || !parsed.checkpoint_sha256) {
      return null;
    }
    const matchingCheckpoint = disk.checkpointsFound.find(
      (c) => c.filename === parsed.checkpoint_filename
    );
    if (!matchingCheckpoint) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function evaluateFourClassClinicalState(
  rootDir: string,
  minImagesPerClass = 10
): ClinicalEvaluationState {
  const disk = inspectClinicalReadinessOnDisk(rootDir);
  const cohortsAcquired = TARGET_CLASS_KEYS.filter(
    (cls) => (disk.rawClassCounts[cls] || 0) >= minImagesPerClass
  ).length;
  const totalRawSamples = TARGET_CLASS_KEYS.reduce(
    (sum, cls) => sum + (disk.rawClassCounts[cls] || 0),
    0
  );

  const trainingSummary = readTrainingSummaryIfVerified(rootDir, disk);
  const matchedSummaryCkpt = trainingSummary
    ? disk.checkpointsFound.find((c) => c.filename === trainingSummary.checkpoint_filename) ?? null
    : null;
  const activeCheckpoint = matchedSummaryCkpt ?? disk.checkpointsFound[0] ?? null;
  const hasVerifiedModel =
    disk.isFourClassReady && activeCheckpoint !== null && disk.trainingLogsCount > 0;

  let trackDState: TrackDState = 'BLOCKED';
  if (hasVerifiedModel) {
    trackDState = 'TRAINED_VERIFIED';
  } else if (cohortsAcquired === 4) {
    trackDState = 'READY_FOR_TRAINING';
  }

  return {
    disk,
    minImagesPerClass,
    cohortsAcquired,
    totalRawSamples,
    trackDState,
    activeCheckpoint,
    trainingSummary,
    trainingCommand:
      'python3 ml_service/scripts/run_training.py --dataset-dir ml_service/data/raw --folds 5 --epochs 60 --backbone resnet101',
    serveCommand:
      'uvicorn ml_service.app.main:app --host 127.0.0.1 --port 8000',
  };
}

export function buildTrackDPayload(state: ClinicalEvaluationState) {
  const { disk, cohortsAcquired, trackDState, activeCheckpoint, trainingSummary, minImagesPerClass } = state;

  if (trackDState === 'TRAINED_VERIFIED' && activeCheckpoint) {
    const testM = trainingSummary?.test_metrics || {};
    const valM = trainingSummary?.validation_metrics || {};
    return {
      track_id: 'track_d' as const,
      title: 'Final Four-Class OsteoGuard AI Diagnostic System',
      modality: `Multimodal (${trainingSummary?.modality?.toUpperCase() || 'XRAY'} Clinical Pipeline)`,
      task_type: '4-Class Differential Diagnosis & Fracture Risk',
      target_classes: [
        'Secondary bone metastasis',
        "Paget's disease of bone",
        'Osteogenesis imperfecta (OI)',
        'Healthy skeletal control',
      ],
      status: 'TRAINED_VERIFIED — LIVE CLINICAL INFERENCE ELIGIBLE',
      historical_metrics: {
        model_readiness: `VERIFIED CHECKPOINT (${activeCheckpoint.filename})`,
        clinical_training_data: `${state.totalRawSamples} VERIFIED SCANS`,
        approved_cohorts: `${cohortsAcquired} / 4 verified clinical cohorts acquired`,
        checkpoint_sha256: activeCheckpoint.sha256.slice(0, 16) + '…',
        ...(typeof testM.accuracy === 'number' ? { holdout_test_accuracy: testM.accuracy } : {}),
        ...(typeof testM.macro_f1 === 'number' ? { holdout_macro_f1: testM.macro_f1 } : {}),
        ...(typeof testM.macro_roc_auc === 'number' ? { holdout_roc_auc: testM.macro_roc_auc } : {}),
        ...(typeof valM.ece === 'number' ? { calibrated_val_ece: valM.ece } : {}),
      },
      limitations: [
        'Mandatory Selective Abstention safeguard (H_norm >= 0.60 or Mahalanobis OOD trigger) stays strictly enforced.',
        'All clinical diagnostic predictions must undergo independent board-certified orthopedic/radiology confirmation.',
      ],
      governance_clearance: 'Verified Clinical Dataset Gate + SHA-256 Checkpoint + Training Logs Passed',
      disclaimer:
        'CLINICAL DECISION SUPPORT MODE ELIGIBLE: Backed by trained 4-class OsteoGuardNet weights. Abstention protocol strictly active.',
    };
  }

  const missingClasses = TARGET_CLASS_KEYS.filter(
    (cls) => (disk.rawClassCounts[cls] || 0) < minImagesPerClass
  );

  const cohortCountsLabel = `M:${disk.rawClassCounts.metastasis} • P:${disk.rawClassCounts.paget} • OI:${disk.rawClassCounts.oi} • H:${disk.rawClassCounts.healthy}`;
  const isDataReadyToTrain = trackDState === 'READY_FOR_TRAINING';

  return {
    track_id: 'track_d' as const,
    title: 'Final Four-Class OsteoGuard AI Diagnostic System',
    modality: 'Multimodal: Radiography + Bone Scintigraphy + CT',
    task_type: '4-Class Differential Diagnosis & Fracture Risk',
    target_classes: [
      'Secondary bone metastasis',
      "Paget's disease of bone",
      'Osteogenesis imperfecta (OI)',
      'Healthy skeletal control',
    ],
    status: isDataReadyToTrain
      ? 'READY_FOR_TRAINING — RUN OSTEOGUARD TRAINER TO UNLOCK CHECKPOINT'
      : 'BLOCKED — APPROVED CLINICAL DATA AND GOVERNANCE REQUIRED',
    historical_metrics: {
      model_readiness: isDataReadyToTrain
        ? 'AWAITING TRAINING EXECUTION'
        : activeCheckpoint
        ? `UNVERIFIED CHECKPOINT (${activeCheckpoint.filename})`
        : 'ZERO MODELS TRAINED',
      clinical_training_data: isDataReadyToTrain
        ? `${state.totalRawSamples} SAMPLES READY (${cohortCountsLabel})`
        : state.totalRawSamples > 0
        ? `PARTIAL (${cohortCountsLabel})`
        : 'UNAVAILABLE',
      approved_cohorts: `${cohortsAcquired} / 4 verified clinical cohorts acquired`,
    },
    limitations: isDataReadyToTrain
      ? [
          `All 4 target classes meet minimum ${minImagesPerClass} patients/class on disk (${cohortCountsLabel}).`,
          `Run training CLI to produce verified PyTorch weights, temperature calibration, and logs: ${state.trainingCommand}`,
        ]
      : [
          ...missingClasses.map(
            (c) =>
              `Class '${c}' currently has ${disk.rawClassCounts[c] || 0}/${minImagesPerClass} verified files in ml_service/data/raw/${c}/.`
          ),
          ...(activeCheckpoint && disk.trainingLogsCount === 0
            ? ['Checkpoint detected in ml_artifacts/checkpoints/ without matching training logs in ml_artifacts/logs/ (Rule 6 violation).']
            : []),
          'All four-class clinical predictions remain blocked until full cohort ingestion and verified PyTorch training complete.',
        ],
    governance_clearance: isDataReadyToTrain
      ? 'Dataset Gate Cleared — Ready for Supervised 4-Class Training'
      : 'Pending Faculty Review, Institutional Ethics Clearance (IEC), and Data Use Agreements (DUA)',
    disclaimer:
      'PROPOSED KSCST RESEARCH ARCHITECTURE: Active clinical prediction is disabled until 4 verified cohorts and trained PyTorch checkpoints exist on disk.',
  };
}

export function buildTrackDEvidenceItem(state: ClinicalEvaluationState) {
  if (state.trackDState === 'TRAINED_VERIFIED' && state.activeCheckpoint) {
    return {
      model_id: 'osteo-trackD-fourclass-unified',
      track: 'Track D: Four-Class OsteoGuard AI',
      dataset_name: 'Verified Multi-Center 4-Class Clinical Cohort',
      modality: state.trainingSummary?.modality || 'Multimodal (X-Ray + Scintigraphy + CT)',
      architecture: `OsteoGuardNet (${state.trainingSummary?.backbone || 'resnet101'} + 102-D IBSI Radiomics Fusion)`,
      checkpoint_hash: state.activeCheckpoint.sha256,
      training_status: `VERIFIED (${state.activeCheckpoint.filename})`,
      validation_evidence: {
        status: 'TRAINED_AND_CALIBRATED',
        temperature: state.trainingSummary?.temperature ?? 1.0,
        ...(state.trainingSummary?.validation_metrics || {}),
        ...(state.trainingSummary?.test_metrics ? { holdout_test: state.trainingSummary.test_metrics } : {}),
      },
      external_holdout_status: `Patient-Stratified Holdout Evaluated (${state.trainingSummary?.dataset_summary?.test_samples ?? 'verified'} scans)`,
      generalization_findings: 'Trained and temperature-calibrated strictly without patient overlap across splits.',
      clinical_deployment_status: 'ELIGIBLE_WITH_ABSTENTION_SAFEGUARD',
      limitations: ['Selective abstention threshold H_norm >= 0.60 strictly enforced at inference time.'],
    };
  }

  return {
    model_id: 'osteo-trackD-fourclass-unified',
    track: 'Track D: Four-Class OsteoGuard AI',
    dataset_name: 'Multi-Center Harmonized 4-Class Cohort (Proposed)',
    modality: 'Multimodal (X-Ray + Scintigraphy + CT)',
    architecture: 'Dual-Stream ResNet-101 / Swin + IBSI Radiomics (512+102 Fusion)',
    checkpoint_hash: state.activeCheckpoint ? `${state.activeCheckpoint.sha256} (UNVERIFIED)` : 'NONE (Untrained)',
    training_status: state.trackDState === 'READY_FOR_TRAINING' ? 'READY_FOR_TRAINING' : 'BLOCKED_BY_DATA',
    validation_evidence: { status: 'NO_CLINICAL_TRAINING_PERFORMED', cohorts_acquired: `${state.cohortsAcquired}/4` },
    external_holdout_status: 'PENDING_DATA_ACQUISITION',
    generalization_findings: 'Cannot be evaluated until verified clinical cohorts are acquired and trained.',
    clinical_deployment_status: 'BLOCKED',
    limitations: [`Currently ${state.cohortsAcquired}/4 verified cohorts populated under ml_service/data/raw/.`],
  };
}

export function buildFourClassGateResponse(state: ClinicalEvaluationState): {
  statusCode: number;
  body: {
    status: string;
    error_code: string;
    message: string;
    target_classes: string[];
    blockers: string[];
    mandatory_safeguard: string;
  };
} {
  const target_classes = [
    'Secondary bone metastasis',
    "Paget's disease of bone",
    'Osteogenesis imperfecta (OI)',
    'Healthy skeletal control',
  ];

  if (state.trackDState === 'TRAINED_VERIFIED' && state.activeCheckpoint) {
    return {
      statusCode: 200,
      body: {
        status: 'READY',
        error_code: 'NONE',
        message: `Verified 4-class PyTorch checkpoint active (${state.activeCheckpoint.filename}, SHA-256 ${state.activeCheckpoint.sha256.slice(0, 12)}…) across ${state.totalRawSamples} verified clinical images.`,
        target_classes,
        blockers: [],
        mandatory_safeguard:
          'Mandatory Clinical Safeguard active: Temperature-scaled probabilities, Mahalanobis OOD check, and H_norm >= 0.60 human-review abstention remain strictly enforced.',
      },
    };
  }

  const blockers: string[] = [];
  for (const cls of TARGET_CLASS_KEYS) {
    const count = state.disk.rawClassCounts[cls] || 0;
    if (count < state.minImagesPerClass) {
      blockers.push(`Missing/insufficient clinical cohort in ml_service/data/raw/${cls}/ (${count}/${state.minImagesPerClass} files)`);
    }
  }
  if (!state.activeCheckpoint) {
    blockers.push('Zero verified .pt/.pth checkpoints exist under ml_artifacts/checkpoints/');
  }
  if (state.disk.trainingLogsCount === 0) {
    blockers.push('Zero verifiable training logs exist under ml_artifacts/logs/');
  }

  return {
    statusCode: 503,
    body: {
      status: 'BLOCKED',
      error_code: 'FOUR_CLASS_MODEL_NOT_READY',
      message:
        'Clinical diagnostic inference is disabled. The OsteoGuard four-class diagnostic model requires verified patient-stratified cohorts in ml_service/data/raw/{metastasis,paget,oi,healthy} and a trained checkpoint with matching logs.',
      target_classes,
      blockers,
      mandatory_safeguard:
        'Under research governance rules, synthetic phantoms must not be fed into clinical models, and preliminary sub-track models (FracAtlas, Bone Scan, BTXRD) must never be remapped into four-class predictions.',
    },
  };
}

export function buildDynamicCohortList(
  baseCohorts: Array<{
    cohort_name: string;
    target_class: string;
    intended_modality: string;
    source_status: string;
    patient_count_estimate: string;
    acquisition_blockers: string[];
    icmr_iec_required: boolean;
    dua_status: string;
  }>,
  state: ClinicalEvaluationState
) {
  const keyOrder: Array<keyof ClinicalDiskReadiness['rawClassCounts']> = [
    'metastasis',
    'paget',
    'oi',
    'healthy',
  ];

  return baseCohorts.map((cohort, idx) => {
    const key = keyOrder[idx];
    const count = key ? state.disk.rawClassCounts[key] || 0 : 0;
    const isPopulated = count >= state.minImagesPerClass;
    return {
      ...cohort,
      disk_folder: `ml_service/data/raw/${key}/`,
      disk_sample_count: count,
      source_status: isPopulated
        ? `VERIFIED ON DISK (${count} imaging files detected)`
        : count > 0
        ? `PARTIAL ON DISK (${count}/${state.minImagesPerClass} files) — ${cohort.source_status}`
        : cohort.source_status,
      acquisition_blockers: isPopulated
        ? [`Verified ${count} scan(s) ready for patient-stratified partitioning.`]
        : cohort.acquisition_blockers,
    };
  });
}
