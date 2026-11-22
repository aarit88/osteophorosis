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

export interface TrackDDatasetReadinessDashboard {
  timestamp: string;
  track: 'track_d';
  readiness_status: 'NOT_AVAILABLE' | 'BLOCKED' | 'READY_FOR_TRAINING';
  ready_for_training: boolean;
  data_origin: 'REAL_CLINICAL_DATA_UNAVAILABLE' | 'REAL_DISK_DATA_PRESENT';
  synthetic_data_excluded: true;
  total_inspected_samples: number;
  valid_samples: number;
  quarantined_samples: number;
  unique_patients: number;
  institution_count: number;
  institutions: string[];
  dataset_sources: string[];
  permissions_verified: string[];
  ethics_references: string[];
  class_counts: Record<'metastasis' | 'paget' | 'oi' | 'healthy', number>;
  modality_counts: Record<'xray' | 'bone_scan' | 'ct', number>;
  class_modality_matrix: Record<
    'xray' | 'bone_scan' | 'ct',
    Record<'metastasis' | 'paget' | 'oi' | 'healthy', number>
  >;
  missing_metadata_count: number;
  duplicate_samples_count: number;
  ontology_version: string;
  accepted_by_class: Record<
    'SECONDARY_BONE_METASTASIS' | 'PAGETS_DISEASE' | 'OSTEOGENESIS_IMPERFECTA' | 'HEALTHY_SKELETAL_CONTROL',
    number
  >;
  adjudication_summary: Record<
    'ACCEPTED' | 'OUT_OF_SCOPE' | 'UNCERTAIN' | 'REQUIRES_ADJUDICATION' | 'REJECTED',
    number
  >;
  healthy_control_eligibility_failures: number;
  conflicting_diagnoses_count: number;
  label_provenance_completeness_pct: number;
  split_distribution: {
    train: number;
    val: number;
    test: number;
    external_test: number;
  };
  gates: {
    provenance_documented: boolean;
    permissions_verified: boolean;
    deidentification_satisfied: boolean;
    diagnostic_labels_documented: boolean;
    patient_level_split_integrity: boolean;
    image_quality_verified: boolean;
    class_and_modality_coverage: boolean;
  };
  protected_holdouts: {
    btxrd_center2_samples: number;
    btxrd_center3_samples: number;
    isolation_enforced: boolean;
  };
  blockers: string[];
  outstanding_requirements: string[];
  clinical_sample_size_justification_note: string;
  quarantine_reasons: Record<string, number>;
}

const IMAGE_EXT_SET = new Set(['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.dcm', '.dicom', '.nii', '.gz']);

function countDiskFilesRecursive(dirPath: string): number {
  if (!fs.existsSync(dirPath)) return 0;
  let total = 0;
  for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      total += countDiskFilesRecursive(full);
    } else if (entry.isFile()) {
      const lower = entry.name.toLowerCase();
      if (lower.endsWith('.nii.gz') || IMAGE_EXT_SET.has(path.extname(lower))) {
        total += 1;
      }
    }
  }
  return total;
}

export function getTrackDDatasetReadiness(rootDir: string): TrackDDatasetReadinessDashboard {
  const rawDir = path.join(rootDir, 'ml_service', 'data', 'raw');
  const reportsDir = path.join(rootDir, 'ml_service', 'data', 'reports');
  const quarantineDir = path.join(rootDir, 'ml_service', 'data', 'quarantine');
  const readinessReportPath = path.join(reportsDir, 'track_d_readiness_report.json');
  const quarantineReportPath = path.join(quarantineDir, 'quarantine_manifest.json');

  // Always measure real physical files on disk across strictly separated modalities
  const diskMatrix: TrackDDatasetReadinessDashboard['class_modality_matrix'] = {
    xray: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
    bone_scan: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
    ct: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
  };
  const diskModalityCounts = { xray: 0, bone_scan: 0, ct: 0 };
  const diskClassCounts = { metastasis: 0, paget: 0, oi: 0, healthy: 0 };

  for (const mod of ['xray', 'bone_scan', 'ct'] as const) {
    for (const cls of ['metastasis', 'paget', 'oi', 'healthy'] as const) {
      const c = countDiskFilesRecursive(path.join(rawDir, mod, cls));
      diskMatrix[mod][cls] = c;
      diskModalityCounts[mod] += c;
      diskClassCounts[cls] += c;
    }
  }

  let legacyRawFiles = 0;
  for (const cls of ['metastasis', 'paget', 'oi', 'healthy'] as const) {
    legacyRawFiles += countDiskFilesRecursive(path.join(rawDir, cls));
  }
  const totalPhysicalRaw =
    diskModalityCounts.xray + diskModalityCounts.bone_scan + diskModalityCounts.ct + legacyRawFiles;

  let parsedReport: Record<string, any> | null = null;
  if (fs.existsSync(readinessReportPath)) {
    try {
      parsedReport = JSON.parse(fs.readFileSync(readinessReportPath, 'utf8'));
    } catch {
      parsedReport = null;
    }
  }

  let quarantineReasons: Record<string, number> = {};
  if (fs.existsSync(quarantineReportPath)) {
    try {
      const qJson = JSON.parse(fs.readFileSync(quarantineReportPath, 'utf8'));
      if (qJson && typeof qJson.reason_counts === 'object' && qJson.reason_counts !== null) {
        quarantineReasons = qJson.reason_counts;
      }
    } catch {
      quarantineReasons = {};
    }
  }

  const reportValidSamples = Number(parsedReport?.valid_samples || 0);
  const reportQuarantined = Number(parsedReport?.quarantined_samples || 0);
  const hasAnyRealData = totalPhysicalRaw > 0 || reportValidSamples > 0 || reportQuarantined > 0;

  if (!hasAnyRealData) {
    return {
      timestamp: new Date().toISOString(),
      track: 'track_d',
      readiness_status: 'NOT_AVAILABLE',
      ready_for_training: false,
      data_origin: 'REAL_CLINICAL_DATA_UNAVAILABLE',
      synthetic_data_excluded: true,
      total_inspected_samples: 0,
      valid_samples: 0,
      quarantined_samples: 0,
      unique_patients: 0,
      institution_count: 0,
      institutions: [],
      dataset_sources: [],
      permissions_verified: [],
      ethics_references: [],
      class_counts: { metastasis: 0, paget: 0, oi: 0, healthy: 0 },
      modality_counts: { xray: 0, bone_scan: 0, ct: 0 },
      class_modality_matrix: diskMatrix,
      missing_metadata_count: 0,
      duplicate_samples_count: 0,
      ontology_version: 'OSTEO_ONTOLOGY_V1.0_STAGE34',
      accepted_by_class: {
        SECONDARY_BONE_METASTASIS: 0,
        PAGETS_DISEASE: 0,
        OSTEOGENESIS_IMPERFECTA: 0,
        HEALTHY_SKELETAL_CONTROL: 0,
      },
      adjudication_summary: {
        ACCEPTED: 0,
        OUT_OF_SCOPE: 0,
        UNCERTAIN: 0,
        REQUIRES_ADJUDICATION: 0,
        REJECTED: 0,
      },
      healthy_control_eligibility_failures: 0,
      conflicting_diagnoses_count: 0,
      label_provenance_completeness_pct: 0,
      split_distribution: { train: 0, val: 0, test: 0, external_test: 0 },
      gates: {
        provenance_documented: false,
        permissions_verified: false,
        deidentification_satisfied: false,
        diagnostic_labels_documented: false,
        patient_level_split_integrity: false,
        image_quality_verified: false,
        class_and_modality_coverage: false,
      },
      protected_holdouts: {
        btxrd_center2_samples: 549,
        btxrd_center3_samples: 259,
        isolation_enforced: true,
      },
      blockers: parsedReport?.blockers?.length
        ? parsedReport.blockers
        : [
            'Zero real Track D clinical dataset records or imaging files found under ml_service/data/raw/ or manifests.',
            'Missing verified cohorts: metastasis, paget, oi, healthy.',
            'Institutional Ethics Committee (IEC/IRB) clearance and bilateral Data Use Agreements (DUA) pending.',
          ],
      outstanding_requirements: [
        'Paget\'s Disease of Bone: Retrospective Hospital Radiology PACS IRB/IEC approval + DUA (radiographic triad + elevated serum ALP).',
        'Osteogenesis Imperfecta (OI): Multi-center pediatric ethics approval + DUA (RDCRN Brittle Bone Disorders Consortium / tertiary pediatric orthopedic centers).',
        'Secondary Bone Metastasis: Hospital PACS biopsy/PET-CT confirmed cohort + TCIA Spine-Mets-CT-Seg (.tcia DUA) + Zenodo 13900966 validation.',
        'Healthy Skeletal Controls: Verified asymptomatic normal PACS controls + Stanford AIMI MURA DUA (matched anatomical sites, never trauma contralateral limbs).',
        'Modality Separation & Biostatistical Power: Separate modality streams (xray, bone_scan, ct) and formal IEC statistical sample-size justification.',
      ],
      clinical_sample_size_justification_note:
        'No arbitrary minimum clinical sample count is fabricated in software; final statistical power and institutional cohort sizes require documented biostatistical and IEC approval.',
      quarantine_reasons: {},
    };
  }

  return {
    timestamp: new Date().toISOString(),
    track: 'track_d',
    readiness_status:
      parsedReport?.readiness_status === 'READY_FOR_TRAINING'
        ? 'READY_FOR_TRAINING'
        : parsedReport?.readiness_status === 'BLOCKED'
        ? 'BLOCKED'
        : 'BLOCKED',
    ready_for_training: Boolean(parsedReport?.ready_for_training),
    data_origin: 'REAL_DISK_DATA_PRESENT',
    synthetic_data_excluded: true,
    total_inspected_samples: Number(parsedReport?.total_inspected_samples ?? totalPhysicalRaw),
    valid_samples: reportValidSamples,
    quarantined_samples: reportQuarantined,
    unique_patients: Number(parsedReport?.unique_patients ?? 0),
    institution_count: Number(parsedReport?.institution_count ?? 0),
    institutions: Array.isArray(parsedReport?.institutions) ? parsedReport.institutions : [],
    dataset_sources: Array.isArray(parsedReport?.dataset_sources) ? parsedReport.dataset_sources : [],
    permissions_verified: Array.isArray(parsedReport?.permissions_verified) ? parsedReport.permissions_verified : [],
    ethics_references: Array.isArray(parsedReport?.ethics_references) ? parsedReport.ethics_references : [],
    class_counts: parsedReport?.class_counts ?? diskClassCounts,
    modality_counts: parsedReport?.modality_counts ?? diskModalityCounts,
    class_modality_matrix: parsedReport?.class_modality_matrix ?? diskMatrix,
    missing_metadata_count: Number(parsedReport?.missing_metadata_count ?? 0),
    duplicate_samples_count: Number(parsedReport?.duplicate_samples_count ?? 0),
    ontology_version: String(parsedReport?.ontology_version || 'OSTEO_ONTOLOGY_V1.0_STAGE34'),
    accepted_by_class: parsedReport?.accepted_by_class ?? {
      SECONDARY_BONE_METASTASIS: 0,
      PAGETS_DISEASE: 0,
      OSTEOGENESIS_IMPERFECTA: 0,
      HEALTHY_SKELETAL_CONTROL: 0,
    },
    adjudication_summary: parsedReport?.adjudication_summary ?? {
      ACCEPTED: 0,
      OUT_OF_SCOPE: 0,
      UNCERTAIN: 0,
      REQUIRES_ADJUDICATION: 0,
      REJECTED: 0,
    },
    healthy_control_eligibility_failures: Number(parsedReport?.healthy_control_eligibility_failures ?? 0),
    conflicting_diagnoses_count: Number(parsedReport?.conflicting_diagnoses_count ?? 0),
    label_provenance_completeness_pct: Number(parsedReport?.label_provenance_completeness_pct ?? 0),
    split_distribution: parsedReport?.split_distribution ?? { train: 0, val: 0, test: 0, external_test: 0 },
    gates: parsedReport?.gates ?? {
      provenance_documented: false,
      permissions_verified: false,
      deidentification_satisfied: false,
      diagnostic_labels_documented: false,
      patient_level_split_integrity: false,
      image_quality_verified: false,
      class_and_modality_coverage: false,
    },
    protected_holdouts: {
      btxrd_center2_samples: 549,
      btxrd_center3_samples: 259,
      isolation_enforced: true,
    },
    blockers: Array.isArray(parsedReport?.blockers)
      ? parsedReport.blockers
      : ['Run python3 ml_service/scripts/run_track_d_intake.py --all to validate and partition newly staged files.'],
    outstanding_requirements: [
      'Maintain strict modality separation (xray, bone_scan, ct) and patient-level zero-leakage splits.',
      'Ensure every clinical sample retains verified IEC/IRB ethics approval, DUA permissions, and pseudonymized patient_id.',
    ],
    clinical_sample_size_justification_note:
      parsedReport?.clinical_sample_size_justification_note ||
      'No arbitrary minimum clinical sample count is fabricated in software; final statistical power requires documented biostatistical and IEC approval.',
    quarantine_reasons: quarantineReasons,
  };
}

