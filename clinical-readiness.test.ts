import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  evaluateFourClassClinicalState,
  buildTrackDPayload,
  buildFourClassGateResponse,
  buildDynamicCohortList,
} from './clinical-readiness.js';

function withTempWorkspace(fn: (rootDir: string) => void) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'osteoguard-readiness-'));
  try {
    fn(tmp);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

function populateClassImages(rootDir: string, cls: string, count: number) {
  const dir = path.join(rootDir, 'ml_service', 'data', 'raw', cls);
  fs.mkdirSync(dir, { recursive: true });
  for (let i = 0; i < count; i++) {
    fs.writeFileSync(path.join(dir, `PT${i.toString().padStart(3, '0')}_01.png`), 'fake-png-bytes');
  }
}

test('empty workspace stays strictly BLOCKED with 0/4 cohorts and HTTP 503 gate', () => {
  withTempWorkspace((rootDir) => {
    const state = evaluateFourClassClinicalState(rootDir);
    assert.equal(state.cohortsAcquired, 0);
    assert.equal(state.totalRawSamples, 0);
    assert.equal(state.trackDState, 'BLOCKED');
    assert.equal(state.activeCheckpoint, null);

    const trackD = buildTrackDPayload(state);
    assert.match(trackD.status, /^BLOCKED/);
    assert.equal(trackD.historical_metrics.approved_cohorts, '0 / 4 verified clinical cohorts acquired');

    const gate = buildFourClassGateResponse(state);
    assert.equal(gate.statusCode, 503);
    assert.equal(gate.body.status, 'BLOCKED');
    assert.ok(gate.body.blockers.length >= 4);
  });
});

test('partial cohorts accurately count acquired classes and stay BLOCKED', () => {
  withTempWorkspace((rootDir) => {
    populateClassImages(rootDir, 'metastasis', 12);
    populateClassImages(rootDir, 'healthy', 10);
    populateClassImages(rootDir, 'paget', 3); // below minimum 10

    const state = evaluateFourClassClinicalState(rootDir);
    assert.equal(state.cohortsAcquired, 2);
    assert.equal(state.totalRawSamples, 25);
    assert.equal(state.trackDState, 'BLOCKED');

    const trackD = buildTrackDPayload(state);
    assert.equal(trackD.historical_metrics.approved_cohorts, '2 / 4 verified clinical cohorts acquired');

    const gate = buildFourClassGateResponse(state);
    assert.equal(gate.statusCode, 503);
  });
});

test('all 4 cohorts without checkpoint transition to READY_FOR_TRAINING (HTTP 503 until trained)', () => {
  withTempWorkspace((rootDir) => {
    for (const cls of ['metastasis', 'paget', 'oi', 'healthy']) {
      populateClassImages(rootDir, cls, 10);
    }

    const state = evaluateFourClassClinicalState(rootDir);
    assert.equal(state.cohortsAcquired, 4);
    assert.equal(state.trackDState, 'READY_FOR_TRAINING');

    const trackD = buildTrackDPayload(state);
    assert.match(trackD.status, /^READY_FOR_TRAINING/);

    const gate = buildFourClassGateResponse(state);
    assert.equal(gate.statusCode, 503);
  });
});

test('checkpoint without training logs violates Rule 6 and stays BLOCKED', () => {
  withTempWorkspace((rootDir) => {
    for (const cls of ['metastasis', 'paget', 'oi', 'healthy']) {
      populateClassImages(rootDir, cls, 10);
    }
    const ckptDir = path.join(rootDir, 'ml_artifacts', 'checkpoints');
    fs.mkdirSync(ckptDir, { recursive: true });
    fs.writeFileSync(path.join(ckptDir, 'osteoguard_4class_best.pt'), 'weights');

    const state = evaluateFourClassClinicalState(rootDir);
    assert.equal(state.disk.isFourClassReady, false);
    assert.notEqual(state.trackDState, 'TRAINED_VERIFIED');
    assert.equal(buildFourClassGateResponse(state).statusCode, 503);
  });
});

test('4 verified cohorts + checkpoint + logs + training summary unlock TRAINED_VERIFIED & HTTP 200 READY', () => {
  withTempWorkspace((rootDir) => {
    for (const cls of ['metastasis', 'paget', 'oi', 'healthy']) {
      populateClassImages(rootDir, cls, 12);
    }
    const ckptDir = path.join(rootDir, 'ml_artifacts', 'checkpoints');
    const logsDir = path.join(rootDir, 'ml_artifacts', 'logs');
    fs.mkdirSync(ckptDir, { recursive: true });
    fs.mkdirSync(logsDir, { recursive: true });

    const ckptBytes = Buffer.from('verified-checkpoint-weights');
    fs.writeFileSync(path.join(ckptDir, 'osteoguard_4class_best.pt'), ckptBytes);
    fs.writeFileSync(path.join(logsDir, 'osteoguard_4class_train.log'), 'Epoch 1 val_macro_f1=0.88');
    fs.writeFileSync(
      path.join(ckptDir, 'four_class_training_summary.json'),
      JSON.stringify({
        status: 'TRAINED_VERIFIED',
        trained_at: '2026-11-18T00:00:00Z',
        backbone: 'resnet101',
        modality: 'xray',
        temperature: 1.18,
        checkpoint_filename: 'osteoguard_4class_best.pt',
        checkpoint_sha256: 'ABCD1234',
        validation_metrics: { ece: 0.032, macro_f1: 0.89 },
        test_metrics: { accuracy: 0.91, macro_f1: 0.90, macro_roc_auc: 0.96 },
      })
    );

    const state = evaluateFourClassClinicalState(rootDir);
    assert.equal(state.cohortsAcquired, 4);
    assert.equal(state.trackDState, 'TRAINED_VERIFIED');
    assert.equal(state.trainingSummary?.temperature, 1.18);

    const trackD = buildTrackDPayload(state);
    assert.match(trackD.status, /^TRAINED_VERIFIED/);
    assert.equal(trackD.historical_metrics.holdout_macro_f1, 0.9);
    assert.equal(trackD.historical_metrics.calibrated_val_ece, 0.032);

    const gate = buildFourClassGateResponse(state);
    assert.equal(gate.statusCode, 200);
    assert.equal(gate.body.status, 'READY');
    assert.equal(gate.body.blockers.length, 0);

    const enriched = buildDynamicCohortList(
      [
        {
          cohort_name: 'M',
          target_class: 'm',
          intended_modality: 'X-Ray',
          source_status: 'BLOCKED',
          patient_count_estimate: '10',
          acquisition_blockers: ['missing'],
          icmr_iec_required: true,
          dua_status: 'DUA',
        },
      ],
      state
    );
    assert.match(enriched[0].source_status, /^VERIFIED ON DISK \(12/);
  });
});

test('multi-fold .pth checkpoints resolve exact summary checkpoint_filename', () => {
  withTempWorkspace((rootDir) => {
    for (const cls of ['metastasis', 'paget', 'oi', 'healthy']) {
      populateClassImages(rootDir, cls, 10);
    }
    const ckptDir = path.join(rootDir, 'ml_artifacts', 'checkpoints');
    const logsDir = path.join(rootDir, 'ml_artifacts', 'logs');
    fs.mkdirSync(ckptDir, { recursive: true });
    fs.mkdirSync(logsDir, { recursive: true });

    fs.writeFileSync(path.join(ckptDir, 'osteoguard_4class_fold_0.pth'), 'fold-0-weights');
    fs.writeFileSync(path.join(ckptDir, 'osteoguard_4class_best.pth'), 'best-production-weights');
    fs.writeFileSync(path.join(logsDir, 'osteoguard_4class_kfold_train.log'), 'Fold 0 Macro-F1=0.92');
    fs.writeFileSync(
      path.join(ckptDir, 'four_class_training_summary.json'),
      JSON.stringify({
        status: 'TRAINED_VERIFIED',
        trained_at: '2026-11-18T00:00:00Z',
        backbone: 'resnet101',
        modality: 'xray',
        temperature: 1.07,
        checkpoint_filename: 'osteoguard_4class_best.pth',
        checkpoint_sha256: 'DEADBEEF',
        validation_metrics: { ece: 0.025, macro_f1: 0.92 },
        test_metrics: { accuracy: 0.93, macro_f1: 0.92, macro_roc_auc: 0.97 },
      })
    );

    const state = evaluateFourClassClinicalState(rootDir);
    assert.equal(state.trackDState, 'TRAINED_VERIFIED');
    assert.equal(state.activeCheckpoint?.filename, 'osteoguard_4class_best.pth');
    assert.match(state.trainingCommand, /ml_service\/scripts\/run_training\.py/);
    assert.equal(buildFourClassGateResponse(state).statusCode, 200);
  });
});

