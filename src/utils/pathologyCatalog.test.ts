import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  evaluateAbstentionAdvisory,
  getModalitySpec,
  getPathologyProfile,
  REFERENCE_SCAN_PRESETS,
  sortProbabilitiesDescending,
} from './pathologyCatalog.js';

describe('pathologyCatalog pure helpers', () => {
  it('returns distinct pathology profiles for all four bone disorder classes', () => {
    const metastasis = getPathologyProfile('metastasis');
    const paget = getPathologyProfile('paget');
    const oi = getPathologyProfile('oi');
    const healthy = getPathologyProfile('healthy');

    assert.equal(metastasis.code, 'SECONDARY_BONE_METASTASIS');
    assert.match(paget.title, /Paget/i);
    assert.match(oi.title, /Osteogenesis Imperfecta/i);
    assert.equal(healthy.accentTone, 'emerald');
  });

  it('supports xray, ct, mri, and bone_scan modalities cleanly', () => {
    assert.match(getModalitySpec('xray').label, /X-ray/i);
    assert.match(getModalitySpec('ct').label, /CT Scan/i);
    assert.match(getModalitySpec('mri').label, /MRI/i);
    assert.match(getModalitySpec('bone_scan').label, /Scintigraphy/i);
  });

  it('includes reference scan presets covering all 4 classes across X-ray and CT', () => {
    const coveredClasses = new Set(REFERENCE_SCAN_PRESETS.map((preset) => preset.sampleCase));
    const coveredModalities = new Set(REFERENCE_SCAN_PRESETS.map((preset) => preset.modality));
    assert.deepEqual(coveredClasses, new Set(['paget', 'metastasis', 'oi', 'healthy']));
    assert.ok(coveredModalities.has('xray'));
    assert.ok(coveredModalities.has('ct'));
  });

  it('evaluates entropy and OOD abstention advisories accurately', () => {
    const validProbs = { metastasis: 0.75, paget: 0.15, oi: 0.05, healthy: 0.05 };
    assert.deepEqual(evaluateAbstentionAdvisory(0.32, false, validProbs), {
      hasAdvisory: false,
      reason: 'none',
      validProbabilities: true,
    });
    assert.equal(evaluateAbstentionAdvisory(0.72, false, validProbs).reason, 'entropy');
    assert.equal(evaluateAbstentionAdvisory(0.25, true, validProbs).reason, 'ood');
    assert.equal(evaluateAbstentionAdvisory(0.75, true, validProbs).reason, 'both');
    assert.equal(
      evaluateAbstentionAdvisory(0.2, false, { metastasis: NaN, paget: 0, oi: 0, healthy: 1 }).reason,
      'invalid_probs',
    );
  });

  it('sorts class probabilities from highest to lowest', () => {
    const sorted = sortProbabilitiesDescending({
      metastasis: 0.12,
      paget: 0.68,
      oi: 0.15,
      healthy: 0.05,
    });
    assert.deepEqual(
      sorted.map((item) => item.id),
      ['paget', 'oi', 'metastasis', 'healthy'],
    );
  });
});
