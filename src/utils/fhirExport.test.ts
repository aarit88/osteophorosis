import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildFhirDiagnosticReport } from './fhirExport.ts';
import type { AnalyzeResponse } from '../api/types';

describe('FHIR R4 DiagnosticReport builder', () => {
  it('builds a valid preliminary HL7 FHIR R4 Bundle with SNOMED CT and Observations', () => {
    const sample: AnalyzeResponse = {
      requestId: 'req_fhir_test_1',
      modality: 'ct',
      timestamp: '2026-11-26T12:00:00Z',
      imageUrl: '/api/v1/ephemeral/scan.png',
      gradcamUrl: '/api/v1/ephemeral/cam.png',
      prediction: {
        topClass: 'metastasis',
        confidence: 0.89,
        probabilities: { metastasis: 0.89, paget: 0.06, oi: 0.03, healthy: 0.02 },
        uncertainty: 'low',
      },
      uncertaintyDetail: {
        entropy: 0.317,
        confidenceInterval: [0.8, 0.99],
        oodFlag: false,
        calibrationBin: { predicted: 0.89, observed: 0.87 },
      },
      fractureRisk: {
        percent: 77,
        level: 'high',
      },
      shap: [
        { name: 'First-order Mean Intensity', value: 0.48 },
        { name: 'Cortical Thickness', value: -0.39 },
      ],
      counterfactuals: [],
      disclaimer: 'RESEARCH PROTOTYPE — Simulated evaluation output.',
    };

    const bundle = buildFhirDiagnosticReport(sample);
    assert.equal(bundle.resourceType, 'Bundle');
    assert.equal(bundle.entry.length, 4);

    const report = bundle.entry[0].resource as Record<string, any>;
    assert.equal(report.resourceType, 'DiagnosticReport');
    assert.equal(report.status, 'preliminary');
    assert.equal(report.conclusionCode[0].coding[0].code, '94222008');
    assert.match(report.conclusion, /RESEARCH PROTOTYPE/);
  });
});
