import { describe, expect, it } from 'vitest';
import { isStrongProductionSecret, isSupportedUploadMediaType, resolveCorsOrigins } from '../security-policy';
import { summarizeClinicalReadiness } from '../security';

describe('security policy', () => {
  it('requires explicit non-wildcard CORS origins in production', () => {
    expect(resolveCorsOrigins('https://clinic.example', true)).toEqual(['https://clinic.example']);
    expect(() => resolveCorsOrigins(undefined, true)).toThrow(/explicit/);
    expect(() => resolveCorsOrigins('*', true)).toThrow(/wildcard/);
    expect(() => resolveCorsOrigins('http://clinic.example', true)).toThrow(/HTTPS/);
    expect(() => resolveCorsOrigins('https://clinic.example/path', true)).toThrow(/exact/);
  });

  it('accepts only sufficiently strong production secrets', () => {
    expect(isStrongProductionSecret('x'.repeat(32))).toBe(true);
    expect(isStrongProductionSecret('short')).toBe(false);
    expect(isStrongProductionSecret(undefined)).toBe(false);
  });

  it('restricts upload media types to supported raster images', () => {
    expect(isSupportedUploadMediaType('image/png')).toBe(true);
    expect(isSupportedUploadMediaType('IMAGE/JPEG')).toBe(true);
    expect(isSupportedUploadMediaType('application/pdf')).toBe(false);
    expect(isSupportedUploadMediaType('application/dicom')).toBe(false);
  });

  it('redacts checkpoint identifiers, hashes, paths, and cohort counts in the public readiness summary', () => {
    const summary = summarizeClinicalReadiness({
      checkpointsDir: '/private/checkpoints',
      checkpointsFound: [{ filename: 'release.pt', sizeBytes: 1024, sha256: 'a'.repeat(64) }],
      trainingLogsCount: 2,
      rawDataDir: '/private/patient-data',
      rawClassCounts: { metastasis: 11, paget: 12, oi: 13, healthy: 14 },
      isFourClassReady: false,
    });
    expect(summary).toEqual({
      checkpointCount: 1,
      trainingLogsAvailable: true,
      clinicalDatasetAvailable: true,
      isFourClassReady: false,
    });
    expect(JSON.stringify(summary)).not.toMatch(/private|release\.pt|a{64}|metastasis|11/);
  });
});
