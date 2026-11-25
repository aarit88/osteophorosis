import { afterEach, describe, expect, it, vi } from 'vitest';
import { getTrackAStatus, getTrackDDatasetReadiness, runTrackAInference } from '../src/api/client';

describe('frontend readiness API contract', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('requests the live readiness resource and consumes schema/holdout fields', async () => {
    const payload = {
      track: 'track_d', readiness_status: 'NOT_AVAILABLE', ready_for_training: false,
      manifest_schema_validation: 'LIVE_JSON_MANIFEST',
      protected_holdouts: { btxrd_center2_samples: 549, btxrd_center3_samples: 259, isolation_enforced: true },
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => payload });
    vi.stubGlobal('fetch', fetchMock);

    const result = await getTrackDDatasetReadiness();
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/datasets/readiness', expect.any(Object));
    expect(result.manifest_schema_validation).toBe('LIVE_JSON_MANIFEST');
    expect(result.protected_holdouts.isolation_enforced).toBe(true);
  });
});

describe('Track A preliminary X-ray API contract', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('requests checkpoint status from the separate preliminary endpoint', async () => {
    const payload = {
      track_id: 'track_a', model_name: 'FracAtlas ResNet-101', modality: 'xray', input_size: [224, 224],
      classes: ['candidate_control_radiograph', 'acute_bone_fracture'], checkpoint_available: false,
      checkpoint_size_bytes: null, status: 'CHECKPOINT_UNAVAILABLE', error_code: 'TRACK_A_CHECKPOINT_NOT_AVAILABLE', clinical_use: false,
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => payload });
    vi.stubGlobal('fetch', fetchMock);

    const result = await getTrackAStatus();
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/preliminary/track-a/status', expect.any(Object));
    expect(result.checkpoint_available).toBe(false);
  });

  it('does not interpret an unauthorized response body as an inference result', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 401, json: async () => ({ error: 'Unauthorized' }) });
    vi.stubGlobal('fetch', fetchMock);

    await expect(getTrackAStatus()).rejects.toThrow('TRACK_A_AUTHENTICATION_REQUIRED');
  });

  it('posts an image only to the Track A endpoint with the selected operating point', async () => {
    const payload = {
      track_id: 'track_a', track: 'test fixture', model_version: '1.0.0', checkpoint_epoch: 1,
      checkpoint_sha256: 'a'.repeat(64), checkpoint_size_bytes: 10, modality: 'xray',
      prob_non_fractured: 0.25, prob_fractured: 0.75, active_threshold: 0.47,
      predicted_class_idx: 1, predicted_class_name: 'acute_bone_fracture', operating_point: 'best_f1',
      timestamp: '2026-11-25T00:00:00Z', research_notice: 'Synthetic test fixture.', disclaimer: 'Synthetic test fixture.',
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => payload });
    vi.stubGlobal('fetch', fetchMock);
    const image = new File(['synthetic fixture only'], 'fixture.png', { type: 'image/png' });

    const result = await runTrackAInference(image, 'best_f1');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/preliminary/track-a/analyze');
    const form = fetchMock.mock.calls[0][1].body as FormData;
    expect(form.get('operating_point')).toBe('best_f1');
    expect(result.predicted_class_name).toBe('acute_bone_fracture');
  });
});
