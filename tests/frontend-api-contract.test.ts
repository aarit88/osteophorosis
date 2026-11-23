import { afterEach, describe, expect, it, vi } from 'vitest';
import { getTrackDDatasetReadiness } from '../src/api/client';

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
