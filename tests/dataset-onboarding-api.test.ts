import { afterEach, describe, expect, it, vi } from 'vitest';
import { getOnboardingSources, registerOnboardingSource } from '../src/api/client';

describe('Stage 38 onboarding API contract', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('uses the authenticated gateway route and returns registry lifecycle state', async () => {
    const payload = { sources: [{ source_id: 'SRC_001', acquisition_status: 'REGISTERED', verification_status: 'NOT_VERIFIED', staging_status: 'NOT_STAGED' }] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => payload });
    vi.stubGlobal('fetch', fetchMock);
    const result = await getOnboardingSources('operator-token');
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/dataset-onboarding/sources', expect.objectContaining({ headers: { Authorization: 'Bearer operator-token' } }));
    expect(result.sources[0].acquisition_status).toBe('REGISTERED');
  });

  it('surfaces authorization errors instead of treating an error response as source data', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 403, json: async () => ({ detail: 'Dataset onboarding role is required.' }) });
    vi.stubGlobal('fetch', fetchMock);
    await expect(registerOnboardingSource('clinical-reviewer-token', {})).rejects.toThrow('Dataset onboarding role is required.');
  });
});
