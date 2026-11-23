import { afterEach, describe, expect, it, vi } from 'vitest';
import { advanceDatasetPreparationRun, createDatasetPreparationRun, getDatasetPreparationRuns } from '../src/api/client';

describe('Stage 39 dataset preparation API contract', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('sends authenticated run creation and lifecycle actions through Express', async () => {
    const run = { run_id: 'prep_fixture', source_id: 'SRC_FIXTURE', source_version: 'v1', state: 'SOURCE_STAGED', validation_status: 'PENDING', config_version: '1.0.0', blockers: [] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ runs: [run] }) });
    vi.stubGlobal('fetch', fetchMock);
    expect((await getDatasetPreparationRuns('operator-token')).runs[0].state).toBe('SOURCE_STAGED');
    await createDatasetPreparationRun('operator-token', 'SRC_FIXTURE');
    await advanceDatasetPreparationRun('operator-token', 'prep_fixture', 'inventory');
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/dataset-preparation/runs', expect.objectContaining({ headers: { Authorization: 'Bearer operator-token' } }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/dataset-preparation/runs', expect.objectContaining({ method: 'POST', body: JSON.stringify({ source_id: 'SRC_FIXTURE' }) }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/v1/dataset-preparation/runs/prep_fixture/inventory', expect.objectContaining({ method: 'POST' }));
  });
});
