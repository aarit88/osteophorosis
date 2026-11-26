// @vitest-environment node
import type { Express, Request, Response, RequestHandler } from 'express';
import { beforeAll, describe, expect, it } from 'vitest';
import { registerResearchRoutes } from '../research-api';

describe('Express research API contracts', () => {
  const handlers = new Map<string, RequestHandler>();

  beforeAll(async () => {
    const app = {
      get(path: string, ...routes: RequestHandler[]) { handlers.set(`GET ${path}`, routes.at(-1)!); return this; },
      post(path: string, ...routes: RequestHandler[]) { handlers.set(`POST ${path}`, routes.at(-1)!); return this; },
    } as unknown as Express;
    registerResearchRoutes(app);
  });

  async function invoke(method: string, path: string) {
    const handler = handlers.get(`${method} ${path}`);
    if (!handler) throw new Error(`No Express route registered: ${method} ${path}`);
    const result: { status: number; body?: any } = { status: 200 };
    const response = {
      status(code: number) { result.status = code; return this; },
      json(body: any) { result.body = body; return this; },
    };
    await handler({ body: {} } as Request, response as Response, (() => undefined) as any);
    return result;
  }

  it('serves readiness and governance fields consumed by the frontend', async () => {
    const [readyRes, governanceRes] = await Promise.all([
      invoke('GET', '/api/v1/datasets/readiness'),
      invoke('GET', '/api/v1/datasets/governance-status'),
    ]);
    expect(readyRes.status).toBe(200);
    const readiness = readyRes.body;
    expect(readiness).toHaveProperty('readiness_status');
    expect(readiness).toHaveProperty('manifest_schema_validation');
    expect(readiness).toHaveProperty('protected_holdouts');
    expect(governanceRes.status).toBe(200);
    expect(governanceRes.body).toHaveProperty('governance_gate_status');
  });

  it('keeps four-class diagnosis blocked with the stable error contract', async () => {
    const response = await invoke('POST', '/api/v1/diagnostic/four-class');
    expect(response.status).toBe(503);
    expect(response.body).toMatchObject({ error_code: 'FOUR_CLASS_MODEL_NOT_READY' });
  });

  it('serves interactive synthetic research demonstrations for radiomics, geometry, explainability, uncertainty, and workbench', async () => {
    const radiomics = await invoke('POST', '/api/v1/radiomics/demo');
    expect(radiomics.status).toBe(200);
    expect(radiomics.body.verified_features_count).toBe(81);

    const geometry = await invoke('POST', '/api/v1/biomechanics/geometry');
    expect(geometry.status).toBe(200);
    expect(geometry.body.pixel_spacing_mm).toEqual([0.5, 0.5]);

    const explain = await invoke('POST', '/api/v1/explainability/demo');
    expect(explain.status).toBe(200);
    expect(Array.isArray(explain.body.heatmap_grid_sample)).toBe(true);

    const uncertainty = await invoke('POST', '/api/v1/uncertainty/demo');
    expect(uncertainty.status).toBe(200);
    expect(uncertainty.body).toHaveProperty('normalized_entropy');

    const workbench = await invoke('POST', '/api/v1/workbench/run-demo');
    expect(workbench.status).toBe(200);
    expect(workbench.body.four_class_system_status).toBe('FOUR_CLASS_MODEL_NOT_READY');
  });
});
