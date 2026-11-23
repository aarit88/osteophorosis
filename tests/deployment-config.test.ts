// @vitest-environment node
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import YAML from 'yaml';

const compose = YAML.parse(readFileSync(new URL('../compose.yaml', import.meta.url), 'utf8'));
const productionCompose = YAML.parse(readFileSync(new URL('../compose.production.yaml', import.meta.url), 'utf8'));

describe('deployment configuration safety contracts', () => {
  it('keeps FastAPI private and gates gateway startup on API process health', () => {
    expect(compose.services['ml-service'].ports).toBeUndefined();
    expect(compose.services['ml-service'].expose).toEqual(['8000']);
    expect(compose.services.gateway.depends_on['ml-service'].condition).toBe('service_healthy');
    expect(compose.services.gateway.ports[0]).toMatch(/^127\.0\.0\.1:/);
  });

  it('mounts clinical data and checkpoint artifacts read-only in the gateway', () => {
    const gatewayVolumes = compose.services.gateway.volumes;
    expect(gatewayVolumes).toContain('./ml_service/data/raw:/app/ml_service/data/raw:ro');
    expect(gatewayVolumes).toContain('model-artifacts:/var/lib/osteoguard/artifacts:ro');
    expect(compose.services['ml-service'].volumes).toContain('./ml_service/data:/app/ml_service/data:ro');
  });

  it('requires an external read-only reviewer registry only in the production overlay', () => {
    const reviewerVolume = productionCompose.services['ml-service'].volumes[0];
    expect(reviewerVolume.read_only).toBe(true);
    expect(reviewerVolume.source).toContain('OSTEO_REVIEWER_REGISTRY_HOST_PATH');
    expect(productionCompose.services['ml-service'].environment.OSTEO_ENV).toBe('production');
    expect(productionCompose.services.gateway.environment.NODE_ENV).toBe('production');
  });

  it('uses read-only roots, drops capabilities, and declares non-root container users', () => {
    for (const service of Object.values(compose.services) as Array<Record<string, any>>) {
      expect(service.read_only).toBe(true);
      expect(service.security_opt).toContain('no-new-privileges:true');
      expect(service.cap_drop).toContain('ALL');
      expect(service.healthcheck).toBeDefined();
    }
    expect(readFileSync(new URL('../Dockerfile', import.meta.url), 'utf8')).toContain('USER node');
    expect(readFileSync(new URL('../Dockerfile.ml-service', import.meta.url), 'utf8')).toContain('USER 10001:10001');
  });

  it('does not bake environment files, clinical imaging, or model weights into the build context', () => {
    const ignore = readFileSync(new URL('../.dockerignore', import.meta.url), 'utf8');
    expect(ignore).toContain('.env');
    expect(ignore).toContain('ml_service/data/**');
    expect(ignore).toContain('datasets/**');
    expect(ignore).toContain('mock-server/data/**');
    expect(ignore).toContain('*.pt');
    expect(ignore).toContain('*.nii.gz');
    expect(ignore).toContain('ml_artifacts/**/*.csv');
    const sampleEnv = readFileSync(new URL('../.env.example', import.meta.url), 'utf8');
    expect(sampleEnv).toContain('OSTEO_CLINICAL_API_TOKEN=');
    expect(sampleEnv).not.toMatch(/OSTEO_(?:CLINICAL_API_TOKEN|GATEWAY_HMAC_SECRET|AUDIT_ENCRYPTION_KEY)=\S+/);
  });

  it('pins Python runtime dependencies without installing test-only packages', () => {
    const runtime = readFileSync(new URL('../ml_service/requirements-runtime.txt', import.meta.url), 'utf8');
    expect(runtime).toContain('torch==2.9.0');
    expect(runtime).toContain('fastapi==');
    expect(runtime).not.toMatch(/pytest|shap|ruff|black/);
  });
});
