import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { timingSafeEqual, randomUUID } from 'node:crypto';
import { fileURLToPath } from 'url';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yaml';
import { registerResearchRoutes } from './research-api.js';
import { simulateAnalysis } from './mock-server/simulation.js';
import {
  ClinicalAnalysisQueue,
  LatencyTracker,
  SlidingWindowRateLimiter,
  formatFeedbackCsv,
} from './clinical-scalability.js';
import { isStrongProductionSecret, isSupportedUploadMediaType, resolveCorsOrigins } from './security-policy.js';
import {
  getOrProvisionEncryptionKey,
  appendEncryptedJsonl,
  readEncryptedJsonl,
  migrateJsonlToEncrypted,
  EphemeralEncryptedAssetStore,
  stripImageMetadataAndEncodePng,
  signGatewayHeaders,
  inspectClinicalReadinessOnDisk,
  summarizeClinicalReadiness,
} from './security.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();

  // Parse command line arguments for host and port
  const args = process.argv.slice(2);
  let PORT = process.env.PORT === undefined ? 3000 : Number(process.env.PORT);
  let HOST = process.env.HOST?.trim() || '0.0.0.0';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port' && args[i + 1]) {
      PORT = Number(args[i + 1]);
      i++;
    } else if (args[i] === '--host' && args[i + 1]) {
      HOST = args[i + 1];
      i++;
    }
  }

  if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) {
    throw new Error('PORT must be an integer between 1 and 65535.');
  }
  if (!HOST || /\s/.test(HOST)) throw new Error('HOST must be a non-empty bind address.');

  const production = process.env.NODE_ENV === 'production';
  const configuredOrigins = resolveCorsOrigins(process.env.OSTEO_CORS_ORIGINS, production);
  app.use(cors({ origin: configuredOrigins, credentials: true, methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'], allowedHeaders: ['Authorization', 'Content-Type', 'X-OsteoGuard-Timestamp', 'X-OsteoGuard-Signature'] }));
  app.use(express.json({ limit: '2mb' }));

  const latencyTracker = new LatencyTracker(500);
  const rateLimitRpm = Math.max(20, Number(process.env.OSTEO_RATE_LIMIT_RPM || 120));
  const uploadRateLimiter = new SlidingWindowRateLimiter({
    maxRequests: rateLimitRpm,
    windowMs: 60_000,
  });

  // Clinical transport, Request-Id tracing & cache-hardening headers across all responses
  app.use((req, res, next) => {
    const reqId = (typeof req.headers['x-request-id'] === 'string' && req.headers['x-request-id'].trim()) || randomUUID();
    res.locals.requestId = reqId;
    res.setHeader('X-Request-Id', reqId);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'SAMEORIGIN');
    res.setHeader('Referrer-Policy', 'no-referrer');
    res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
    if (req.secure || req.headers['x-forwarded-proto'] === 'https') {
      res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
    }
    if (req.path.startsWith('/api')) {
      res.setHeader('Cache-Control', 'no-store, private');
      res.setHeader('Pragma', 'no-cache');
      const startedAt = Date.now();
      res.on('finish', () => {
        if (!req.path.startsWith('/api/v1/system') && req.path !== '/api/metrics') {
          latencyTracker.record(Date.now() - startedAt, res.statusCode >= 500);
        }
      });
    }
    next();
  });

  const enforceRateLimit: express.RequestHandler = (req, res, next) => {
    const clientIp = (req.ip || req.socket.remoteAddress || 'unknown').replace(/[^a-zA-Z0-9.:_-]/g, '');
    const verdict = uploadRateLimiter.check(clientIp);
    res.setHeader('X-RateLimit-Limit', String(verdict.limit));
    res.setHeader('X-RateLimit-Remaining', String(verdict.remaining));
    if (!verdict.allowed) {
      res.setHeader('Retry-After', String(verdict.resetSeconds));
      return res.status(429).json({
        error_code: 'RATE_LIMIT_EXCEEDED',
        error: `Too many requests. Retry after ${verdict.resetSeconds}s.`,
      });
    }
    next();
  };

  const EXPLICIT_BACKEND_URL = process.env.BACKEND_URL?.trim() || '';
  const CLINICAL_API_TOKEN = process.env.OSTEO_CLINICAL_API_TOKEN?.trim() || '';
  const GATEWAY_HMAC_SECRET = process.env.OSTEO_GATEWAY_HMAC_SECRET?.trim() || '';
  const STRICT_CLINICAL_MODE = process.env.OSTEO_STRICT_CLINICAL_MODE === 'true';
  const ALLOW_SIMULATION_FALLBACK =
    process.env.OSTEO_DISABLE_SIMULATION !== 'true' && !(production && STRICT_CLINICAL_MODE);
  const probeMLProcessReadiness = async () => {
    if (!EXPLICIT_BACKEND_URL) return false;
    try {
      const response = await fetch(new URL('/health/ready', EXPLICIT_BACKEND_URL), { signal: AbortSignal.timeout(2000) });
      return response.ok;
    } catch {
      return false;
    }
  };
  if (production && (!isStrongProductionSecret(CLINICAL_API_TOKEN) || !isStrongProductionSecret(GATEWAY_HMAC_SECRET) || !isStrongProductionSecret(process.env.OSTEO_AUDIT_ENCRYPTION_KEY))) {
    throw new Error('Production requires externally supplied clinical, gateway HMAC, and audit encryption secrets of at least 32 characters.');
  }

  const requireClinicalToken: express.RequestHandler = (req, res, next) => {
    if (!CLINICAL_API_TOKEN) return production ? res.status(503).json({ error: 'Authentication is not configured.' }) : next();
    const authHeader = req.headers.authorization || '';
    const supplied = Buffer.from(authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '', 'utf8');
    const expected = Buffer.from(CLINICAL_API_TOKEN, 'utf8');
    if (supplied.length === expected.length && timingSafeEqual(supplied, expected)) {
      return next();
    }
    return res.status(401).json({
      error: 'Unauthorized: valid clinical Bearer token required.',
    });
  };

  // Optional proxy to external FastAPI ML service when BACKEND_URL is explicitly configured;
  // otherwise handled directly by the built-in Node.js Stage 27-31 research endpoints below.
  if (EXPLICIT_BACKEND_URL) {
    app.use('/api', async (req, res, next) => {
      const researchPaths = [
        '/v1/research',
        '/v1/evidence',
        '/v1/architecture',
        '/v1/datasets',
        '/v1/dataset-onboarding',
        '/v1/dataset-preparation',
        '/v1/dataset-qualification',
        '/v1/radiomics',
        '/v1/biomechanics',
        '/v1/explainability',
        '/v1/uncertainty',
        '/v1/workbench',
        '/v1/diagnostic',
      ];

      const shouldProxy = researchPaths.some((p) => req.path.startsWith(p));
      if (!shouldProxy) {
        return next();
      }

      try {
        const targetUrl = `${EXPLICIT_BACKEND_URL.replace(/\/+$/, '')}/api${req.url}`;
        const fetchHeaders: Record<string, string> = {
          accept: (req.headers.accept as string) || 'application/json',
        };
        if (req.headers.authorization) {
          fetchHeaders.authorization = req.headers.authorization as string;
        }

        let serializedBody = '';
        const fetchOptions: RequestInit = {
          method: req.method,
          headers: fetchHeaders,
        };

        if (req.method !== 'GET' && req.method !== 'HEAD' && req.body && Object.keys(req.body).length > 0) {
          serializedBody = JSON.stringify(req.body);
          fetchOptions.body = serializedBody;
          fetchHeaders['content-type'] = 'application/json';
        }

        if (GATEWAY_HMAC_SECRET) {
          Object.assign(
            fetchHeaders,
            signGatewayHeaders(req.method, `/api${req.path}`, serializedBody, GATEWAY_HMAC_SECRET)
          );
        }

        const backendRes = await fetch(targetUrl, fetchOptions);
        res.status(backendRes.status);
        backendRes.headers.forEach((val, key) => {
          const lower = key.toLowerCase();
          if (lower !== 'content-encoding' && lower !== 'content-length' && lower !== 'transfer-encoding') {
            res.setHeader(key, val);
          }
        });
        const data = await backendRes.text();
        res.send(data);
      } catch (err: any) {
        console.warn('[Proxy Fallback] ML backend is unreachable; returning the configured gateway fallback response.');
        return next();
      }
    });
  }

  registerResearchRoutes(app);

  // Direct /health route for server health probes
  app.get('/health', (_req, res) => {
    res.json({
      status: 'ok',
      service: 'osteoguard-frontend-server',
    });
  });

  // Readiness for the gateway workload includes the configured ML service. This is
  // distinct from Track D clinical/model readiness, reported by FastAPI model status.
  app.get('/readyz', async (_req, res) => {
    if (!EXPLICIT_BACKEND_URL) {
      if (production) return res.status(503).json({ status: 'not_ready', dependency: 'ml_service_not_configured' });
      return res.json({ status: 'ready', mode: 'standalone_development', clinical_readiness: 'separate' });
    }
    try {
      if (!(await probeMLProcessReadiness())) {
        return res.status(503).json({ status: 'not_ready', dependency: 'ml_service_unavailable' });
      }
      return res.json({ status: 'ready', dependencies: { ml_service: 'ready' }, clinical_readiness: 'separate' });
    } catch {
      return res.status(503).json({ status: 'not_ready', dependency: 'ml_service_unavailable' });
    }
  });

  // Purge any legacy unencrypted images under public/uploads/ so patient uploads remain RAM-only
  const publicUploadsDir = path.resolve(__dirname, 'public/uploads');
  if (fs.existsSync(publicUploadsDir)) {
    for (const file of fs.readdirSync(publicUploadsDir)) {
      if (/\.(png|jpe?g|webp|tif?f|dcm)$/i.test(file)) {
        try {
          fs.unlinkSync(path.join(publicUploadsDir, file));
        } catch {
          // Ignore non-removable files
        }
      }
    }
  }

  // AES-256-GCM encrypted RAM-only store for ephemeral radiograph & Grad-CAM assets (configurable TTL/capacity)
  const ephemeralMaxEntries = Math.max(60, Number(process.env.OSTEO_EPHEMERAL_MAX_ENTRIES || 240));
  const ephemeralTtlSeconds = Math.max(120, Number(process.env.OSTEO_EPHEMERAL_TTL_SECONDS || 900));
  const ephemeralStore = new EphemeralEncryptedAssetStore({
    ttlMs: ephemeralTtlSeconds * 1000,
    maxEntries: ephemeralMaxEntries,
  });

  const maxConcurrentJobs = Math.max(1, Number(process.env.OSTEO_MAX_CONCURRENT_JOBS || 4));
  const maxQueueSize = Math.max(10, Number(process.env.OSTEO_MAX_QUEUE_SIZE || 60));
  const analysisQueue = new ClinicalAnalysisQueue<Record<string, any>>({
    maxConcurrency: maxConcurrentJobs,
    maxQueueSize,
    ttlMs: 30 * 60 * 1000,
  });

  // Serve trained model evaluation PNG plots safely from ml_artifacts/
  const mlArtifactsDir = path.resolve(__dirname, 'ml_artifacts');
  if (fs.existsSync(mlArtifactsDir)) {
    app.use('/ml-artifacts', (req, res, next) => {
      if (!req.path.toLowerCase().endsWith('.png')) {
        return res.status(403).json({ error: 'Only PNG evaluation plot assets are served here.' });
      }
      next();
    }, express.static(mlArtifactsDir));
  }

  // Data persistence directory for AES-256-GCM encrypted audit feedback and telemetry
  const dataDir = path.resolve(process.env.OSTEO_NODE_DATA_DIR || path.join(__dirname, 'mock-server/data'));
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  const auditEncryptionKey = getOrProvisionEncryptionKey(dataDir, process.env.OSTEO_AUDIT_ENCRYPTION_KEY);
  const feedbackFile = path.join(dataDir, 'feedback.jsonl');
  const eventsFile = path.join(dataDir, 'events.jsonl');

  // Seed initial demo feedback encrypted at rest if empty, and migrate any legacy plaintext records
  if (production || process.env.OSTEO_DISABLE_DEMO_SEED === 'true') {
    if (fs.existsSync(feedbackFile) && fs.readFileSync(feedbackFile, 'utf-8').trim() !== '') {
      migrateJsonlToEncrypted(feedbackFile, auditEncryptionKey);
    }
  } else if (fs.existsSync(feedbackFile) && fs.readFileSync(feedbackFile, 'utf-8').trim() !== '') {
    const migratedFb = migrateJsonlToEncrypted(feedbackFile, auditEncryptionKey);
    if (migratedFb > 0) {
      console.log(`[OsteoGuard Security] Migrated ${migratedFb} legacy feedback record(s) to AES-256-GCM encryption.`);
    }
  }

  if (fs.existsSync(eventsFile)) {
    const migratedEv = migrateJsonlToEncrypted(eventsFile, auditEncryptionKey);
    if (migratedEv > 0) {
      console.log(`[OsteoGuard Security] Migrated ${migratedEv} legacy telemetry event(s) to AES-256-GCM encryption.`);
    }
  }

  // Swagger UI documentation at /docs
  try {
    const openApiPath = path.resolve(__dirname, 'openapi.yaml');
    if (fs.existsSync(openApiPath)) {
      const file = fs.readFileSync(openApiPath, 'utf8');
      const swaggerDocument = YAML.parse(file);
      app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));
      console.log(`[OsteoGuard Server] Interactive OpenAPI 3.1 Swagger UI mounted at /docs`);
    }
  } catch (err) {
    console.warn('Failed to mount Swagger UI at /docs:', err);
  }

  // Memory storage for uploads (never touches disk)
  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 15 * 1024 * 1024 },
    fileFilter: (_req, file, callback) => {
      if (!isSupportedUploadMediaType(file.mimetype)) return callback(new Error('Unsupported image media type.'));
      callback(null, true);
    },
  });

  // GET /api/health (unversioned)
  app.get('/api/health', async (_req, res) => {
    if (!EXPLICIT_BACKEND_URL) {
      if (!ALLOW_SIMULATION_FALLBACK) {
        return res.status(503).json({ status: 'error', mode: 'unavailable', error_code: 'ML_SERVICE_NOT_CONFIGURED' });
      }
      return res.json({ status: 'ok', mode: 'simulation', clinical_readiness: 'blocked' });
    }
    if (!(await probeMLProcessReadiness())) {
      if (ALLOW_SIMULATION_FALLBACK) {
        return res.json({ status: 'ok', mode: 'simulation', clinical_readiness: 'blocked' });
      }
      return res.status(503).json({ status: 'error', mode: 'unavailable', error_code: 'ML_SERVICE_UNAVAILABLE' });
    }
    try {
      const modelResponse = await fetch(new URL('/api/ml/models/status', EXPLICIT_BACKEND_URL), {
        signal: AbortSignal.timeout(2500),
      });
      if (!modelResponse.ok) {
        if (ALLOW_SIMULATION_FALLBACK) {
          return res.json({ status: 'ok', mode: 'simulation', clinical_readiness: 'blocked' });
        }
        return res.status(503).json({ status: 'error', mode: 'unavailable', error_code: 'ML_READINESS_UNAVAILABLE' });
      }
      const model = await modelResponse.json() as { can_infer?: boolean; checkpoint_verified?: boolean };
      const clinicallyReady = model.can_infer === true && model.checkpoint_verified === true;
      return res.json({ status: 'ok', mode: clinicallyReady ? 'production' : 'simulation', clinical_readiness: clinicallyReady ? 'ready' : 'blocked' });
    } catch {
      if (ALLOW_SIMULATION_FALLBACK) {
        return res.json({ status: 'ok', mode: 'simulation', clinical_readiness: 'blocked' });
      }
      return res.status(503).json({ status: 'error', mode: 'unavailable', error_code: 'ML_READINESS_UNAVAILABLE' });
    }
  });

  // GET /api/v1/security/status — live cryptographic & clinical checkpoint readiness telemetry
  app.get('/api/v1/security/status', async (_req, res) => {
    const fbRead = readEncryptedJsonl(feedbackFile, auditEncryptionKey);
    const evRead = readEncryptedJsonl(eventsFile, auditEncryptionKey);
    const diskReadiness = inspectClinicalReadinessOnDisk(__dirname);

    let diskUploadsRemaining = 0;
    if (fs.existsSync(publicUploadsDir)) {
      diskUploadsRemaining = fs
        .readdirSync(publicUploadsDir)
        .filter((f) => /\.(png|jpe?g|webp|tif?f|dcm)$/i.test(f)).length;
    }

    let mlBackendReachable = false;
    let mlBackendCanInfer = false;
    if (EXPLICIT_BACKEND_URL) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 1500);
        const probe = await fetch(`${EXPLICIT_BACKEND_URL.replace(/\/+$/, '')}/api/ml/models/status`, {
          signal: controller.signal,
        });
        clearTimeout(timer);
        if (probe.ok) {
          mlBackendReachable = true;
          const body = (await probe.json()) as { can_infer?: boolean };
          mlBackendCanInfer = Boolean(body.can_infer);
        }
      } catch {
        mlBackendReachable = false;
      }
    }

    res.json({
      timestamp: new Date().toISOString(),
      encryptionAtRest: {
        enabled: true,
        algorithm: 'AES-256-GCM',
        keySource: process.env.OSTEO_AUDIT_ENCRYPTION_KEY ? 'environment' : 'local_0600_keyfile',
        feedbackEncryptedCount: fbRead.encryptedCount,
        feedbackPlaintextCount: fbRead.legacyPlaintextCount,
        eventsEncryptedCount: evRead.encryptedCount,
        eventsPlaintextCount: evRead.legacyPlaintextCount,
      },
      ephemeralMedia: {
        ramOnly: true,
        exifStrippingActive: true,
        diskUploadsRemaining,
        ttlSeconds: 900,
        activeRamAssets: ephemeralStore.getActiveCount(),
      },
      gatewayTransport: {
        hstsHeaderReady: true,
        strictNoStoreApiCache: true,
        bearerAuthRequired: Boolean(CLINICAL_API_TOKEN),
        hmacSigningConfigured: Boolean(GATEWAY_HMAC_SECRET),
        mlBackendConfigured: Boolean(EXPLICIT_BACKEND_URL),
        mlBackendReachable,
        mlBackendCanInfer,
        strictClinicalMode: STRICT_CLINICAL_MODE,
      },
      clinicalReadiness: summarizeClinicalReadiness(diskReadiness),
    });
  });

  // GET /api/v1/ephemeral/:assetId — decrypts and streams RAM-only radiograph / Grad-CAM PNG
  app.get('/api/v1/ephemeral/:assetId', requireClinicalToken, (req, res) => {
    const assetId = String(req.params.assetId || '').replace(/[^a-zA-Z0-9._-]/g, '');
    if (!assetId) {
      return res.status(400).json({ error: 'Invalid asset identifier.' });
    }
    const item = ephemeralStore.get(assetId);
    if (!item) {
      return res.status(404).json({ error: 'Ephemeral clinical image expired or unavailable in RAM.' });
    }
    res.setHeader('Content-Type', item.contentType);
    res.setHeader('Cache-Control', 'no-store, private');
    res.setHeader('X-OsteoGuard-Ephemeral', 'ram-aes-256-gcm');
    res.send(item.buffer);
  });

  class ClinicalPipelineRejectionError extends Error {
    statusCode: number;
    error_code: string;
    payload: Record<string, unknown>;
    constructor(statusCode: number, errorCode: string, payload: Record<string, unknown>) {
      super(String(payload.detail || payload.message || errorCode));
      this.statusCode = statusCode;
      this.error_code = errorCode;
      this.payload = payload;
    }
  }

  const executeAnalysisPipeline = async (params: {
    sanitizedBuffer: Buffer;
    originalname: string;
    modality: string;
    sampleCase?: string;
  }): Promise<Record<string, any>> => {
    const { sanitizedBuffer, originalname, modality, sampleCase } = params;

    const runSimulatedPipeline = async () => {
      return await simulateAnalysis(sanitizedBuffer, originalname, modality, publicUploadsDir, {
        sampleCase,
        storeAsset: (filename: string, buf: Buffer, contentType: string) => {
          ephemeralStore.put(filename, buf, contentType);
          return `/api/v1/ephemeral/${encodeURIComponent(filename)}`;
        },
      });
    };

    if (!EXPLICIT_BACKEND_URL) {
      if (!ALLOW_SIMULATION_FALLBACK) {
        throw new ClinicalPipelineRejectionError(503, 'FOUR_CLASS_MODEL_NOT_READY', {
          status: 'BLOCKED',
          error_code: 'FOUR_CLASS_MODEL_NOT_READY',
          detail: 'The four-class model is unavailable because no configured ML service has cleared the dataset, governance, training, and validation gates.',
        });
      }
      return await runSimulatedPipeline();
    }

    const form = new FormData();
    form.append('image', new Blob([new Uint8Array(sanitizedBuffer)], { type: 'image/png' }), 'sanitized.png');
    form.append('modality', modality);

    const mlHeaders: Record<string, string> = {};
    if (GATEWAY_HMAC_SECRET) {
      Object.assign(mlHeaders, signGatewayHeaders('POST', '/api/v1/analyze', sanitizedBuffer, GATEWAY_HMAC_SECRET));
    }

    let liveRes: Response | null = null;
    try {
      liveRes = await fetch(`${EXPLICIT_BACKEND_URL.replace(/\/+$/, '')}/api/v1/analyze`, {
        method: 'POST',
        headers: mlHeaders,
        body: form,
        signal: AbortSignal.timeout(4000),
      });
    } catch {
      liveRes = null;
    }

    if (!liveRes) {
      if (ALLOW_SIMULATION_FALLBACK) {
        return await runSimulatedPipeline();
      }
      throw new ClinicalPipelineRejectionError(503, 'ML_SERVICE_UNAVAILABLE', {
        status: 'BLOCKED',
        error_code: 'ML_SERVICE_UNAVAILABLE',
        detail: 'The ML service could not be reached. No simulation fallback is available.',
      });
    }

    const body = (await liveRes.json().catch(() => null)) as Record<string, any> | null;
    if (!liveRes.ok) {
      if (liveRes.status === 503 && ALLOW_SIMULATION_FALLBACK) {
        return await runSimulatedPipeline();
      }
      const errCode = liveRes.status === 503 ? 'FOUR_CLASS_MODEL_NOT_READY' : 'ML_INFERENCE_REJECTED';
      throw new ClinicalPipelineRejectionError(
        liveRes.status,
        String(body?.error_code || errCode),
        body ?? { status: 'BLOCKED', error_code: errCode, detail: 'The ML service rejected this request.' },
      );
    }

    if (!body || typeof body.prediction !== 'object' || body.prediction === null) {
      throw new ClinicalPipelineRejectionError(502, 'INVALID_ML_RESPONSE', {
        status: 'ERROR',
        error_code: 'INVALID_ML_RESPONSE',
        detail: 'The ML service returned an invalid inference response.',
      });
    }

    const reqTag = String(body.requestId || `live_${Date.now()}`).replace(/[^a-zA-Z0-9_-]/g, '');
    if (typeof body.imageBase64 === 'string' && body.imageBase64.length > 0) {
      const imgId = `${reqTag}_scan.png`;
      ephemeralStore.put(imgId, Buffer.from(body.imageBase64, 'base64'), 'image/png');
      body.imageUrl = `/api/v1/ephemeral/${encodeURIComponent(imgId)}`;
      delete body.imageBase64;
    }
    if (typeof body.gradcamBase64 === 'string' && body.gradcamBase64.length > 0) {
      const camId = `${reqTag}_gradcam.png`;
      ephemeralStore.put(camId, Buffer.from(body.gradcamBase64, 'base64'), 'image/png');
      body.gradcamUrl = `/api/v1/ephemeral/${encodeURIComponent(camId)}`;
      delete body.gradcamBase64;
    }
    return body;
  };

  // Analyze handler (EXIF-stripped in RAM -> live FastAPI inference when available, or deterministic simulation fallback)
  const handleAnalyze = async (req: express.Request, res: express.Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image file uploaded.' });
      }

      const rawModality = String(req.body.modality || 'xray').toLowerCase();
      if (!['xray', 'ct', 'mri', 'bone_scan'].includes(rawModality)) {
        return res.status(422).json({ error: 'Unsupported modality.' });
      }

      const rawSampleCase = typeof req.body.sample_case === 'string' ? req.body.sample_case.trim().toLowerCase() : '';
      const sampleCase = ['metastasis', 'paget', 'oi', 'healthy'].includes(rawSampleCase) ? rawSampleCase : undefined;

      let sanitizedBuffer: Buffer;
      try {
        sanitizedBuffer = await stripImageMetadataAndEncodePng(req.file.buffer);
      } catch {
        return res.status(415).json({ error: 'Uploaded file is not a supported, valid image.' });
      }

      const result = await executeAnalysisPipeline({
        sanitizedBuffer,
        originalname: req.file.originalname,
        modality: rawModality,
        sampleCase,
      });
      return res.json(result);
    } catch (err: unknown) {
      if (err instanceof ClinicalPipelineRejectionError) {
        return res.status(err.statusCode).json(err.payload);
      }
      // Do not log image metadata, model inputs, or raw exceptions from inference handling.
      return res.status(500).json({ error_code: 'ANALYSIS_REQUEST_FAILED', error: 'The analysis request could not be completed.' });
    }
  };

  // POST /api/v1/analyze & POST /api/analyze
  app.post('/api/v1/analyze', enforceRateLimit, requireClinicalToken, upload.single('image'), handleAnalyze);
  app.post('/api/analyze', enforceRateLimit, requireClinicalToken, upload.single('image'), handleAnalyze);

  // POST /api/v1/jobs/analyze — Asynchronous bounded-concurrency scan queueing for batch clinical triage
  app.post('/api/v1/jobs/analyze', enforceRateLimit, requireClinicalToken, upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image file uploaded.' });
      }
      const rawModality = String(req.body.modality || 'xray').toLowerCase();
      if (!['xray', 'ct', 'mri', 'bone_scan'].includes(rawModality)) {
        return res.status(422).json({ error: 'Unsupported modality.' });
      }
      const rawSampleCase = typeof req.body.sample_case === 'string' ? req.body.sample_case.trim().toLowerCase() : '';
      const sampleCase = ['metastasis', 'paget', 'oi', 'healthy'].includes(rawSampleCase) ? rawSampleCase : undefined;

      let sanitizedBuffer: Buffer;
      try {
        sanitizedBuffer = await stripImageMetadataAndEncodePng(req.file.buffer);
      } catch {
        return res.status(415).json({ error: 'Uploaded file is not a supported, valid image.' });
      }

      const originalname = req.file.originalname;
      const queued = analysisQueue.enqueue(rawModality, async () => {
        const result = await executeAnalysisPipeline({
          sanitizedBuffer,
          originalname,
          modality: rawModality,
          sampleCase,
        });
        return {
          result,
          fractureRiskPercent: Number(result?.fractureRisk?.percent ?? 0),
          entropy: Number(result?.uncertaintyDetail?.entropy ?? 0),
          oodFlag: Boolean(result?.uncertaintyDetail?.oodFlag),
        };
      });

      if (!queued.accepted) {
        return res.status(503).json({
          error_code: 'CLINICAL_QUEUE_AT_CAPACITY',
          error: queued.reason,
        });
      }

      return res.status(202).json({
        id: queued.job.jobId,
        ...queued.job,
        position: queued.position,
      });
    } catch {
      return res.status(500).json({ error_code: 'JOB_ENQUEUE_FAILED', error: 'Unable to enqueue analysis job.' });
    }
  });

  // Track A is a separate binary FracAtlas X-ray research model; it never calls the Track D route.
  app.get('/api/v1/preliminary/track-a/status', requireClinicalToken, async (_req, res) => {
    if (!EXPLICIT_BACKEND_URL) {
      return res.json({
        track_id: 'track_a',
        model_name: 'FracAtlas ResNet-101 binary fracture research classifier',
        modality: 'xray',
        input_size: [224, 224],
        classes: ['candidate_control_radiograph', 'acute_bone_fracture'],
        checkpoint_available: false,
        checkpoint_size_bytes: null,
        status: 'ML_SERVICE_NOT_CONFIGURED',
        error_code: 'ML_SERVICE_NOT_CONFIGURED',
        clinical_use: false,
      });
    }
    try {
      const response = await fetch(`${EXPLICIT_BACKEND_URL.replace(/\/+$/, '')}/api/v1/preliminary/track-a/status`, {
        signal: AbortSignal.timeout(5000),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok || !body || typeof body.checkpoint_available !== 'boolean') {
        return res.status(response.status || 502).json({ error_code: 'TRACK_A_STATUS_UNAVAILABLE', detail: 'Track A checkpoint status could not be retrieved.' });
      }
      return res.json(body);
    } catch {
      return res.json({
        track_id: 'track_a',
        model_name: 'FracAtlas ResNet-101 binary fracture research classifier',
        modality: 'xray',
        input_size: [224, 224],
        classes: ['candidate_control_radiograph', 'acute_bone_fracture'],
        checkpoint_available: false,
        checkpoint_size_bytes: null,
        status: 'ML_SERVICE_NOT_CONFIGURED',
        error_code: 'ML_SERVICE_NOT_CONFIGURED',
        clinical_use: false,
      });
    }
  });

  app.post('/api/v1/preliminary/track-a/analyze', requireClinicalToken, upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error_code: 'IMAGE_REQUIRED', detail: 'Upload an X-ray image to run Track A research inference.' });
    if (!EXPLICIT_BACKEND_URL) {
      return res.status(503).json({ error_code: 'ML_SERVICE_NOT_CONFIGURED', detail: 'The Track A inference service is not configured.' });
    }

    let sanitizedBuffer: Buffer;
    try {
      sanitizedBuffer = await stripImageMetadataAndEncodePng(req.file.buffer);
    } catch {
      return res.status(415).json({ error_code: 'INVALID_XRAY_IMAGE', detail: 'Upload a valid PNG, JPEG, or WebP radiograph image.' });
    }

    const selectedOperatingPoint = String(req.body.operating_point || 'high_sensitivity_screening');
    const allowedOperatingPoints = new Set(['high_sensitivity_screening', 'best_youden_j', 'best_f1', 'default_0_50']);
    if (!allowedOperatingPoints.has(selectedOperatingPoint)) {
      return res.status(400).json({ error_code: 'INVALID_OPERATING_POINT', detail: 'Select a documented Track A operating point.' });
    }
    const form = new FormData();
    form.append('image', new Blob([new Uint8Array(sanitizedBuffer)], { type: 'image/png' }), 'radiograph.png');
    form.append('operating_point', selectedOperatingPoint);
    const headers: Record<string, string> = {};
    if (GATEWAY_HMAC_SECRET) {
      const signedPath = `/api/v1/preliminary/track-a/analyze?operating_point=${selectedOperatingPoint}`;
      Object.assign(headers, signGatewayHeaders('POST', signedPath, sanitizedBuffer, GATEWAY_HMAC_SECRET));
    }

    try {
      const response = await fetch(`${EXPLICIT_BACKEND_URL.replace(/\/+$/, '')}/api/v1/preliminary/track-a/analyze`, {
        method: 'POST',
        headers,
        body: form,
        signal: AbortSignal.timeout(120000),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) return res.status(response.status).json(body ?? { error_code: 'TRACK_A_INFERENCE_REJECTED', detail: 'Track A rejected the image; no prediction was generated.' });
      if (!body || body.track_id !== 'track_a' || !Number.isFinite(body.prob_fractured) || !Number.isFinite(body.prob_non_fractured)) {
        return res.status(502).json({ error_code: 'INVALID_TRACK_A_RESPONSE', detail: 'The ML service returned an invalid Track A response.' });
      }
      return res.json(body);
    } catch {
      return res.status(503).json({ error_code: 'ML_SERVICE_UNAVAILABLE', detail: 'The ML service could not complete Track A inference.' });
    }
  });

  // POST /api/v1/feedback (AES-256-GCM encrypted at rest)
  app.post('/api/v1/feedback', requireClinicalToken, (req, res) => {
    try {
      const { requestId, decision, predictedClass, reasonCode, reasonText, timestamp } = req.body;
      const allowedDecisions = new Set(['agree', 'disagree', 'unsure']);
      const allowedClasses = new Set(['metastasis', 'paget', 'oi', 'healthy']);
      if (typeof requestId !== 'string' || requestId.length < 1 || requestId.length > 128 || !allowedDecisions.has(decision) || !allowedClasses.has(predictedClass)) {
        return res.status(400).json({ error: 'A valid request reference, review decision, and predicted class are required.' });
      }
      if (typeof reasonText === 'string' && reasonText.length > 2000) return res.status(400).json({ error: 'Review text exceeds the allowed length.' });

      const feedbackRecord = {
        feedbackId: `fb_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        requestId,
        decision,
        predictedClass,
        reasonCode: reasonCode || null,
        reasonText: reasonText || '',
        timestamp: timestamp || new Date().toISOString(),
      };

      appendEncryptedJsonl(feedbackFile, feedbackRecord, auditEncryptionKey);
      res.json({ success: true, feedbackId: feedbackRecord.feedbackId });
    } catch {
      res.status(500).json({ error: 'Failed to record encrypted feedback' });
    }
  });

  function readFeedbackRecords() {
    const records = readEncryptedJsonl<Record<string, any>>(feedbackFile, auditEncryptionKey).records;
    // Historical bootstrap records were synthetic examples. Keep the on-disk source untouched,
    // but exclude those exact IDs from operational counts and history.
    return records.filter((record) => !(['fb_01', 'fb_02', 'fb_03', 'fb_04'].includes(record.feedbackId)
      && /^req_demo[1-4]$/.test(record.requestId || '')));
  }

  // GET /api/v1/feedback/stats
  app.get('/api/v1/feedback/stats', requireClinicalToken, (_req, res) => {
    const records = readFeedbackRecords();
    let agree = 0;
    let disagree = 0;
    let unsure = 0;

    const byClass: Record<string, { agree: number; disagree: number; unsure: number }> = {
      metastasis: { agree: 0, disagree: 0, unsure: 0 },
      paget: { agree: 0, disagree: 0, unsure: 0 },
      oi: { agree: 0, disagree: 0, unsure: 0 },
      healthy: { agree: 0, disagree: 0, unsure: 0 },
    };

    records.forEach((r: any) => {
      if (r.decision === 'agree') agree++;
      else if (r.decision === 'disagree') disagree++;
      else if (r.decision === 'unsure') unsure++;

      const cls = r.predictedClass || 'healthy';
      if (byClass[cls]) {
        byClass[cls][r.decision] = (byClass[cls][r.decision] || 0) + 1;
      }
    });

    const total = records.length;
    const disagreementRate = total > 0 ? parseFloat((disagree / total).toFixed(3)) : 0;

    res.json({
      total,
      agree,
      disagree,
      unsure,
      disagreementRate,
      byClass,
    });
  });

  // GET /api/v1/feedback/list
  app.get('/api/v1/feedback/list', requireClinicalToken, (_req, res) => {
    const records = readFeedbackRecords();
    res.json({ items: records.reverse() });
  });

  // GET /api/v1/feedback/export — De-identified clinical QA export (CSV or JSON)
  app.get('/api/v1/feedback/export', requireClinicalToken, (req, res) => {
    const format = String(req.query.format || 'csv').toLowerCase();
    const records = readFeedbackRecords().reverse();
    const stamp = new Date().toISOString().slice(0, 10);
    if (format === 'json') {
      res.setHeader('Content-Disposition', `attachment; filename="osteoguard-audit-${stamp}.json"`);
      return res.json({
        exportedAt: new Date().toISOString(),
        recordCount: records.length,
        encryptionAtRest: 'AES-256-GCM',
        items: records,
      });
    }
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="osteoguard-audit-${stamp}.csv"`);
    return res.send(formatFeedbackCsv(records));
  });

  // POST /api/v1/events (Telemetry encrypted at rest)
  app.post('/api/v1/events', requireClinicalToken, (req, res) => {
    try {
      const { type, requestId, timestamp, metadata } = req.body;
      const entry = {
        type: type || 'unknown',
        requestId: requestId || null,
        timestamp: timestamp || new Date().toISOString(),
        metadata: metadata || {},
      };
      appendEncryptedJsonl(eventsFile, entry, auditEncryptionKey);
      res.json({ logged: true });
    } catch {
      res.status(500).json({ logged: false });
    }
  });

  // GET /api/v1/jobs (List queued/completed worklist jobs)
  app.get('/api/v1/jobs', requireClinicalToken, (_req, res) => {
    const items = analysisQueue.listJobs(50).map((job) => ({
      id: job.jobId,
      ...job,
    }));
    res.json({ items, stats: analysisQueue.getStats() });
  });

  // GET /api/v1/jobs/:id (Live async worker job polling)
  app.get('/api/v1/jobs/:id', requireClinicalToken, (req, res) => {
    const id = String(req.params.id || '').trim();
    const job = analysisQueue.getJob(id);
    if (!job) {
      return res.status(404).json({ error_code: 'JOB_NOT_FOUND', error: 'Analysis job not found or expired.' });
    }
    return res.json({
      id: job.jobId,
      ...job,
    });
  });

  // GET /api/v1/system/scalability — Operational scalability, queue worker, memory & latency telemetry
  app.get('/api/v1/system/scalability', (_req, res) => {
    const mem = process.memoryUsage();
    const workerPool = analysisQueue.getStats();
    const latency = latencyTracker.getSummary();
    res.json({
      timestamp: new Date().toISOString(),
      requestId: res.locals.requestId || 'unknown',
      nodeVersion: process.version,
      uptimeSeconds: Math.round(process.uptime()),
      memoryMb: {
        rss: Math.round(mem.rss / (1024 * 1024)),
        heapUsed: Math.round(mem.heapUsed / (1024 * 1024)),
      },
      workerPool,
      ephemeralStore: {
        activeEntries: ephemeralStore.getActiveCount(),
        maxEntries: ephemeralMaxEntries,
        ttlSeconds: ephemeralTtlSeconds,
      },
      rateLimiter: {
        maxRequestsPerMinute: rateLimitRpm,
        trackedClients: uploadRateLimiter.getTrackedClientCount(),
      },
      latency,
    });
  });

  // GET /api/metrics — Prometheus exposition endpoint for Kubernetes / clinical cluster scraping
  app.get('/api/metrics', (_req, res) => {
    const mem = process.memoryUsage();
    const pool = analysisQueue.getStats();
    const lat = latencyTracker.getSummary();
    const lines = [
      '# HELP osteoguard_api_requests_total Total API requests observed by Express gateway',
      '# TYPE osteoguard_api_requests_total counter',
      `osteoguard_api_requests_total ${lat.totalRequests}`,
      '# HELP osteoguard_api_errors_total Total 5xx API errors observed',
      '# TYPE osteoguard_api_errors_total counter',
      `osteoguard_api_errors_total ${lat.totalErrors}`,
      '# HELP osteoguard_request_latency_p95_ms 95th percentile API latency in milliseconds',
      '# TYPE osteoguard_request_latency_p95_ms gauge',
      `osteoguard_request_latency_p95_ms ${lat.p95Ms}`,
      '# HELP osteoguard_worker_active Active scan processing workers',
      '# TYPE osteoguard_worker_active gauge',
      `osteoguard_worker_active ${pool.activeWorkers}`,
      '# HELP osteoguard_worker_queued Queued scan jobs waiting for worker slot',
      '# TYPE osteoguard_worker_queued gauge',
      `osteoguard_worker_queued ${pool.queuedDepth}`,
      '# HELP osteoguard_ephemeral_assets Active AES-256-GCM RAM-only image assets',
      '# TYPE osteoguard_ephemeral_assets gauge',
      `osteoguard_ephemeral_assets ${ephemeralStore.getActiveCount()}`,
      '# HELP osteoguard_process_rss_bytes Resident set size in bytes',
      '# TYPE osteoguard_process_rss_bytes gauge',
      `osteoguard_process_rss_bytes ${mem.rss}`,
    ];
    res.setHeader('Content-Type', 'text/plain; version=0.0.4; charset=utf-8');
    return res.send(lines.join('\n') + '\n');
  });

  app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    if (!req.path.startsWith('/api') || res.headersSent) return next(err);
    if (err?.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: 'Upload exceeds the 15 MiB limit.' });
    if (String(err?.message || '').includes('Unsupported image media type')) return res.status(415).json({ error: 'Unsupported image media type.' });
    return res.status(400).json({ error: 'Invalid API request or upload.' });
  });

  // Serve static production build when NODE_ENV=production and dist/ exists; otherwise mount Vite middleware
  const distDir = path.resolve(__dirname, 'dist');
  if (process.env.NODE_ENV === 'production' && fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    app.use((_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
    app.locals.closeDevelopmentServer = () => vite.close();
  }

  const httpServer = app.listen(PORT, HOST, () => {
    console.log(`[OsteoGuard Server] Full-stack application running at http://${HOST}:${PORT}`);
    console.log(`[OsteoGuard Server] AES-256-GCM at-rest ledger encryption + RAM-only EXIF-stripped image processing active`);
    console.log(`[OsteoGuard Server] Swagger UI available at http://${HOST}:${PORT}/docs`);
  });

  let shuttingDown = false;
  const shutdown = (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`[OsteoGuard Server] Received ${signal}; closing HTTP listener.`);
    const closeDev = app.locals.closeDevelopmentServer as (() => Promise<void>) | undefined;
    const closeHttp = () => httpServer.close((error) => {
      if (error) {
        console.error('[OsteoGuard Server] Graceful shutdown failed.');
        process.exitCode = 1;
      }
    });
    if (closeDev) void closeDev().catch(() => undefined).finally(closeHttp);
    else closeHttp();
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

startServer().catch((err) => {
  console.error('Failed to start unified server:', err);
  process.exit(1);
});
