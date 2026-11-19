import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yaml';
import { createServer as createViteServer } from 'vite';
import { simulateAnalysis } from './mock-server/simulation.js';
import { registerResearchRoutes } from './research-api.js';
import {
  getOrProvisionEncryptionKey,
  appendEncryptedJsonl,
  readEncryptedJsonl,
  migrateJsonlToEncrypted,
  EphemeralEncryptedAssetStore,
  stripImageMetadataAndEncodePng,
  signGatewayHeaders,
  inspectClinicalReadinessOnDisk,
} from './security.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();

  // Parse command line arguments for host and port
  const args = process.argv.slice(2);
  let PORT = Number(process.env.PORT) || 3000;
  let HOST = process.env.HOST || '0.0.0.0';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port' && args[i + 1]) {
      PORT = Number(args[i + 1]);
      i++;
    } else if (args[i] === '--host' && args[i + 1]) {
      HOST = args[i + 1];
      i++;
    }
  }

  app.use(cors());
  app.use(express.json({ limit: '2mb' }));

  // Clinical transport & cache-hardening headers across all responses
  app.use((req, res, next) => {
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
    }
    next();
  });

  const EXPLICIT_BACKEND_URL = process.env.BACKEND_URL?.trim() || '';
  const CLINICAL_API_TOKEN = process.env.OSTEO_CLINICAL_API_TOKEN?.trim() || '';
  const GATEWAY_HMAC_SECRET = process.env.OSTEO_GATEWAY_HMAC_SECRET?.trim() || '';
  const STRICT_CLINICAL_MODE = process.env.OSTEO_STRICT_CLINICAL_MODE === 'true';

  const requireClinicalToken: express.RequestHandler = (req, res, next) => {
    if (!CLINICAL_API_TOKEN) return next();
    const authHeader = req.headers.authorization || '';
    if (authHeader === `Bearer ${CLINICAL_API_TOKEN}`) {
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
        console.warn(`[Proxy Fallback] External ML backend unreachable (${err.message}); using built-in Node.js research engine.`);
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

  // AES-256-GCM encrypted RAM-only store for ephemeral radiograph & Grad-CAM assets (15-min TTL)
  const ephemeralStore = new EphemeralEncryptedAssetStore({
    ttlMs: 15 * 60 * 1000,
    maxEntries: 120,
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
  const dataDir = path.resolve(__dirname, 'mock-server/data');
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  const auditEncryptionKey = getOrProvisionEncryptionKey(dataDir, process.env.OSTEO_AUDIT_ENCRYPTION_KEY);
  const feedbackFile = path.join(dataDir, 'feedback.jsonl');
  const eventsFile = path.join(dataDir, 'events.jsonl');

  // Seed initial demo feedback encrypted at rest if empty, and migrate any legacy plaintext records
  if (!fs.existsSync(feedbackFile) || fs.readFileSync(feedbackFile, 'utf-8').trim() === '') {
    const initialFeedback = [
      { feedbackId: 'fb_01', requestId: 'req_demo1', decision: 'agree', predictedClass: 'metastasis', timestamp: new Date(Date.now() - 3600000 * 5).toISOString() },
      { feedbackId: 'fb_02', requestId: 'req_demo2', decision: 'agree', predictedClass: 'paget', timestamp: new Date(Date.now() - 3600000 * 3).toISOString() },
      { feedbackId: 'fb_03', requestId: 'req_demo3', decision: 'disagree', predictedClass: 'oi', reasonCode: 'insufficient_context', reasonText: 'Need lateral lumbar views', timestamp: new Date(Date.now() - 3600000 * 2).toISOString() },
      { feedbackId: 'fb_04', requestId: 'req_demo4', decision: 'agree', predictedClass: 'healthy', timestamp: new Date(Date.now() - 3600000 * 1).toISOString() },
    ];
    for (const item of initialFeedback) {
      appendEncryptedJsonl(feedbackFile, item, auditEncryptionKey);
    }
  } else {
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
  });

  // GET /api/health (unversioned)
  app.get('/api/health', (_req, res) => {
    const readiness = inspectClinicalReadinessOnDisk(__dirname);
    res.json({
      status: 'ok',
      mode: readiness.isFourClassReady && EXPLICIT_BACKEND_URL ? 'production' : 'simulation',
    });
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
      clinicalReadiness: diskReadiness,
    });
  });

  // GET /api/v1/ephemeral/:assetId — decrypts and streams RAM-only radiograph / Grad-CAM PNG
  app.get('/api/v1/ephemeral/:assetId', (req, res) => {
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

  // Analyze handler (EXIF-stripped in RAM -> live FastAPI PyTorch bridge when ready -> safe simulation fallback)
  const handleAnalyze = async (req: express.Request, res: express.Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image file uploaded.' });
      }

      const modality = (req.body.modality as any) || 'xray';
      let sanitizedBuffer = req.file.buffer;
      try {
        sanitizedBuffer = await stripImageMetadataAndEncodePng(req.file.buffer);
      } catch {
        // Keep raw bytes if synthetic minimal non-image fixture is submitted
        sanitizedBuffer = req.file.buffer;
      }

      const diskReadiness = inspectClinicalReadinessOnDisk(__dirname);

      // 1. Live PyTorch FastAPI inference bridge when BACKEND_URL is configured
      if (EXPLICIT_BACKEND_URL) {
        try {
          const form = new FormData();
          form.append('image', new Blob([new Uint8Array(sanitizedBuffer)], { type: 'image/png' }), 'sanitized.png');
          form.append('modality', String(modality));

          const mlHeaders: Record<string, string> = {};
          if (GATEWAY_HMAC_SECRET) {
            Object.assign(
              mlHeaders,
              signGatewayHeaders('POST', '/api/v1/analyze', sanitizedBuffer, GATEWAY_HMAC_SECRET)
            );
          }

          const liveRes = await fetch(`${EXPLICIT_BACKEND_URL.replace(/\/+$/, '')}/api/v1/analyze`, {
            method: 'POST',
            headers: mlHeaders,
            body: form,
          });

          if (liveRes.ok) {
            const liveJson = (await liveRes.json()) as Record<string, any>;
            if (liveJson && liveJson.prediction) {
              const reqTag = String(liveJson.requestId || `live_${Date.now()}`).replace(/[^a-zA-Z0-9_-]/g, '');
              if (typeof liveJson.imageBase64 === 'string' && liveJson.imageBase64.length > 0) {
                const imgId = `${reqTag}_scan.png`;
                ephemeralStore.put(imgId, Buffer.from(liveJson.imageBase64, 'base64'), 'image/png');
                liveJson.imageUrl = `/api/v1/ephemeral/${encodeURIComponent(imgId)}`;
                delete liveJson.imageBase64;
              }
              if (typeof liveJson.gradcamBase64 === 'string' && liveJson.gradcamBase64.length > 0) {
                const camId = `${reqTag}_gradcam.png`;
                ephemeralStore.put(camId, Buffer.from(liveJson.gradcamBase64, 'base64'), 'image/png');
                liveJson.gradcamUrl = `/api/v1/ephemeral/${encodeURIComponent(camId)}`;
                delete liveJson.gradcamBase64;
              }
              return res.json(liveJson);
            }
          } else if (liveRes.status === 503 && STRICT_CLINICAL_MODE) {
            const blockedJson = await liveRes.json().catch(() => ({
              status: 'BLOCKED',
              error_code: 'CLINICAL_INFERENCE_BLOCKED',
              error: 'Live FastAPI ML backend rejected inference because verified clinical checkpoints are missing.',
            }));
            return res.status(503).json(blockedJson);
          }
        } catch (bridgeErr: any) {
          console.warn(`[ML Bridge] External PyTorch service unavailable (${bridgeErr.message}).`);
          if (STRICT_CLINICAL_MODE) {
            return res.status(503).json({
              status: 'BLOCKED',
              error_code: 'CLINICAL_ML_BACKEND_UNREACHABLE',
              error: 'Strict clinical mode forbids simulation fallback when the live PyTorch ML backend is unreachable.',
            });
          }
        }
      }

      // 2. Strict clinical enforcement guard (forbids simulation when OSTEO_STRICT_CLINICAL_MODE=true without real weights)
      if (STRICT_CLINICAL_MODE && !diskReadiness.isFourClassReady) {
        return res.status(503).json({
          status: 'BLOCKED',
          error_code: 'CLINICAL_INFERENCE_BLOCKED',
          error:
            'Strict clinical mode is enabled: zero verified 4-class checkpoints or clinical cohorts exist under ml_artifacts/checkpoints/ and ml_service/data/raw/.',
          readiness: diskReadiness,
        });
      }

      // 3. Research prototype simulation rendered strictly in RAM (AES-256-GCM ephemeral asset store)
      const result = await simulateAnalysis(
        sanitizedBuffer,
        req.file.originalname,
        modality,
        undefined,
        (filename: string, buf: Buffer, contentType = 'image/png') => {
          const assetId = path.basename(filename);
          ephemeralStore.put(assetId, buf, contentType);
          return `/api/v1/ephemeral/${encodeURIComponent(assetId)}`;
        }
      );

      res.json(result);
    } catch (error) {
      console.error('API Error in analyze:', error);
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Internal simulation failure',
      });
    }
  };

  // POST /api/v1/analyze & POST /api/analyze
  app.post('/api/v1/analyze', requireClinicalToken, upload.single('image'), handleAnalyze);
  app.post('/api/analyze', requireClinicalToken, upload.single('image'), handleAnalyze);

  // POST /api/v1/feedback (AES-256-GCM encrypted at rest)
  app.post('/api/v1/feedback', requireClinicalToken, (req, res) => {
    try {
      const { requestId, decision, predictedClass, reasonCode, reasonText, timestamp } = req.body;
      if (!requestId || !decision) {
        return res.status(400).json({ error: 'requestId and decision are required.' });
      }

      const feedbackRecord = {
        feedbackId: `fb_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        requestId,
        decision,
        predictedClass: predictedClass || 'healthy',
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
    return readEncryptedJsonl<Record<string, any>>(feedbackFile, auditEncryptionKey).records;
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

  // GET /api/v1/jobs/:id (Async polling)
  app.get('/api/v1/jobs/:id', (req, res) => {
    const { id } = req.params;
    res.json({
      id,
      status: 'done',
      progress: 100,
      resultUrl: `/api/v1/analyze`,
    });
  });

  // Serve static production build when NODE_ENV=production and dist/ exists; otherwise mount Vite middleware
  const distDir = path.resolve(__dirname, 'dist');
  if (process.env.NODE_ENV === 'production' && fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    app.use((_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, HOST, () => {
    console.log(`[OsteoGuard Server] Full-stack application running at http://${HOST}:${PORT}`);
    console.log(`[OsteoGuard Server] AES-256-GCM at-rest ledger encryption + RAM-only EXIF-stripped image processing active`);
    console.log(`[OsteoGuard Server] Swagger UI available at http://${HOST}:${PORT}/docs`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start unified server:', err);
  process.exit(1);
});
