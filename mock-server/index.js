import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8787;

app.use(cors());
app.use(express.json());

// Data persistence directory for feedback and telemetry
const dataDir = path.resolve(__dirname, 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}
const feedbackFile = path.join(dataDir, 'feedback.jsonl');
const eventsFile = path.join(dataDir, 'events.jsonl');

// Load OpenAPI Spec for Swagger UI
try {
  const openApiPath = path.resolve(__dirname, '../openapi.yaml');
  if (fs.existsSync(openApiPath)) {
    const file = fs.readFileSync(openApiPath, 'utf8');
    const swaggerDocument = YAML.parse(file);
    app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));
  }
} catch (err) {
  console.warn('Failed to load openapi.yaml for Swagger UI:', err);
}

// ━━━ Unversioned Routes ━━━
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    mode: 'simulation',
  });
});

// ━━━ Version 1 Routes (/api/v1/*) ━━━

// This standalone mock server never fabricates image-derived clinical output.
const modelUnavailable = (_req, res) => res.status(503).json({
  status: 'BLOCKED',
  error_code: 'FOUR_CLASS_MODEL_NOT_READY',
  detail: 'The mock server has no verified clinical model and does not generate simulated patient predictions.',
});
app.post('/api/v1/analyze', modelUnavailable);
app.post('/api/analyze', modelUnavailable);

// POST /api/v1/feedback
app.post('/api/v1/feedback', (req, res) => {
  try {
    const { requestId, decision, predictedClass, reasonCode, reasonText, timestamp } = req.body;
    if (typeof requestId !== 'string' || requestId.length < 1 || requestId.length > 128
      || !['agree', 'disagree', 'unsure'].includes(decision)
      || !['metastasis', 'paget', 'oi', 'healthy'].includes(predictedClass)
      || (typeof reasonText === 'string' && reasonText.length > 2000)) {
      return res.status(400).json({ error: 'A valid request reference, review decision, and predicted class are required.' });
    }

    const feedbackRecord = {
      feedbackId: `fb_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      requestId,
      decision,
      predictedClass,
      reasonCode: reasonCode || null,
      reasonText: reasonText || '',
      timestamp: timestamp || new Date().toISOString(),
    };

    fs.appendFileSync(feedbackFile, JSON.stringify(feedbackRecord) + '\n');
    res.json({ success: true, feedbackId: feedbackRecord.feedbackId });
  } catch (err) {
    console.error('Error recording feedback:', err);
    res.status(500).json({ error: 'Failed to record feedback' });
  }
});

// Helper to read feedback records
function readFeedbackRecords() {
  if (!fs.existsSync(feedbackFile)) return [];
  const lines = fs.readFileSync(feedbackFile, 'utf-8').split('\n').filter(Boolean);
  return lines.map((line) => {
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  }).filter((record) => record && !(['fb_01', 'fb_02', 'fb_03', 'fb_04'].includes(record.feedbackId)
    && /^req_demo[1-4]$/.test(record.requestId || '')));
}

// GET /api/v1/feedback/stats
app.get('/api/v1/feedback/stats', (req, res) => {
  const records = readFeedbackRecords();
  let agree = 0;
  let disagree = 0;
  let unsure = 0;

  const byClass = {
    metastasis: { agree: 0, disagree: 0, unsure: 0 },
    paget: { agree: 0, disagree: 0, unsure: 0 },
    oi: { agree: 0, disagree: 0, unsure: 0 },
    healthy: { agree: 0, disagree: 0, unsure: 0 },
  };

  records.forEach((r) => {
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
app.get('/api/v1/feedback/list', (req, res) => {
  const records = readFeedbackRecords();
  res.json({ items: records.reverse() });
});

// POST /api/v1/events (Telemetry)
app.post('/api/v1/events', (req, res) => {
  try {
    const { type, requestId, timestamp, metadata } = req.body;
    const entry = {
      type: type || 'unknown',
      requestId: requestId || null,
      timestamp: timestamp || new Date().toISOString(),
      metadata: metadata || {},
    };
    fs.appendFileSync(eventsFile, JSON.stringify(entry) + '\n');
    res.json({ logged: true });
  } catch (err) {
    res.status(500).json({ logged: false });
  }
});

// GET /api/v1/jobs/:id (Async Inference Seam)
app.get('/api/v1/jobs/:id', (req, res) => {
  const { id } = req.params;
  res.json({
    id,
    status: 'done',
    progress: 100,
    resultUrl: `/api/v1/analyze`,
  });
});

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[OsteoGuard Mock Server] Running at http://0.0.0.0:${PORT}`);
    console.log(`[OsteoGuard Mock Server] Swagger docs available at http://0.0.0.0:${PORT}/docs`);
  });
}

export { app, publicUploadsDir };
