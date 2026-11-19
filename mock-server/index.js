import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yaml';
import { simulateAnalysis } from './simulation.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8787;

app.use(cors());
app.use(express.json());

// Public uploads directory
const publicUploadsDir = path.resolve(__dirname, '../public/uploads');
if (!fs.existsSync(publicUploadsDir)) {
  fs.mkdirSync(publicUploadsDir, { recursive: true });
}
app.use('/uploads', express.static(publicUploadsDir));

// Data persistence directory for feedback and telemetry
const dataDir = path.resolve(__dirname, 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}
const feedbackFile = path.join(dataDir, 'feedback.jsonl');
const eventsFile = path.join(dataDir, 'events.jsonl');

// Seed initial demo feedback if empty
if (!fs.existsSync(feedbackFile) || fs.readFileSync(feedbackFile, 'utf-8').trim() === '') {
  const initialFeedback = [
    { feedbackId: 'fb_01', requestId: 'req_demo1', decision: 'agree', predictedClass: 'metastasis', timestamp: new Date(Date.now() - 3600000 * 5).toISOString() },
    { feedbackId: 'fb_02', requestId: 'req_demo2', decision: 'agree', predictedClass: 'paget', timestamp: new Date(Date.now() - 3600000 * 3).toISOString() },
    { feedbackId: 'fb_03', requestId: 'req_demo3', decision: 'disagree', predictedClass: 'oi', reasonCode: 'insufficient_context', reasonText: 'Need lateral lumbar views', timestamp: new Date(Date.now() - 3600000 * 2).toISOString() },
    { feedbackId: 'fb_04', requestId: 'req_demo4', decision: 'agree', predictedClass: 'healthy', timestamp: new Date(Date.now() - 3600000 * 1).toISOString() },
  ];
  fs.writeFileSync(feedbackFile, initialFeedback.map((f) => JSON.stringify(f)).join('\n') + '\n');
}

// Memory storage for uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
});

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

// POST /api/v1/analyze
const handleAnalyze = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file was uploaded. Please attach an image.' });
    }

    const modality = req.body.modality || 'xray';
    const result = await simulateAnalysis(
      req.file.buffer,
      req.file.originalname,
      modality,
      publicUploadsDir
    );

    res.json(result);
  } catch (error) {
    console.error('Error during simulated analysis:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Internal simulation failure',
    });
  }
};

app.post('/api/v1/analyze', upload.single('image'), handleAnalyze);
// Backward compatibility alias:
app.post('/api/analyze', upload.single('image'), handleAnalyze);

// POST /api/v1/feedback
app.post('/api/v1/feedback', (req, res) => {
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
  }).filter(Boolean);
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
