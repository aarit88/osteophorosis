import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';

export interface EncryptedEnvelope {
  v: 1;
  alg: 'aes-256-gcm';
  iv: string;
  tag: string;
  ct: string;
}

const AUDIT_KEY_FILENAME = '.audit-secret.key';

/**
 * Resolves a 32-byte AES-256 encryption key from env or provisions a local 0600 keyfile.
 */
export function getOrProvisionEncryptionKey(dataDir: string, envKey?: string): Buffer {
  const configured = (envKey || '').trim();
  if (configured) {
    if (/^[0-9a-fA-F]{64}$/.test(configured)) {
      return Buffer.from(configured, 'hex');
    }
    try {
      const b64 = Buffer.from(configured, 'base64');
      if (b64.length === 32) {
        return b64;
      }
    } catch {
      // fall through to SHA-256 key derivation
    }
    return crypto.createHash('sha256').update(configured, 'utf8').digest();
  }

  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  const keyPath = path.join(dataDir, AUDIT_KEY_FILENAME);
  if (fs.existsSync(keyPath)) {
    const raw = fs.readFileSync(keyPath, 'utf8').trim();
    if (/^[0-9a-fA-F]{64}$/.test(raw)) {
      return Buffer.from(raw, 'hex');
    }
  }

  const newKey = crypto.randomBytes(32);
  fs.writeFileSync(keyPath, newKey.toString('hex') + '\n', { mode: 0o600 });
  try {
    fs.chmodSync(keyPath, 0o600);
  } catch {
    // Ignore on filesystems that do not support POSIX chmod
  }
  return newKey;
}

export function encryptPayload(plaintext: string | Buffer, key: Buffer): string {
  if (key.length !== 32) {
    throw new Error('AES-256-GCM requires a 32-byte encryption key');
  }
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  const input = typeof plaintext === 'string' ? Buffer.from(plaintext, 'utf8') : plaintext;
  const encrypted = Buffer.concat([cipher.update(input), cipher.final()]);
  const tag = cipher.getAuthTag();

  const envelope: EncryptedEnvelope = {
    v: 1,
    alg: 'aes-256-gcm',
    iv: iv.toString('base64'),
    tag: tag.toString('base64'),
    ct: encrypted.toString('base64'),
  };
  return JSON.stringify(envelope);
}

export function decryptPayload(envelopeJson: string, key: Buffer): Buffer {
  if (key.length !== 32) {
    throw new Error('AES-256-GCM requires a 32-byte encryption key');
  }
  const parsed = JSON.parse(envelopeJson) as Partial<EncryptedEnvelope>;
  if (parsed.v !== 1 || parsed.alg !== 'aes-256-gcm' || !parsed.iv || !parsed.tag || !parsed.ct) {
    throw new Error('Invalid encrypted payload envelope');
  }

  const decipher = crypto.createDecipheriv(
    'aes-256-gcm',
    key,
    Buffer.from(parsed.iv, 'base64')
  );
  decipher.setAuthTag(Buffer.from(parsed.tag, 'base64'));
  return Buffer.concat([
    decipher.update(Buffer.from(parsed.ct, 'base64')),
    decipher.final(),
  ]);
}

export function isEncryptedEnvelopeLine(line: string): boolean {
  try {
    const parsed = JSON.parse(line) as Partial<EncryptedEnvelope>;
    return parsed.v === 1 && parsed.alg === 'aes-256-gcm' && typeof parsed.ct === 'string';
  } catch {
    return false;
  }
}

export function appendEncryptedJsonl(filePath: string, record: unknown, key: Buffer): void {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const encryptedLine = encryptPayload(JSON.stringify(record), key);
  fs.appendFileSync(filePath, encryptedLine + '\n', { mode: 0o600 });
}

export function readEncryptedJsonl<T = Record<string, unknown>>(
  filePath: string,
  key: Buffer
): { records: T[]; encryptedCount: number; legacyPlaintextCount: number } {
  if (!fs.existsSync(filePath)) {
    return { records: [], encryptedCount: 0, legacyPlaintextCount: 0 };
  }
  const lines = fs
    .readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0);

  const records: T[] = [];
  let encryptedCount = 0;
  let legacyPlaintextCount = 0;

  for (const line of lines) {
    try {
      if (isEncryptedEnvelopeLine(line)) {
        const plainBuf = decryptPayload(line, key);
        records.push(JSON.parse(plainBuf.toString('utf8')) as T);
        encryptedCount++;
      } else {
        records.push(JSON.parse(line) as T);
        legacyPlaintextCount++;
      }
    } catch {
      // Skip malformed/unreadable lines without crashing reader
    }
  }

  return { records, encryptedCount, legacyPlaintextCount };
}

/**
 * Re-encrypts any legacy plaintext lines in a JSONL file so all stored records use AES-256-GCM.
 */
export function migrateJsonlToEncrypted(filePath: string, key: Buffer): number {
  const { records, legacyPlaintextCount } = readEncryptedJsonl(filePath, key);
  if (legacyPlaintextCount === 0) return 0;

  const lines = records.map((r) => encryptPayload(JSON.stringify(r), key));
  fs.writeFileSync(filePath, lines.join('\n') + (lines.length ? '\n' : ''), { mode: 0o600 });
  return legacyPlaintextCount;
}

interface StoredRamAsset {
  iv: Buffer;
  tag: Buffer;
  ciphertext: Buffer;
  contentType: string;
  expiresAt: number;
}

/**
 * Stores ephemeral radiographic / Grad-CAM PNG buffers encrypted in RAM (AES-256-GCM)
 * with TTL expiration and zero disk persistence.
 */
export class EphemeralEncryptedAssetStore {
  private readonly key: Buffer;
  private readonly ttlMs: number;
  private readonly maxEntries: number;
  private readonly store = new Map<string, StoredRamAsset>();

  constructor(options: { ttlMs?: number; maxEntries?: number } = {}) {
    this.key = crypto.randomBytes(32);
    this.ttlMs = options.ttlMs ?? 15 * 60 * 1000;
    this.maxEntries = options.maxEntries ?? 120;
  }

  put(id: string, data: Buffer, contentType = 'image/png'): string {
    this.pruneExpired();
    if (this.store.size >= this.maxEntries) {
      const oldestKey = this.store.keys().next().value;
      if (oldestKey) this.store.delete(oldestKey);
    }

    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv('aes-256-gcm', this.key, iv);
    const ciphertext = Buffer.concat([cipher.update(data), cipher.final()]);
    const tag = cipher.getAuthTag();

    this.store.set(id, {
      iv,
      tag,
      ciphertext,
      contentType,
      expiresAt: Date.now() + this.ttlMs,
    });
    return id;
  }

  get(id: string): { buffer: Buffer; contentType: string } | null {
    const entry = this.store.get(id);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(id);
      return null;
    }

    const decipher = crypto.createDecipheriv('aes-256-gcm', this.key, entry.iv);
    decipher.setAuthTag(entry.tag);
    const buffer = Buffer.concat([decipher.update(entry.ciphertext), decipher.final()]);
    return { buffer, contentType: entry.contentType };
  }

  getActiveCount(): number {
    this.pruneExpired();
    return this.store.size;
  }

  pruneExpired(): number {
    const now = Date.now();
    let removed = 0;
    for (const [k, v] of this.store.entries()) {
      if (now > v.expiresAt) {
        this.store.delete(k);
        removed++;
      }
    }
    return removed;
  }
}

/**
 * Strips EXIF, XMP, IPTC, and GPS metadata from an uploaded image buffer in RAM
 * and normalizes orientation into a clean PNG buffer.
 */
export async function stripImageMetadataAndEncodePng(imageBuffer: Buffer): Promise<Buffer> {
  return sharp(imageBuffer).rotate().png().toBuffer();
}

export function signGatewayHeaders(
  method: string,
  reqPath: string,
  bodyBytes: Buffer | string,
  secret: string,
  timestampSec = Math.floor(Date.now() / 1000)
): { 'x-osteoguard-timestamp': string; 'x-osteoguard-signature': string } {
  const bodyHash = crypto.createHash('sha256').update(bodyBytes).digest('hex');
  const payload = `${timestampSec}:${method.toUpperCase()}:${reqPath}:${bodyHash}`;
  const sig = crypto.createHmac('sha256', secret).update(payload).digest('hex');
  return {
    'x-osteoguard-timestamp': String(timestampSec),
    'x-osteoguard-signature': `sha256=${sig}`,
  };
}

export function verifyGatewaySignature(
  method: string,
  reqPath: string,
  bodyBytes: Buffer | string,
  secret: string,
  timestampHeader: string,
  signatureHeader: string,
  maxSkewSeconds = 300,
  nowSec = Math.floor(Date.now() / 1000)
): boolean {
  const ts = Number(timestampHeader);
  if (!Number.isFinite(ts) || Math.abs(nowSec - ts) > maxSkewSeconds) {
    return false;
  }
  const expected = signGatewayHeaders(method, reqPath, bodyBytes, secret, ts)['x-osteoguard-signature'];
  const actualBuf = Buffer.from(signatureHeader || '', 'utf8');
  const expectedBuf = Buffer.from(expected, 'utf8');
  if (actualBuf.length !== expectedBuf.length) return false;
  return crypto.timingSafeEqual(actualBuf, expectedBuf);
}

export interface CheckpointArtifactInfo {
  filename: string;
  sizeBytes: number;
  sha256: string;
}

export interface ClinicalDiskReadiness {
  checkpointsDir: string;
  checkpointsFound: CheckpointArtifactInfo[];
  trainingLogsCount: number;
  rawDataDir: string;
  rawClassCounts: Record<'metastasis' | 'paget' | 'oi' | 'healthy', number>;
  isFourClassReady: boolean;
}

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.dcm', '.dicom', '.tif', '.tiff']);

/**
 * Inspects actual on-disk PyTorch checkpoints, logs, and 4-class clinical datasets.
 */
export function inspectClinicalReadinessOnDisk(rootDir: string): ClinicalDiskReadiness {
  const checkpointsDir = path.join(rootDir, 'ml_artifacts', 'checkpoints');
  const logsDir = path.join(rootDir, 'ml_artifacts', 'logs');
  const rawDataDir = path.join(rootDir, 'ml_service', 'data', 'raw');

  const checkpointsFound: CheckpointArtifactInfo[] = [];
  if (fs.existsSync(checkpointsDir)) {
    for (const name of fs.readdirSync(checkpointsDir)) {
      if (name.endsWith('.pt') || name.endsWith('.pth') || name.endsWith('.safetensors')) {
        const full = path.join(checkpointsDir, name);
        const stat = fs.statSync(full);
        if (stat.isFile()) {
          const sha256 = crypto.createHash('sha256').update(fs.readFileSync(full)).digest('hex').toUpperCase();
          checkpointsFound.push({ filename: name, sizeBytes: stat.size, sha256 });
        }
      }
    }
  }

  let trainingLogsCount = 0;
  if (fs.existsSync(logsDir)) {
    trainingLogsCount = fs
      .readdirSync(logsDir)
      .filter((f) => f.endsWith('.log') || f.endsWith('.csv') || f.endsWith('.json')).length;
  }

  const rawClassCounts: Record<'metastasis' | 'paget' | 'oi' | 'healthy', number> = {
    metastasis: 0,
    paget: 0,
    oi: 0,
    healthy: 0,
  };

  for (const cls of ['metastasis', 'paget', 'oi', 'healthy'] as const) {
    const clsDir = path.join(rawDataDir, cls);
    if (fs.existsSync(clsDir)) {
      rawClassCounts[cls] = fs
        .readdirSync(clsDir)
        .filter((f) => IMAGE_EXTENSIONS.has(path.extname(f).toLowerCase())).length;
    }
  }

  const allClassesPopulated = Object.values(rawClassCounts).every((n) => n >= 10);
  const isFourClassReady = checkpointsFound.length > 0 && trainingLogsCount > 0 && allClassesPopulated;

  return {
    checkpointsDir: 'ml_artifacts/checkpoints',
    checkpointsFound,
    trainingLogsCount,
    rawDataDir: 'ml_service/data/raw',
    rawClassCounts,
    isFourClassReady,
  };
}
