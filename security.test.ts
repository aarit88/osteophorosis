import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import sharp from 'sharp';
import {
  getOrProvisionEncryptionKey,
  encryptPayload,
  decryptPayload,
  appendEncryptedJsonl,
  readEncryptedJsonl,
  migrateJsonlToEncrypted,
  EphemeralEncryptedAssetStore,
  stripImageMetadataAndEncodePng,
  signGatewayHeaders,
  verifyGatewaySignature,
  inspectClinicalReadinessOnDisk,
} from './security';

test('AES-256-GCM encryptPayload and decryptPayload round-trip and detect tampering', () => {
  const key = crypto.randomBytes(32);
  const original = JSON.stringify({ requestId: 'req_123', decision: 'agree' });
  const envelope = encryptPayload(original, key);

  assert.equal(envelope.includes('req_123'), false, 'Ciphertext must not leak plaintext');
  const decrypted = decryptPayload(envelope, key).toString('utf8');
  assert.equal(decrypted, original);

  const parsed = JSON.parse(envelope);
  parsed.ct = Buffer.from('tampered_payload_bytes').toString('base64');
  assert.throws(() => decryptPayload(JSON.stringify(parsed), key));
});

test('Encrypted JSONL append, read, and legacy migration work cleanly', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'osteoguard-sec-'));
  try {
    const key = getOrProvisionEncryptionKey(tmpDir);
    const jsonlPath = path.join(tmpDir, 'feedback.jsonl');

    // Seed one legacy plaintext record + one encrypted record
    fs.writeFileSync(jsonlPath, JSON.stringify({ id: 'legacy_1', decision: 'agree' }) + '\n');
    appendEncryptedJsonl(jsonlPath, { id: 'enc_2', decision: 'disagree' }, key);

    const before = readEncryptedJsonl<{ id: string }>(jsonlPath, key);
    assert.equal(before.records.length, 2);
    assert.equal(before.legacyPlaintextCount, 1);
    assert.equal(before.encryptedCount, 1);

    const migrated = migrateJsonlToEncrypted(jsonlPath, key);
    assert.equal(migrated, 1);

    const after = readEncryptedJsonl<{ id: string }>(jsonlPath, key);
    assert.equal(after.records.length, 2);
    assert.equal(after.legacyPlaintextCount, 0);
    assert.equal(after.encryptedCount, 2);
    assert.equal(after.records[0].id, 'legacy_1');
    assert.equal(after.records[1].id, 'enc_2');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('EphemeralEncryptedAssetStore stores buffers encrypted in RAM and expires by TTL', async () => {
  const store = new EphemeralEncryptedAssetStore({ ttlMs: 40, maxEntries: 5 });
  const rawBytes = Buffer.from('sensitive-radiograph-png-bytes');
  store.put('asset_1.png', rawBytes, 'image/png');

  const fetched = store.get('asset_1.png');
  assert.ok(fetched);
  assert.equal(fetched.contentType, 'image/png');
  assert.deepEqual(fetched.buffer, rawBytes);

  await new Promise((res) => setTimeout(res, 60));
  assert.equal(store.get('asset_1.png'), null);
});

test('stripImageMetadataAndEncodePng removes EXIF metadata while preserving image dimensions', async () => {
  const rawWithExif = await sharp({
    create: { width: 32, height: 32, channels: 3, background: { r: 40, g: 40, b: 40 } },
  })
    .jpeg()
    .withMetadata({ exif: { IFD0: { ImageDescription: 'PATIENT_JOHN_DOE_MRN_999' } } })
    .toBuffer();

  const metaBefore = await sharp(rawWithExif).metadata();
  assert.ok(metaBefore.exif, 'Test fixture should contain EXIF metadata before stripping');

  const sanitized = await stripImageMetadataAndEncodePng(rawWithExif);
  const metaAfter = await sharp(sanitized).metadata();
  assert.equal(metaAfter.exif, undefined);
  assert.equal(metaAfter.format, 'png');
  assert.equal(metaAfter.width, 32);
  assert.equal(metaAfter.height, 32);
});

test('HMAC-SHA256 gateway signing validates authentic requests and rejects tampering or replay', () => {
  const secret = 'test-gateway-hmac-secret';
  const body = Buffer.from('{"modality":"xray"}');
  const headers = signGatewayHeaders('POST', '/api/v1/analyze', body, secret, 1700000000);

  assert.equal(
    verifyGatewaySignature(
      'POST',
      '/api/v1/analyze',
      body,
      secret,
      headers['x-osteoguard-timestamp'],
      headers['x-osteoguard-signature'],
      300,
      1700000100
    ),
    true
  );

  // Tampered body fails
  assert.equal(
    verifyGatewaySignature(
      'POST',
      '/api/v1/analyze',
      Buffer.from('{"modality":"ct"}'),
      secret,
      headers['x-osteoguard-timestamp'],
      headers['x-osteoguard-signature'],
      300,
      1700000100
    ),
    false
  );

  // Expired timestamp fails
  assert.equal(
    verifyGatewaySignature(
      'POST',
      '/api/v1/analyze',
      body,
      secret,
      headers['x-osteoguard-timestamp'],
      headers['x-osteoguard-signature'],
      300,
      1700001000
    ),
    false
  );
});

test('inspectClinicalReadinessOnDisk detects checkpoints and raw class folders accurately', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'osteoguard-ready-'));
  try {
    const empty = inspectClinicalReadinessOnDisk(tmpRoot);
    assert.equal(empty.checkpointsFound.length, 0);
    assert.equal(empty.isFourClassReady, false);

    fs.mkdirSync(path.join(tmpRoot, 'ml_artifacts', 'checkpoints'), { recursive: true });
    fs.mkdirSync(path.join(tmpRoot, 'ml_artifacts', 'logs'), { recursive: true });
    fs.writeFileSync(path.join(tmpRoot, 'ml_artifacts', 'checkpoints', 'model.pt'), 'fake-weights-bytes');
    fs.writeFileSync(path.join(tmpRoot, 'ml_artifacts', 'logs', 'train.log'), 'epoch 1 loss 0.2');

    for (const cls of ['metastasis', 'paget', 'oi', 'healthy']) {
      const dir = path.join(tmpRoot, 'ml_service', 'data', 'raw', cls);
      fs.mkdirSync(dir, { recursive: true });
      for (let i = 0; i < 10; i++) {
        fs.writeFileSync(path.join(dir, `img_${i}.png`), 'png');
      }
    }

    const ready = inspectClinicalReadinessOnDisk(tmpRoot);
    assert.equal(ready.checkpointsFound.length, 1);
    assert.equal(ready.trainingLogsCount, 1);
    assert.equal(ready.isFourClassReady, true);
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});
