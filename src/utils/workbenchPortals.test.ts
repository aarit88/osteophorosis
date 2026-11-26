import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildPortalResearchPayload,
  DIAGNOSTIC_UPLOAD_PORTALS,
  MAX_PORTAL_IMAGE_BYTES,
  PORTAL_ORDER,
  validatePortalScanFile,
} from './workbenchPortals.js';

describe('workbenchPortals pure helpers', () => {
  it('defines distinct portal configurations for X-ray, CT scan, and MRI', () => {
    assert.deepEqual([...PORTAL_ORDER], ['xray', 'ct', 'mri']);
    assert.match(DIAGNOSTIC_UPLOAD_PORTALS.xray.title, /X-ray/i);
    assert.match(DIAGNOSTIC_UPLOAD_PORTALS.ct.title, /CT Scan/i);
    assert.match(DIAGNOSTIC_UPLOAD_PORTALS.mri.title, /MRI/i);
    assert.match(DIAGNOSTIC_UPLOAD_PORTALS.ct.calibrationProtocol, /Hounsfield/i);
    assert.match(DIAGNOSTIC_UPLOAD_PORTALS.mri.calibrationProtocol, /T1\/T2/i);
  });

  it('validates MIME types and 15 MB limits cleanly across portals', () => {
    assert.deepEqual(
      validatePortalScanFile({ type: 'image/png', size: 1024 }, 'ct'),
      { valid: true, reason: null },
    );
    assert.deepEqual(
      validatePortalScanFile({ type: 'image/webp', size: 4096 }, 'mri'),
      { valid: true, reason: null },
    );
    const badMime = validatePortalScanFile({ type: 'application/pdf', size: 512 }, 'mri');
    assert.equal(badMime.valid, false);
    assert.match(badMime.reason || '', /MRI/i);

    const tooLarge = validatePortalScanFile({ type: 'image/png', size: MAX_PORTAL_IMAGE_BYTES + 1 }, 'xray');
    assert.equal(tooLarge.valid, false);
    assert.match(tooLarge.reason || '', /15 MB/);
  });

  it('builds distinct modality-aligned FastAPI research payloads', () => {
    const xray = buildPortalResearchPayload('xray');
    const ct = buildPortalResearchPayload('ct');
    const mri = buildPortalResearchPayload('mri');

    assert.equal(xray.modality, 'xray');
    assert.equal(xray.radiomics_phantom, 'checkerboard');
    assert.equal(ct.modality, 'ct');
    assert.equal(ct.radiomics_phantom, 'lytic_metastasis');
    assert.equal(ct.geometry_phantom, 'focal_anterior_defect');
    assert.equal(mri.modality, 'mri');
    assert.equal(mri.radiomics_phantom, 'trabecular_paget');
    assert.equal(mri.geometry_phantom, 'eccentric_thinning');
  });
});
