import type { ModalityType } from '../api/types';

export type DiagnosticPortalModality = 'xray' | 'ct' | 'mri';

export interface WorkbenchPortalSpec {
  modality: DiagnosticPortalModality;
  title: string;
  shortTitle: string;
  endpointLabel: string;
  researchEndpoint: string;
  calibrationProtocol: string;
  acceptedFormatsLabel: string;
  targetModules: Array<{ label: string; path: string }>;
  defaultWorkbenchParams: {
    radiomics_phantom: string;
    geometry_phantom: string;
    grid_size: number;
    spacing_mm: number;
    voxel_size_mm: number;
    site: number;
    pain: number;
    radiographic_nature: number;
    lesion_size: number;
    target_class: number;
    distribution_preset: string;
  };
}

export const DIAGNOSTIC_UPLOAD_PORTALS: Record<DiagnosticPortalModality, WorkbenchPortalSpec> = {
  xray: {
    modality: 'xray',
    title: 'X-ray Upload Portal',
    shortTitle: 'X-ray',
    endpointLabel: 'POST /api/v1/preliminary/track-a/analyze · POST /api/v1/analyze',
    researchEndpoint: '/api/v1/workbench/run-demo',
    calibrationProtocol: '8×8 CLAHE contrast normalization (clip 2.0) · 224×224 Track A / 512×512 fusion intake',
    acceptedFormatsLabel: 'PNG, JPEG, or WebP radiograph up to 15 MB',
    targetModules: [
      { label: 'Track A Status', path: '/api/v1/preliminary/track-a/status' },
      { label: 'Radiomics 102-D', path: '/api/v1/radiomics/demo' },
      { label: 'Grad-CAM XAI', path: '/api/v1/explainability/demo' },
    ],
    defaultWorkbenchParams: {
      radiomics_phantom: 'checkerboard',
      geometry_phantom: 'intact_femur_diaphysis',
      grid_size: 32,
      spacing_mm: 1.0,
      voxel_size_mm: 0.5,
      site: 2,
      pain: 2,
      radiographic_nature: 2,
      lesion_size: 2,
      target_class: 1,
      distribution_preset: 'moderately_confident',
    },
  },
  ct: {
    modality: 'ct',
    title: 'CT Scan Upload Portal',
    shortTitle: 'CT Scan',
    endpointLabel: 'POST /api/v1/analyze · POST /api/v1/diagnostic/four-class',
    researchEndpoint: '/api/v1/workbench/run-demo',
    calibrationProtocol: 'Hounsfield Bone Window (WL +400 HU · WW 1500 HU) · 0.5 mm cortical cross-section profiling',
    acceptedFormatsLabel: 'PNG, JPEG, or WebP bone-window CT slice up to 15 MB',
    targetModules: [
      { label: 'Track D Gate', path: '/api/v1/diagnostic/four-class' },
      { label: 'Biomechanics J', path: '/api/v1/biomechanics/geometry' },
      { label: 'Radiomics 102-D', path: '/api/v1/radiomics/demo' },
    ],
    defaultWorkbenchParams: {
      radiomics_phantom: 'lytic_metastasis',
      geometry_phantom: 'focal_anterior_defect',
      grid_size: 64,
      spacing_mm: 0.5,
      voxel_size_mm: 0.5,
      site: 3,
      pain: 2,
      radiographic_nature: 3,
      lesion_size: 2,
      target_class: 0,
      distribution_preset: 'highly_confident',
    },
  },
  mri: {
    modality: 'mri',
    title: 'MRI Upload Portal',
    shortTitle: 'MRI',
    endpointLabel: 'POST /api/v1/analyze · POST /api/v1/workbench/run-demo',
    researchEndpoint: '/api/v1/workbench/run-demo',
    calibrationProtocol: 'T1/T2 marrow percentile normalization (1%–99%) · intramedullary edema & trabecular heterogeneity',
    acceptedFormatsLabel: 'PNG, JPEG, or WebP MRI marrow slice up to 15 MB',
    targetModules: [
      { label: 'Track D Gate', path: '/api/v1/diagnostic/four-class' },
      { label: 'Uncertainty H', path: '/api/v1/uncertainty/demo' },
      { label: 'Grad-CAM XAI', path: '/api/v1/explainability/demo' },
    ],
    defaultWorkbenchParams: {
      radiomics_phantom: 'trabecular_paget',
      geometry_phantom: 'eccentric_thinning',
      grid_size: 32,
      spacing_mm: 1.0,
      voxel_size_mm: 0.5,
      site: 2,
      pain: 2,
      radiographic_nature: 2,
      lesion_size: 2,
      target_class: 1,
      distribution_preset: 'moderately_confident',
    },
  },
};

export const PORTAL_ORDER: readonly DiagnosticPortalModality[] = ['xray', 'ct', 'mri'];
export const MAX_PORTAL_IMAGE_BYTES = 15 * 1024 * 1024;
const ALLOWED_PORTAL_MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/webp']);

export interface PortalFileValidationResult {
  valid: boolean;
  reason: string | null;
}

export function validatePortalScanFile(
  file: { type: string; size: number } | null | undefined,
  modality: DiagnosticPortalModality,
): PortalFileValidationResult {
  if (!file) {
    return {
      valid: false,
      reason: `Select a ${DIAGNOSTIC_UPLOAD_PORTALS[modality].shortTitle} scan before dispatching.`,
    };
  }
  const normalizedType = (file.type || '').trim().toLowerCase();
  if (!ALLOWED_PORTAL_MIME_TYPES.has(normalizedType)) {
    return {
      valid: false,
      reason: `Choose a PNG, JPEG, or WebP ${DIAGNOSTIC_UPLOAD_PORTALS[modality].shortTitle} image. Unconverted DICOM and arbitrary binary files are rejected by the gateway.`,
    };
  }
  if (!Number.isFinite(file.size) || file.size <= 0 || file.size > MAX_PORTAL_IMAGE_BYTES) {
    return {
      valid: false,
      reason: 'Image exceeds the 15 MB RAM-only intake limit.',
    };
  }
  return { valid: true, reason: null };
}

export function buildPortalResearchPayload(modality: DiagnosticPortalModality): Record<string, string | number | boolean> {
  const spec = DIAGNOSTIC_UPLOAD_PORTALS[modality];
  return {
    modality,
    ...spec.defaultWorkbenchParams,
  };
}

export function isDiagnosticPortalModality(modality: ModalityType): modality is DiagnosticPortalModality {
  return modality === 'xray' || modality === 'ct' || modality === 'mri';
}
