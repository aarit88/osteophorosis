import type { Express, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'url';
import {
  evaluateFourClassClinicalState,
  buildTrackDPayload,
  buildTrackDEvidenceItem,
  buildFourClassGateResponse,
  buildDynamicCohortList,
  getTrackDDatasetReadiness,
} from './clinical-readiness.js';
import {
  getDatasetSourcesResponse,
  getDatasetAcquisitionStatusResponse,
  getDatasetGovernanceStatusResponse,
  getCohortPlanningResponse,
  validateWorkflowTransitionRequest,
} from './dataset-governance.js';

const __researchFilename = fileURLToPath(import.meta.url);
const __researchDirname = path.dirname(__researchFilename);
const ML_ARTIFACTS_DIR = path.resolve(__researchDirname, 'ml_artifacts');
const TRACK_C_CHECKPOINT_SHA256 = '21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD';

function verifyFrozenTrackCCheckpoint(): boolean {
  const checkpoint = path.join(ML_ARTIFACTS_DIR, 'btxrd_frozen_model', 'best_checkpoint.pt');
  try {
    const stat = fs.lstatSync(checkpoint);
    if (!stat.isFile() || stat.isSymbolicLink()) return false;
    const realPath = fs.realpathSync(checkpoint);
    const relative = path.relative(ML_ARTIFACTS_DIR, realPath);
    if (relative.startsWith('..') || path.isAbsolute(relative)) return false;
    const digest = crypto.createHash('sha256').update(fs.readFileSync(realPath)).digest('hex').toUpperCase();
    return digest === TRACK_C_CHECKPOINT_SHA256;
  } catch {
    return false;
  }
}

const TARGET_CLASSES = ['metastasis', 'paget', 'oi', 'healthy'];
const CLASS_DISPLAY_NAMES: Record<string, string> = {
  metastasis: 'Secondary Bone Metastasis',
  paget: "Paget's Disease of Bone",
  oi: 'Osteogenesis Imperfecta (OI)',
  healthy: 'Healthy Skeletal Control',
};

// ── Digital Phantom & Radiomics Engine (Stage 27) ────────────────────────────
function generateRadiomicsPhantom(phantomType: string, gridSize: number): { image: Float64Array; mask: Uint8Array } {
  const n = gridSize * gridSize;
  const image = new Float64Array(n);
  const mask = new Uint8Array(n);
  const ptype = phantomType.toLowerCase();

  if (ptype === 'checkerboard') {
    const tile = Math.max(2, Math.floor(gridSize / 4));
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const idx = y * gridSize + x;
        const parity = (Math.floor(y / tile) + Math.floor(x / tile)) % 2;
        image[idx] = parity === 0 ? 50.0 : 200.0;
        mask[idx] = 1;
      }
    }
  } else if (ptype === 'linear_gradient') {
    const denom = Math.max(1, 2 * (gridSize - 1));
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const idx = y * gridSize + x;
        image[idx] = 20.0 + (220.0 * (x + y)) / denom;
        mask[idx] = 1;
      }
    }
  } else if (ptype === 'annulus') {
    const cy = (gridSize - 1) / 2.0;
    const cx = (gridSize - 1) / 2.0;
    const rOuter = gridSize * 0.42;
    const rInner = gridSize * 0.22;
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const idx = y * gridSize + x;
        const r = Math.hypot(x - cx, y - cy);
        if (r >= rInner && r <= rOuter) {
          image[idx] = 180.0 + 40.0 * Math.cos((r - rInner) / Math.max(1, rOuter - rInner) * Math.PI);
          mask[idx] = 1;
        } else {
          image[idx] = 0.0;
          mask[idx] = 0;
        }
      }
    }
  } else {
    for (let i = 0; i < n; i++) {
      image[i] = 100.0;
      mask[i] = 1;
    }
  }

  return { image, mask };
}

function computeRadiomicsSample(phantomType: string, gridSize: number, spacingMm: number) {
  const { image, mask } = generateRadiomicsPhantom(phantomType, gridSize);
  const roiVals: number[] = [];
  for (let i = 0; i < image.length; i++) {
    if (mask[i] > 0) roiVals.push(image[i]);
  }
  const n = roiVals.length || 1;
  const mean = roiVals.reduce((a, b) => a + b, 0) / n;
  const variance = roiVals.reduce((a, v) => a + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const skewness = std > 1e-7 ? roiVals.reduce((a, v) => a + ((v - mean) / std) ** 3, 0) / n : 0.0;
  const kurtosis = std > 1e-7 ? roiVals.reduce((a, v) => a + ((v - mean) / std) ** 4, 0) / n : 0.0;

  const sorted = [...roiVals].sort((a, b) => a - b);
  const minVal = sorted[0] ?? 0;
  const maxVal = sorted[sorted.length - 1] ?? 0;
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  const p25 = sorted[Math.floor(sorted.length * 0.25)] ?? 0;
  const p75 = sorted[Math.floor(sorted.length * 0.75)] ?? 0;
  const iqr = p75 - p25;
  const energy = roiVals.reduce((a, v) => a + v * v, 0);
  const totalEnergy = (energy * spacingMm * spacingMm) / n;

  // Discretize into 64 bins for GLDM/NGTDM sample heuristics
  const numBins = 64;
  const binned = new Int32Array(image.length).fill(-1);
  if (maxVal === minVal) {
    for (let i = 0; i < image.length; i++) {
      if (mask[i] > 0) binned[i] = 0;
    }
  } else {
    for (let i = 0; i < image.length; i++) {
      if (mask[i] > 0) {
        const s = ((image[i] - minVal) / (maxVal - minVal + 1e-12)) * (numBins - 1);
        binned[i] = Math.min(numBins - 1, Math.max(0, Math.round(s)));
      }
    }
  }

  const diffs: number[] = [];
  const roiBins: number[] = [];
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const idx = y * gridSize + x;
      if (mask[idx] === 0) continue;
      let sumN = 0;
      let cntN = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dy === 0 && dx === 0) continue;
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < gridSize && nx >= 0 && nx < gridSize) {
            const nIdx = ny * gridSize + nx;
            if (mask[nIdx] > 0) {
              sumN += binned[nIdx];
              cntN++;
            }
          }
        }
      }
      const localMean = cntN > 0 ? sumN / cntN : 0;
      diffs.push(Math.abs(binned[idx] - localMean));
      roiBins.push(binned[idx]);
    }
  }

  const dCount = diffs.length || 1;
  const diffMeanSq = diffs.reduce((a, d) => a + d * d, 0) / dCount;
  const dnu = diffs.reduce((a, d) => a + d * d, 0);
  const dnun = dnu / (dCount * dCount + 1e-4);
  const dcp = dCount / Math.max(image.length, 1);
  const glnuM = roiBins.reduce((a, b) => a + b * b, 0);
  const glnunM = glnuM / (dCount * dCount + 1e-4);
  const binMean = roiBins.reduce((a, b) => a + b, 0) / dCount;
  const varianceM = roiBins.reduce((a, b) => a + (b - binMean) ** 2, 0) / dCount;
  const sumDiff = diffs.reduce((a, d) => a + d, 0);
  let de = 0;
  if (sumDiff > 0) {
    for (const d of diffs) {
      if (d > 0) {
        const p = d / sumDiff;
        de -= p * Math.log2(p + 1e-12);
      }
    }
  }
  const entropyM = de * 0.8;
  const energyM = dnu / dCount;
  const homogeneityM = diffs.reduce((a, d) => a + 1.0 / (1.0 + d), 0) / dCount;
  const contrastM = diffMeanSq * 1.2;

  return {
    verified_features_sample: {
      first_order_mean: Number(mean.toFixed(6)),
      first_order_variance: Number(variance.toFixed(6)),
      first_order_skewness: Number(skewness.toFixed(6)),
      first_order_kurtosis: Number(kurtosis.toFixed(6)),
      first_order_median: Number(median.toFixed(6)),
      first_order_min: Number(minVal.toFixed(6)),
      first_order_max: Number(maxVal.toFixed(6)),
      first_order_iqr: Number(iqr.toFixed(6)),
      first_order_total_energy: Number(totalEnergy.toFixed(6)),
    },
    unverified_features_sample: {
      gldm_dcp: Number(dcp.toFixed(6)),
      gldm_glnu: Number(glnuM.toFixed(6)),
      gldm_glnun: Number(glnunM.toFixed(6)),
      gldm_variance: Number(varianceM.toFixed(6)),
      gldm_entropy: Number(entropyM.toFixed(6)),
      gldm_energy: Number(energyM.toFixed(6)),
      gldm_homogeneity: Number(homogeneityM.toFixed(6)),
      gldm_contrast: Number(contrastM.toFixed(6)),
      gldm_dnun: Number(dnun.toFixed(6)),
    },
  };
}

// ── Biomechanical Cross-Sectional Geometry Engine (Stage 28) ─────────────────
function generateBoneCrossSectionMask(phantomType: string, gridSize = 100, spacingMm = 0.5): Uint8Array {
  const mask = new Uint8Array(gridSize * gridSize);
  const cy = ((gridSize - 1) / 2.0) * spacingMm;
  const cx = ((gridSize - 1) / 2.0) * spacingMm;
  const outerR = 15.0;
  const innerR = 10.0;

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const xm = x * spacingMm;
      const ym = y * spacingMm;
      const dx = xm - cx;
      const dy = ym - cy;
      const r = Math.hypot(dx, dy);
      const angle = Math.atan2(dy, dx);
      let active = false;

      if (phantomType === 'mild_osteopenia') {
        active = r >= 11.5 && r <= outerR;
      } else if (phantomType === 'severe_osteopenia' || phantomType === 'uniform_cortical_thinning') {
        active = r >= 13.0 && r <= outerR;
      } else if (phantomType === 'small_lytic_defect') {
        const inAnnulus = r >= innerR && r <= outerR;
        const inCavity = Math.hypot(xm - (cx + 12.5), ym - cy) <= 2.2;
        active = inAnnulus && !inCavity;
      } else if (phantomType === 'med_lytic_defect' || phantomType === 'cortical_defect_33') {
        const inAnnulus = r >= innerR && r <= outerR;
        const inWedge = Math.abs(angle) <= (Math.PI / 3.0);
        active = inAnnulus && !inWedge;
      } else if (phantomType === 'large_lytic_defect') {
        const inAnnulus = r >= innerR && r <= outerR;
        const inWedge = Math.abs(angle) <= (2.0 * Math.PI / 3.0);
        active = inAnnulus && !inWedge;
      } else if (phantomType === 'sclerotic_rim') {
        active = r >= 8.2 && r <= 15.8;
      } else {
        // intact_femur, intact_femur_diaphysis, intact_circular_ring
        active = r >= innerR && r <= outerR;
      }

      mask[y * gridSize + x] = active ? 1 : 0;
    }
  }
  return mask;
}

function computeGeometryMoments(mask: Uint8Array, gridSize: number, spacingMm: number) {
  const dA = spacingMm * spacingMm;
  let pixelCount = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      if (mask[y * gridSize + x] > 0) {
        pixelCount++;
        sumX += x * spacingMm;
        sumY += y * spacingMm;
      }
    }
  }

  const area = pixelCount * dA;
  const xBar = pixelCount > 0 ? sumX / pixelCount : 0;
  const yBar = pixelCount > 0 ? sumY / pixelCount : 0;

  let ix = 0;
  let iy = 0;
  let ixy = 0;

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      if (mask[y * gridSize + x] > 0) {
        const dx = x * spacingMm - xBar;
        const dy = y * spacingMm - yBar;
        ix += dy * dy * dA;
        iy += dx * dx * dA;
        ixy += dx * dy * dA;
      }
    }
  }

  const polarJ = ix + iy;
  return { area, xBar, yBar, ix, iy, ixy, polarJ };
}

function runRadiomicsDemoData(req: { phantom_type?: string; grid_size?: number; spacing_mm?: number }) {
  const phantomType = req.phantom_type || 'checkerboard';
  const gridSize = Math.min(128, Math.max(8, Number(req.grid_size) || 32));
  const spacingMm = Math.min(5.0, Math.max(0.1, Number(req.spacing_mm) || 1.0));
  const sample = computeRadiomicsSample(phantomType, gridSize, spacingMm);

  return {
    phantom_type: phantomType,
    grid_size: gridSize,
    spacing_mm: spacingMm,
    verified_features_count: 81,
    unverified_features_count: 21,
    total_features_count: 102,
    specification_status: '102-D SPECIFICATION NOT VERIFIED (81 Verified / 21 Unverified Heuristic)',
    verified_features_sample: sample.verified_features_sample,
    unverified_features_sample: sample.unverified_features_sample,
    disclaimer:
      'DEMONSTRATION ONLY: Extracted from deterministic digital phantom. 21 engineering features remain unverified against IBSI benchmark standards.',
  };
}

function runBiomechanicsGeometryData(req: { phantom_type?: string; voxel_size_mm?: number }) {
  const phantomType = req.phantom_type || 'intact_femur_diaphysis';
  const voxelSize = Math.min(5.0, Math.max(0.1, Number(req.voxel_size_mm) || 0.5));
  const gridSize = voxelSize >= 1.0 ? 50 : 100;

  const intactMask = generateBoneCrossSectionMask('intact_circular_ring', gridSize, voxelSize);
  const intact = computeGeometryMoments(intactMask, gridSize, voxelSize);

  const targetMask = generateBoneCrossSectionMask(phantomType, gridSize, voxelSize);
  const res = computeGeometryMoments(targetMask, gridSize, voxelSize);

  return {
    phantom_type: phantomType,
    voxel_size_mm: voxelSize,
    area_mm2: Number(res.area.toFixed(2)),
    centroid_x_mm: Number(res.xBar.toFixed(2)),
    centroid_y_mm: Number(res.yBar.toFixed(2)),
    I_xx_mm4: Number(res.ix.toFixed(2)),
    I_yy_mm4: Number(res.iy.toFixed(2)),
    I_xy_mm4: Number(res.ixy.toFixed(2)),
    polar_J_mm4: Number(res.polarJ.toFixed(2)),
    area_retention_ratio: Number((intact.area > 0 ? res.area / intact.area : 1.0).toFixed(4)),
    bending_retention_ratio_x: Number((intact.ix > 0 ? res.ix / intact.ix : 1.0).toFixed(4)),
    bending_retention_ratio_y: Number((intact.iy > 0 ? res.iy / intact.iy : 1.0).toFixed(4)),
    torsion_retention_ratio: Number((intact.polarJ > 0 ? res.polarJ / intact.polarJ : 1.0).toFixed(4)),
    warning:
      'MATHEMATICAL DEMONSTRATION ONLY: Evaluated on 2D synthetic bone cross-section. Does NOT compute patient-specific fracture risk or clinical bone strength.',
  };
}

function runBiomechanicsMirelsData(req: {
  site: number;
  pain: number;
  radiographic_nature: number;
  lesion_size: number;
}) {
  const clampFactor = (v: number) => Math.min(3, Math.max(1, Math.round(Number(v) || 1)));
  const sSite = clampFactor(req.site);
  const sPain = clampFactor(req.pain);
  const sNature = clampFactor(req.radiographic_nature);
  const sSize = clampFactor(req.lesion_size);
  const total = sSite + sPain + sNature + sSize;

  const riskMap: Record<number, number> = {
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: 4.0,
    8: 15.0,
    9: 33.3,
    10: 50.0,
    11: 66.7,
    12: 100.0,
  };
  const riskPct = riskMap[total] ?? 50.0;

  let tier = 'LOW_RISK';
  let fixation = false;
  let guidance = '';

  if (total <= 7) {
    tier = 'LOW_RISK';
    fixation = false;
    guidance = `Total Score ${total}: Low risk of pathological fracture (~${Math.round(riskPct)}% in Mirels 1989 cohort). Historically managed non-operatively with radiotherapy and clinical surveillance. Valid for metastatic carcinoma of long bones.`;
  } else if (total === 8) {
    tier = 'INTERMEDIATE_RISK';
    fixation = false;
    guidance = `Total Score ${total}: Borderline / Intermediate fracture risk (~${Math.round(riskPct)}% in Mirels 1989 cohort). Requires careful clinical judgment, consideration of anatomical demands, and close surveillance. Valid for metastatic carcinoma of long bones.`;
  } else {
    tier = 'HIGH_RISK';
    fixation = true;
    guidance = `Total Score ${total}: High risk of impending pathological fracture (~${Math.round(riskPct)}% in Mirels 1989 cohort). Prophylactic internal fixation strongly indicated prior to radiation or systemic therapy. Valid for metastatic carcinoma of long bones.`;
  }

  return {
    total_score: total,
    fracture_risk_percent: riskPct,
    recommendation_tier: tier,
    prophylactic_fixation_indicated: fixation,
    clinical_guidance: guidance,
    disclaimer:
      'CONVENTIONAL RULE ENGINE: Classical Mirels (1989) scoring based solely on user-supplied parameters. Independent orthopedic consultation required.',
  };
}

function runExplainabilityDemoData(req: { target_class?: number; shift_horizontal?: number }) {
  const targetClass = Math.min(3, Math.max(0, Math.round(Number(req.target_class) || 0)));
  const shift = Math.min(8, Math.max(-8, Math.round(Number(req.shift_horizontal) || 0)));

  const size = 32;
  const centerY = 16;
  const centerX = Math.max(4, Math.min(28, 16 + shift));
  const heatmap: number[][] = Array.from({ length: size }, () => Array(size).fill(0));

  let maxVal = 0;
  let peakY = centerY;
  let peakX = centerX;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dist2 = (y - centerY) ** 2 + (x - centerX) ** 2;
      const sigma = 4.2 + targetClass * 0.35;
      const val = Math.exp(-dist2 / (2 * sigma * sigma));
      heatmap[y][x] = val;
      if (val > maxVal) {
        maxVal = val;
        peakY = y;
        peakX = x;
      }
    }
  }

  let activeCount = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      heatmap[y][x] = maxVal > 0 ? heatmap[y][x] / maxVal : 0;
      if (heatmap[y][x] > 0.5) activeCount++;
    }
  }

  const subsampled: number[][] = [];
  for (let y = 0; y < size; y += 4) {
    const row: number[] = [];
    for (let x = 0; x < size; x += 4) {
      row.push(Number(heatmap[y][x].toFixed(4)));
    }
    subsampled.push(row);
  }

  return {
    synthetic_input_shape: [1, 1, 32, 32],
    target_class: targetClass,
    active_region_ratio: Number((activeCount / (size * size)).toFixed(4)),
    peak_coordinates: [peakY, peakX],
    heatmap_preview_base64: null,
    heatmap_grid_sample: subsampled,
    method: 'Grad-CAM (Gradient-Weighted Class Activation Mapping)',
    shap_status: 'PARTIAL (Exact low-dimensional Shapley test calculator available; production SHAP dependency not installed)',
    warning:
      'SYNTHETIC DEMO ONLY: Generated on controlled synthetic CNN fixture. Does NOT explain real patient radiographs or clinical predictions.',
  };
}

function runUncertaintyDemoData(req: {
  distribution_preset?: string;
  custom_probabilities?: number[];
  ood_flag?: boolean;
}) {
  const presets: Record<string, number[]> = {
    highly_confident: [0.94, 0.02, 0.02, 0.02],
    moderately_confident: [0.65, 0.15, 0.10, 0.10],
    ambiguous_two_class: [0.48, 0.48, 0.02, 0.02],
    uniform_four_class: [0.25, 0.25, 0.25, 0.25],
    near_equal: [0.28, 0.26, 0.24, 0.22],
  };

  let p: number[];
  if (Array.isArray(req.custom_probabilities) && req.custom_probabilities.length === 4) {
    const sum = req.custom_probabilities.reduce((a, b) => a + b, 0) || 1;
    p = req.custom_probabilities.map((v) => v / sum);
  } else {
    p = presets[req.distribution_preset || 'highly_confident'] || presets.highly_confident;
  }

  let hNats = 0;
  for (const val of p) {
    if (val > 1e-12) {
      hNats -= val * Math.log(val);
    }
  }
  const hNorm = hNats / Math.log(4);
  const sorted = [...p].sort((a, b) => b - a);
  const margin = (sorted[0] ?? 0) - (sorted[1] ?? 0);

  let predIdx = 0;
  for (let i = 1; i < p.length; i++) {
    if (p[i] > p[predIdx]) predIdx = i;
  }
  const predKey = TARGET_CLASSES[predIdx];
  const predName = CLASS_DISPLAY_NAMES[predKey] || predKey;
  const oodDetected = Boolean(req.ood_flag);
  const threshold = 0.6;
  const entropyExceeded = hNorm > threshold;

  let decision = 'ACCEPT_PREDICTION';
  let humanReview = false;
  let abstentionReason = 'NONE';
  let reasonDescription = 'Verified high-confidence prediction. Entropy and manifold bounds satisfied.';

  if (entropyExceeded && oodDetected) {
    decision = 'ABSTAIN_HUMAN_REVIEW_REQUIRED';
    humanReview = true;
    abstentionReason = 'COMBINED_ENTROPY_AND_OOD';
    reasonDescription = `CRITICAL ABSTENTION: High entropy (${hNorm.toFixed(3)} > ${threshold.toFixed(2)}) and Out-of-Distribution detector triggered simultaneously.`;
  } else if (entropyExceeded) {
    decision = 'ABSTAIN_HUMAN_REVIEW_REQUIRED';
    humanReview = true;
    abstentionReason = 'ELEVATED_ENTROPY';
    reasonDescription = `ABSTENTION: Prediction entropy (${hNorm.toFixed(3)}) exceeds safety limit (${threshold.toFixed(2)}). Diagnostic probabilities are overly dispersed across differential targets.`;
  } else if (oodDetected) {
    decision = 'ABSTAIN_HUMAN_REVIEW_REQUIRED';
    humanReview = true;
    abstentionReason = 'OUT_OF_DISTRIBUTION';
    reasonDescription =
      'ABSTENTION: Anatomical feature representation lies outside training manifold bounds. Mandatory clinical radiologist review required.';
  } else if (hNorm >= 0.35) {
    decision = 'WARN_BORDERLINE_CONFIDENCE';
    humanReview = false;
    abstentionReason = 'NONE';
    reasonDescription = `BORDERLINE CONFIDENCE: Normalized entropy (${hNorm.toFixed(3)}) is elevated (0.35 <= H_norm < 0.60); double-reading recommended.`;
  }

  return {
    probabilities: p.map((v) => Number(v.toFixed(4))),
    predicted_class_idx: predIdx,
    predicted_class_name: predName,
    max_probability: Number(p[predIdx].toFixed(4)),
    top_two_margin: Number(margin.toFixed(4)),
    shannon_entropy_nats: Number(hNats.toFixed(4)),
    normalized_entropy: Number(hNorm.toFixed(4)),
    candidate_threshold: threshold,
    decision,
    human_review_required: humanReview,
    abstention_reason: abstentionReason,
    reason_description: reasonDescription,
    ood_detected: oodDetected,
    clinical_guidance: humanReview
      ? 'Human specialist review mandated. An abstention does NOT indicate absence of disease or healthy bone.'
      : 'Acceptable predictive entropy within candidate engineering threshold.',
    disclaimer:
      'RESEARCH ONLY: Threshold H_norm > 0.60 is an unvalidated candidate engineering threshold. Evaluated on synthetic distributions.',
  };
}

const RESEARCH_TRACKS_DATA = [
  {
    track_id: 'track_a',
    title: 'FracAtlas Fracture Research',
    modality: 'Plain Radiography (X-Ray)',
    task_type: 'Binary Fracture Detection',
    target_classes: ['Fractured', 'Non-fractured'],
    status: 'HISTORICAL_RESEARCH_FROZEN',
    historical_metrics: {
      evaluation_stage: 'Stage 10 (Holdout Test N=614)',
      roc_auc: 0.9117,
      pr_auc: 0.7537,
      sensitivity_theta_0_31: 0.9,
      specificity_theta_0_31: 0.744,
      balanced_accuracy_theta_0_31: 0.822,
      accuracy_theta_0_31: 0.772,
      accuracy_theta_0_50: 0.8827,
      npv_theta_0_31: 0.9715,
      f1_score_theta_0_31: 0.5858,
      test_samples: 614,
    },
    limitations: [
      'Patient IDs absent from source metadata; true patient-level independence cannot be verified across multi-projection views.',
      'Binary task only; does NOT differentiate bone pathologies, metabolic conditions, or secondary metastases.',
      'Burned holdout test partition (N=614) is preserved and must not be re-evaluated.',
    ],
    governance_clearance: 'Academic Research Use (FracAtlas Open Dataset)',
    disclaimer: 'RESEARCH TRACK ONLY: Binary fracture classification model. Not a four-class differential diagnosis.',
  },
  {
    track_id: 'track_b',
    title: 'Bone Scintigraphy Metastasis Research',
    modality: 'Tc-99m Whole-Body Bone Scintigraphy',
    task_type: 'Binary Skeletal Metastasis Assessment',
    target_classes: ['Metastasis-positive', 'No detected bone metastasis in breast cancer staging cohort'],
    status: 'HISTORICAL_RESEARCH_FROZEN',
    historical_metrics: {
      evaluation_stage: 'Stage 14 (Holdout Test N=44 Patients)',
      roc_auc: 0.8317,
      pr_auc: 0.7257,
      accuracy_primary_theta_0_60: 0.8409,
      balanced_accuracy_theta_0_60: 0.7762,
      sensitivity_primary_theta_0_60: 0.6667,
      specificity_primary_theta_0_60: 0.8857,
      sensitivity_screening_theta_0_45: 0.7778,
      precision_ppv: 0.6,
      npv: 0.9118,
      f1_score: 0.6316,
      brier_score: 0.1791,
      test_patients: 44,
      test_scans: 88,
    },
    limitations: [
      'Negative cohort comprises breast cancer staging patients with no detectable metastasis; NOT healthy normal controls.',
      'Planar scintigraphy prone to false-positive degenerative joint disease and trauma uptake.',
      'Burned holdout test partition (44 patients, 88 scans) is preserved and must not be re-evaluated.',
    ],
    governance_clearance: 'Academic Retrospective Cohort Research',
    disclaimer:
      'RESEARCH TRACK ONLY: Secondary bone metastasis screening in staging breast cancer. Negative scans are NOT physiological healthy controls.',
  },
  {
    track_id: 'track_c',
    title: 'BTXRD Primary Bone Tumour Research',
    modality: 'Planar Radiography (X-Ray)',
    task_type: '3-Class Primary Tumour Classification',
    target_classes: ['Normal/non-tumour bone', 'Benign primary bone tumour', 'Primary malignant bone tumour'],
    status: 'HISTORICAL_FROZEN_MODEL (GENERALIZATION LIMITED)',
    historical_metrics: {
      center1_val_accuracy: 0.818,
      center1_val_balanced_acc: 0.829,
      center1_val_macro_f1: 0.8103,
      center1_val_malignant_sens: 0.9024,
      center1_val_samples: 588,
      center2_test_accuracy: 0.5173,
      center2_test_balanced_acc: 0.5374,
      center2_test_macro_f1: 0.5069,
      center2_test_malignant_sens: 0.5921,
      center2_test_samples: 549,
      center3_test_accuracy: 0.471,
      center3_test_balanced_acc: 0.4467,
      center3_test_macro_f1: 0.3592,
      center3_test_malignant_sens: 0.6774,
      center3_test_samples: 259,
    },
    limitations: [
      'Severe cross-center generalization collapse: Accuracy drops from 81.80% (Center 1) to 51.73% (Center 2) and 47.10% (Center 3).',
      'Macro-F1 collapses from 0.8103 (Center 1) to 0.5069 (Center 2) and 0.3592 (Center 3).',
      'Balanced accuracy drops from 82.90% (Center 1) to 53.74% (Center 2) and 44.67% (Center 3).',
      'Primary malignant bone tumours (Osteosarcoma, Ewing) are biologically and radiographically distinct from secondary bone metastases.',
      'Center 3 referral skew (89.58% tumours) caused normal-class sensitivity collapse to 18.52%.',
      'Not clinically reliable or deployable; frozen model checkpoint SHA-256 strictly preserved.',
    ],
    governance_clearance: 'Retrospective Tri-Center Multi-Site Audit Only',
    disclaimer:
      'RESEARCH AUDIT ONLY: Model displays severe domain shift degradation on external hospital centers. Not approved for clinical decision support.',
  },
  {
    track_id: 'track_d',
    title: 'Final Four-Class OsteoGuard AI Diagnostic System',
    modality: 'Multimodal: Radiography + Bone Scintigraphy + CT',
    task_type: '4-Class Differential Diagnosis & Fracture Risk',
    target_classes: [
      'Secondary bone metastasis',
      "Paget's disease of bone",
      'Osteogenesis imperfecta (OI)',
      'Healthy skeletal control',
    ],
    status: 'BLOCKED — APPROVED CLINICAL DATA AND GOVERNANCE REQUIRED',
    historical_metrics: {
      model_readiness: 'ZERO MODELS TRAINED',
      clinical_training_data: 'UNAVAILABLE',
      approved_cohorts: '0 / 4 verified clinical cohorts acquired',
    },
    limitations: [
      "Clinical Paget's disease radiograph cohort is unavailable.",
      'Human Osteogenesis Imperfecta (OI) data pending multi-center institutional ethics approval and DUA.',
      'True matched healthy skeletal control cohort requires clinical site harmonization.',
      'All diagnostic predictions are strictly disabled until full IRB/IEC approval and data acquisition.',
    ],
    governance_clearance: 'Pending Faculty Review, Institutional Ethics Clearance (IEC), and Data Use Agreements (DUA)',
    disclaimer:
      'PROPOSED KSCST RESEARCH ARCHITECTURE: Active clinical prediction is completely disabled. System operates in synthetic research demonstration mode only.',
  },
];

const EVIDENCE_MODELS_DATA = [
  {
    model_id: 'osteo-trackA-fracatlas-v1',
    track: 'Track A: FracAtlas Fracture Detection',
    dataset_name: 'FracAtlas Open Orthopedic Dataset',
    modality: 'Planar Radiography',
    architecture: 'ResNet-101 (Pretrained IMAGENET1K_V2)',
    checkpoint_hash: 'STAGE_09_EPOCH9_HASH_VERIFIED',
    training_status: 'FROZEN (Stage 09)',
    validation_evidence: {
      roc_auc: 0.9117,
      pr_auc: 0.7537,
      sensitivity_theta_0_31: 0.9,
      specificity_theta_0_31: 0.744,
      balanced_accuracy_theta_0_31: 0.822,
      accuracy_theta_0_31: 0.772,
      accuracy_theta_0_50: 0.8827,
      npv_theta_0_31: 0.9715,
      f1_score_theta_0_31: 0.5858,
      holdout_test_samples: 614,
    },
    external_holdout_status: 'Evaluated in Stage 10 (Burned Holdout)',
    generalization_findings:
      'High discriminative ROC-AUC (0.9117) and PR-AUC (0.7537); patient-level independence unverified due to absent patient IDs in FracAtlas metadata.',
    clinical_deployment_status: 'RESEARCH_ONLY',
    limitations: ['No patient identifiers', 'Binary fracture detection only; not differential diagnosis'],
  },
  {
    model_id: 'osteo-trackB-scintigraphy-v1',
    track: 'Track B: Bone Scintigraphy Metastasis',
    dataset_name: 'Whole-Body Bone Scintigraphy Staging Cohort',
    modality: 'Tc-99m Scintigraphy (Anterior/Posterior)',
    architecture: '5-Fold ResNet-18 Patient-Aware Ensemble (Clinical Max-Pooling)',
    checkpoint_hash: 'STAGE_12_5FOLD_ENSEMBLE_VERIFIED',
    training_status: 'FROZEN (Stage 13)',
    validation_evidence: {
      roc_auc: 0.8317,
      pr_auc: 0.7257,
      accuracy_primary_theta_0_60: 0.8409,
      balanced_accuracy_theta_0_60: 0.7762,
      sensitivity_primary_theta_0_60: 0.6667,
      specificity_primary_theta_0_60: 0.8857,
      sensitivity_screening_theta_0_45: 0.7778,
      brier_score: 0.1791,
      holdout_patients: 44,
      holdout_scans: 88,
    },
    external_holdout_status: 'Evaluated in Stage 14 (Burned Holdout)',
    generalization_findings:
      'PR-AUC (0.7257) matches validation (0.7276) with delta -0.0019; negative scans are staging breast cancer, not healthy controls.',
    clinical_deployment_status: 'RESEARCH_ONLY',
    limitations: ['Negative class is not healthy controls', 'Hotspots confounded by osteoarthritis'],
  },
  {
    model_id: 'osteo-trackC-btxrd-frozen-v1',
    track: 'Track C: BTXRD Primary Bone Tumour',
    dataset_name: 'BTXRD Multi-Center Radiograph Cohort',
    modality: 'Planar Radiography',
    architecture: 'ResNet-34 Aspect-Ratio Letterboxed (512x512) + Focal Loss',
    checkpoint_hash: '21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD',
    training_status: 'FROZEN (Stage 18 Cryptographically Locked)',
    validation_evidence: {
      center1_val_accuracy: 0.818,
      center1_val_balanced_acc: 0.829,
      center1_val_macro_f1: 0.8103,
      center1_val_samples: 588,
      center2_test_accuracy: 0.5173,
      center2_test_balanced_acc: 0.5374,
      center2_test_macro_f1: 0.5069,
      center2_test_samples: 549,
      center3_test_accuracy: 0.471,
      center3_test_balanced_acc: 0.4467,
      center3_test_macro_f1: 0.3592,
      center3_test_samples: 259,
    },
    external_holdout_status: 'Center 2 (N=549) & Center 3 (N=259) Burned External Holdouts',
    generalization_findings:
      'Catastrophic cross-center degradation: Accuracy drops from 81.80% (Center 1) to 51.73% (Center 2) and 47.10% (Center 3). Macro-F1 collapses from 0.8103 to 0.5069 and 0.3592.',
    clinical_deployment_status: 'NOT_SUITABLE_FOR_DEPLOYMENT',
    limitations: [
      'Severe domain shift across hospital centers',
      'Primary bone tumours != secondary bone metastases',
      'Center 3 referral bias collapsed normal sensitivity to 18.52%',
    ],
  },
  {
    model_id: 'osteo-trackD-fourclass-unified',
    track: 'Track D: Four-Class OsteoGuard AI',
    dataset_name: 'Multi-Center Harmonized 4-Class Cohort (Proposed)',
    modality: 'Multimodal (X-Ray + Scintigraphy + CT)',
    architecture: 'Dual-Stream Swin-Transformer + IBSI Radiomics (512+102 Fusion)',
    checkpoint_hash: 'NONE (Untrained)',
    training_status: 'BLOCKED_BY_DATA',
    validation_evidence: { status: 'NO_CLINICAL_TRAINING_PERFORMED' },
    external_holdout_status: 'PENDING_DATA_ACQUISITION',
    generalization_findings: 'Cannot be evaluated until verified clinical cohorts are legally acquired.',
    clinical_deployment_status: 'BLOCKED',
    limitations: ['Approved clinical Paget, human OI, and matched controls are missing.'],
  },
];

const ARCHITECTURE_COMPONENTS_DATA = [
  {
    component_name: 'Modality Input Pathways (Radiography, Scintigraphy, CT)',
    layer_group: 'Data Intake & Normalization',
    status: 'PARTIALLY_IMPLEMENTED',
    implemented_features: 'Planar radiograph and scintigraphy loaders; synthetic digital phantoms.',
    unresolved_gaps: 'Volumetric CT DICOM pipeline unverified on real bone window series.',
    governance_dependency: 'Clinical imaging protocol harmonization across centers.',
  },
  {
    component_name: 'IBSI Radiomics Feature Engine',
    layer_group: 'Feature Engineering',
    status: 'PARTIALLY_IMPLEMENTED',
    implemented_features: '81 verified IBSI features (First Order, GLCM, GLRLM, GLSZM, GLDM).',
    unresolved_gaps: '21 engineering heuristic features unverified; 102-D spec marked NOT VERIFIED.',
    governance_dependency: 'IBSI Benchmark 2 compliance review.',
  },
  {
    component_name: 'Cross-Sectional Biomechanics Geometry Engine',
    layer_group: 'Biomechanical Risk',
    status: 'IMPLEMENTED',
    implemented_features: 'Area, centroid, principal second moments of area, polar moment of area, retention ratios.',
    unresolved_gaps: 'Evaluated strictly on 2D synthetic phantoms; does NOT compute clinical fracture probabilities.',
    governance_dependency: 'Correlation against clinical CT-FEA fracture outcomes required.',
  },
  {
    component_name: 'Conventional Mirels Scorer',
    layer_group: 'Clinical Rule Engine',
    status: 'IMPLEMENTED',
    implemented_features: 'Deterministic 4-factor scoring (site, pain, nature, size) with prophylactic fixation thresholds.',
    unresolved_gaps: 'Requires explicit manual inputs; no automated anatomical segmentation inputs.',
    governance_dependency: 'Orthopedic surgical guideline verification.',
  },
  {
    component_name: 'Grad-CAM Visual Explainability Engine',
    layer_group: 'Explainable AI',
    status: 'IMPLEMENTED',
    implemented_features: 'Production PyTorch hook manager, spatial localization verified, Adebayo sanity check passed.',
    unresolved_gaps: 'Verified exclusively on synthetic controlled CNNs; does NOT explain real patient predictions.',
    governance_dependency: 'Radiologist visual utility study.',
  },
  {
    component_name: 'SHAP Radiomic Explainability Engine',
    layer_group: 'Explainable AI',
    status: 'PARTIALLY_IMPLEMENTED',
    implemented_features: 'Exact combinatorial Shapley calculator (Efficiency Axiom verified).',
    unresolved_gaps: 'Production SHAP library is not installed; full 102-D feature SHAP is disabled.',
    governance_dependency: 'Computational feasibility review on high-dimensional radiomics.',
  },
  {
    component_name: 'Uncertainty Quantification & Selective Prediction',
    layer_group: 'Decision Support & Safety Gate',
    status: 'IMPLEMENTED',
    implemented_features:
      'Predictive entropy, normalized entropy, top-two margin, epistemic/aleatoric MI decomposition, tri-state triage.',
    unresolved_gaps: 'Candidate threshold H_norm > 0.60 is an unvalidated engineering heuristic.',
    governance_dependency: 'Pre-clinical ROC/Cost optimization on patient cohort.',
  },
  {
    component_name: 'Multimodal Late Fusion (512 Deep + 102 Radiomics = 614)',
    layer_group: 'Multimodal Fusion',
    status: 'ARCHITECTURALLY_SPECIFIED',
    implemented_features: 'Concatenation and projection tensor schemas defined in architecture documents.',
    unresolved_gaps: 'Scientifically NOT VERIFIED until 102 radiomics features are resolved and clinical encoders are trained.',
    governance_dependency: 'Data acquisition for all 4 classes.',
  },
  {
    component_name: 'Four-Class Diagnostic Classifier Head',
    layer_group: 'Inference & Output',
    status: 'BLOCKED_BY_DATA',
    implemented_features: 'Output taxonomy defined: metastasis, paget, oi, healthy.',
    unresolved_gaps: 'Zero trained weights; training blocked due to unavailable clinical datasets.',
    governance_dependency: 'IEC/IRB approval, DUA execution, and clinical cohort acquisition.',
  },
];

const DATASET_COHORTS_DATA = [
  {
    cohort_name: 'Secondary Bone Metastasis Cohort',
    target_class: 'Secondary bone metastasis',
    intended_modality: 'X-Ray / Bone Scintigraphy / CT',
    source_status: 'PARTIALLY_IDENTIFIED (Scintigraphy screening cohort available; harmonized plain radiographs required)',
    patient_count_estimate: 'Target: >= 250 patients',
    acquisition_blockers: [
      'Need institutional clinical radiographs with confirmed biopsy or PET-CT metastasis ground truth.',
      'Existing scintigraphy cohort contains breast cancer only; cannot represent all secondary metastases.',
    ],
    icmr_iec_required: true,
    dua_status: 'Institutional DUA Required',
  },
  {
    cohort_name: "Paget's Disease of Bone Cohort",
    target_class: "Paget's disease of bone",
    intended_modality: 'Plain Radiography (Pelvis, Spine, Femur, Skull)',
    source_status: 'UNAVAILABLE (No open-access human imaging repository exists)',
    patient_count_estimate: 'Target: >= 100 patients',
    acquisition_blockers: [
      "Paget's disease is rare in Indian subcontinent; requires collaborative outreach to UK/European registries (PRISM trial, Paget's Association).",
      'Requires international cross-border data transfer clearance (ICMR-HMSC).',
    ],
    icmr_iec_required: true,
    dua_status: 'International Collaborative DUA Required',
  },
  {
    cohort_name: 'Osteogenesis Imperfecta (OI) Cohort',
    target_class: 'Osteogenesis imperfecta (OI)',
    intended_modality: 'Plain Radiography (Long bones, Spine)',
    source_status: 'BLOCKED (Candidate study PMC11324408 investigated: 0 human radiographs available; restricted clinical data)',
    patient_count_estimate: 'Target: >= 100 pediatric/adult patients',
    acquisition_blockers: [
      'Rare genetic disorder; necessitates dedicated multi-center pediatric orthopedic network collaboration (OIF, Brittle Bone Society, AIIMS).',
      'Strict ethical safeguards for vulnerable pediatric populations.',
    ],
    icmr_iec_required: true,
    dua_status: 'Pediatric Ethics & Clinical DUA Required',
  },
  {
    cohort_name: 'Healthy Skeletal Control Cohort',
    target_class: 'Healthy skeletal control',
    intended_modality: 'Planar Radiography (Matched anatomical sites)',
    source_status: 'PENDING_HARMONIZATION (FracAtlas negative cohort unverified for metabolic health)',
    patient_count_estimate: 'Target: >= 250 age/sex-matched subjects',
    acquisition_blockers: [
      "Existing 'non-fractured' radiographs from trauma centers may contain underlying osteoporosis, osteopenia, or systemic disease.",
      'Requires verified asymptomatic controls with documented normal bone mineral density.',
    ],
    icmr_iec_required: true,
    dua_status: 'Institutional DUA Required',
  },
];

export function registerResearchRoutes(app: Express): void {
  const staticTracks = RESEARCH_TRACKS_DATA.filter((t) => t.track_id !== 'track_d');
  const staticEvidence = EVIDENCE_MODELS_DATA.filter((m) => m.model_id !== 'osteo-trackD-fourclass-unified');

  app.get('/api/v1/clinical/training-status', (_req: Request, res: Response) => {
    const state = evaluateFourClassClinicalState(__researchDirname);
    res.json(state);
  });

  app.get('/api/v1/research/status', (_req: Request, res: Response) => {
    const state = evaluateFourClassClinicalState(__researchDirname);
    const trackD = buildTrackDPayload(state);
    const tracks = [...staticTracks, trackD];
    const ready = state.trackDState === 'TRAINED_VERIFIED';

    res.json({
      status: ready ? 'CLINICAL_AND_RESEARCH_ONLINE' : 'RESEARCH_PROTOTYPE_ONLINE',
      prototype_version: '1.0.0',
      four_class_inference_status: trackD.status,
      active_tracks_count: tracks.length,
      tracks: tracks.map((t) => t.title),
      architecture_readiness: ready
        ? 'CLINICAL_CHECKPOINT_VERIFIED__ABSTENTION_SAFEGUARD_ACTIVE'
        : state.trackDState === 'READY_FOR_TRAINING'
        ? 'CLINICAL_COHORTS_READY__AWAITING_TRAINING_RUN'
        : 'ENGINEERING_COMPONENTS_VERIFIED__CLINICAL_MODEL_BLOCKED',
      frozen_checkpoint_status: verifyFrozenTrackCCheckpoint()
        ? 'EXPECTED SHA-256 VERIFIED; CHECKPOINT REMAINS FROZEN'
        : 'EXPECTED SHA-256 RECORDED; CHECKPOINT FILE MISSING OR NOT VERIFIED',
      synthetic_workbench_available: true,
      disclaimer: ready
        ? 'CLINICAL SUPPORT ELIGIBLE: Verified 4-class weights present. Mandatory H_norm >= 0.60 abstention active.'
        : 'RESEARCH PROTOTYPE ONLY: Demonstrates verified engineering components on synthetic phantoms.',
    });
  });

  app.get('/api/v1/research/tracks', (_req: Request, res: Response) => {
    const state = evaluateFourClassClinicalState(__researchDirname);
    const tracks = [...staticTracks, buildTrackDPayload(state)];
    res.json({
      timestamp: new Date().toISOString(),
      active_tracks_count: tracks.length,
      tracks,
      disclaimer:
        state.trackDState === 'TRAINED_VERIFIED'
          ? 'Track D verified on disk; Tracks A-C remain frozen historical sub-track benchmarks.'
          : 'All models are experimental research artifacts. No clinical diagnostic claims are made.',
    });
  });

  app.get('/api/v1/evidence/registry', (_req: Request, res: Response) => {
    const state = evaluateFourClassClinicalState(__researchDirname);
    const models = [...staticEvidence, buildTrackDEvidenceItem(state)];
    res.json({
      timestamp: new Date().toISOString(),
      total_models: models.length,
      models,
      frozen_checkpoint_verified: verifyFrozenTrackCCheckpoint(),
      disclaimer: 'Historical evidence and current artifact integrity are distinct. Track C is verified only when the frozen checkpoint file is present and matches its expected SHA-256; absent Track D metrics remain unavailable.',
    });
  });

  app.get('/api/v1/architecture/readiness', (_req: Request, res: Response) => {
    const state = evaluateFourClassClinicalState(__researchDirname);
    const ready = state.trackDState === 'TRAINED_VERIFIED';
    const dynamicComponents = ARCHITECTURE_COMPONENTS_DATA.map((comp) => {
      if (comp.layer_group === 'Inference & Output' && ready && state.activeCheckpoint) {
        return {
          ...comp,
          status: 'IMPLEMENTED' as const,
          implemented_features: `Verified 4-class checkpoint (${state.activeCheckpoint.filename}) trained & temperature calibrated.`,
          unresolved_gaps: 'Prospective external multi-center audit & H_norm threshold tuning ongoing.',
          governance_dependency: 'Clinical governance review & clinician human-in-the-loop verification.',
        };
      }
      return comp;
    });

    res.json({
      timestamp: new Date().toISOString(),
      system_readiness: ready
        ? 'FOUR_CLASS_CHECKPOINT_VERIFIED'
        : 'ENGINEERING_COMPONENTS_VERIFIED__CLINICAL_MODEL_BLOCKED',
      components: dynamicComponents,
      fusion_contract_status: ready
        ? '512 + 102 = 614 Late Fusion Contract trained with post-hoc temperature scaling.'
        : '512 + 102 = 614 Late Fusion Contract is ARCHITECTURALLY_SPECIFIED but NOT_VERIFIED',
      inference_eligibility: ready
        ? 'CLINICAL_INFERENCE_ELIGIBLE (Supervised 4-class weights verified + H_norm >= 0.60 gate active)'
        : 'STRICTLY_BLOCKED (Diagnostic inference disabled; synthetic demo workbench active)',
      disclaimer: ready
        ? 'Live 4-class weights verified on disk. Always preserve human-in-the-loop oversight.'
        : 'Components are modular engineering fixtures validated on synthetic phantoms only.',
    });
  });

  app.get('/api/v1/datasets/status', (_req: Request, res: Response) => {
    const state = evaluateFourClassClinicalState(__researchDirname);
    const cohorts = buildDynamicCohortList(DATASET_COHORTS_DATA, state);
    const ready = state.cohortsAcquired === 4;

    res.json({
      timestamp: new Date().toISOString(),
      target_classes: TARGET_CLASSES,
      cohorts,
      overall_acquisition_gate:
        state.trackDState === 'TRAINED_VERIFIED'
          ? `UNLOCKED — 4 / 4 cohorts verified on disk (${state.totalRawSamples} files) & 4-class checkpoint trained.`
          : ready
          ? `READY FOR TRAINING — 4 / 4 cohorts populated on disk (${state.totalRawSamples} files). Run train_cli to unlock checkpoint.`
          : `BLOCKED (${state.cohortsAcquired} / 4 verified cohorts) — Formal institutional collaboration and clinical cohort ingestion required.`,
      disclaimer: 'Research governance prohibits training on surrogate or unverified datasets.',
    });
  });

  app.get('/api/v1/datasets/readiness', (_req: Request, res: Response) => {
    res.json(getTrackDDatasetReadiness(__researchDirname));
  });

  app.get('/api/v1/datasets/sources', (_req: Request, res: Response) => {
    res.json(getDatasetSourcesResponse(__researchDirname));
  });

  app.get('/api/v1/datasets/acquisition-status', (_req: Request, res: Response) => {
    res.json(getDatasetAcquisitionStatusResponse(__researchDirname));
  });

  app.get('/api/v1/datasets/governance-status', (_req: Request, res: Response) => {
    res.json(getDatasetGovernanceStatusResponse(__researchDirname));
  });

  app.get('/api/v1/datasets/cohort-planning', (_req: Request, res: Response) => {
    res.json(getCohortPlanningResponse(__researchDirname));
  });

  app.post('/api/v1/datasets/workflow/transition', (req: Request, res: Response) => {
    const clinicalToken = process.env.OSTEO_CLINICAL_API_TOKEN?.trim() || '';
    if (clinicalToken) {
      const authHeader = req.headers.authorization || '';
      if (authHeader !== `Bearer ${clinicalToken}`) {
        return res.status(401).json({
          error: 'Unauthorized: valid clinical Bearer token required for workflow governance transitions.',
        });
      }
    }
    const result = validateWorkflowTransitionRequest(__researchDirname, req.body || {});
    if (!result.ok) {
      return res.status(result.statusCode).json({ error: result.error });
    }
    return res.status(200).json(result.transition);
  });

  const denySyntheticDemo = (_req: Request, res: Response) => res.status(404).json({
    error_code: 'SYNTHETIC_DEMO_NOT_AVAILABLE',
    detail: 'Synthetic phantom demonstrations are available only in isolated test runs.',
  });
  app.post('/api/v1/radiomics/demo', denySyntheticDemo);
  app.post('/api/v1/biomechanics/geometry', denySyntheticDemo);

  app.post('/api/v1/biomechanics/mirels', (req: Request, res: Response) => {
    res.json(runBiomechanicsMirelsData(req.body || {}));
  });

  app.post('/api/v1/explainability/demo', denySyntheticDemo);
  app.post('/api/v1/uncertainty/demo', denySyntheticDemo);
  app.post('/api/v1/workbench/run-demo', denySyntheticDemo);

  app.post('/api/v1/diagnostic/four-class', (_req: Request, res: Response) => {
    const state = evaluateFourClassClinicalState(__researchDirname);
    const gate = buildFourClassGateResponse(state);
    res.status(gate.statusCode).json(gate.body);
  });

  app.get('/api/v1/trained-models/overview', (_req: Request, res: Response) => {
    res.json(loadTrainedModelsOverview());
  });
}

function safeReadJson(relPath: string): Record<string, any> | null {
  try {
    const fullPath = path.join(ML_ARTIFACTS_DIR, relPath);
    if (!fs.existsSync(fullPath)) return null;
    return JSON.parse(fs.readFileSync(fullPath, 'utf-8'));
  } catch {
    return null;
  }
}

function metricOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null;
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseCsvRows(relPath: string): Array<Record<string, string>> {
  try {
    const fullPath = path.join(ML_ARTIFACTS_DIR, relPath);
    if (!fs.existsSync(fullPath)) return [];
    const lines = fs
      .readFileSync(fullPath, 'utf-8')
      .split(/\r?\n/)
      .filter((l) => l.trim().length > 0);
    if (lines.length < 2) return [];
    const headers = lines[0].split(',').map((h) => h.trim());
    const rows: Array<Record<string, string>> = [];
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',');
      const row: Record<string, string> = {};
      headers.forEach((h, idx) => {
        row[h] = (vals[idx] ?? '').trim();
      });
      rows.push(row);
    }
    return rows;
  } catch {
    return [];
  }
}

function filterExistingPlots(plots: Array<{ title: string; url: string; caption: string }>) {
  return plots.filter((p) => {
    const rel = p.url.replace(/^\/ml-artifacts\//, '');
    return fs.existsSync(path.join(ML_ARTIFACTS_DIR, rel));
  });
}

function loadTrainedModelsOverview() {
  // 1. Track A: FracAtlas ResNet-101
  const fracMetrics = safeReadJson('fracatlas_test_evaluation/test_metrics.json');
  const fracRows = parseCsvRows('fracatlas_test_evaluation/test_predictions.csv');
  // 2. Track B: Bone Scintigraphy 5-Fold ResNet-18 Ensemble
  const bsMetrics = safeReadJson('bone_scan_test_evaluation/test_metrics.json');
  const bsRows = parseCsvRows('bone_scan_test_evaluation/test_patient_predictions.csv');
  // 3. Track C: BTXRD Multi-Center ResNet-34
  const c2Metrics = safeReadJson('btxrd_center2_external_evaluation/classification_metrics.json');
  const c3Metrics = safeReadJson('btxrd_center3_external_evaluation/classification_metrics.json');
  const c2Rows = parseCsvRows('btxrd_center2_external_evaluation/center2_predictions.csv');
  const c3Rows = parseCsvRows('btxrd_center3_external_evaluation/center3_predictions.csv');

  const models = [
    {
      model_id: 'osteo-trackA-fracatlas-v1',
      track: 'track_a',
      title: 'FracAtlas Fracture Detection (ResNet-101)',
      architecture: 'ResNet-101 (IMAGENET1K_V2, Epoch 9)',
      modality: 'Plain Radiography (X-Ray)',
      total_holdout_evaluated: fracMetrics?.total_test_samples ?? fracRows.length,
      headline_metrics: {
        roc_auc: metricOrNull(fracMetrics?.roc_auc),
        pr_auc: metricOrNull(fracMetrics?.pr_auc),
        sensitivity: metricOrNull(fracMetrics?.frozen_primary_threshold_031?.sensitivity_recall),
        specificity: metricOrNull(fracMetrics?.frozen_primary_threshold_031?.specificity),
        balanced_accuracy: metricOrNull(fracMetrics?.frozen_primary_threshold_031?.balanced_accuracy),
      },
      plots: filterExistingPlots([
        {
          title: 'Holdout ROC Curve',
          url: '/ml-artifacts/fracatlas_test_evaluation/roc_curve_test.png',
          caption: 'Archived evaluation chart available in the local research artifact store.',
        },
        {
          title: 'Precision-Recall Curve',
          url: '/ml-artifacts/fracatlas_test_evaluation/precision_recall_curve_test.png',
          caption: 'Archived evaluation chart available in the local research artifact store.',
        },
        {
          title: 'Confusion Matrix (θ=0.31)',
          url: '/ml-artifacts/fracatlas_test_evaluation/confusion_matrix_test.png',
          caption: 'Archived evaluation chart available in the local research artifact store.',
        },
        {
          title: 'Validation vs Holdout Test',
          url: '/ml-artifacts/fracatlas_test_evaluation/validation_vs_test_comparison.png',
          caption: 'Archived evaluation chart available in the local research artifact store.',
        },
      ]),
      // Individual holdout rows are not exposed through the unauthenticated public summary route.
      sample_predictions: [],
    },
    {
      model_id: 'osteo-trackB-scintigraphy-v1',
      track: 'track_b',
      title: 'Bone Scintigraphy Metastasis (5-Fold ResNet-18 Ensemble)',
      architecture: '5-Fold ResNet-18 Patient-Aware Ensemble (Clinical Max-Pooling)',
      modality: 'Tc-99m Whole-Body Bone Scintigraphy',
      total_holdout_evaluated: bsMetrics?.evaluation_cohort?.total_patients ?? bsRows.length,
      headline_metrics: {
        roc_auc: metricOrNull(bsMetrics?.discriminative_performance?.roc_auc),
        pr_auc: metricOrNull(bsMetrics?.discriminative_performance?.pr_auc),
        sensitivity: metricOrNull(bsMetrics?.primary_operating_point_060?.sensitivity),
        specificity: metricOrNull(bsMetrics?.primary_operating_point_060?.specificity),
        accuracy: metricOrNull(bsMetrics?.primary_operating_point_060?.accuracy),
        mean_fold_std: metricOrNull(bsMetrics?.model_disagreement?.overall_mean_std),
      },
      plots: filterExistingPlots([
        {
          title: 'Ensemble Holdout ROC Curve',
          url: '/ml-artifacts/bone_scan_test_evaluation/roc_curve_test.png',
          caption: 'Archived patient-level evaluation chart available in the local research artifact store.',
        },
        {
          title: 'Ensemble Precision-Recall Curve',
          url: '/ml-artifacts/bone_scan_test_evaluation/precision_recall_curve_test.png',
          caption: 'Archived evaluation chart available in the local research artifact store.',
        },
        {
          title: '5-Fold Model Disagreement',
          url: '/ml-artifacts/bone_scan_test_evaluation/ensemble_disagreement_test.png',
          caption: 'Archived fold-disagreement chart available in the local research artifact store.',
        },
        {
          title: 'Confusion Matrix (θ=0.60)',
          url: '/ml-artifacts/bone_scan_test_evaluation/confusion_matrix_test.png',
          caption: 'Archived evaluation chart available in the local research artifact store.',
        },
      ]),
      // Patient-level holdout rows are not exposed through the unauthenticated public summary route.
      sample_predictions: [],
    },
    {
      model_id: 'osteo-trackC-btxrd-frozen-v1',
      track: 'track_c',
      title: 'BTXRD Multi-Center Primary Bone Tumour Audit (ResNet-34)',
      architecture: 'ResNet-34 Aspect-Ratio Letterboxed (512x512) + Focal Loss',
      modality: 'Planar Radiography (Tri-Center Hospital Audit)',
      total_holdout_evaluated: c2Rows.length + c3Rows.length,
      headline_metrics: {
        center2_accuracy: metricOrNull(c2Metrics?.overall_accuracy?.estimate),
        center2_macro_f1: metricOrNull(c2Metrics?.macro_f1?.estimate),
        center2_roc_auc: metricOrNull(c2Metrics?.macro_roc_auc_ovr?.estimate),
        center3_accuracy: metricOrNull(c3Metrics?.overall_accuracy?.estimate),
        center3_macro_f1: metricOrNull(c3Metrics?.macro_f1?.estimate),
      },
      plots: filterExistingPlots([
        {
          title: 'Multi-Center Metric Comparison',
          url: '/ml-artifacts/btxrd_multicenter_audit/multicenter_metric_comparison.png',
          caption: 'Archived aggregate evaluation chart available in the local research artifact store.',
        },
        {
          title: 'Multi-Center Confusion Matrices',
          url: '/ml-artifacts/btxrd_multicenter_audit/multicenter_confusion_matrices.png',
          caption: 'Archived aggregate evaluation chart available in the local research artifact store.',
        },
        {
          title: 'Center 2 External ROC Curves',
          url: '/ml-artifacts/btxrd_center2_external_evaluation/center2_roc_curves.png',
          caption: 'Archived external-site evaluation chart available in the local research artifact store.',
        },
        {
          title: 'Center 3 External ROC Curves',
          url: '/ml-artifacts/btxrd_center3_external_evaluation/center3_roc_curves.png',
          caption: 'Archived external-site evaluation chart available in the local research artifact store.',
        },
      ]),
      // BTXRD external holdout sample predictions remain sealed; only approved aggregate evidence may be displayed.
      sample_predictions: [],
    },
  ];

  const clinicalState = evaluateFourClassClinicalState(__researchDirname);
  if (clinicalState.trackDState === 'TRAINED_VERIFIED' && clinicalState.trainingSummary) {
    const s = clinicalState.trainingSummary;
    const testM = s.test_metrics || {};
    models.push({
      model_id: 'osteo-trackD-fourclass-unified',
      track: 'track_d',
      title: `Four-Class OsteoGuardNet (${s.backbone.toUpperCase()} + 102-D Radiomics)`,
      architecture: `OsteoGuardNet (${s.backbone} + 102-D IBSI Late Fusion, T=${Number(s.temperature || 1).toFixed(2)})`,
      modality: `${s.modality.toUpperCase()} (Patient-Stratified Holdout)`,
      total_holdout_evaluated: Number(testM.n_samples || s.dataset_summary?.test_samples || 0),
      headline_metrics: {
        accuracy: metricOrNull(testM.accuracy),
        macro_f1: metricOrNull(testM.macro_f1),
        macro_roc_auc: metricOrNull(testM.macro_roc_auc),
        calibrated_ece: metricOrNull(testM.ece),
        brier_score: metricOrNull(testM.brier_score),
      },
      plots: [],
      // Track D records also remain aggregate-only on this public summary route.
      sample_predictions: [],
    });
  }

  const totalSamplesEvaluated = models.reduce((acc, m) => acc + (Number(m.total_holdout_evaluated) || 0), 0);

  return {
    timestamp: new Date().toISOString(),
    trained_models_count: models.length,
    total_holdout_samples_evaluated: totalSamplesEvaluated,
    models,
    disclaimer:
      'Historical research artifact summary. Metrics are included only when present in readable local evidence files; absent metrics are null. Individual prediction rows and protected holdout samples are not returned by this public route. Historical research evidence does not establish clinical validity.',
  };
}
