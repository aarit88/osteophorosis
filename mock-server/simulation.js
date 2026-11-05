import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';

// 4 diagnostic classes supported by OsteoGuard AI
const CLASSES = ['metastasis', 'paget', 'oi', 'healthy'];

const SHAP_FEATURE_NAMES = [
  'GLCM Contrast',
  'Cortical Thickness',
  'Texture Entropy',
  'First-order Mean Intensity',
  'GLRLM Long-Run Emphasis',
  'Bone Volume Fraction',
];

/**
 * Creates a deterministic Linear Congruential Generator (LCG) from a seed number.
 */
function createRng(seed) {
  let state = seed >>> 0;
  return function () {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Calculates normalized Shannon entropy H = -sum(p * log4(p)) -> 0.0 to 1.0.
 */
function calculateEntropy(probabilities) {
  let entropy = 0;
  const probs = Object.values(probabilities);
  for (const p of probs) {
    if (p > 0.0001) {
      entropy -= p * (Math.log(p) / Math.log(4));
    }
  }
  return parseFloat(Math.min(Math.max(entropy, 0.0), 1.0).toFixed(3));
}

/**
 * Deterministically simulates an OsteoGuard AI analysis for an uploaded image.
 */
export async function simulateAnalysis(imageBuffer, originalFilename, modality = 'xray', outputDir) {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // 1. Compute SHA-256 hash from image bytes for deterministic reproducibility
  const sha256 = crypto.createHash('sha256').update(imageBuffer).digest('hex');
  const seed = parseInt(sha256.substring(0, 8), 16);
  const rng = createRng(seed);

  const requestId = `req_${sha256.substring(0, 12)}`;
  const timestamp = new Date().toISOString();

  // 2. Determine top class and probabilities
  const classIndex = Math.floor(rng() * 4);
  const topClass = CLASSES[classIndex];

  // Top class confidence between 0.56 and 0.94
  const confidence = parseFloat((0.56 + rng() * 0.36).toFixed(3));
  const remaining = 1.0 - confidence;

  // Split remaining amongst the other 3 classes
  const rawOthers = [rng() + 0.05, rng() + 0.05, rng() + 0.05];
  const sumRaw = rawOthers.reduce((a, b) => a + b, 0);
  const otherProbabilities = rawOthers.map((v) => parseFloat(((v / sumRaw) * remaining).toFixed(3)));

  const probabilities = { metastasis: 0, paget: 0, oi: 0, healthy: 0 };
  probabilities[topClass] = confidence;

  let otherIdx = 0;
  CLASSES.forEach((c) => {
    if (c !== topClass) {
      probabilities[c] = otherProbabilities[otherIdx++];
    }
  });

  // Re-normalize to guarantee exact sum = 1.0
  const totalSum = Object.values(probabilities).reduce((a, b) => a + b, 0);
  Object.keys(probabilities).forEach((k) => {
    probabilities[k] = parseFloat((probabilities[k] / totalSum).toFixed(3));
  });

  // 3. Uncertainty Details & Novel Abstention Heuristics
  const entropy = calculateEntropy(probabilities);

  // 95% Confidence Interval on top-class prediction
  const ciMargin = parseFloat((0.04 + rng() * 0.06).toFixed(3));
  const confidenceInterval = [
    parseFloat(Math.max(0.35, confidence - ciMargin).toFixed(2)),
    parseFloat(Math.min(0.99, confidence + ciMargin).toFixed(2)),
  ];

  // Out-Of-Distribution (OOD) flag triggered ~10% of the time based on hash
  const oodFlag = (seed % 10) === 0;

  // Calibration reliability bin (predicted vs empirical observed)
  const calibrationBin = {
    predicted: confidence,
    observed: parseFloat((confidence * (0.91 + rng() * 0.14)).toFixed(2)),
  };

  let uncertainty = 'low';
  if (entropy > 0.6 || oodFlag) {
    uncertainty = 'high';
  } else if (entropy > 0.3) {
    uncertainty = 'medium';
  }

  // 4. Generate SHAP values (6 features, signed, sorted by absolute magnitude desc)
  const shap = SHAP_FEATURE_NAMES.map((name) => {
    let sign = rng() > 0.45 ? 1 : -1;
    if (topClass === 'metastasis' && name === 'Cortical Thickness') sign = -1;
    if (topClass === 'metastasis' && name === 'GLCM Contrast') sign = 1;
    if (topClass === 'healthy' && name === 'Bone Volume Fraction') sign = 1;
    if (topClass === 'oi' && name === 'Cortical Thickness') sign = -1;

    const magnitude = parseFloat((0.1 + rng() * 0.45).toFixed(2));
    return {
      name,
      value: parseFloat((sign * magnitude).toFixed(2)),
    };
  });

  shap.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  // 5. Counterfactual Diagnostic Sensitivity Scenarios (Novelty 3)
  const otherClasses = CLASSES.filter((c) => c !== topClass);
  const counterfactuals = [
    {
      feature: 'Cortical Thickness',
      deltaDirection: topClass === 'healthy' ? 'decrease' : 'increase',
      percentChange: Math.round(15 + rng() * 15),
      flipTo: topClass === 'healthy' ? 'metastasis' : 'healthy',
      probabilityShift: parseFloat((0.14 + rng() * 0.12).toFixed(2)),
    },
    {
      feature: 'GLCM Contrast',
      deltaDirection: topClass === 'paget' ? 'decrease' : 'increase',
      percentChange: Math.round(18 + rng() * 14),
      flipTo: otherClasses[0],
      probabilityShift: parseFloat((0.12 + rng() * 0.1).toFixed(2)),
    },
    {
      feature: 'Bone Volume Fraction',
      deltaDirection: topClass === 'oi' ? 'increase' : 'decrease',
      percentChange: Math.round(20 + rng() * 15),
      flipTo: otherClasses[1] || 'healthy',
      probabilityShift: parseFloat((0.15 + rng() * 0.11).toFixed(2)),
    },
  ];

  // 6. Derive Fracture Risk
  let riskPercent = 20;
  let riskLevel = 'low';

  if (topClass === 'metastasis') {
    riskPercent = Math.round(72 + rng() * 24);
    riskLevel = 'high';
  } else if (topClass === 'oi') {
    riskPercent = Math.round(68 + rng() * 26);
    riskLevel = 'high';
  } else if (topClass === 'paget') {
    riskPercent = Math.round(38 + rng() * 28);
    riskLevel = riskPercent >= 60 ? 'high' : 'medium';
  } else {
    // healthy
    riskPercent = Math.round(6 + rng() * 16);
    riskLevel = 'low';
  }

  // 7. Generate image assets (uploaded image copy + Grad-CAM heatmap)
  const ext = path.extname(originalFilename || 'image.png') || '.png';
  const uploadFilename = `upload_${requestId}${ext}`;
  const gradcamFilename = `gradcam_${requestId}.png`;

  const uploadFilePath = path.join(outputDir, uploadFilename);
  const gradcamFilePath = path.join(outputDir, gradcamFilename);

  fs.writeFileSync(uploadFilePath, imageBuffer);

  try {
    const meta = await sharp(imageBuffer).metadata();
    const width = meta.width || 512;
    const height = meta.height || 512;

    const hotspotX = Math.round(width * (0.35 + rng() * 0.35));
    const hotspotY = Math.round(height * (0.35 + rng() * 0.35));
    const radius = Math.round(Math.min(width, height) * (0.22 + rng() * 0.12));

    const svgOverlay = `
      <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <radialGradient id="gradcamHeat" cx="${hotspotX}" cy="${hotspotY}" r="${radius}" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stop-color="#ff0000" stop-opacity="0.82" />
            <stop offset="35%" stop-color="#ff9900" stop-opacity="0.68" />
            <stop offset="65%" stop-color="#ffff00" stop-opacity="0.48" />
            <stop offset="85%" stop-color="#00ddff" stop-opacity="0.25" />
            <stop offset="100%" stop-color="#0000ff" stop-opacity="0.0" />
          </radialGradient>
        </defs>
        <rect x="0" y="0" width="${width}" height="${height}" fill="url(#gradcamHeat)" />
      </svg>
    `;

    await sharp(imageBuffer)
      .grayscale()
      .composite([{ input: Buffer.from(svgOverlay), blend: 'over' }])
      .png()
      .toFile(gradcamFilePath);
  } catch (err) {
    console.error('Error rendering Grad-CAM image with sharp, creating fallback:', err);
    fs.writeFileSync(gradcamFilePath, imageBuffer);
  }

  return {
    requestId,
    timestamp,
    modality,
    imageUrl: `/uploads/${uploadFilename}`,
    gradcamUrl: `/uploads/${gradcamFilename}`,
    prediction: {
      topClass,
      confidence,
      probabilities,
      uncertainty,
    },
    uncertaintyDetail: {
      entropy,
      confidenceInterval,
      oodFlag,
      calibrationBin,
    },
    shap,
    counterfactuals,
    fractureRisk: {
      level: riskLevel,
      percent: riskPercent,
    },
    disclaimer: '⚠️ RESEARCH PROTOTYPE — SIMULATED OUTPUT — NOT FOR CLINICAL USE. This tool is designed strictly for investigational validation and engineering demonstration.',
  };
}
