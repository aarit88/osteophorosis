import type {
  BonePathologyClass,
  ClassProbabilities,
  ModalityType,
} from '../api/types';

export interface PathologyProfile {
  id: BonePathologyClass;
  code: string;
  title: string;
  shortTitle: string;
  subtitle: string;
  radiographicHallmarks: string[];
  pathologySummary: string;
  keyRadiomicsFeatures: string[];
  accentTone: 'rose' | 'amber' | 'sky' | 'emerald';
}

export interface ModalitySpec {
  id: ModalityType;
  label: string;
  shortLabel: string;
  protocolNote: string;
  typicalFindings: string;
}

export interface ReferenceScanPreset {
  id: string;
  sampleCase: BonePathologyClass;
  modality: ModalityType;
  title: string;
  anatomy: string;
  imagePath: string;
  filename: string;
  clinicalNote: string;
}

export const PATHOLOGY_PROFILES: Record<BonePathologyClass, PathologyProfile> = {
  metastasis: {
    id: 'metastasis',
    code: 'SECONDARY_BONE_METASTASIS',
    title: 'Secondary Bone Metastasis',
    shortTitle: 'Bone Metastasis',
    subtitle: 'Focal osteolytic, osteoblastic, or mixed skeletal invasion',
    radiographicHallmarks: [
      'Ill-defined permeative or moth-eaten osteolytic destruction',
      'Endosteal scalloping and focal cortical thinning or breach',
      'Asymmetric trabecular loss with high Mirels pathological fracture risk',
    ],
    pathologySummary:
      'Tumor cell seeding disrupts coupled osteoblast–osteoclast homeostasis in trabecular marrow, creating focal bone loss or reactive sclerosis requiring urgent orthopedic risk triage.',
    keyRadiomicsFeatures: ['GLCM Contrast', 'Cortical Thickness', 'Texture Entropy'],
    accentTone: 'rose',
  },
  paget: {
    id: 'paget',
    code: 'PAGETS_DISEASE',
    title: "Paget's Disease of Bone",
    shortTitle: "Paget's Disease",
    subtitle: 'Osteitis deformans — accelerated disorganized bone remodeling',
    radiographicHallmarks: [
      'Thickened, coarse trabecular markings and widened cortical bone',
      'Mixed lytic–sclerotic remodeling including "cotton-wool" radiodensity',
      'Long-bone bowing or pelvic iliopectineal cortical expansion (brim sign)',
    ],
    pathologySummary:
      'Excessive multinucleated osteoclastic resorption followed by chaotic osteoblastic woven bone formation produces mechanically inferior yet radiographically dense enlarged bone.',
    keyRadiomicsFeatures: ['GLCM Contrast', 'GLRLM Long-Run Emphasis', 'Cortical Thickness'],
    accentTone: 'amber',
  },
  oi: {
    id: 'oi',
    code: 'OSTEOGENESIS_IMPERFECTA',
    title: 'Osteogenesis Imperfecta (OI)',
    shortTitle: 'Osteogenesis Imperfecta',
    subtitle: 'Hereditary Type I collagen brittle-bone skeletal disorder',
    radiographicHallmarks: [
      'Diffuse generalized osteopenia with hyperlucent diaphyseal cortex',
      'Gracile over-tubulated long bones with angular bowing deformities',
      'Multiple transverse fractures at varying healing stages & hyperplastic callus',
    ],
    pathologySummary:
      'Mutations in COL1A1/COL1A2 destabilize fibrillar Type I collagen, reducing volumetric mineral density and structural bending rigidity across appendicular and axial skeletons.',
    keyRadiomicsFeatures: ['Bone Volume Fraction', 'Cortical Thickness', 'First-order Mean Intensity'],
    accentTone: 'sky',
  },
  healthy: {
    id: 'healthy',
    code: 'HEALTHY_SKELETAL_CONTROL',
    title: 'Healthy Skeletal Control',
    shortTitle: 'Healthy Control',
    subtitle: 'Normal trabecular architecture & intact cortical margin',
    radiographicHallmarks: [
      'Uniform dense cortical contour without endosteal scalloping or breach',
      'Ordered primary compressive and tensile trabecular latticework',
      'Absence of focal lytic lucency, cotton-wool sclerosis, or pathologic bowing',
    ],
    pathologySummary:
      'Baseline physiological skeletal mineralization and micro-architectural continuity with low biomechanical fracture vulnerability.',
    keyRadiomicsFeatures: ['Bone Volume Fraction', 'Cortical Thickness', 'First-order Mean Intensity'],
    accentTone: 'emerald',
  },
};

export const MODALITY_SPECS: Record<ModalityType, ModalitySpec> = {
  xray: {
    id: 'xray',
    label: '2D Digital X-ray',
    shortLabel: 'X-ray',
    protocolNote: 'Planar radiography (AP/lateral) — evaluates cortical thickness, bowing deformity, & focal trabecular lucency.',
    typicalFindings: '224×224 standardized projection with cortical envelope contrast enhancement',
  },
  ct: {
    id: 'ct',
    label: 'CT Scan (Bone Window)',
    shortLabel: 'CT Scan',
    protocolNote: 'Cross-sectional computed tomography slice — resolves endosteal scalloping, fine trabecular mosaic, and Hounsfield attenuation.',
    typicalFindings: 'Multi-planar trabecular & cortical bone-window radiomic profiling',
  },
  bone_scan: {
    id: 'bone_scan',
    label: 'Bone Scintigraphy (Tc-99m)',
    shortLabel: 'Bone Scan',
    protocolNote: 'Whole-body nuclear MDP uptake — detects focal osteoblastic hotspot flare across axial & appendicular skeleton.',
    typicalFindings: 'High-sensitivity metabolic turnover & multifocal hotspot screening',
  },
};

export const REFERENCE_SCAN_PRESETS: ReferenceScanPreset[] = [
  {
    id: 'ref-paget-pelvis',
    sampleCase: 'paget',
    modality: 'xray',
    title: "Paget's Disease (Pelvis X-ray)",
    anatomy: 'Pelvis / Proximal Femur · X-ray',
    imagePath: '/demo/pelvis-xray-2.png',
    filename: 'paget-pelvis-xray.png',
    clinicalNote: 'Coarse thickened trabeculae & iliopectineal cortical expansion',
  },
  {
    id: 'ref-metastasis-femur-ct',
    sampleCase: 'metastasis',
    modality: 'ct',
    title: 'Bone Metastasis (Femur CT)',
    anatomy: 'Femoral Shaft · CT Bone Window',
    imagePath: '/demo/femur-xray-1.png',
    filename: 'metastasis-femur-ct.png',
    clinicalNote: 'Focal lytic lesion & cortical thinning requiring Mirels assessment',
  },
  {
    id: 'ref-oi-tibia-xray',
    sampleCase: 'oi',
    modality: 'xray',
    title: 'Osteogenesis Imperfecta (Tibia)',
    anatomy: 'Tibial Diaphysis · X-ray',
    imagePath: '/demo/tibia-xray-3.png',
    filename: 'oi-tibia-xray.png',
    clinicalNote: 'Diffuse osteopenia, gracile cortex & anterior bowing deformity',
  },
  {
    id: 'ref-healthy-femur',
    sampleCase: 'healthy',
    modality: 'xray',
    title: 'Healthy Skeletal Control',
    anatomy: 'Proximal Femur · X-ray',
    imagePath: '/demo/femur-xray-1.png',
    filename: 'healthy-femur-xray.png',
    clinicalNote: 'Intact cortical contour and physiological trabecular density',
  },
];

export function getPathologyProfile(cls: BonePathologyClass): PathologyProfile {
  return PATHOLOGY_PROFILES[cls] || PATHOLOGY_PROFILES.healthy;
}

export function getModalitySpec(modality: ModalityType): ModalitySpec {
  return MODALITY_SPECS[modality] || MODALITY_SPECS.xray;
}

export interface AbstentionAdvisorySummary {
  hasAdvisory: boolean;
  reason: 'none' | 'entropy' | 'ood' | 'both' | 'invalid_probs';
  validProbabilities: boolean;
}

export function evaluateAbstentionAdvisory(
  entropy: number | undefined,
  oodFlag: boolean | undefined,
  probabilities: ClassProbabilities,
): AbstentionAdvisorySummary {
  const vals = Object.values(probabilities);
  const validProbabilities =
    vals.length === 4 &&
    vals.every((v) => Number.isFinite(v) && v >= 0 && v <= 1) &&
    Math.abs(vals.reduce((sum, v) => sum + v, 0) - 1) < 0.02;

  if (!validProbabilities) {
    return { hasAdvisory: true, reason: 'invalid_probs', validProbabilities: false };
  }

  const highEntropy = !Number.isFinite(entropy) || (entropy as number) > 0.6;
  const ood = Boolean(oodFlag);

  if (highEntropy && ood) {
    return { hasAdvisory: true, reason: 'both', validProbabilities: true };
  }
  if (highEntropy) {
    return { hasAdvisory: true, reason: 'entropy', validProbabilities: true };
  }
  if (ood) {
    return { hasAdvisory: true, reason: 'ood', validProbabilities: true };
  }
  return { hasAdvisory: false, reason: 'none', validProbabilities: true };
}

export function sortProbabilitiesDescending(
  probabilities: ClassProbabilities,
): Array<{ id: BonePathologyClass; probability: number; profile: PathologyProfile }> {
  return (Object.entries(probabilities) as Array<[BonePathologyClass, number]>)
    .sort((a, b) => b[1] - a[1])
    .map(([id, probability]) => ({
      id,
      probability,
      profile: getPathologyProfile(id),
    }));
}
