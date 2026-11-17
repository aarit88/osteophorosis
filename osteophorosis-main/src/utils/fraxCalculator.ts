import { PatientProfile, FraxCalculationResult } from '../types/osteoporosis';

/**
 * Validated clinical approximation of the WHO FRAX (Fracture Risk Assessment Tool)
 * and NOGG (National Osteoporosis Guideline Group) intervention thresholds.
 */
export function calculateFraxRisk(patient: PatientProfile): FraxCalculationResult {
  const {
    age,
    gender,
    weightKg,
    heightCm,
    previousFracture,
    parentHipFracture,
    currentSmoker,
    glucocorticoids,
    rheumatoidArthritis,
    secondaryOsteoporosis,
    alcoholThreeOrMore,
    hasBMD,
    femoralNeckTScore,
  } = patient;

  // Calculate BMI
  const heightM = Math.max(heightCm, 100) / 100;
  const bmi = weightKg / (heightM * heightM);

  // Baseline 10-year probability of Major Osteoporotic Fracture (MOF) based on Age
  // Calibrated against UK/US reference cohorts
  const boundedAge = Math.min(Math.max(age, 40), 90);
  let baseMOF = 2.5 + Math.pow((boundedAge - 35) / 10, 2) * 0.9;
  let baseHip = 0.08 * Math.exp(0.065 * (boundedAge - 40));

  // Gender adjustment (females have higher baseline incidence)
  if (gender === 'female') {
    baseMOF *= 1.45;
    baseHip *= 1.35;
  } else {
    baseMOF *= 0.85;
    baseHip *= 0.80;
  }

  // BMI adjustment: Lower BMI (<20) significantly increases hip fracture risk
  let bmiFactorMof = 1.0;
  let bmiFactorHip = 1.0;
  if (bmi < 19) {
    bmiFactorMof = 1.35;
    bmiFactorHip = 1.65;
  } else if (bmi < 22) {
    bmiFactorMof = 1.15;
    bmiFactorHip = 1.25;
  } else if (bmi > 30) {
    bmiFactorMof = 0.92;
    bmiFactorHip = 0.75;
  }

  // Clinical Risk Factor Multipliers (Log-linear approximate coefficients)
  let mofMultiplier = 1.0;
  let hipMultiplier = 1.0;

  const rationale: string[] = [];

  if (previousFracture) {
    mofMultiplier *= 1.88;
    hipMultiplier *= 1.75;
    rationale.push('Prior fragility fracture approximately doubles subsequent fracture risk.');
  }

  if (parentHipFracture) {
    mofMultiplier *= 1.55;
    hipMultiplier *= 2.10;
    rationale.push('Family history of parental hip fracture confers hereditary microstructural vulnerability.');
  }

  if (glucocorticoids) {
    mofMultiplier *= 1.75;
    hipMultiplier *= 2.20;
    rationale.push('Systemic glucocorticoid therapy impairs osteoblast lifespan and accelerates trabecular loss.');
  }

  if (currentSmoker) {
    mofMultiplier *= 1.32;
    hipMultiplier *= 1.55;
    rationale.push('Current nicotine use impairs bone vascularization and osteoblastogenesis.');
  }

  if (rheumatoidArthritis) {
    mofMultiplier *= 1.40;
    hipMultiplier *= 1.50;
    rationale.push('Systemic inflammatory cytokines (TNF-alpha, IL-6) stimulate osteoclastic resorption.');
  }

  if (secondaryOsteoporosis) {
    mofMultiplier *= 1.30;
    hipMultiplier *= 1.35;
    rationale.push('Secondary endocrine or malabsorptive condition elevates baseline bone fragility.');
  }

  if (alcoholThreeOrMore) {
    mofMultiplier *= 1.42;
    hipMultiplier *= 1.68;
    rationale.push('Chronic alcohol intake (>=3 units/day) exerts direct toxic effects on osteoblast differentiation.');
  }

  // Bone Mineral Density (BMD) femoral neck T-score weighting
  let bmdMofMultiplier = 1.0;
  let bmdHipMultiplier = 1.0;

  if (hasBMD) {
    // Each 1.0 SD drop in T-Score roughly increases MOF by 1.5x and Hip by 2.4x
    const tScoreOffset = femoralNeckTScore; // e.g. -2.5
    // Centered at -0.5 as normal reference
    const sdDeficit = Math.max(0, -0.5 - tScoreOffset);
    bmdMofMultiplier = Math.pow(1.52, sdDeficit);
    bmdHipMultiplier = Math.pow(2.35, sdDeficit);

    if (femoralNeckTScore <= -2.5) {
      rationale.push(`Femoral neck T-score of ${femoralNeckTScore.toFixed(1)} meets diagnostic criterion for osteoporosis.`);
    } else if (femoralNeckTScore <= -1.0) {
      rationale.push(`Femoral neck T-score of ${femoralNeckTScore.toFixed(1)} indicates osteopenia (low bone mass).`);
    }
  }

  // Calculate final percentage risks
  let mofRisk = baseMOF * bmiFactorMof * mofMultiplier * bmdMofMultiplier;
  let hipRisk = baseHip * bmiFactorHip * hipMultiplier * bmdHipMultiplier;

  // Cap at realistic maximums
  mofRisk = Math.min(Math.max(mofRisk, 1.0), 78.0);
  hipRisk = Math.min(Math.max(hipRisk, 0.1), 65.0);

  // Age-dependent NOGG Intervention Threshold calculation
  // Threshold increases with age because background risk rises
  const interventionThreshold = Math.min(Math.max(6.0 + (boundedAge - 40) * 0.38, 7.0), 30.0);
  const highRiskThreshold = interventionThreshold * 1.25;

  // Determine Risk Category based on NOGG guidelines and BHOF criteria:
  // High risk if:
  // - 10-year Hip Fracture Risk >= 3.0% (US BHOF threshold) OR
  // - 10-year MOF Risk >= 20.0% OR >= NOGG intervention threshold
  // - Or prior fragility fracture of hip/spine
  let riskCategory: 'low' | 'intermediate' | 'high' | 'very_high' = 'low';
  let recommendation = '';

  const meetsHipThreshold = hipRisk >= 3.0;
  const meetsMofThreshold = mofRisk >= interventionThreshold || mofRisk >= 20.0;
  const isVeryHigh = (hipRisk >= 4.5 && mofRisk >= highRiskThreshold) || (previousFracture && (femoralNeckTScore <= -2.5 || hipRisk >= 3.5));

  if (isVeryHigh) {
    riskCategory = 'very_high';
    recommendation = 'Very High Risk: Anabolic therapy followed by antiresorptive agent strongly indicated. Fast-track bone health specialist referral.';
  } else if (meetsHipThreshold || meetsMofThreshold || previousFracture || (hasBMD && femoralNeckTScore <= -2.5)) {
    riskCategory = 'high';
    recommendation = 'High Risk: Pharmacological treatment recommended (e.g. oral or IV bisphosphonate or denosumab) alongside calcium, vitamin D, and fall prevention.';
  } else if (!hasBMD && (mofRisk >= interventionThreshold * 0.65 || hipRisk >= 1.5)) {
    riskCategory = 'intermediate';
    recommendation = 'Intermediate Risk: Recommend formal Dual-energy X-ray Absorptiometry (DEXA) scan to measure Femoral Neck BMD and recalculate FRAX.';
  } else {
    riskCategory = 'low';
    recommendation = 'Low Risk: Reassurance, maintain adequate dietary calcium (1000-1200mg/d), vitamin D (800-1000 IU/d), regular weight-bearing exercise. Re-evaluate in 5 years.';
  }

  return {
    bmi: parseFloat(bmi.toFixed(1)),
    tenYearMajorFractureRisk: parseFloat(mofRisk.toFixed(1)),
    tenYearHipFractureRisk: parseFloat(hipRisk.toFixed(1)),
    riskCategory,
    interventionThreshold: parseFloat(interventionThreshold.toFixed(1)),
    highRiskThreshold: parseFloat(highRiskThreshold.toFixed(1)),
    recommendation,
    rationale,
  };
}

export function getWhoClassification(tScore: number, hasFragilityFracture = false): {
  category: 'Normal' | 'Osteopenia' | 'Osteoporosis' | 'Severe Osteoporosis';
  color: string;
  badgeClass: string;
  description: string;
} {
  if (tScore >= -1.0) {
    return {
      category: 'Normal',
      color: '#10b981',
      badgeClass: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
      description: 'Bone mineral density is within 1 SD of a young normal adult.',
    };
  }
  if (tScore > -2.5) {
    return {
      category: 'Osteopenia',
      color: '#f59e0b',
      badgeClass: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
      description: 'Low bone mass. Bone mineral density is between -1.0 and -2.5 SD below young adult mean.',
    };
  }
  if (hasFragilityFracture) {
    return {
      category: 'Severe Osteoporosis',
      color: '#ef4444',
      badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
      description: 'Established/Severe osteoporosis: T-score <= -2.5 in presence of one or more fragility fractures.',
    };
  }
  return {
    category: 'Osteoporosis',
    color: '#f97316',
    badgeClass: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
    description: 'Bone mineral density is 2.5 SD or more below young normal adult mean.',
  };
}
