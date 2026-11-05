export type Gender = 'female' | 'male';

export interface PatientProfile {
  id: string;
  name: string;
  age: number;
  gender: Gender;
  weightKg: number;
  heightCm: number;
  previousFracture: boolean;
  parentHipFracture: boolean;
  currentSmoker: boolean;
  glucocorticoids: boolean; // >= 5mg prednisolone daily for >= 3 months
  rheumatoidArthritis: boolean;
  secondaryOsteoporosis: boolean; // e.g. T1D, hyperparathyroidism, hypogonadism, malabsorption
  alcoholThreeOrMore: boolean; // 3 or more units daily
  femoralNeckTScore: number; // e.g. -2.4
  hasBMD: boolean;
}

export interface FraxCalculationResult {
  bmi: number;
  tenYearMajorFractureRisk: number; // percentage
  tenYearHipFractureRisk: number; // percentage
  riskCategory: 'low' | 'intermediate' | 'high' | 'very_high';
  interventionThreshold: number; // Major fracture intervention threshold for age
  highRiskThreshold: number;
  recommendation: string;
  rationale: string[];
}

export interface DexaSite {
  siteName: string;
  bmd: number; // g/cm^2
  tScore: number;
  zScore: number;
  referenceMean: number;
  referenceSd: number;
  previousBmd?: number;
  scanIntervalMonths?: number;
}

export interface DexaAnalysis {
  sites: DexaSite[];
  lowestTScore: number;
  lowestSite: string;
  whoCategory: 'Normal' | 'Osteopenia' | 'Osteoporosis' | 'Severe Osteoporosis';
  fragilityFracturePresent: boolean;
  significantChangeObserved?: boolean;
}

export interface Medication {
  id: string;
  name: string;
  brandName: string;
  drugClass: 'Bisphosphonate' | 'RANKL Inhibitor' | 'Anabolic (PTH/PTHrP analog)' | 'Dual Action (Sclerostin Inhibitor)' | 'SERM';
  route: string;
  frequency: string;
  vertebralReduction: string;
  hipReduction: string;
  nonVertebralReduction: string;
  keyContraindications: string[];
  adverseEffects: string[];
  monitoringReqs: string;
  durationRecommendation: string;
  dentalPrecautions: string;
}

export interface FoodCalciumItem {
  id: string;
  name: string;
  category: 'Dairy' | 'Fortified' | 'Fish/Seafood' | 'Vegetables' | 'Nuts/Seeds';
  serving: string;
  calciumMg: number;
  vitaminDIU: number;
}

export interface CaseStudy {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  patient: PatientProfile;
  dexa: DexaAnalysis;
  clinicalKeyTakeaway: string;
}
