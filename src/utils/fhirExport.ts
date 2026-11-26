import type { AnalyzeResponse, BonePathologyClass } from '../api/types';

export interface FhirBundle {
  resourceType: 'Bundle';
  id: string;
  type: 'document' | 'collection';
  timestamp: string;
  meta: {
    profile: string[];
    tag: Array<{ system: string; code: string; display: string }>;
  };
  entry: Array<{
    fullUrl: string;
    resource: Record<string, unknown>;
  }>;
}

const SNOMED_MAP: Record<BonePathologyClass, { code: string; display: string }> = {
  metastasis: { code: '94222008', display: 'Secondary malignant neoplasm of bone' },
  paget: { code: '2089002', display: 'Osteitis deformans (Paget disease of bone)' },
  oi: { code: '78314001', display: 'Osteogenesis imperfecta' },
  healthy: { code: '17621005', display: 'Normal skeletal finding' },
};

const LOINC_MODALITY_MAP: Record<string, { code: string; display: string }> = {
  xray: { code: '30746-2', display: 'Skeletal X-ray Study' },
  ct: { code: '24627-2', display: 'CT Musculoskeletal Bone Window Study' },
  bone_scan: { code: '39840-5', display: 'Nuclear Medicine Tc-99m Bone Scintigraphy' },
};

/**
 * Generates an HL7 FHIR R4 compliant Bundle containing a DiagnosticReport
 * and structured Observations (classification score, Mirels fracture risk,
 * uncertainty entropy, and top radiomics SHAP features) for PACS/EHR exchange.
 */
export function buildFhirDiagnosticReport(result: AnalyzeResponse): FhirBundle {
  const snomed = SNOMED_MAP[result.prediction.topClass];
  const modalityCode = LOINC_MODALITY_MAP[result.modality] || LOINC_MODALITY_MAP.xray;
  const confidencePct = Math.round(result.prediction.confidence * 100);
  const reportUuid = `urn:uuid:report-${result.requestId}`;
  const obsFractureUuid = `urn:uuid:obs-fracture-${result.requestId}`;
  const obsUncertaintyUuid = `urn:uuid:obs-uncertainty-${result.requestId}`;
  const obsRadiomicsUuid = `urn:uuid:obs-radiomics-${result.requestId}`;

  return {
    resourceType: 'Bundle',
    id: `osteoguard-fhir-${result.requestId}`,
    type: 'collection',
    timestamp: result.timestamp || new Date().toISOString(),
    meta: {
      profile: ['http://hl7.org/fhir/StructureDefinition/DiagnosticReport'],
      tag: [
        {
          system: 'https://osteoguard.ai/fhir/CodeSystem/execution-mode',
          code: 'research-prototype-preliminary',
          display: 'OsteoGuard AI Preliminary Research Decision Support (Requires Radiologist Sign-Off)',
        },
      ],
    },
    entry: [
      {
        fullUrl: reportUuid,
        resource: {
          resourceType: 'DiagnosticReport',
          id: result.requestId,
          status: 'preliminary',
          category: [
            {
              coding: [
                {
                  system: 'http://terminology.hl7.org/CodeSystem/v2-0074',
                  code: 'RAD',
                  display: 'Radiology',
                },
              ],
            },
          ],
          code: {
            coding: [
              {
                system: 'http://loinc.org',
                code: modalityCode.code,
                display: modalityCode.display,
              },
            ],
            text: `OsteoGuard AI Differential Skeletal Report (${result.modality.toUpperCase()})`,
          },
          effectiveDateTime: result.timestamp,
          issued: new Date().toISOString(),
          result: [
            { reference: obsFractureUuid, display: 'Pathological Fracture Risk Assessment' },
            { reference: obsUncertaintyUuid, display: 'Selective Prediction Uncertainty & OOD Gate' },
            { reference: obsRadiomicsUuid, display: 'Quantitative IBSI Radiomics SHAP Summary' },
          ],
          conclusionCode: [
            {
              coding: [
                {
                  system: 'http://snomed.info/sct',
                  code: snomed.code,
                  display: snomed.display,
                },
              ],
              text: `${snomed.display} (${confidencePct}% model score)`,
            },
          ],
          conclusion: `${snomed.display} (${confidencePct}% model score; Fracture risk: ${result.fractureRisk.percent}% [${result.fractureRisk.level}]; Normalized entropy: ${result.uncertaintyDetail.entropy.toFixed(3)}). ${result.disclaimer}`,
        },
      },
      {
        fullUrl: obsFractureUuid,
        resource: {
          resourceType: 'Observation',
          id: `obs-fracture-${result.requestId}`,
          status: 'preliminary',
          code: {
            coding: [
              {
                system: 'https://osteoguard.ai/fhir/CodeSystem/biomechanics',
                code: 'mirels-fracture-risk',
                display: 'Impending Pathological Fracture Risk Index',
              },
            ],
          },
          valueQuantity: {
            value: result.fractureRisk.percent,
            unit: '%',
            system: 'http://unitsofmeasure.org',
            code: '%',
          },
          interpretation: [
            {
              text: `${result.fractureRisk.level.toUpperCase()} — Cortical breach & Mirels vulnerability risk (${result.fractureRisk.percent}%)`,
            },
          ],
        },
      },
      {
        fullUrl: obsUncertaintyUuid,
        resource: {
          resourceType: 'Observation',
          id: `obs-uncertainty-${result.requestId}`,
          status: 'preliminary',
          code: {
            coding: [
              {
                system: 'https://osteoguard.ai/fhir/CodeSystem/uncertainty',
                code: 'normalized-shannon-entropy',
                display: 'Predictive Shannon Entropy & OOD Check',
              },
            ],
          },
          valueQuantity: {
            value: Number(result.uncertaintyDetail.entropy.toFixed(4)),
            unit: 'nats',
          },
          component: [
            {
              code: { text: 'Out-of-Distribution Flag' },
              valueBoolean: result.uncertaintyDetail.oodFlag,
            },
            {
              code: { text: 'Requires Human Radiologist Review' },
              valueBoolean: result.uncertaintyDetail.oodFlag || result.uncertaintyDetail.entropy >= 0.6,
            },
          ],
        },
      },
      {
        fullUrl: obsRadiomicsUuid,
        resource: {
          resourceType: 'Observation',
          id: `obs-radiomics-${result.requestId}`,
          status: 'preliminary',
          code: {
            coding: [
              {
                system: 'https://osteoguard.ai/fhir/CodeSystem/radiomics',
                code: 'shap-ibsi-attributions',
                display: 'Quantitative Radiomics Feature Attributions (SHAP)',
              },
            ],
          },
          component: result.shap.map((item) => ({
            code: { text: item.name },
            valueQuantity: { value: item.value, unit: 'shap-weight' },
          })),
        },
      },
    ],
  };
}
