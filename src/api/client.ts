import {
  AnalyzePayload,
  AnalyzeResponse,
  HealthResponse,
  FeedbackPayload,
  FeedbackRecord,
  FeedbackStats,
  EventPayload,
  JobStatusResponse,
  NetworkError,
  TimeoutError,
  ValidationError,
  ServerError,
  OsteoGuardError,
} from './types';

const RAW_BASE_URL = ((import.meta as unknown as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL) ?? '';

function resolveApiBaseUrl(rawUrl: string): string {
  const trimmed = (rawUrl || '').trim().replace(/\/+$/, '');
  if (!trimmed) return '';
  if (typeof window !== 'undefined') {
    try {
      const parsed = new URL(trimmed, window.location.origin);
      const isLegacyLocalPort =
        (parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1') &&
        (parsed.port === '8787' || parsed.port === '8000') &&
        parsed.port !== window.location.port;
      if (isLegacyLocalPort) {
        return '';
      }
    } catch {
      return '';
    }
  }
  return trimmed;
}

export const API_BASE_URL = resolveApiBaseUrl(RAW_BASE_URL);

const DEFAULT_TIMEOUT_MS = 10000;
const MAX_RETRIES = 2;

/**
 * Normalizes relative URLs (e.g. /uploads/xyz.png) to be prefixed with API_BASE_URL.
 */
export function resolveApiAssetUrl(relativeOrAbsoluteUrl: string): string {
  if (!relativeOrAbsoluteUrl) return '';
  if (relativeOrAbsoluteUrl.startsWith('http://') || relativeOrAbsoluteUrl.startsWith('https://')) {
    return relativeOrAbsoluteUrl;
  }
  if (!API_BASE_URL) {
    return relativeOrAbsoluteUrl;
  }
  const cleanPath = relativeOrAbsoluteUrl.startsWith('/') ? relativeOrAbsoluteUrl : `/${relativeOrAbsoluteUrl}`;
  return `${API_BASE_URL}${cleanPath}`;
}

/**
 * Fetch wrapper with timeout and exponential backoff retry.
 */
async function fetchWithResilience(
  url: string,
  options: RequestInit,
  retries = MAX_RETRIES,
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  let attempt = 0;

  while (attempt <= retries) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.status === 400 || response.status === 422) {
        let errMessage = 'Validation failed';
        try {
          const json = await response.json();
          if (json.error) errMessage = json.error;
        } catch {}
        throw new ValidationError(errMessage);
      }

      if (response.status >= 500) {
        let errMessage = `Server encountered error (${response.status})`;
        try {
          const json = await response.json();
          if (json.error) errMessage = json.error;
        } catch {}
        throw new ServerError(errMessage, response.status);
      }

      return response;
    } catch (err: unknown) {
      clearTimeout(timeoutId);

      const isAbort = (err as { name?: string })?.name === 'AbortError';
      const isValidation = err instanceof ValidationError;

      if (isValidation) {
        throw err;
      }

      if (attempt === retries) {
        if (isAbort) {
          throw new TimeoutError(`Request to ${url} timed out after ${timeoutMs}ms`);
        }
        if (err instanceof ServerError) {
          throw err;
        }
        const message = err instanceof Error ? err.message : 'Network connection failure';
        throw new NetworkError(`Unable to reach OsteoGuard service: ${message}`);
      }

      // Exponential backoff wait: 300ms, 600ms
      const backoffWait = Math.pow(2, attempt) * 300;
      await new Promise((res) => setTimeout(res, backoffWait));
      attempt++;
    }
  }

  throw new NetworkError('Exceeded maximum retry attempts');
}

/**
 * GET /api/health
 */
export async function checkHealth(customBaseUrl?: string): Promise<HealthResponse> {
  const base = (customBaseUrl ?? API_BASE_URL).replace(/\/+$/, '');
  const url = `${base}/api/health`;

  try {
    const response = await fetchWithResilience(url, { method: 'GET' }, 1, 4000);
    return (await response.json()) as HealthResponse;
  } catch (error) {
    if (error instanceof OsteoGuardError) throw error;
    throw new NetworkError(`Health check failed: ${(error as Error).message}`);
  }
}

/**
 * POST /api/v1/analyze
 */
export async function analyzeImage(payload: AnalyzePayload, customBaseUrl?: string): Promise<AnalyzeResponse> {
  const base = (customBaseUrl ?? API_BASE_URL).replace(/\/+$/, '');
  const url = `${base}/api/v1/analyze`;

  const formData = new FormData();
  formData.append('image', payload.image);
  formData.append('modality', payload.modality);

  const response = await fetchWithResilience(url, {
    method: 'POST',
    body: formData,
  });

  return (await response.json()) as AnalyzeResponse;
}

/**
 * POST /api/v1/feedback
 */
export async function submitFeedback(payload: FeedbackPayload): Promise<{ success: boolean; feedbackId: string }> {
  const url = `${API_BASE_URL}/api/v1/feedback`;
  const response = await fetchWithResilience(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return (await response.json()) as { success: boolean; feedbackId: string };
}

/**
 * GET /api/v1/feedback/stats
 */
export async function getFeedbackStats(): Promise<FeedbackStats> {
  const url = `${API_BASE_URL}/api/v1/feedback/stats`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as FeedbackStats;
}

/**
 * GET /api/v1/feedback/list
 */
export async function getFeedbackList(): Promise<FeedbackRecord[]> {
  const url = `${API_BASE_URL}/api/v1/feedback/list`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  const data = (await response.json()) as { items: FeedbackRecord[] };
  return data.items || [];
}

/**
 * POST /api/v1/events (Telemetry logging)
 */
export async function logTelemetryEvent(event: Omit<EventPayload, 'timestamp'>): Promise<void> {
  try {
    const url = `${API_BASE_URL}/api/v1/events`;
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...event,
        timestamp: new Date().toISOString(),
      }),
    });
  } catch (err) {
    // Non-blocking telemetry
    console.debug('Telemetry event log skipped:', err);
  }
}

/**
 * GET /api/v1/jobs/:id (Async polling seam)
 */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const url = `${API_BASE_URL}/api/v1/jobs/${jobId}`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as JobStatusResponse;
}

/**
 * WebSocket / EventSource subscription stub for real-time model jobs
 */
export function onJobUpdate(jobId: string, callback: (job: JobStatusResponse) => void): () => void {
  const interval = setInterval(async () => {
    try {
      const status = await getJobStatus(jobId);
      callback(status);
      if (status.status === 'done' || status.status === 'failed') {
        clearInterval(interval);
      }
    } catch (err) {
      console.warn('Job polling error:', err);
    }
  }, 1000);

  return () => clearInterval(interval);
}

// ━━━ Stage 31 Research API Client Functions ━━━

export async function getResearchStatus(): Promise<any> {
  const url = `${API_BASE_URL}/api/v1/research/status`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return await response.json();
}

export async function getResearchTracks(): Promise<import('./types').ResearchTracksResponse> {
  const url = `${API_BASE_URL}/api/v1/research/tracks`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').ResearchTracksResponse;
}

export async function getEvidenceRegistry(): Promise<import('./types').EvidenceRegistryResponse> {
  const url = `${API_BASE_URL}/api/v1/evidence/registry`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').EvidenceRegistryResponse;
}

export async function getArchitectureReadiness(): Promise<import('./types').ArchitectureReadinessResponse> {
  const url = `${API_BASE_URL}/api/v1/architecture/readiness`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').ArchitectureReadinessResponse;
}

export async function getDatasetGovernance(): Promise<import('./types').DatasetGovernanceResponse> {
  const url = `${API_BASE_URL}/api/v1/datasets/status`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').DatasetGovernanceResponse;
}

export async function getTrackDDatasetReadiness(): Promise<import('./types').TrackDDatasetReadinessResponse> {
  const url = `${API_BASE_URL}/api/v1/datasets/readiness`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').TrackDDatasetReadinessResponse;
}

export async function runRadiomicsDemo(payload: { phantom_type: string; grid_size?: number; spacing_mm?: number }): Promise<import('./types').RadiomicsDemoResponse> {
  const url = `${API_BASE_URL}/api/v1/radiomics/demo`;
  const response = await fetchWithResilience(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return (await response.json()) as import('./types').RadiomicsDemoResponse;
}

export async function runBiomechanicsGeometry(payload: { phantom_type: string; voxel_size_mm?: number }): Promise<import('./types').BiomechanicsGeometryResponse> {
  const url = `${API_BASE_URL}/api/v1/biomechanics/geometry`;
  const response = await fetchWithResilience(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return (await response.json()) as import('./types').BiomechanicsGeometryResponse;
}

export async function runBiomechanicsMirels(payload: { site: number; pain: number; radiographic_nature: number; lesion_size: number }): Promise<import('./types').BiomechanicsMirelsResponse> {
  const url = `${API_BASE_URL}/api/v1/biomechanics/mirels`;
  const response = await fetchWithResilience(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return (await response.json()) as import('./types').BiomechanicsMirelsResponse;
}

export async function runExplainabilityDemo(payload: { target_class: number; shift_horizontal?: number }): Promise<import('./types').ExplainabilityDemoResponse> {
  const url = `${API_BASE_URL}/api/v1/explainability/demo`;
  const response = await fetchWithResilience(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return (await response.json()) as import('./types').ExplainabilityDemoResponse;
}

export async function runUncertaintyDemo(payload: { distribution_preset?: string; custom_probabilities?: number[]; ood_flag?: boolean }): Promise<import('./types').UncertaintyDemoResponse> {
  const url = `${API_BASE_URL}/api/v1/uncertainty/demo`;
  const response = await fetchWithResilience(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return (await response.json()) as import('./types').UncertaintyDemoResponse;
}

export async function runSyntheticWorkbench(payload: Record<string, any> = {}): Promise<import('./types').SyntheticWorkbenchRunResponse> {
  const url = `${API_BASE_URL}/api/v1/workbench/run-demo`;
  const response = await fetchWithResilience(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return (await response.json()) as import('./types').SyntheticWorkbenchRunResponse;
}

export async function requestFourClassInference(): Promise<any> {
  const url = `${API_BASE_URL}/api/v1/diagnostic/four-class`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    return await response.json();
  } catch (err: any) {
    return { status: 'BLOCKED', message: err.message };
  }
}

export async function getTrainedModelsOverview(): Promise<import('./types').TrainedModelsOverviewResponse> {
  const url = `${API_BASE_URL}/api/v1/trained-models/overview`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').TrainedModelsOverviewResponse;
}

export async function getSecurityStatus(): Promise<import('./types').SecurityStatusResponse> {
  const url = `${API_BASE_URL}/api/v1/security/status`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').SecurityStatusResponse;
}

export async function getClinicalTrainingStatus(): Promise<import('./types').ClinicalTrainingStatusResponse> {
  const url = `${API_BASE_URL}/api/v1/clinical/training-status`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').ClinicalTrainingStatusResponse;
}

// ━━━ Stage 35 Institutional Acquisition, Governance & Cohort Planning Clients ━━━

export async function getDatasetSources(): Promise<import('./types').DatasetSourcesResponse> {
  const url = `${API_BASE_URL}/api/v1/datasets/sources`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').DatasetSourcesResponse;
}

export async function getDatasetAcquisitionStatus(): Promise<import('./types').DatasetAcquisitionStatusResponse> {
  const url = `${API_BASE_URL}/api/v1/datasets/acquisition-status`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').DatasetAcquisitionStatusResponse;
}

export async function getDatasetGovernanceStatus(): Promise<import('./types').DatasetGovernanceStatusResponse> {
  const url = `${API_BASE_URL}/api/v1/datasets/governance-status`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').DatasetGovernanceStatusResponse;
}

export async function getCohortPlanning(): Promise<import('./types').CohortPlanningResponse> {
  const url = `${API_BASE_URL}/api/v1/datasets/cohort-planning`;
  const response = await fetchWithResilience(url, { method: 'GET' });
  return (await response.json()) as import('./types').CohortPlanningResponse;
}

export interface OnboardingSource {
  source_id: string;
  name: string;
  source_type: string;
  source_reference: string;
  description: string;
  modalities: string[];
  target_categories: string[];
  version: string;
  acquisition_status: string;
  verification_status: string;
  integrity_status: string;
  staging_status: string;
  governance_approved: boolean;
  package: { valid: boolean; errors: string[]; inventory_sha256: string; clinical_validity: string; training_authorized: boolean } | null;
}

async function onboardingJson<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Dataset onboarding request failed (${response.status}).`);
  return body as T;
}

export async function getOnboardingSources(bearerToken: string): Promise<{ sources: OnboardingSource[] }> {
  const response = await fetchWithResilience(`${API_BASE_URL}/api/v1/dataset-onboarding/sources`, {
    method: 'GET', headers: { Authorization: `Bearer ${bearerToken}` },
  });
  return onboardingJson<{ sources: OnboardingSource[] }>(response);
}

export async function registerOnboardingSource(bearerToken: string, payload: Record<string, unknown>): Promise<OnboardingSource> {
  const response = await fetchWithResilience(`${API_BASE_URL}/api/v1/dataset-onboarding/sources`, {
    method: 'POST', headers: { Authorization: `Bearer ${bearerToken}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return onboardingJson<OnboardingSource>(response);
}

export async function transitionOnboardingSource(bearerToken: string, sourceId: string, payload: Record<string, unknown>): Promise<OnboardingSource> {
  const response = await fetchWithResilience(`${API_BASE_URL}/api/v1/dataset-onboarding/sources/${encodeURIComponent(sourceId)}/events`, {
    method: 'POST', headers: { Authorization: `Bearer ${bearerToken}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return onboardingJson<OnboardingSource>(response);
}

export async function verifyOnboardingSource(bearerToken: string, sourceId: string): Promise<NonNullable<OnboardingSource['package']>> {
  const response = await fetchWithResilience(`${API_BASE_URL}/api/v1/dataset-onboarding/sources/${encodeURIComponent(sourceId)}/verify`, {
    method: 'POST', headers: { Authorization: `Bearer ${bearerToken}` },
  });
  return onboardingJson<NonNullable<OnboardingSource['package']>>(response);
}

export async function stageOnboardingSource(bearerToken: string, sourceId: string): Promise<{ status: string; training_authorized: boolean }> {
  const response = await fetchWithResilience(`${API_BASE_URL}/api/v1/dataset-onboarding/sources/${encodeURIComponent(sourceId)}/stage`, {
    method: 'POST', headers: { Authorization: `Bearer ${bearerToken}` },
  });
  return onboardingJson<{ status: string; training_authorized: boolean }>(response);
}



