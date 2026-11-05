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
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, '');

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
