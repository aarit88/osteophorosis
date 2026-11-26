import crypto from 'node:crypto';

export type TriagePriorityTier = 'STAT_URGENT' | 'PRIORITY_REVIEW' | 'ROUTINE';
export type AnalysisJobStatus = 'queued' | 'processing' | 'done' | 'failed';

export interface TriageEvaluation {
  tier: TriagePriorityTier;
  reason: string;
  rankWeight: number;
}

export interface RateLimitCheckResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  resetSeconds: number;
}

/**
 * Per-client sliding-window rate limiter with automatic stale bucket pruning
 * so sustained high-volume clinical API traffic cannot exhaust heap memory.
 */
export class SlidingWindowRateLimiter {
  private readonly maxRequests: number;
  private readonly windowMs: number;
  private readonly maxTrackedClients: number;
  private readonly buckets = new Map<string, number[]>();

  constructor(options: { maxRequests?: number; windowMs?: number; maxTrackedClients?: number } = {}) {
    this.maxRequests = Math.max(1, options.maxRequests ?? 120);
    this.windowMs = Math.max(1000, options.windowMs ?? 60_000);
    this.maxTrackedClients = Math.max(100, options.maxTrackedClients ?? 5000);
  }

  check(clientKey: string, nowMs = Date.now()): RateLimitCheckResult {
    const cutoff = nowMs - this.windowMs;
    const existing = this.buckets.get(clientKey) || [];
    const active = existing.filter((ts) => ts > cutoff);

    if (active.length >= this.maxRequests) {
      this.buckets.set(clientKey, active);
      const oldest = active[0] ?? nowMs;
      const resetSeconds = Math.max(1, Math.ceil((oldest + this.windowMs - nowMs) / 1000));
      return {
        allowed: false,
        limit: this.maxRequests,
        remaining: 0,
        resetSeconds,
      };
    }

    active.push(nowMs);
    if (!this.buckets.has(clientKey) && this.buckets.size >= this.maxTrackedClients) {
      this.prune(cutoff);
      if (this.buckets.size >= this.maxTrackedClients) {
        const oldestKey = this.buckets.keys().next().value;
        if (oldestKey) this.buckets.delete(oldestKey);
      }
    }
    this.buckets.set(clientKey, active);

    return {
      allowed: true,
      limit: this.maxRequests,
      remaining: Math.max(0, this.maxRequests - active.length),
      resetSeconds: Math.ceil(this.windowMs / 1000),
    };
  }

  prune(cutoffMs = Date.now() - this.windowMs): number {
    let removed = 0;
    for (const [key, timestamps] of this.buckets.entries()) {
      const fresh = timestamps.filter((ts) => ts > cutoffMs);
      if (fresh.length === 0) {
        this.buckets.delete(key);
        removed += 1;
      } else {
        this.buckets.set(key, fresh);
      }
    }
    return removed;
  }

  getTrackedClientCount(): number {
    return this.buckets.size;
  }
}

/**
 * Fixed-capacity ring buffer for computing p50 / p95 / p99 inference latency percentiles.
 */
export class LatencyTracker {
  private readonly samples: number[] = [];
  private readonly maxSamples: number;
  private totalRequests = 0;
  private totalErrors = 0;

  constructor(maxSamples = 500) {
    this.maxSamples = Math.max(10, maxSamples);
  }

  record(durationMs: number, isError = false): void {
    const clean = Number.isFinite(durationMs) && durationMs >= 0 ? durationMs : 0;
    this.totalRequests += 1;
    if (isError) this.totalErrors += 1;
    if (this.samples.length >= this.maxSamples) {
      this.samples.shift();
    }
    this.samples.push(clean);
  }

  getSummary(): {
    totalRequests: number;
    totalErrors: number;
    sampleCount: number;
    avgMs: number;
    p50Ms: number;
    p95Ms: number;
    p99Ms: number;
  } {
    if (this.samples.length === 0) {
      return {
        totalRequests: this.totalRequests,
        totalErrors: this.totalErrors,
        sampleCount: 0,
        avgMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
      };
    }

    const sorted = [...this.samples].sort((a, b) => a - b);
    const sum = sorted.reduce((acc, v) => acc + v, 0);
    const pct = (q: number) => {
      const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
      return Number(sorted[idx].toFixed(2));
    };

    return {
      totalRequests: this.totalRequests,
      totalErrors: this.totalErrors,
      sampleCount: sorted.length,
      avgMs: Number((sum / sorted.length).toFixed(2)),
      p50Ms: pct(0.5),
      p95Ms: pct(0.95),
      p99Ms: pct(0.99),
    };
  }
}

/**
 * Evaluates clinical reading priority from Mirels/cortical fracture vulnerability and epistemic uncertainty.
 */
export function evaluateTriagePriority(
  fractureRiskPercent: number,
  normalizedEntropy: number,
  oodFlag: boolean,
): TriageEvaluation {
  if (fractureRiskPercent >= 70) {
    return {
      tier: 'STAT_URGENT',
      reason: `High impending pathological fracture risk (${Math.round(fractureRiskPercent)}%)`,
      rankWeight: 300 + fractureRiskPercent,
    };
  }
  if (oodFlag || normalizedEntropy >= 0.6) {
    return {
      tier: 'STAT_URGENT',
      reason: oodFlag
        ? 'Out-of-distribution input flagged for immediate radiologist review'
        : `High predictive entropy (${normalizedEntropy.toFixed(2)} >= 0.60) requires specialist review`,
      rankWeight: 260 + normalizedEntropy * 40,
    };
  }
  if (fractureRiskPercent >= 35 || normalizedEntropy >= 0.45) {
    return {
      tier: 'PRIORITY_REVIEW',
      reason: `Moderate structural risk (${Math.round(fractureRiskPercent)}%) or elevated ambiguity`,
      rankWeight: 100 + fractureRiskPercent,
    };
  }
  return {
    tier: 'ROUTINE',
    reason: 'Standard screening priority — intact cortex & confident output',
    rankWeight: fractureRiskPercent,
  };
}

export interface RankedWorklistCandidate {
  jobId: string;
  priorityTier?: TriagePriorityTier;
  fractureRiskPercent?: number;
  entropy?: number;
  oodFlag?: boolean;
  createdAtMs: number;
}

export function rankWorklistItems<T extends RankedWorklistCandidate>(items: T[]): T[] {
  return [...items].sort((a, b) => {
    const scoreA = evaluateTriagePriority(a.fractureRiskPercent ?? 0, a.entropy ?? 0, Boolean(a.oodFlag)).rankWeight;
    const scoreB = evaluateTriagePriority(b.fractureRiskPercent ?? 0, b.entropy ?? 0, Boolean(b.oodFlag)).rankWeight;
    if (scoreB !== scoreA) return scoreB - scoreA;
    return b.createdAtMs - a.createdAtMs;
  });
}

export interface ClinicalJobRecord<T = unknown> {
  jobId: string;
  modality: string;
  status: AnalysisJobStatus;
  progress: number;
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
  latencyMs: number | null;
  triagePriority: TriagePriorityTier | null;
  triageReason: string | null;
  result?: T;
  error?: { error_code: string; message: string; statusCode?: number };
  resultUrl: string;
}

interface InternalQueuedTask<T> {
  record: ClinicalJobRecord<T>;
  createdAtMs: number;
  expiresAtMs: number;
  execute: () => Promise<{ result: T; fractureRiskPercent?: number; entropy?: number; oodFlag?: boolean }>;
}

/**
 * Bounded-concurrency asynchronous worker queue for scalable multi-scan batch triage.
 * Isolates inference worker load and prevents unbounded memory retention via TTL + maxJobs pruning.
 */
export class ClinicalAnalysisQueue<T = unknown> {
  private readonly maxConcurrency: number;
  private readonly maxQueueSize: number;
  private readonly ttlMs: number;
  private activeWorkers = 0;
  private completedCount = 0;
  private failedCount = 0;
  private readonly jobs = new Map<string, InternalQueuedTask<T>>();
  private readonly pendingIds: string[] = [];

  constructor(options: { maxConcurrency?: number; maxQueueSize?: number; ttlMs?: number } = {}) {
    this.maxConcurrency = Math.max(1, options.maxConcurrency ?? 4);
    this.maxQueueSize = Math.max(1, options.maxQueueSize ?? 60);
    this.ttlMs = Math.max(1000, options.ttlMs ?? 30 * 60 * 1000);
  }

  enqueue(
    modality: string,
    execute: () => Promise<{ result: T; fractureRiskPercent?: number; entropy?: number; oodFlag?: boolean }>,
  ): { accepted: true; job: ClinicalJobRecord<T>; position: number } | { accepted: false; reason: string } {
    const now = Date.now();
    this.pruneExpired(now);

    const inFlightCount = this.activeWorkers + this.pendingIds.length;
    if (inFlightCount >= this.maxQueueSize) {
      return {
        accepted: false,
        reason: `Clinical job queue at capacity (${this.maxQueueSize} queued/active tasks). Retry shortly.`,
      };
    }

    const jobId = `job_${now.toString(36)}_${crypto.randomBytes(4).toString('hex')}`;
    const record: ClinicalJobRecord<T> = {
      jobId,
      modality,
      status: 'queued',
      progress: 10,
      createdAt: new Date(now).toISOString(),
      startedAt: null,
      completedAt: null,
      latencyMs: null,
      triagePriority: null,
      triageReason: null,
      resultUrl: `/api/v1/jobs/${encodeURIComponent(jobId)}`,
    };

    this.jobs.set(jobId, {
      record,
      createdAtMs: now,
      expiresAtMs: now + this.ttlMs,
      execute,
    });
    this.pendingIds.push(jobId);

    const position = this.pendingIds.length;
    this.pump();

    return { accepted: true, job: { ...record }, position };
  }

  private pump(): void {
    while (this.activeWorkers < this.maxConcurrency && this.pendingIds.length > 0) {
      const nextId = this.pendingIds.shift();
      if (!nextId) continue;
      const entry = this.jobs.get(nextId);
      if (!entry) continue;

      this.activeWorkers += 1;
      const startedMs = Date.now();
      entry.record.status = 'processing';
      entry.record.progress = 45;
      entry.record.startedAt = new Date(startedMs).toISOString();

      void entry
        .execute()
        .then((payload) => {
          const completedMs = Date.now();
          const triage = evaluateTriagePriority(
            payload.fractureRiskPercent ?? 0,
            payload.entropy ?? 0,
            Boolean(payload.oodFlag),
          );
          entry.record.status = 'done';
          entry.record.progress = 100;
          entry.record.completedAt = new Date(completedMs).toISOString();
          entry.record.latencyMs = completedMs - startedMs;
          entry.record.triagePriority = triage.tier;
          entry.record.triageReason = triage.reason;
          entry.record.result = payload.result;
          entry.expiresAtMs = completedMs + this.ttlMs;
          this.completedCount += 1;
        })
        .catch((err: unknown) => {
          const completedMs = Date.now();
          const errObj = err as { error_code?: string; message?: string; statusCode?: number };
          entry.record.status = 'failed';
          entry.record.progress = 100;
          entry.record.completedAt = new Date(completedMs).toISOString();
          entry.record.latencyMs = completedMs - startedMs;
          entry.record.error = {
            error_code: errObj?.error_code || 'JOB_EXECUTION_FAILED',
            message: errObj?.message || 'Clinical analysis job failed.',
            ...(typeof errObj?.statusCode === 'number' ? { statusCode: errObj.statusCode } : {}),
          };
          entry.expiresAtMs = completedMs + this.ttlMs;
          this.failedCount += 1;
        })
        .finally(() => {
          this.activeWorkers = Math.max(0, this.activeWorkers - 1);
          this.pump();
        });
    }
  }

  getJob(jobId: string): ClinicalJobRecord<T> | null {
    const entry = this.jobs.get(jobId);
    if (!entry) return null;
    if (Date.now() > entry.expiresAtMs && entry.record.status !== 'processing') {
      this.jobs.delete(jobId);
      return null;
    }
    return { ...entry.record };
  }

  listJobs(limit = 50): ClinicalJobRecord<T>[] {
    this.pruneExpired();
    return Array.from(this.jobs.values())
      .sort((a, b) => b.createdAtMs - a.createdAtMs)
      .slice(0, limit)
      .map((entry) => ({ ...entry.record }));
  }

  pruneExpired(now = Date.now()): number {
    let removed = 0;
    for (const [id, entry] of this.jobs.entries()) {
      if (now > entry.expiresAtMs && entry.record.status !== 'processing' && entry.record.status !== 'queued') {
        this.jobs.delete(id);
        removed += 1;
      }
    }
    return removed;
  }

  getStats() {
    this.pruneExpired();
    return {
      maxConcurrency: this.maxConcurrency,
      maxQueueSize: this.maxQueueSize,
      activeWorkers: this.activeWorkers,
      queuedDepth: this.pendingIds.length,
      retainedJobs: this.jobs.size,
      completedCount: this.completedCount,
      failedCount: this.failedCount,
    };
  }
}

/**
 * Neutralizes CSV formula injection (CWE-1236) and safely quotes fields per RFC 4180.
 */
export function escapeCsvCell(raw: unknown): string {
  const str = raw === null || raw === undefined ? '' : String(raw);
  const neutralized = /^[=+\-@\t\r]/.test(str) ? `'${str}` : str;
  return `"${neutralized.replace(/"/g, '""')}"`;
}

export interface FeedbackExportRecord {
  feedbackId?: string;
  requestId?: string;
  decision?: string;
  predictedClass?: string;
  reasonCode?: string | null;
  reasonText?: string;
  timestamp?: string;
}

export function formatFeedbackCsv(records: FeedbackExportRecord[]): string {
  const header = ['feedbackId', 'requestId', 'predictedClass', 'decision', 'reasonCode', 'reasonText', 'timestamp']
    .map(escapeCsvCell)
    .join(',');
  const rows = records.map((r) =>
    [
      r.feedbackId ?? '',
      r.requestId ?? '',
      r.predictedClass ?? '',
      r.decision ?? '',
      r.reasonCode ?? '',
      r.reasonText ?? '',
      r.timestamp ?? '',
    ]
      .map(escapeCsvCell)
      .join(','),
  );
  return [header, ...rows].join('\n') + '\n';
}
