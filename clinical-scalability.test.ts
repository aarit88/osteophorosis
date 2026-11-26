import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  ClinicalAnalysisQueue,
  LatencyTracker,
  SlidingWindowRateLimiter,
  escapeCsvCell,
  evaluateTriagePriority,
  formatFeedbackCsv,
  rankWorklistItems,
} from './clinical-scalability.ts';

describe('clinical-scalability pure logic & worker queue', () => {
  it('enforces sliding-window rate limits and recovers after window expiry', () => {
    const limiter = new SlidingWindowRateLimiter({ maxRequests: 2, windowMs: 10_000 });
    assert.equal(limiter.check('10.0.0.1', 1000).allowed, true);
    assert.equal(limiter.check('10.0.0.1', 2000).allowed, true);
    const blocked = limiter.check('10.0.0.1', 3000);
    assert.equal(blocked.allowed, false);
    assert.equal(blocked.remaining, 0);
    assert.ok(blocked.resetSeconds >= 1);

    // Different client stays unblocked
    assert.equal(limiter.check('10.0.0.2', 3000).allowed, true);

    // Original client allowed after 10s window elapses
    assert.equal(limiter.check('10.0.0.1', 12_001).allowed, true);
  });

  it('tracks bounded latency samples and computes p50/p95/p99 percentiles accurately', () => {
    const tracker = new LatencyTracker(20);
    for (let i = 1; i <= 20; i++) {
      tracker.record(i * 10, i === 20);
    }
    const summary = tracker.getSummary();
    assert.equal(summary.totalRequests, 20);
    assert.equal(summary.totalErrors, 1);
    assert.equal(summary.p50Ms, 100);
    assert.equal(summary.p95Ms, 190);
    assert.equal(summary.p99Ms, 200);
  });

  it('prioritizes STAT_URGENT cases (high Mirels risk or high entropy/OOD) ahead of routine studies', () => {
    const routine = evaluateTriagePriority(15, 0.25, false);
    const moderate = evaluateTriagePriority(48, 0.35, false);
    const statOod = evaluateTriagePriority(20, 0.4, true);
    const statFracture = evaluateTriagePriority(82, 0.3, false);

    assert.equal(routine.tier, 'ROUTINE');
    assert.equal(moderate.tier, 'PRIORITY_REVIEW');
    assert.equal(statOod.tier, 'STAT_URGENT');
    assert.equal(statFracture.tier, 'STAT_URGENT');

    const ranked = rankWorklistItems([
      { jobId: 'routine', fractureRiskPercent: 15, entropy: 0.25, oodFlag: false, createdAtMs: 300 },
      { jobId: 'moderate', fractureRiskPercent: 48, entropy: 0.35, oodFlag: false, createdAtMs: 200 },
      { jobId: 'urgent', fractureRiskPercent: 82, entropy: 0.3, oodFlag: false, createdAtMs: 100 },
    ]);
    assert.deepEqual(
      ranked.map((r) => r.jobId),
      ['urgent', 'moderate', 'routine'],
    );
  });

  it('executes bounded concurrent jobs and records completion telemetry and failure isolation', async () => {
    const queue = new ClinicalAnalysisQueue<{ ok: boolean }>({ maxConcurrency: 2, maxQueueSize: 4 });
    let activeObserved = 0;
    let peakActive = 0;

    const makeTask = (delayMs: number, shouldFail = false) => async () => {
      activeObserved += 1;
      peakActive = Math.max(peakActive, activeObserved);
      await new Promise((r) => setTimeout(r, delayMs));
      activeObserved -= 1;
      if (shouldFail) {
        const err = new Error('Simulated clinical gate block') as Error & { error_code: string; statusCode: number };
        err.error_code = 'FOUR_CLASS_MODEL_NOT_READY';
        err.statusCode = 503;
        throw err;
      }
      return { result: { ok: true }, fractureRiskPercent: 78, entropy: 0.31, oodFlag: false };
    };

    const j1 = queue.enqueue('xray', makeTask(25));
    const j2 = queue.enqueue('ct', makeTask(25));
    const j3 = queue.enqueue('bone_scan', makeTask(15, true));
    assert.equal(j1.accepted, true);
    assert.equal(j2.accepted, true);
    assert.equal(j3.accepted, true);

    await new Promise((r) => setTimeout(r, 90));

    assert.ok(peakActive <= 2, `Expected peak concurrency <= 2, got ${peakActive}`);
    if (j1.accepted && j3.accepted) {
      const s1 = queue.getJob(j1.job.jobId);
      const s3 = queue.getJob(j3.job.jobId);
      assert.equal(s1?.status, 'done');
      assert.equal(s1?.triagePriority, 'STAT_URGENT');
      assert.equal(s3?.status, 'failed');
      assert.equal(s3?.error?.error_code, 'FOUR_CLASS_MODEL_NOT_READY');
    }
    const stats = queue.getStats();
    assert.equal(stats.completedCount, 2);
    assert.equal(stats.failedCount, 1);
  });

  it('escapes formula injection prefixes in clinical audit CSV exports', () => {
    assert.equal(escapeCsvCell('=CMD("calc")'), '"\'=CMD(""calc"")"');
    const csv = formatFeedbackCsv([
      {
        feedbackId: 'fb_1',
        requestId: 'req_1',
        predictedClass: 'metastasis',
        decision: 'agree',
        reasonCode: null,
        reasonText: '+injected',
        timestamp: '2026-11-26T00:00:00Z',
      },
    ]);
    assert.match(csv, /"feedbackId","requestId"/);
    assert.match(csv, /"'\+injected"/);
  });
});
