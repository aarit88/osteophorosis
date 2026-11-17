import { PredictionEngine, EngineMode } from './PredictionEngine';
import { MockEngine } from './MockEngine';
import { LiveEngine } from './LiveEngine';

export * from './PredictionEngine';
export * from './MockEngine';
export * from './LiveEngine';

const CONFIGURED_ENGINE_SETTING = (
  ((import.meta as unknown as { env?: { VITE_ENGINE?: string } }).env?.VITE_ENGINE) || 'auto'
).toLowerCase();

let cachedEngine: PredictionEngine | null = null;

export async function createPredictionEngine(): Promise<PredictionEngine> {
  if (CONFIGURED_ENGINE_SETTING === 'mock') {
    return new MockEngine(false);
  }

  if (CONFIGURED_ENGINE_SETTING === 'live') {
    const live = new LiveEngine();
    return live;
  }

  // 'auto' mode: Probe LiveEngine first, fallback to MockEngine
  const live = new LiveEngine();
  const isLiveAvailable = await live.checkAvailability();

  if (isLiveAvailable) {
    return live;
  }

  // Graceful fallback to simulation
  return new MockEngine(true);
}

export function getPredictionEngineSync(): PredictionEngine {
  if (!cachedEngine) {
    if (CONFIGURED_ENGINE_SETTING === 'live') {
      cachedEngine = new LiveEngine();
    } else if (CONFIGURED_ENGINE_SETTING === 'mock') {
      cachedEngine = new MockEngine(false);
    } else {
      // default auto sync initialization
      cachedEngine = new MockEngine(true);
    }
  }
  return cachedEngine;
}
