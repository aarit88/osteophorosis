import { PredictionEngine, EngineMode } from './PredictionEngine';
import { GatewayEngine } from './MockEngine';
import { LiveEngine } from './LiveEngine';

export * from './PredictionEngine';
export * from './MockEngine';
export * from './LiveEngine';

const CONFIGURED_ENGINE_SETTING = (
  ((import.meta as unknown as { env?: { VITE_ENGINE?: string } }).env?.VITE_ENGINE) || 'auto'
).toLowerCase();

const CONFIGURED_LIVE_URL = (
  ((import.meta as unknown as { env?: { VITE_API_BASE_URL_LIVE?: string } }).env?.VITE_API_BASE_URL_LIVE) || ''
).trim();

let cachedEngine: PredictionEngine | null = null;

export async function createPredictionEngine(): Promise<PredictionEngine> {
  if (CONFIGURED_ENGINE_SETTING === 'mock') {
    return new GatewayEngine(false);
  }

  if (CONFIGURED_ENGINE_SETTING === 'live') {
    const live = new LiveEngine();
    return live;
  }

  // 'auto' mode: Probe LiveEngine only when VITE_API_BASE_URL_LIVE is configured
  if (CONFIGURED_LIVE_URL) {
    const live = new LiveEngine();
    const isLiveAvailable = await live.checkAvailability();
    if (isLiveAvailable) {
      return live;
    }
    return new GatewayEngine(true);
  }

  // Unified deployment checks the API gateway; inference remains server-gated.
  const gateway = new GatewayEngine(false);
  const isGatewayHealthy = await gateway.checkAvailability();
  return isGatewayHealthy ? gateway : new GatewayEngine(true);
}

export function getPredictionEngineSync(): PredictionEngine {
  if (!cachedEngine) {
    if (CONFIGURED_ENGINE_SETTING === 'live') {
      cachedEngine = new LiveEngine();
    } else if (CONFIGURED_ENGINE_SETTING === 'mock' || !CONFIGURED_LIVE_URL) {
      cachedEngine = new GatewayEngine(false);
    } else {
      cachedEngine = new GatewayEngine(true);
    }
  }
  return cachedEngine;
}
