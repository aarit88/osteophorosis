import { AnalyzePayload, AnalyzeResponse } from '../api/types';
import { PredictionEngine, EngineMode } from './PredictionEngine';
import { analyzeImage, checkHealth } from '../api/client';

export class GatewayEngine implements PredictionEngine {
  readonly name = 'Server-gated research API';
  readonly mode: EngineMode;

  constructor(isFallback = false) {
    this.mode = isFallback ? 'unavailable' : 'gateway';
  }

  async analyze(payload: AnalyzePayload): Promise<AnalyzeResponse> {
    return analyzeImage(payload);
  }

  async checkAvailability(): Promise<boolean> {
    try {
      const res = await checkHealth();
      return res.status === 'ok';
    } catch {
      return false;
    }
  }
}

/** @deprecated Retained as an import alias for older integrations. This class never simulates inference. */
export { GatewayEngine as MockEngine };
