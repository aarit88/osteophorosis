import { AnalyzePayload, AnalyzeResponse } from '../api/types';
import { PredictionEngine, EngineMode } from './PredictionEngine';
import { analyzeImage, checkHealth } from '../api/client';

export class GatewayEngine implements PredictionEngine {
  readonly name: string;
  mode: EngineMode;

  constructor(modeOrFallback: EngineMode | boolean = false) {
    if (typeof modeOrFallback === 'boolean') {
      this.mode = modeOrFallback ? 'unavailable' : 'gateway';
    } else {
      this.mode = modeOrFallback;
    }
    this.name =
      this.mode === 'simulation'
        ? 'Research prototype simulator'
        : 'Server-gated research API';
  }

  async analyze(payload: AnalyzePayload): Promise<AnalyzeResponse> {
    return analyzeImage(payload);
  }

  async checkAvailability(): Promise<boolean> {
    try {
      const res = await checkHealth();
      if (res.status === 'ok') {
        this.mode = res.mode === 'simulation' ? 'simulation' : 'gateway';
        return true;
      }
      this.mode = 'unavailable';
      return false;
    } catch {
      this.mode = 'unavailable';
      return false;
    }
  }
}

/** @deprecated Retained as an import alias for older integrations. */
export { GatewayEngine as MockEngine };
