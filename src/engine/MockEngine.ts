import { AnalyzePayload, AnalyzeResponse } from '../api/types';
import { PredictionEngine, EngineMode } from './PredictionEngine';
import { analyzeImage, checkHealth } from '../api/client';

export class MockEngine implements PredictionEngine {
  readonly name = 'Simulation Mock Engine (Deterministic ViT+CNN)';
  readonly mode: EngineMode;

  constructor(isFallback = false) {
    this.mode = isFallback ? 'fallback' : 'mock';
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
