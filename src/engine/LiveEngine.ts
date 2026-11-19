import { AnalyzePayload, AnalyzeResponse } from '../api/types';
import { PredictionEngine, EngineMode } from './PredictionEngine';
import { analyzeImage, checkHealth } from '../api/client';

export class LiveEngine implements PredictionEngine {
  readonly name = 'Live PyTorch/TensorRT Production Engine';
  readonly mode: EngineMode = 'live';
  private liveBaseUrl: string;

  constructor(liveUrl?: string) {
    this.liveBaseUrl = (
      liveUrl ||
      ((import.meta as unknown as { env?: { VITE_API_BASE_URL_LIVE?: string } }).env?.VITE_API_BASE_URL_LIVE) ||
      ''
    ).trim();
  }

  async checkAvailability(): Promise<boolean> {
    if (!this.liveBaseUrl) return false;
    try {
      const res = await checkHealth(this.liveBaseUrl);
      return res.status === 'ok';
    } catch {
      return false;
    }
  }

  async analyze(payload: AnalyzePayload): Promise<AnalyzeResponse> {
    if (!this.liveBaseUrl) {
      throw new Error(
        'Live backend is not configured. Teammates must specify VITE_API_BASE_URL_LIVE in environment.'
      );
    }
    return analyzeImage(payload, this.liveBaseUrl);
  }
}
