import { AnalyzePayload, AnalyzeResponse } from '../api/types';

export type EngineMode = 'mock' | 'live' | 'fallback';

export interface PredictionEngine {
  readonly name: string;
  readonly mode: EngineMode;
  analyze(payload: AnalyzePayload): Promise<AnalyzeResponse>;
  checkAvailability(): Promise<boolean>;
}
