import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { AnalyzeResponse, AnalyzePayload, OsteoGuardError, NetworkError } from '../api/types';
import { EngineMode, createPredictionEngine, PredictionEngine } from '../engine';
import { checkHealth } from '../api/client';

interface AnalysisContextType {
  lastResult: AnalyzeResponse | null;
  setLastResult: (result: AnalyzeResponse | null) => void;
  isAnalyzing: boolean;
  error: string | null;
  setError: (err: string | null) => void;
  clearResult: () => void;
  recentAnalyses: AnalyzeResponse[];
  addRecentAnalysis: (res: AnalyzeResponse) => void;
  clearRecentAnalyses: () => void;
  engineMode: EngineMode;
  setEngineMode: (mode: EngineMode) => void;
  activeQueueCount: number;
  queuedJobsCount: number;
  isServerOffline: boolean;
  retryConnectivity: () => Promise<void>;
  executeAnalysisWithQueue: (payload: AnalyzePayload) => Promise<AnalyzeResponse>;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

const STORAGE_LAST_KEY = 'osteoguard_last_result';
const STORAGE_HISTORY_KEY = 'osteoguard_history_list';

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [lastResult, setLastResultState] = useState<AnalyzeResponse | null>(() => {
    if (typeof window !== 'undefined') {
      try {
        const cached = sessionStorage.getItem(STORAGE_LAST_KEY);
        if (cached) return JSON.parse(cached) as AnalyzeResponse;
      } catch (err) {
        console.warn('Failed to parse cached analysis result:', err);
      }
    }
    return null;
  });

  const [recentAnalyses, setRecentAnalyses] = useState<AnalyzeResponse[]>(() => {
    if (typeof window !== 'undefined') {
      try {
        const cached = sessionStorage.getItem(STORAGE_HISTORY_KEY);
        if (cached) return JSON.parse(cached) as AnalyzeResponse[];
      } catch {}
    }
    return [];
  });

  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [engineMode, setEngineMode] = useState<EngineMode>('mock');
  const [activeQueueCount, setActiveQueueCount] = useState<number>(0);
  const [queuedJobsCount, setQueuedJobsCount] = useState<number>(0);
  const [isServerOffline, setIsServerOffline] = useState<boolean>(false);

  // Initialize engine
  useEffect(() => {
    createPredictionEngine()
      .then((eng) => setEngineMode(eng.mode))
      .catch(() => setEngineMode('fallback'));
  }, []);

  const retryConnectivity = useCallback(async () => {
    try {
      await checkHealth();
      setIsServerOffline(false);
      setError(null);
    } catch {
      setIsServerOffline(true);
    }
  }, []);

  const setLastResult = (result: AnalyzeResponse | null) => {
    setLastResultState(result);
    if (typeof window !== 'undefined') {
      if (result) {
        sessionStorage.setItem(STORAGE_LAST_KEY, JSON.stringify(result));
        addRecentAnalysis(result);
      } else {
        sessionStorage.removeItem(STORAGE_LAST_KEY);
      }
    }
  };

  const addRecentAnalysis = (res: AnalyzeResponse) => {
    setRecentAnalyses((prev) => {
      const filtered = prev.filter((item) => item.requestId !== res.requestId);
      const updated = [res, ...filtered].slice(0, 20); // Keep max N=20
      if (typeof window !== 'undefined') {
        sessionStorage.setItem(STORAGE_HISTORY_KEY, JSON.stringify(updated));
      }
      return updated;
    });
  };

  const clearRecentAnalyses = () => {
    setRecentAnalyses([]);
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem(STORAGE_HISTORY_KEY);
    }
  };

  const clearResult = () => {
    setLastResult(null);
    setError(null);
  };

  // Client-side concurrency queue (max 2 concurrent requests)
  const executeAnalysisWithQueue = async (payload: AnalyzePayload): Promise<AnalyzeResponse> => {
    const engine = await createPredictionEngine();
    setEngineMode(engine.mode);

    if (activeQueueCount >= 2) {
      setQueuedJobsCount((prev) => prev + 1);
      // Wait for a slot
      while (activeQueueCount >= 2) {
        await new Promise((res) => setTimeout(res, 300));
      }
      setQueuedJobsCount((prev) => Math.max(0, prev - 1));
    }

    setActiveQueueCount((prev) => prev + 1);
    setIsAnalyzing(true);
    setError(null);

    try {
      const res = await engine.analyze(payload);
      setLastResult(res);
      setIsServerOffline(false);
      return res;
    } catch (err: unknown) {
      if (err instanceof NetworkError) {
        setIsServerOffline(true);
      }
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setError(message);
      throw err;
    } finally {
      setActiveQueueCount((prev) => Math.max(0, prev - 1));
      setIsAnalyzing(false);
    }
  };

  return (
    <AnalysisContext.Provider
      value={{
        lastResult,
        setLastResult,
        isAnalyzing,
        error,
        setError,
        clearResult,
        recentAnalyses,
        addRecentAnalysis,
        clearRecentAnalyses,
        engineMode,
        setEngineMode,
        activeQueueCount,
        queuedJobsCount,
        isServerOffline,
        retryConnectivity,
        executeAnalysisWithQueue,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
}
