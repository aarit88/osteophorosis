import React, { useEffect, useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  FileCheck2,
  AlertTriangle,
  CheckCircle2,
  Hash,
  BarChart2,
  ZoomIn,
  X,
} from 'lucide-react';
import { getEvidenceRegistry, getTrainedModelsOverview } from '../api/client';
import {
  EvidenceRegistryResponse,
  ModelEvidenceItem,
  TrainedModelsOverviewResponse,
  TrainedModelPlot,
} from '../api/types';

export const EvidenceRegistryPage: React.FC = () => {
  const [data, setData] = useState<EvidenceRegistryResponse | null>(null);
  const [trainedData, setTrainedData] = useState<TrainedModelsOverviewResponse | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>('osteo-trackA-fracatlas-v1');
  const [filterMode, setFilterMode] = useState<'all' | 'correct' | 'error'>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [zoomedPlot, setZoomedPlot] = useState<TrainedModelPlot | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getEvidenceRegistry(), getTrainedModelsOverview()])
      .then(([registryRes, trainedRes]) => {
        setData(registryRes);
        setTrainedData(trainedRes);
        if (trainedRes.models.length > 0) {
          setSelectedModelId(trainedRes.models[0].model_id);
        }
        setLoading(false);
      })
      .catch(() => {
        setError('Evidence summaries could not be loaded. Check the API connection and retry.');
        setLoading(false);
      });
  }, []);

  const activeTrainedModel = useMemo(() => {
    if (!trainedData) return null;
    return (
      trainedData.models.find((m) => m.model_id === selectedModelId) ||
      trainedData.models[0] ||
      null
    );
  }, [trainedData, selectedModelId]);

  // Individual rows remain withheld even if a stale or misconfigured API returns them.
  const filteredSamples = useMemo<TrainedModelsOverviewResponse['models'][number]['sample_predictions']>(() => [], []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-xs font-semibold">
          <FileCheck2 className="w-3.5 h-3.5" />
          <span>Historical Research Evidence</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Model Evidence & Multi-Center Evaluation Registry
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Historical aggregate evaluation records are shown separately from current checkpoint verification. Individual prediction rows and protected holdout samples are not exposed by this page.
        </p>
      </div>

      {loading && (
        <div className="p-8 text-center text-slate-400 animate-pulse">
          Loading evidence registry and trained model artifacts...
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="space-y-8">
          {/* Checkpoint Hash Verification Card */}
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider block">
                Track C Expected Checkpoint Digest
              </span>
              <div className="flex items-center space-x-2 font-mono text-xs text-slate-300">
                <Hash className="w-4 h-4 text-slate-500" />
                <span className="break-all font-bold text-slate-200">
                  Expected SHA-256: 21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD
                </span>
              </div>
            </div>
            <div>
              <span className={`inline-flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-bold border ${data.frozen_checkpoint_verified ? 'bg-emerald-950 text-emerald-300 border-emerald-800' : 'bg-amber-950 text-amber-200 border-amber-800'}`}>
                {data.frozen_checkpoint_verified ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                <span>{data.frozen_checkpoint_verified ? 'CHECKPOINT VERIFIED IN THIS RUNTIME' : 'CHECKPOINT NOT VERIFIED IN THIS RUNTIME'}</span>
              </span>
            </div>
          </div>

              {/* Historical aggregate evidence; individual rows stay withheld. */}
          {trainedData && activeTrainedModel && (
            <div className="p-6 rounded-2xl bg-slate-900/75 border border-cyan-900/60 space-y-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
                <div>
                  <div className="flex items-center space-x-2 text-xs font-bold text-cyan-400 uppercase tracking-wider">
                    <BarChart2 className="w-4 h-4" />
                    <span>Historical Aggregate Evaluation Artifacts</span>
                  </div>
                  <h2 className="text-xl font-extrabold text-white mt-1">
                    {activeTrainedModel.title}
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {activeTrainedModel.architecture} •{' '}
                    <span className="text-slate-200 font-semibold">
                      {activeTrainedModel.total_holdout_evaluated.toLocaleString()} holdout samples evaluated
                    </span>
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  {trainedData.models.map((m) => {
                    const active = m.model_id === activeTrainedModel.model_id;
                    return (
                      <button
                        key={m.model_id}
                        type="button"
                        onClick={() => setSelectedModelId(m.model_id)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-bold border transition ${
                          active
                            ? 'bg-cyan-500 text-slate-950 border-cyan-400'
                            : 'bg-slate-950 text-slate-300 border-slate-800 hover:border-slate-700'
                        }`}
                      >
                        {m.track.replace('_', ' ').toUpperCase()}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Evaluation Plots Gallery */}
              {activeTrainedModel.plots.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Evaluation Charts (Click to Enlarge)
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {activeTrainedModel.plots.map((plot) => (
                      <button
                        key={plot.url}
                        type="button"
                        onClick={() => setZoomedPlot(plot)}
                        className="group rounded-xl bg-slate-950 border border-slate-800 hover:border-cyan-500/60 overflow-hidden text-left transition flex flex-col"
                      >
                        <div className="relative aspect-[4/3] w-full bg-white/95 overflow-hidden">
                          <img
                            src={plot.url}
                            alt={plot.title}
                            loading="lazy"
                            className="w-full h-full object-contain p-1 group-hover:scale-105 transition-transform duration-200"
                          />
                          <div className="absolute top-2 right-2 p-1 rounded-lg bg-slate-900/80 text-white opacity-0 group-hover:opacity-100 transition">
                            <ZoomIn className="w-3.5 h-3.5" />
                          </div>
                        </div>
                        <div className="p-2.5 space-y-0.5">
                          <div className="text-xs font-bold text-slate-200 truncate">{plot.title}</div>
                          <div className="text-[11px] text-slate-400 line-clamp-2">{plot.caption}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Do not expose individual prediction or holdout rows through the public UI. */}
              <div className="space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Individual prediction rows are always withheld from this public summary
                  </h3>
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="relative">
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Individual records are not exposed"
                        disabled
                        className="pl-8 pr-3 py-1 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                    {(['all', 'correct', 'error'] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => setFilterMode(mode)}
                        disabled
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold capitalize border transition ${
                          filterMode === mode
                            ? 'bg-slate-800 text-cyan-300 border-cyan-600'
                            : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-900/90 text-slate-400 uppercase text-[10px]">
                      <tr>
                        <th className="px-3 py-2.5">Sample ID</th>
                        <th className="px-3 py-2.5">Cohort & Details</th>
                        <th className="px-3 py-2.5">Ground Truth</th>
                        <th className="px-3 py-2.5">Model Prediction</th>
                        <th className="px-3 py-2.5 text-right">Score</th>
                        <th className="px-3 py-2.5 text-center">Match</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70">
                      {filteredSamples.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="px-4 py-6 text-center text-slate-500">
                            Individual prediction rows are withheld by the public UI. Protected holdout samples remain undisclosed.
                          </td>
                        </tr>
                      ) : (
                        filteredSamples.map((row) => (
                          <tr key={`${row.cohort}-${row.sample_id}`} className="hover:bg-slate-900/40">
                            <td className="px-3 py-2 font-mono font-semibold text-slate-200">
                              {row.sample_id}
                            </td>
                            <td className="px-3 py-2 text-slate-400">
                              <div className="text-slate-300 font-medium">{row.cohort}</div>
                              <div className="text-[10px] font-mono text-slate-500">{row.metadata}</div>
                            </td>
                            <td className="px-3 py-2 text-slate-300 font-mono text-[11px]">
                              {row.ground_truth}
                            </td>
                            <td className="px-3 py-2 text-slate-200 font-mono text-[11px]">
                              {row.predicted_label}
                            </td>
                            <td className="px-3 py-2 text-right font-mono font-bold text-cyan-300">
                              {(row.probability * 100).toFixed(1)}%
                            </td>
                            <td className="px-3 py-2 text-center">
                              <span
                                className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                  row.is_correct
                                    ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                                    : 'bg-rose-950 text-rose-300 border border-rose-800'
                                }`}
                              >
                                {row.is_correct ? 'PASS' : 'MISS'}
                              </span>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                <p className="text-[11px] text-slate-500">{trainedData.disclaimer}</p>
              </div>
            </div>
          )}

          {/* Model Cards */}
          <div className="space-y-6">
            {data.models.map((model: ModelEvidenceItem, idx: number) => {
              const isFourClass =
                model.track.toLowerCase().includes('track d') ||
                model.model_id.includes('fourclass');
              const isBTXRD =
                model.track.toLowerCase().includes('track c') ||
                model.model_id.includes('btxrd');

              return (
                <motion.div
                  key={model.model_id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`p-6 rounded-2xl border space-y-4 ${
                    isFourClass
                      ? 'bg-rose-950/15 border-rose-900/50'
                      : isBTXRD
                      ? 'bg-amber-950/15 border-amber-900/50'
                      : 'bg-slate-900/60 border-slate-800'
                  }`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                    <div>
                      <div className="flex items-center space-x-2 text-xs font-bold text-cyan-400 mb-0.5">
                        <span className="uppercase">{model.track.replace('_', ' ')}</span>
                        <span>•</span>
                        <span>{model.modality}</span>
                      </div>
                      <h2 className="text-xl font-extrabold text-white">{model.model_id}</h2>
                      <p className="text-xs text-slate-400">
                        Architecture: <span className="font-mono text-slate-300">{model.architecture}</span> | Dataset: <span className="text-slate-300">{model.dataset_name}</span>
                      </p>
                    </div>
                    <div>
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-slate-950 border border-slate-800 text-slate-300">
                        {model.training_status}
                      </span>
                    </div>
                  </div>

                  {/* Validation Metrics Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    {Object.entries(model.validation_evidence).map(([k, v]) => (
                      <div key={k} className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                        <span className="text-slate-400 block text-[11px] capitalize">
                          {k.replace(/_/g, ' ')}
                        </span>
                        <span className="text-sm font-bold font-mono text-white mt-0.5 block">
                          {typeof v === 'number'
                            ? v < 1
                              ? `${(v * 100).toFixed(1)}%`
                              : v.toLocaleString()
                            : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Multi-Center Findings */}
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2">
                    <div className="flex items-center space-x-2 font-bold text-amber-300">
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span>Generalization & Holdout Audit:</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed">{model.generalization_findings}</p>
                    <div className="text-[11px] text-slate-400">
                      <strong>External Holdout Status:</strong> {model.external_holdout_status}
                    </div>
                  </div>

                  {/* Disclosures & Limitations */}
                  <div className="space-y-1 text-xs">
                    <span className="font-semibold text-slate-300 block">Reported Gaps & Governance Disclosures:</span>
                    <ul className="space-y-1 text-slate-400 list-disc list-inside">
                      {model.limitations.map((lim, lIdx) => (
                        <li key={lIdx}>{lim}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Clinical Clearance Status */}
                  <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                    <span>Historical artifact deployment status:</span>
                    <span className="font-bold text-rose-400">{model.clinical_deployment_status}</span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      )}

      {/* Plot Zoom Modal */}
      {zoomedPlot && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
          onClick={() => setZoomedPlot(null)}
        >
          <div
            className="max-w-4xl w-full rounded-2xl bg-slate-900 border border-slate-700 p-4 space-y-3 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div>
                <h4 className="text-sm font-bold text-white">{zoomedPlot.title}</h4>
                <p className="text-xs text-slate-400">{zoomedPlot.caption}</p>
              </div>
              <button
                type="button"
                onClick={() => setZoomedPlot(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
                aria-label="Close chart preview"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="bg-white rounded-xl p-2 max-h-[75vh] overflow-auto flex items-center justify-center">
              <img src={zoomedPlot.url} alt={zoomedPlot.title} className="max-h-[70vh] w-auto object-contain" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EvidenceRegistryPage;
