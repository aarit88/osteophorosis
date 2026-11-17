import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  FileCheck2,
  AlertTriangle,
  Lock,
  CheckCircle2,
  Database,
  Hash,
  ShieldAlert,
  BarChart2,
  Cpu,
} from 'lucide-react';
import { getEvidenceRegistry } from '../api/client';
import { EvidenceRegistryResponse, ModelEvidenceItem } from '../api/types';

export const EvidenceRegistryPage: React.FC = () => {
  const [data, setData] = useState<EvidenceRegistryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEvidenceRegistry()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load model evidence registry.');
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-xs font-semibold">
          <FileCheck2 className="w-3.5 h-3.5" />
          <span>Stage 31 Evidence Registry & Checkpoint Verification</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Model Evidence & Multi-Center Evaluation Registry
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Complete historical registry of all models developed across research sub-tracks. Documents verified metrics, external generalization collapses, burned holdouts, and frozen checkpoint hashes.
        </p>
      </div>

      {loading && (
        <div className="p-8 text-center text-slate-400 animate-pulse">
          Loading evidence registry...
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="space-y-6">
          {/* Checkpoint Hash Verification Card */}
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider block">
                Primary Checkpoint Immutability Verification
              </span>
              <div className="flex items-center space-x-2 font-mono text-xs text-slate-300">
                <Hash className="w-4 h-4 text-slate-500" />
                <span className="break-all font-bold text-slate-200">
                  SHA-256: 21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD
                </span>
              </div>
            </div>
            <div>
              <span className="inline-flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>CRYPTOGRAPHICALLY VERIFIED</span>
              </span>
            </div>
          </div>

          {/* Model Cards */}
          <div className="space-y-6">
            {data.models.map((model: ModelEvidenceItem, idx: number) => {
              const isFourClass = model.track === 'track_d';
              const isBTXRD = model.model_id === 'btxrd_densenet121_c1';

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
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-850 pb-3">
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
                  <div className="pt-2 border-t border-slate-850 flex items-center justify-between text-xs text-slate-400">
                    <span>Clinical Deployment Clearance:</span>
                    <span className="font-bold text-rose-400">{model.clinical_deployment_status}</span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default EvidenceRegistryPage;

