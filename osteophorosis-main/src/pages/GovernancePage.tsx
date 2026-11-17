import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldAlert,
  Database,
  Lock,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Calendar,
  Building,
} from 'lucide-react';
import { getDatasetGovernance } from '../api/client';
import { DatasetGovernanceResponse, DatasetGovernanceItem } from '../api/types';

export const GovernancePage: React.FC = () => {
  const [data, setData] = useState<DatasetGovernanceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDatasetGovernance()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load dataset governance status.');
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-rose-950/70 border border-rose-800 text-rose-300 text-xs font-semibold">
          <Database className="w-3.5 h-3.5" />
          <span>Stage 24 / 25 Governance & Clinical Readiness</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Dataset Governance & Clinical Acquisition Gate
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Tracks the ethical, regulatory, and legal readiness of clinical data for the intended four-class differential diagnosis. In accordance with ICMR Guidelines, unapproved clinical data cannot be ingested.
        </p>
      </div>

      {loading && (
        <div className="p-8 text-center text-slate-400 animate-pulse">
          Loading governance status...
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="space-y-6">
          {/* Acquisition Gate Status Banner */}
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs text-slate-400 block mb-1">Overall Institutional Acquisition Gate</span>
              <span className="text-lg font-bold text-amber-400 font-mono">
                {data.overall_acquisition_gate}
              </span>
            </div>
            <div>
              <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-950 text-rose-300 border border-rose-800">
                <Lock className="w-3.5 h-3.5" />
                <span>INSPECTION & DUA REQUIRED</span>
              </span>
            </div>
          </div>

          {/* 4 Cohorts Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {data.cohorts.map((cohort: DatasetGovernanceItem, idx: number) => {
              return (
                <motion.div
                  key={cohort.cohort_name}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4"
                >
                  <div className="border-b border-slate-850 pb-3">
                    <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider block mb-1">
                      Target Class {idx + 1}: {cohort.target_class}
                    </span>
                    <h3 className="text-lg font-extrabold text-white">{cohort.cohort_name}</h3>
                    <p className="text-xs text-slate-400 mt-0.5">Modality: {cohort.intended_modality}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Source Status</span>
                      <span className="font-bold text-rose-300 block truncate">{cohort.source_status}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Target Cohort Size</span>
                      <span className="font-bold text-white block">{cohort.patient_count_estimate}</span>
                    </div>
                  </div>

                  <div className="space-y-1.5 text-xs">
                    <span className="font-semibold text-slate-300 block">Identified Acquisition Blockers:</span>
                    <ul className="space-y-1 text-slate-400 list-disc list-inside">
                      {cohort.acquisition_blockers.map((blk, bIdx) => (
                        <li key={bIdx}>{blk}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="pt-3 border-t border-slate-850 flex items-center justify-between text-xs text-slate-400">
                    <span>ICMR/IEC Protocol Required:</span>
                    <span className={`font-bold ${cohort.icmr_iec_required ? 'text-amber-400' : 'text-slate-400'}`}>
                      {cohort.icmr_iec_required ? 'Mandatory Institutional Approval' : 'Exempt'}
                    </span>
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* Institutional Compliance Card */}
          <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
              <Building className="w-4 h-4 text-cyan-400" />
              <span>Institutional Research Preparation (Stage 24 / 25 Documents)</span>
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Formal protocols prepared under Stage 24 and Stage 25 include institutional ethics applications (IEC Form A), bilateral Data Use Agreements (DUA), Clinical Trial Registry of India (CTRI) prospective documentation, and anonymization pipelines compliant with Indian Personal Data Protection (DPDP) Act 2023.
            </p>
            <div className="text-[11px] text-slate-500 italic">
              Notice: Applications are documented in <code>ml_artifacts/stage24_collaboration_prep/</code> and awaiting institutional faculty submission.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GovernancePage;

