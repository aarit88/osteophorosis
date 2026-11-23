import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldAlert,
  Database,
  Lock,
  AlertTriangle,
  CheckCircle2,
  Building,
  Layers,
  Users,
  FileWarning,
  CopyCheck,
  ShieldCheck,
  Split,
  ClipboardCheck,
  FileText,
  Scale,
} from 'lucide-react';
import {
  getDatasetGovernance,
  getTrackDDatasetReadiness,
  getDatasetSources,
  getDatasetAcquisitionStatus,
  getDatasetGovernanceStatus,
  getCohortPlanning,
} from '../api/client';
import {
  DatasetGovernanceResponse,
  DatasetGovernanceItem,
  TrackDDatasetReadinessResponse,
  DatasetSourcesResponse,
  DatasetAcquisitionStatusResponse,
  DatasetGovernanceStatusResponse,
  CohortPlanningResponse,
} from '../api/types';
import DatasetOnboardingPanel from '../components/DatasetOnboardingPanel';

const CLASS_LABELS: Array<{ key: 'metastasis' | 'paget' | 'oi' | 'healthy'; label: string }> = [
  { key: 'metastasis', label: 'Secondary Bone Metastasis' },
  { key: 'paget', label: "Paget's Disease of Bone" },
  { key: 'oi', label: 'Osteogenesis Imperfecta (OI)' },
  { key: 'healthy', label: 'Healthy Skeletal Control' },
];

const MODALITY_LABELS: Array<{ key: 'xray' | 'bone_scan' | 'ct'; label: string }> = [
  { key: 'xray', label: 'Plain Radiography (X-Ray)' },
  { key: 'bone_scan', label: 'Bone Scintigraphy (99mTc-MDP)' },
  { key: 'ct', label: 'Computed Tomography (CT)' },
];

const GATE_LABELS: Array<{ key: keyof TrackDDatasetReadinessResponse['gates']; label: string }> = [
  { key: 'provenance_documented', label: 'Documented Institutional Provenance' },
  { key: 'permissions_verified', label: 'Verified Usage Permissions / DUA' },
  { key: 'deidentification_satisfied', label: 'De-Identification & Zero Direct PHI' },
  { key: 'diagnostic_labels_documented', label: 'Documented Diagnostic Label Evidence' },
  { key: 'patient_level_split_integrity', label: 'Patient-Level Zero-Leakage Splits' },
  { key: 'image_quality_verified', label: 'Image Format / Dimension QC Passed' },
  { key: 'class_and_modality_coverage', label: '4-Class Coverage Verified' },
];

export const GovernancePage: React.FC = () => {
  const [data, setData] = useState<DatasetGovernanceResponse | null>(null);
  const [readiness, setReadiness] = useState<TrackDDatasetReadinessResponse | null>(null);
  const [sourcesData, setSourcesData] = useState<DatasetSourcesResponse | null>(null);
  const [acquisitionStatus, setAcquisitionStatus] = useState<DatasetAcquisitionStatusResponse | null>(null);
  const [governanceStatus, setGovernanceStatus] = useState<DatasetGovernanceStatusResponse | null>(null);
  const [cohortPlanning, setCohortPlanning] = useState<CohortPlanningResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getDatasetGovernance(),
      getTrackDDatasetReadiness(),
      getDatasetSources().catch(() => null),
      getDatasetAcquisitionStatus().catch(() => null),
      getDatasetGovernanceStatus().catch(() => null),
      getCohortPlanning().catch(() => null),
    ])
      .then(([govRes, readinessRes, srcRes, acqRes, govStatusRes, planRes]) => {
        setData(govRes);
        setReadiness(readinessRes);
        setSourcesData(srcRes);
        setAcquisitionStatus(acqRes);
        setGovernanceStatus(govStatusRes);
        setCohortPlanning(planRes);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load dataset governance & intake readiness.');
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-rose-950/70 border border-rose-800 text-rose-300 text-xs font-semibold">
          <Database className="w-3.5 h-3.5" />
          <span>Stage 33–35 Track D Dataset Readiness, Ontology &amp; Institutional Acquisition Governance</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Dataset Governance, Intake &amp; Readiness Gate
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Monitors real on-disk four-class clinical cohorts, strict modality separation (X-Ray, Bone Scintigraphy, CT),
          quarantine quality checks, patient-level leakage-safe splits, and institutional clearance gates.
          Synthetic demonstration outputs are strictly excluded from real dataset counts.
        </p>
      </div>

      <DatasetOnboardingPanel />

      {loading && (
        <div className="p-8 text-center text-slate-400 animate-pulse">
          Loading dataset governance &amp; Track D intake status...
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Stage 33 Track D Real Dataset Readiness Dashboard */}
      {readiness && (
        <section className="space-y-6" aria-label="Track D Dataset Readiness Dashboard">
          {/* Status & Origin Distinction Banner */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <span className="text-xs text-slate-400 block mb-1">
                  Track D Clinical Dataset Readiness Status (Real Disk Data Only)
                </span>
                <div className="flex flex-wrap items-center gap-3">
                  <span
                    className={`text-xl font-black font-mono ${
                      readiness.readiness_status === 'READY_FOR_TRAINING'
                        ? 'text-emerald-400'
                        : readiness.readiness_status === 'BLOCKED'
                        ? 'text-amber-400'
                        : 'text-rose-400'
                    }`}
                  >
                    {readiness.readiness_status}
                  </span>
                  <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-950 border border-slate-700 text-slate-300">
                    {readiness.data_origin === 'REAL_CLINICAL_DATA_UNAVAILABLE'
                      ? '0 REAL SAMPLES — SYNTHETIC DEMO EXCLUDED'
                      : `${readiness.valid_samples} VALID REAL SAMPLES ON DISK`}
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-cyan-950/60 text-cyan-300 border border-cyan-800">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>
                    BTXRD Center 2 (N={readiness.protected_holdouts.btxrd_center2_samples}) &amp; Center 3 (N=
                    {readiness.protected_holdouts.btxrd_center3_samples}) Protected Holdout
                  </span>
                </span>
              </div>
            </div>

            {readiness.readiness_status === 'NOT_AVAILABLE' && (
              <div className="p-3.5 rounded-xl bg-rose-950/35 border border-rose-900/70 flex items-start gap-3 text-xs text-rose-200">
                <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold">Real Clinical Training Cohorts: NOT AVAILABLE.</span>{' '}
                  Zero real uploaded clinical images exist under <code>ml_service/data/raw/</code> or live manifests.
                  Interactive workbench phantoms remain strictly separated and are never counted as training data.
                </div>
              </div>
            )}
          </div>

          {/* Core Summary KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Unique Patients</span>
                <Users className="w-3.5 h-3.5 text-cyan-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">{readiness.unique_patients}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Institutions</span>
                <Building className="w-3.5 h-3.5 text-cyan-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">{readiness.institution_count}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Valid Samples</span>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">{readiness.valid_samples}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Missing Metadata</span>
                <FileWarning className="w-3.5 h-3.5 text-amber-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">{readiness.missing_metadata_count}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Duplicate Samples</span>
                <CopyCheck className="w-3.5 h-3.5 text-amber-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">{readiness.duplicate_samples_count}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span>Quarantined</span>
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">{readiness.quarantined_samples}</div>
            </div>
          </div>

          {/* Class Counts, Modality Counts, & Split Distribution */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* By Class */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Database className="w-4 h-4 text-cyan-400" />
                <span>Sample Counts by Class</span>
              </h2>
              <div className="space-y-2">
                {CLASS_LABELS.map(({ key, label }) => (
                  <div
                    key={key}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs"
                  >
                    <span className="text-slate-300">{label}</span>
                    <span className="font-mono font-bold text-white">{readiness.class_counts[key] ?? 0}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* By Modality (Strictly Separated) */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>Sample Counts by Modality</span>
              </h2>
              <div className="space-y-2">
                {MODALITY_LABELS.map(({ key, label }) => (
                  <div
                    key={key}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs"
                  >
                    <span className="text-slate-300">{label}</span>
                    <span className="font-mono font-bold text-white">{readiness.modality_counts[key] ?? 0}</span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-500">
                Modalities are strictly isolated in separate intake, validation, and split pools.
              </p>
            </div>

            {/* Train / Val / Test Split Distribution */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Split className="w-4 h-4 text-cyan-400" />
                <span>Patient-Level Split Distribution</span>
              </h2>
              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Train Split</span>
                  <span className="text-lg font-mono font-bold text-white">{readiness.split_distribution.train}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Validation Split</span>
                  <span className="text-lg font-mono font-bold text-white">{readiness.split_distribution.val}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Internal Test Split</span>
                  <span className="text-lg font-mono font-bold text-white">{readiness.split_distribution.test}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">External Site Test</span>
                  <span className="text-lg font-mono font-bold text-white">
                    {readiness.split_distribution.external_test}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Modality x Class Matrix + Permission / Readiness Gates */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 overflow-x-auto">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Class × Modality Separation Matrix
              </h2>
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2 pr-3">Modality</th>
                    <th className="py-2 px-2">Metastasis</th>
                    <th className="py-2 px-2">Paget</th>
                    <th className="py-2 px-2">OI</th>
                    <th className="py-2 pl-2">Healthy</th>
                  </tr>
                </thead>
                <tbody>
                  {MODALITY_LABELS.map(({ key, label }) => {
                    const row = readiness.class_modality_matrix[key] || {
                      metastasis: 0,
                      paget: 0,
                      oi: 0,
                      healthy: 0,
                    };
                    return (
                      <tr key={key} className="border-b border-slate-850 text-slate-300">
                        <td className="py-2.5 pr-3 font-medium">{label}</td>
                        <td className="py-2.5 px-2 font-mono">{row.metastasis}</td>
                        <td className="py-2.5 px-2 font-mono">{row.paget}</td>
                        <td className="py-2.5 px-2 font-mono">{row.oi}</td>
                        <td className="py-2.5 pl-2 font-mono">{row.healthy}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="text-[11px] text-slate-400 italic">
                {readiness.clinical_sample_size_justification_note}
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Permission, Provenance &amp; Readiness Gate Checks
              </h2>
              <div className="space-y-2">
                {GATE_LABELS.map(({ key, label }) => {
                  const passed = readiness.gates[key];
                  return (
                    <div
                      key={key}
                      className="flex items-center justify-between p-2 rounded-lg bg-slate-950 border border-slate-800 text-xs"
                    >
                      <span className="text-slate-300">{label}</span>
                      <span
                        className={`px-2 py-0.5 rounded font-mono font-bold text-[11px] ${
                          passed
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                            : 'bg-rose-950 text-rose-300 border border-rose-800'
                        }`}
                      >
                        {passed ? 'PASS' : 'BLOCKED'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Stage 34 Clinical Label Ontology, Adjudication & Healthy-Control Audit */}
          <div
            className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-4"
            aria-label="Clinical Label Ontology and Diagnostic Adjudication Audit"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                  Clinical Label Ontology &amp; Diagnostic Adjudication Audit
                </h2>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Zero cross-pathology equivalences (Osteoporosis != OI, Osteoarthritis != Paget&apos;s,
                  Contralateral != Healthy, Primary Malignancy != Metastasis). Actual patient counts only; synthetic examples excluded.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className="px-2.5 py-1 rounded bg-slate-950 border border-cyan-900 text-cyan-300 font-mono text-xs font-bold">
                  Ontology: {readiness.ontology_version}
                </span>
                <span className="px-2.5 py-1 rounded bg-slate-950 border border-slate-700 text-slate-300 font-mono text-xs font-bold">
                  Manifest Schema: {readiness.manifest_schema_validation}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5 text-xs">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Accepted</span>
                <span className="text-lg font-mono font-bold text-emerald-400">
                  {readiness.adjudication_summary.ACCEPTED}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Out of Scope</span>
                <span className="text-lg font-mono font-bold text-amber-300">
                  {readiness.adjudication_summary.OUT_OF_SCOPE}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Uncertain</span>
                <span className="text-lg font-mono font-bold text-amber-300">
                  {readiness.adjudication_summary.UNCERTAIN}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Awaiting Review</span>
                <span className="text-lg font-mono font-bold text-cyan-300">
                  {readiness.adjudication_summary.REQUIRES_ADJUDICATION}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Rejected</span>
                <span className="text-lg font-mono font-bold text-rose-400">
                  {readiness.adjudication_summary.REJECTED}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Healthy Failures</span>
                <span className="text-lg font-mono font-bold text-rose-300">
                  {readiness.healthy_control_eligibility_failures}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Conflicts</span>
                <span className="text-lg font-mono font-bold text-rose-300">
                  {readiness.conflicting_diagnoses_count}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Provenance %</span>
                <span className="text-lg font-mono font-bold text-white">
                  {readiness.label_provenance_completeness_pct.toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
              {(
                [
                  ['SECONDARY_BONE_METASTASIS', 'Secondary Bone Metastasis'],
                  ['PAGETS_DISEASE', "Paget's Disease"],
                  ['OSTEOGENESIS_IMPERFECTA', 'Osteogenesis Imperfecta'],
                  ['HEALTHY_SKELETAL_CONTROL', 'Healthy Skeletal Control'],
                ] as const
              ).map(([ontKey, title]) => (
                <div
                  key={ontKey}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800"
                >
                  <div className="min-w-0 pr-2">
                    <div className="text-slate-300 truncate">{title}</div>
                    <div className="text-[10px] font-mono text-slate-500 truncate">{ontKey}</div>
                  </div>
                  <span className="font-mono font-bold text-white">
                    {readiness.accepted_by_class[ontKey] ?? 0}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Outstanding Dataset Acquisition & Ethics Requirements */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Outstanding Dataset Acquisition &amp; Ethics Requirements
            </h2>
            <ul className="space-y-1.5 text-xs text-slate-300 list-disc list-inside">
              {readiness.outstanding_requirements.map((req, idx) => (
                <li key={idx}>{req}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {/* Stage 35 Institutional Dataset Acquisition, Governance & Cohort Planning */}
      {(sourcesData || acquisitionStatus || governanceStatus || cohortPlanning) && (
        <section
          className="space-y-6"
          aria-label="Institutional Dataset Acquisition, Governance and Cohort Planning"
        >
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-base font-extrabold text-white uppercase tracking-wider flex items-center gap-2">
                  <ClipboardCheck className="w-4 h-4 text-cyan-400" />
                  <span>Institutional Dataset Acquisition &amp; Governance Registry</span>
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Tracks verified external vs proposed institutional sources, IRB/IEC &amp; DUA clearances, sequential workflow
                  stages, and planned targets strictly separated from 0 actual validated clinical patients.
                </p>
              </div>
              {governanceStatus && (
                <span
                  className={`px-3 py-1 rounded font-mono text-xs font-bold border ${
                    governanceStatus.governance_gate_status === 'ALL_GOVERNANCE_VERIFIED'
                      ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                      : 'bg-amber-950/60 text-amber-300 border-amber-800'
                  }`}
                >
                  {governanceStatus.governance_gate_status}
                </span>
              )}
            </div>

            {/* Stage 35 Summary KPIs */}
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5 text-xs">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Identified Sources</span>
                <span className="text-lg font-mono font-bold text-white">
                  {sourcesData?.identified_sources_count ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Verified Candidates</span>
                <span className="text-lg font-mono font-bold text-cyan-300">
                  {sourcesData?.verified_external_or_in_tree_candidates_count ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Pending Ethics</span>
                <span className="text-lg font-mono font-bold text-amber-300">
                  {governanceStatus?.pending_ethics_approvals_count ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Pending DUAs</span>
                <span className="text-lg font-mono font-bold text-amber-300">
                  {governanceStatus?.pending_duas_count ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Approved Sources</span>
                <span className="text-lg font-mono font-bold text-emerald-400">
                  {governanceStatus?.approved_sources_count ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Received Scans</span>
                <span className="text-lg font-mono font-bold text-white">
                  {acquisitionStatus?.total_received_valid_images ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Validated Patients</span>
                <span className="text-lg font-mono font-bold text-rose-400">
                  {acquisitionStatus?.total_validated_patients ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Evidence Complete</span>
                <span className="text-lg font-mono font-bold text-white">
                  {(governanceStatus?.mean_evidence_completeness_pct ?? 0).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {/* 4-Cohort Acquisition Workflow Status */}
          {acquisitionStatus && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {acquisitionStatus.cohorts.map((coh) => {
                const label =
                  CLASS_LABELS.find((c) => c.key === coh.cohort_class)?.label || coh.ontology_target_class;
                return (
                  <div
                    key={coh.cohort_class}
                    className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <span className="text-[10px] font-mono text-cyan-400 block">
                          {coh.ontology_target_class}
                        </span>
                        <h3 className="text-sm font-bold text-white">{label}</h3>
                      </div>
                      <span className="px-2 py-0.5 rounded bg-slate-950 border border-amber-800 text-amber-300 font-mono text-[11px] font-bold">
                        {coh.current_workflow_stage}
                      </span>
                    </div>

                    <div className="grid grid-cols-4 gap-2 text-[11px]">
                      <div className="p-2 rounded bg-slate-950 border border-slate-800">
                        <span className="text-slate-500 block">Identified</span>
                        <span className="font-mono font-bold text-white">{coh.identified_sources_count}</span>
                      </div>
                      <div className="p-2 rounded bg-slate-950 border border-slate-800">
                        <span className="text-slate-500 block">Verified</span>
                        <span className="font-mono font-bold text-cyan-300">{coh.verified_sources_count}</span>
                      </div>
                      <div className="p-2 rounded bg-slate-950 border border-slate-800">
                        <span className="text-slate-500 block">Approved</span>
                        <span className="font-mono font-bold text-emerald-400">{coh.approved_sources_count}</span>
                      </div>
                      <div className="p-2 rounded bg-slate-950 border border-slate-800">
                        <span className="text-slate-500 block">Validated Pts</span>
                        <span className="font-mono font-bold text-rose-400">{coh.validated_patients_count}</span>
                      </div>
                    </div>

                    {coh.outstanding_blockers.length > 0 && (
                      <ul className="space-y-1 text-[11px] text-slate-400 list-disc list-inside">
                        {coh.outstanding_blockers.slice(0, 3).map((b, i) => (
                          <li key={i}>{b}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Planned vs Actual Cohort Sizes (Strict Distinction) */}
          {cohortPlanning && (
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 overflow-x-auto">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Scale className="w-4 h-4 text-cyan-400" />
                    <span>Planned vs Actual Cohort Planning Matrix</span>
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Planned Targets ({cohortPlanning.totals.planned_patient_count} patients /{' '}
                    {cohortPlanning.totals.planned_image_count} scans) vs Actual Validated On-Disk (
                    {cohortPlanning.totals.actual_validated_patients} patients /{' '}
                    {cohortPlanning.totals.actual_validated_images} scans). Sample Size Ref:{' '}
                    <code>{cohortPlanning.statistical_sample_size_justification_ref}</code>
                  </p>
                </div>
                {cohortPlanning.proposed_patient_level_split_allocation && (
                  <span className="px-2.5 py-1 rounded bg-slate-950 border border-slate-700 font-mono text-[11px] text-slate-300">
                    Proposed Split: {cohortPlanning.proposed_patient_level_split_allocation.train_pct}/
                    {cohortPlanning.proposed_patient_level_split_allocation.val_pct}/
                    {cohortPlanning.proposed_patient_level_split_allocation.internal_test_pct} (by patient_id)
                  </span>
                )}
              </div>

              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2 pr-3">Cohort Class</th>
                    <th className="py-2 px-2">Modality</th>
                    <th className="py-2 px-2">Planned Pts / Scans</th>
                    <th className="py-2 px-2">Actual Received</th>
                    <th className="py-2 px-2">Actual Validated Pts</th>
                    <th className="py-2 pl-2">Biostatistical Ref</th>
                  </tr>
                </thead>
                <tbody>
                  {cohortPlanning.planned_cohorts.map((row) => (
                    <tr key={row.plan_id} className="border-b border-slate-850 text-slate-300">
                      <td className="py-2 pr-3 font-medium">{row.ontology_target_class}</td>
                      <td className="py-2 px-2 font-mono uppercase">{row.imaging_modality}</td>
                      <td className="py-2 px-2 font-mono text-cyan-300">
                        {row.planned_patient_count} pts / {row.planned_image_count} imgs
                      </td>
                      <td className="py-2 px-2 font-mono text-white">{row.actual_received_patients}</td>
                      <td className="py-2 px-2 font-mono text-rose-400">{row.actual_validated_patients}</td>
                      <td className="py-2 pl-2 font-mono text-[11px] text-amber-300">
                        {row.statistical_sample_size_justification_ref}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {cohortPlanning.statistical_disclaimer && (
                <p className="text-[11px] text-slate-400 italic">{cohortPlanning.statistical_disclaimer}</p>
              )}
            </div>
          )}

          {/* Source Registry Table & Draft Institutional Templates */}
          {sourcesData && (
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 overflow-x-auto">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  <span>Dataset Source Registry &amp; Evidence Completeness</span>
                </h3>
                {governanceStatus && (
                  <span className="px-2.5 py-1 rounded bg-slate-950 border border-amber-900 text-amber-300 font-mono text-[11px]">
                    {governanceStatus.templates_status} ({governanceStatus.available_templates.length} templates)
                  </span>
                )}
              </div>

              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2 pr-3">Source</th>
                    <th className="py-2 px-2">Cohort</th>
                    <th className="py-2 px-2">Modalities</th>
                    <th className="py-2 px-2">Classification</th>
                    <th className="py-2 px-2">Acquisition Status</th>
                    <th className="py-2 pl-2">Evidence %</th>
                  </tr>
                </thead>
                <tbody>
                  {sourcesData.sources.map((src) => (
                    <tr key={src.source_id} className="border-b border-slate-850 text-slate-300">
                      <td className="py-2 pr-3">
                        <div className="font-medium text-white">{src.source_name}</div>
                        <div className="text-[11px] text-slate-500">{src.institution}</div>
                      </td>
                      <td className="py-2 px-2 font-mono">{src.target_cohort}</td>
                      <td className="py-2 px-2 font-mono">
                        {src.available_modalities.length ? src.available_modalities.join(', ') : 'NONE'}
                      </td>
                      <td className="py-2 px-2 font-mono text-[11px]">{src.source_classification}</td>
                      <td className="py-2 px-2">
                        <span
                          className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold border ${
                            src.acquisition_status === 'REJECTED'
                              ? 'bg-rose-950 text-rose-300 border-rose-800'
                              : src.acquisition_status === 'APPROVED' || src.acquisition_status === 'VALIDATED'
                              ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                              : 'bg-amber-950/60 text-amber-300 border-amber-800'
                          }`}
                        >
                          {src.acquisition_status}
                        </span>
                      </td>
                      <td className="py-2 pl-2 font-mono">{src.evidence_completeness_pct.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {data && (
        <div className="space-y-6">
          {/* Acquisition Gate Status Banner */}
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs text-slate-400 block mb-1">Overall Institutional Acquisition Gate</span>
              <span
                className={`text-lg font-bold font-mono ${
                  data.overall_acquisition_gate.startsWith('UNLOCKED')
                    ? 'text-emerald-400'
                    : data.overall_acquisition_gate.startsWith('READY')
                    ? 'text-cyan-400'
                    : 'text-amber-400'
                }`}
              >
                {data.overall_acquisition_gate}
              </span>
            </div>
            <div>
              {data.overall_acquisition_gate.startsWith('UNLOCKED') ? (
                <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>4 / 4 COHORTS VERIFIED</span>
                </span>
              ) : (
                <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-950 text-rose-300 border border-rose-800">
                  <Lock className="w-3.5 h-3.5" />
                  <span>INSPECTION &amp; DUA REQUIRED</span>
                </span>
              )}
            </div>
          </div>

          {/* 4 Cohorts Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {data.cohorts.map((cohort: DatasetGovernanceItem, idx: number) => {
              const isVerifiedOnDisk = cohort.source_status.startsWith('VERIFIED ON DISK');
              return (
                <motion.div
                  key={cohort.cohort_name}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`p-6 rounded-2xl border space-y-4 ${
                    isVerifiedOnDisk
                      ? 'bg-emerald-950/15 border-emerald-800/60'
                      : 'bg-slate-900/60 border-slate-800'
                  }`}
                >
                  <div className="border-b border-slate-850 pb-3">
                    <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider block mb-1">
                      Target Class {idx + 1}: {cohort.target_class}
                    </span>
                    <h3 className="text-lg font-extrabold text-white">{cohort.cohort_name}</h3>
                    <p className="text-xs text-slate-400 mt-0.5 truncate" title={`Modality: ${cohort.intended_modality}`}>
                      Modality: {cohort.intended_modality}
                    </p>
                    {cohort.disk_folder && (
                      <p className="text-[11px] font-mono text-slate-400 mt-1 truncate" title={cohort.disk_folder}>
                        Path: {cohort.disk_folder}
                      </p>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Source Status</span>
                      <span
                        className={`font-bold block truncate ${
                          isVerifiedOnDisk ? 'text-emerald-300' : 'text-rose-300'
                        }`}
                      >
                        {cohort.source_status}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Target Cohort Size</span>
                      <span className="font-bold text-white block">{cohort.patient_count_estimate}</span>
                    </div>
                  </div>

                  <div className="space-y-1.5 text-xs">
                    <span className="font-semibold text-slate-300 block">
                      {isVerifiedOnDisk ? 'Disk Validation Verdict:' : 'Identified Acquisition Blockers:'}
                    </span>
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
              Formal protocols prepared under Stage 24 and Stage 25 include institutional ethics applications (IEC Form A),
              bilateral Data Use Agreements (DUA), Clinical Trial Registry of India (CTRI) prospective documentation,
              and anonymization pipelines compliant with Indian Personal Data Protection (DPDP) Act 2023.
            </p>
            <div className="text-[11px] text-slate-500 italic">
              Notice: Applications are documented in <code>ml_artifacts/stage24_collaboration_prep/</code> and awaiting
              institutional faculty submission.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GovernancePage;
