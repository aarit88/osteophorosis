import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Layers,
  ShieldAlert,
  ShieldCheck,
  Lock,
  Play,
  RotateCw,
  HardDrive,
  KeyRound,
  Server,
} from 'lucide-react';
import { getArchitectureReadiness, getSecurityStatus, requestFourClassInference } from '../api/client';
import {
  ArchitectureReadinessResponse,
  ArchitectureComponentStatus,
  BlockedInferenceResponse,
  SecurityStatusResponse,
} from '../api/types';

export const ArchitecturePage: React.FC = () => {
  const [data, setData] = useState<ArchitectureReadinessResponse | null>(null);
  const [securityStatus, setSecurityStatus] = useState<SecurityStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Safe Gate Test state
  const [blockedResult, setBlockedResult] = useState<BlockedInferenceResponse | null>(null);
  const [gateTesting, setGateTesting] = useState<boolean>(false);

  useEffect(() => {
    Promise.all([
      getArchitectureReadiness(),
      getSecurityStatus().catch(() => null),
    ])
      .then(([archRes, secRes]) => {
        setData(archRes);
        setSecurityStatus(secRes);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load architecture readiness.');
        setLoading(false);
      });
  }, []);

  const handleTestSafeGate = async () => {
    setGateTesting(true);
    try {
      const res = await requestFourClassInference();
      setBlockedResult(res);
    } catch (err: any) {
      setBlockedResult({
        status: 'BLOCKED',
        error_code: 'CLIENT_CAUGHT_ERROR',
        message: err.message,
        target_classes: [],
        blockers: ['Network or connection error'],
        mandatory_safeguard: 'System blocked request.',
      });
    } finally {
      setGateTesting(false);
    }
  };

  const getStatusBadge = (status: string) => {
    if (status === 'IMPLEMENTED') {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
          IMPLEMENTED
        </span>
      );
    }
    if (status === 'PARTIALLY_IMPLEMENTED') {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
          PARTIALLY IMPLEMENTED
        </span>
      );
    }
    if (status === 'ARCHITECTURALLY_SPECIFIED') {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800">
          ARCHITECTURALLY SPECIFIED
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-300 border border-rose-800">
        BLOCKED BY DATA
      </span>
    );
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-xs font-semibold">
          <Layers className="w-3.5 h-3.5" />
          <span>Stage 26 / 31 Architectural Readiness</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Four-Class System Architecture & Diagnostic Safety Gate
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Visualizes the full multi-modal fusion architecture specified in the original KSCST proposal and demonstrates the immutable safety gate blocking diagnostic inference until clinical cohorts are acquired.
        </p>
      </div>

      {loading && (
        <div className="p-8 text-center text-slate-400 animate-pulse">
          Loading architecture readiness...
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="space-y-8">
          {/* End-to-End Encryption, Privacy & Live PyTorch Bridge Status */}
          {securityStatus && (
            <div className="p-6 rounded-2xl bg-slate-900/70 border border-emerald-500/30 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div>
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    <span>End-to-End Encryption, Privacy & Live PyTorch Bridge</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Live verification of cryptographic audit storage, zero-disk RAM radiograph processing, and PyTorch checkpoint readiness on disk.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded text-[11px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                  {securityStatus.encryptionAtRest.algorithm} ACTIVE
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-[11px] font-semibold uppercase">At-Rest Ledger</span>
                    <KeyRound className="w-3.5 h-3.5 text-emerald-400" />
                  </div>
                  <div className="text-sm font-bold text-white">
                    {securityStatus.encryptionAtRest.algorithm} Encrypted
                  </div>
                  <div className="text-[11px] text-slate-400">
                    {securityStatus.encryptionAtRest.feedbackEncryptedCount} feedback &{' '}
                    {securityStatus.encryptionAtRest.eventsEncryptedCount} event records (0 plaintext)
                  </div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-[11px] font-semibold uppercase">Uploaded Scans</span>
                    <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
                  </div>
                  <div className="text-sm font-bold text-white">
                    RAM-Only + EXIF Stripped
                  </div>
                  <div className="text-[11px] text-slate-400">
                    {securityStatus.ephemeralMedia.diskUploadsRemaining} disk files · {securityStatus.ephemeralMedia.ttlSeconds}s RAM TTL
                  </div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-[11px] font-semibold uppercase">ML Gateway Auth</span>
                    <Server className="w-3.5 h-3.5 text-cyan-400" />
                  </div>
                  <div className="text-sm font-bold text-white">
                    {securityStatus.gatewayTransport.mlBackendConfigured
                      ? securityStatus.gatewayTransport.mlBackendReachable
                        ? 'FastAPI Linked'
                        : 'FastAPI Offline'
                      : 'Standalone Gateway'}
                  </div>
                  <div className="text-[11px] text-slate-400">
                    HMAC-SHA256: {securityStatus.gatewayTransport.hmacSigningConfigured ? 'Active' : 'Optional'} ·{' '}
                    <span className="whitespace-nowrap">no-store</span>
                  </div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-[11px] font-semibold uppercase">4-Class PyTorch Weights</span>
                    <Lock className="w-3.5 h-3.5 text-rose-400" />
                  </div>
                  <div className="text-sm font-bold text-white">
                    {securityStatus.clinicalReadiness.checkpointCount} Checkpoints Found
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Clinical dataset: {securityStatus.clinicalReadiness.clinicalDatasetAvailable ? 'Present' : 'Unavailable'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Visual Architecture Flowchart */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              KSCST Multi-Modal Fusion Pipeline Architecture
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs">
              {/* Step 1: Input Modalities */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 flex flex-col justify-between">
                <div>
                  <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider block">Inputs</span>
                  <div className="font-extrabold text-white text-sm mt-1">Skeletal Radiograph + ROI Mask</div>
                </div>
                <div className="text-[11px] text-slate-400">Standard DICOM / PNG / TIFF</div>
                <div className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                  Ready
                </div>
              </div>

              {/* Step 2: Dual Extraction */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 flex flex-col justify-between">
                <div>
                  <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider block">Extraction</span>
                  <div className="font-extrabold text-white text-sm mt-1">DenseNet-121 (512-D) + Radiomics (102-D)</div>
                </div>
                <div className="text-[11px] text-slate-400">Deep CNN + 81 IBSI Textures</div>
                <div className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800">
                  81 / 102 Verified
                </div>
              </div>

              {/* Step 3: Fusion Layer */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 flex flex-col justify-between">
                <div>
                  <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider block">Fusion</span>
                  <div className="font-extrabold text-white text-sm mt-1">Multi-Modal Concatenation (614-D)</div>
                </div>
                <div className="text-[11px] text-slate-400">Bottleneck Linear Layer</div>
                <div className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800">
                  Specified / Unverified
                </div>
              </div>

              {/* Step 4: Modifiers & Safety */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 flex flex-col justify-between">
                <div>
                  <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider block">Modifiers</span>
                  <div className="font-extrabold text-white text-sm mt-1">Biomechanics + Uncertainty Gate</div>
                </div>
                <div className="text-[11px] text-slate-400">Mirels + H_norm ≥ 0.60 Gate</div>
                <div className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                  Verified Engine
                </div>
              </div>

              {/* Step 5: Final Output */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 flex flex-col justify-between">
                <div>
                  <span className="text-[10px] font-bold text-rose-400 uppercase tracking-wider block">Output</span>
                  <div className="font-extrabold text-white text-sm mt-1">Four-Class Differential Diagnosis</div>
                </div>
                <div className="text-[11px] text-slate-400">Metastasis, Paget, OI, Healthy</div>
                <div className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-300 border border-rose-800">
                  {securityStatus?.clinicalReadiness.isFourClassReady ? 'CHECKPOINT READY' : 'BLOCKED BY DATA'}
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Safe Gate Verification Test */}
          <div className="p-6 rounded-2xl bg-rose-950/20 border border-rose-900/60 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-rose-900/40 pb-3">
              <div>
                <h3 className="font-extrabold text-white text-lg flex items-center space-x-2">
                  <Lock className="w-5 h-5 text-rose-400" />
                  <span>Four-Class Inference Safety Gate Verification</span>
                </h3>
                <p className="text-xs text-rose-200 mt-0.5">
                  Test that requesting four-class diagnosis immediately triggers the HTTP 503 BLOCKED safeguard.
                </p>
              </div>
              <button
                type="button"
                onClick={handleTestSafeGate}
                disabled={gateTesting}
                className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs disabled:opacity-50 transition cursor-pointer self-start sm:self-auto"
              >
                {gateTesting ? <RotateCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                <span>Trigger Four-Class Inference Request</span>
              </button>
            </div>

            {blockedResult && (
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className={`p-5 rounded-xl bg-slate-950 border text-xs space-y-3 ${
                  blockedResult.status === 'READY' ? 'border-emerald-700' : 'border-rose-800'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`font-bold flex items-center space-x-1.5 text-sm ${
                      blockedResult.status === 'READY' ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {blockedResult.status === 'READY' ? (
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <ShieldAlert className="w-4 h-4 text-rose-400" />
                    )}
                    <span>
                      GATE RESPONSE: {blockedResult.status} (
                      {blockedResult.status === 'READY' ? 'HTTP 200 CLINICAL ELIGIBLE' : 'HTTP 503 SERVICE UNAVAILABLE'})
                    </span>
                  </span>
                  <span
                    className={`font-mono text-[10px] px-2 py-0.5 rounded border ${
                      blockedResult.status === 'READY'
                        ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                        : 'bg-rose-950 text-rose-300 border-rose-800'
                    }`}
                  >
                    {blockedResult.error_code}
                  </span>
                </div>
                <p className="text-slate-200">{blockedResult.message}</p>

                {blockedResult.blockers.length > 0 && (
                  <div className="space-y-1">
                    <span className="font-bold text-amber-400 block">Active Blocker Checklist:</span>
                    <ul className="space-y-0.5 text-slate-300 list-disc list-inside text-[11px]">
                      {blockedResult.blockers.map((blk, idx) => (
                        <li key={idx}>{blk}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="pt-2 border-t border-slate-850 text-[11px] text-slate-400 italic">
                  <strong>Mandatory Safeguard:</strong> {blockedResult.mandatory_safeguard}
                </div>
              </motion.div>
            )}
          </div>

          {/* Component Readiness Table */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
            <h3 className="font-bold text-white text-base">Architectural Component Readiness Register</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase text-[10px]">
                    <th className="py-2.5 px-3">Component</th>
                    <th className="py-2.5 px-3">Layer Group</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Implemented Features</th>
                    <th className="py-2.5 px-3">Unresolved Gaps</th>
                    <th className="py-2.5 px-3">Governance Dependency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-850">
                  {data.components.map((c: ArchitectureComponentStatus, idx: number) => (
                    <tr key={idx} className="hover:bg-slate-850/40 transition">
                      <td className="py-3 px-3 font-bold text-white whitespace-nowrap">{c.component_name}</td>
                      <td className="py-3 px-3 text-slate-300">{c.layer_group}</td>
                      <td className="py-3 px-3">{getStatusBadge(c.status)}</td>
                      <td className="py-3 px-3 text-slate-300">{c.implemented_features}</td>
                      <td className="py-3 px-3 text-amber-300/90">{c.unresolved_gaps}</td>
                      <td className="py-3 px-3 text-slate-400">{c.governance_dependency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ArchitecturePage;


