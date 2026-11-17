import React from 'react';
import { PatientProfile, FraxCalculationResult, DexaAnalysis } from '../types/osteoporosis';
import { Printer, X, Download, ShieldCheck, Bone, CheckCircle2 } from 'lucide-react';

interface ClinicalReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  patient: PatientProfile;
  fraxResult: FraxCalculationResult;
  dexaData: DexaAnalysis;
}

export const ClinicalReportModal: React.FC<ClinicalReportModalProps> = ({
  isOpen,
  onClose,
  patient,
  fraxResult,
  dexaData,
}) => {
  if (!isOpen) return null;

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm overflow-y-auto">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-3xl shadow-2xl overflow-hidden my-8 print:border-none print:shadow-none print:bg-white print:text-black">
        {/* Header bar (no-print) */}
        <div className="no-print flex items-center justify-between px-6 py-4 bg-slate-800/80 border-b border-slate-700">
          <div className="flex items-center space-x-2">
            <Bone className="w-5 h-5 text-cyan-400" />
            <span className="text-base font-bold text-white">Comprehensive Clinical Osteoporosis Assessment Report</span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={handlePrint}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-md transition"
            >
              <Printer className="w-4 h-4" />
              <span>Print / Save PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable Document Body */}
        <div className="p-6 sm:p-8 space-y-6 text-slate-200 print:text-slate-900 print:p-0">
          {/* Institutional / Clinical Header */}
          <div className="border-b-2 border-cyan-500/80 pb-4 flex justify-between items-start">
            <div>
              <h1 className="text-2xl font-extrabold text-white print:text-black tracking-tight">
                Osteo<span className="text-cyan-400 print:text-cyan-600">Shield</span> Bone Health Assessment
              </h1>
              <p className="text-xs text-slate-400 print:text-slate-600 mt-0.5">
                Clinical Fracture Risk & Dual-Energy X-Ray Absorptiometry Summary
              </p>
            </div>
            <div className="text-right text-xs text-slate-400 print:text-slate-600">
              <div>Date: {new Date().toLocaleDateString()}</div>
              <div className="font-mono text-cyan-400 print:text-slate-800">Ref ID: {patient.id}</div>
            </div>
          </div>

          {/* Patient Demographic Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950 print:bg-slate-100 p-4 rounded-xl border border-slate-800 print:border-slate-300 text-xs">
            <div>
              <span className="text-slate-400 print:text-slate-600 block">Patient Name:</span>
              <span className="font-bold text-white print:text-black">{patient.name}</span>
            </div>
            <div>
              <span className="text-slate-400 print:text-slate-600 block">Age / Gender:</span>
              <span className="font-bold text-white print:text-black">{patient.age}y / {patient.gender === 'female' ? 'Female' : 'Male'}</span>
            </div>
            <div>
              <span className="text-slate-400 print:text-slate-600 block">Weight & Height:</span>
              <span className="font-bold text-white print:text-black">{patient.weightKg} kg / {patient.heightCm} cm</span>
            </div>
            <div>
              <span className="text-slate-400 print:text-slate-600 block">Body Mass Index:</span>
              <span className="font-mono font-bold text-white print:text-black">{fraxResult.bmi} kg/m²</span>
            </div>
          </div>

          {/* FRAX 10-Year Fracture Risk Assessment */}
          <div className="border border-slate-800 print:border-slate-300 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-cyan-400 print:text-cyan-700 uppercase tracking-wider">
              1. WHO FRAX® 10-Year Probability of Fracture
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-200">
                <span className="text-[11px] text-slate-400 print:text-slate-600 block">Major Osteoporotic Fracture</span>
                <span className="text-2xl font-extrabold text-white print:text-black">{fraxResult.tenYearMajorFractureRisk}%</span>
                <span className="text-[10px] text-slate-500 block mt-0.5">Threshold: {fraxResult.interventionThreshold}%</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-200">
                <span className="text-[11px] text-slate-400 print:text-slate-600 block">Hip Fracture</span>
                <span className="text-2xl font-extrabold text-white print:text-black">{fraxResult.tenYearHipFractureRisk}%</span>
                <span className="text-[10px] text-rose-400 print:text-rose-600 block mt-0.5">BHOF High-Risk Cutoff: 3.0%</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 print:bg-slate-50 border border-slate-800 print:border-slate-200">
                <span className="text-[11px] text-slate-400 print:text-slate-600 block">NOGG Risk Tier</span>
                <span className="text-lg font-bold text-cyan-300 print:text-cyan-700 uppercase block mt-1">
                  {fraxResult.riskCategory.replace('_', ' ')}
                </span>
              </div>
            </div>
            <div className="text-xs text-slate-300 print:text-slate-800 bg-slate-950/60 print:bg-slate-100 p-3 rounded-lg">
              <strong>Recommendation:</strong> {fraxResult.recommendation}
            </div>
          </div>

          {/* DXA Bone Mineral Density Measurements */}
          <div className="border border-slate-800 print:border-slate-300 rounded-xl p-4 space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-xs font-bold text-cyan-400 print:text-cyan-700 uppercase tracking-wider">
                2. Central DXA Bone Mineral Density Measurements
              </h3>
              <span className="text-xs font-bold px-2 py-0.5 rounded bg-slate-800 print:bg-slate-200 text-slate-200 print:text-black">
                WHO Diagnosis: {dexaData.whoCategory}
              </span>
            </div>

            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-800 print:border-slate-300 text-slate-400 print:text-slate-600">
                  <th className="py-2">Anatomical Region</th>
                  <th className="py-2">BMD (g/cm²)</th>
                  <th className="py-2">T-Score (SD)</th>
                  <th className="py-2">Z-Score (SD)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 print:divide-slate-200 font-mono">
                {dexaData.sites.map((site) => (
                  <tr key={site.siteName}>
                    <td className="py-2 font-sans font-medium text-slate-200 print:text-black">{site.siteName}</td>
                    <td className="py-2">{site.bmd.toFixed(3)}</td>
                    <td className="py-2 font-bold">{site.tScore > 0 ? `+${site.tScore}` : site.tScore}</td>
                    <td className="py-2">{site.zScore > 0 ? `+${site.zScore}` : site.zScore}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Action Plan & Safety Guidelines */}
          <div className="border border-slate-800 print:border-slate-300 rounded-xl p-4 space-y-2 text-xs">
            <h3 className="text-xs font-bold text-cyan-400 print:text-cyan-700 uppercase tracking-wider mb-2">
              3. Clinical Care Plan & Non-Pharmacological Orders
            </h3>
            <ul className="space-y-1.5 text-slate-300 print:text-slate-800">
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400 font-bold">•</span>
                <span><strong>Nutrition:</strong> Target 1,200 mg elemental calcium daily (dietary preference; supplement if deficit exists) + Vitamin D3 1,000–2,000 IU daily to maintain 25(OH)D &gt; 30 ng/mL.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400 font-bold">•</span>
                <span><strong>Physical Therapy:</strong> Progressive weight-bearing & resistance exercise. Avoid spinal hyperflexion (sit-ups) and rapid axial torsion.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400 font-bold">•</span>
                <span><strong>Fall Mitigation:</strong> Home environmental safety check (throw rugs, nightlights, bathroom grab bars). Medication review for sedatives/hypotensives.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400 font-bold">•</span>
                <span><strong>Follow-up Surveillance:</strong> Repeat DXA scan in 24 months on the same scanner to assess Least Significant Change (LSC).</span>
              </li>
            </ul>
          </div>

          {/* Signature / Attestation */}
          <div className="pt-6 border-t border-slate-800 print:border-slate-300 flex justify-between items-end text-xs text-slate-400 print:text-slate-600">
            <div>
              <div className="w-48 border-b border-slate-700 print:border-slate-400 mb-1" />
              <span>Attending Physician / Specialist Signature</span>
            </div>
            <div>
              <span>Generated via OsteoShield Clinical Decision Support</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
