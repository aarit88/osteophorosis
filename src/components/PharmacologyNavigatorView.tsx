import React, { useState } from 'react';
import { MEDICATIONS_DATABASE } from '../data/mockData';
import { Medication } from '../types/osteoporosis';
import { Pill, ShieldCheck, AlertOctagon, CheckCircle2, Info, ArrowRight, Clock, HelpCircle } from 'lucide-react';

export const PharmacologyNavigatorView: React.FC = () => {
  const [selectedMed, setSelectedMed] = useState<Medication>(MEDICATIONS_DATABASE[0]);
  const [activeClassFilter, setActiveClassFilter] = useState<string>('All');
  const [showHolidayHelper, setShowHolidayHelper] = useState<boolean>(false);

  const classes = ['All', 'Bisphosphonate', 'RANKL Inhibitor', 'Anabolic (PTH/PTHrP analog)', 'Dual Action (Sclerostin Inhibitor)', 'SERM'];

  const filteredMeds = activeClassFilter === 'All'
    ? MEDICATIONS_DATABASE
    : MEDICATIONS_DATABASE.filter((m) => m.drugClass === activeClassFilter);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center space-x-2">
              <Pill className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">Pharmacotherapy Navigator & Evidence-Based Guidelines</h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Comparative efficacy, administration, drug holidays, and safety checklists (BHOF, AACE, ESCEO guidelines)
            </p>
          </div>

          <button
            onClick={() => setShowHolidayHelper(!showHolidayHelper)}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-cyan-950/70 text-cyan-300 border border-cyan-800 text-xs font-semibold hover:bg-cyan-900/80 transition"
          >
            <Clock className="w-4 h-4" />
            <span>{showHolidayHelper ? 'Hide Drug Holiday Decision Tree' : 'Bisphosphonate Drug Holiday Guide'}</span>
          </button>
        </div>

        {/* Drug Holiday Interactive Helper */}
        {showHolidayHelper && (
          <div className="p-4 rounded-xl bg-slate-950 border border-cyan-800/60 mb-6 space-y-3">
            <div className="flex items-center space-x-2 text-cyan-300 text-xs font-bold uppercase tracking-wider">
              <ShieldCheck className="w-4 h-4" />
              <span>Bisphosphonate "Drug Holiday" Evaluation Protocol</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="font-semibold text-white block mb-1">1. Duration of Therapy</span>
                <p className="text-slate-300 text-[11px]">
                  Oral bisphosphonate &ge; 5 years, OR intravenous zoledronic acid &ge; 3 annual infusions.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="font-semibold text-white block mb-1">2. Fracture Risk Reassessment</span>
                <p className="text-slate-300 text-[11px]">
                  If Femoral Neck T &gt; -2.5 and NO incident fragility fractures: Safe to pause for 2-3 years with DXA monitoring.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="font-semibold text-rose-300 block mb-1">3. Persistent High Risk</span>
                <p className="text-slate-300 text-[11px]">
                  If T &le; -2.5 or prior hip/vertebral fracture: Continue therapy up to 10 years or switch to anabolic / denosumab.
                </p>
              </div>
            </div>
            <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/40 text-xs text-rose-200">
              <strong>CRITICAL WARNING (Denosumab):</strong> Never institute a "drug holiday" with Denosumab (Prolia). Discontinuation causes rapid rebound osteoclast activation and multiple vertebral compression fractures. Transition to bisphosphonate is mandatory.
            </div>
          </div>
        )}

        {/* Drug Class Filter Pills */}
        <div className="flex overflow-x-auto no-scrollbar gap-2 pt-2">
          {classes.map((cls) => (
            <button
              key={cls}
              onClick={() => setActiveClassFilter(cls)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                activeClassFilter === cls
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50'
                  : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200'
              }`}
            >
              {cls}
            </button>
          ))}
        </div>
      </div>

      {/* Main Medications Grid & Detail Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Medication Cards List */}
        <div className="lg:col-span-7 space-y-3">
          {filteredMeds.map((med) => {
            const isSelected = med.id === selectedMed.id;
            return (
              <div
                key={med.id}
                onClick={() => setSelectedMed(med)}
                className={`p-4 rounded-xl border cursor-pointer transition ${
                  isSelected
                    ? 'bg-cyan-950/30 border-cyan-500/60 shadow-lg shadow-cyan-950/40'
                    : 'bg-slate-900/90 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-base font-bold text-white">{med.name}</span>
                    <span className="text-xs font-mono text-cyan-400">({med.brandName})</span>
                  </div>
                  <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 font-medium border border-slate-700">
                    {med.drugClass}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-xs py-2 bg-slate-950/60 rounded-lg px-3 border border-slate-800/80 mb-2">
                  <div>
                    <span className="text-[10px] text-slate-500 block">Vertebral Fx</span>
                    <span className="font-semibold text-emerald-400">{med.vertebralReduction}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Hip Fx</span>
                    <span className="font-semibold text-emerald-400">{med.hipReduction}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Administration</span>
                    <span className="font-semibold text-slate-300 truncate block">{med.frequency}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="truncate max-w-[80%]">{med.route}</span>
                  <span className="text-cyan-400 flex items-center space-x-1 font-medium">
                    <span>Details</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Medication Deep Dive Panel */}
        <div className="lg:col-span-5 rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <span className="text-xs text-cyan-400 font-semibold uppercase tracking-wider block">
              {selectedMed.drugClass}
            </span>
            <div className="text-xl font-bold text-white mt-1">
              {selectedMed.name} <span className="text-slate-400 text-sm">({selectedMed.brandName})</span>
            </div>
            <div className="text-xs text-slate-300 mt-1 font-mono">{selectedMed.route} • {selectedMed.frequency}</div>
          </div>

          {/* Fracture Reduction Efficacy */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Clinical Trial Fracture Reduction Rates
            </h4>
            <div className="grid grid-cols-3 gap-2">
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-center">
                <span className="text-[10px] text-slate-400 block">Vertebral</span>
                <span className="text-xs font-bold text-emerald-400 block mt-0.5">{selectedMed.vertebralReduction}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-center">
                <span className="text-[10px] text-slate-400 block">Hip</span>
                <span className="text-xs font-bold text-emerald-400 block mt-0.5">{selectedMed.hipReduction}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-center">
                <span className="text-[10px] text-slate-400 block">Non-Vertebral</span>
                <span className="text-xs font-bold text-emerald-400 block mt-0.5">{selectedMed.nonVertebralReduction}</span>
              </div>
            </div>
          </div>

          {/* Key Contraindications */}
          <div>
            <h4 className="text-xs font-semibold text-rose-400 uppercase tracking-wider flex items-center space-x-1.5 mb-1.5">
              <AlertOctagon className="w-3.5 h-3.5" />
              <span>Contraindications</span>
            </h4>
            <ul className="space-y-1 text-xs text-slate-300">
              {selectedMed.keyContraindications.map((c, i) => (
                <li key={i} className="flex items-start space-x-1.5">
                  <span className="text-rose-400 mt-0.5">•</span>
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Potential Adverse Effects */}
          <div>
            <h4 className="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center space-x-1.5 mb-1.5">
              <Info className="w-3.5 h-3.5" />
              <span>Key Adverse Effects & Risks</span>
            </h4>
            <ul className="space-y-1 text-xs text-slate-300">
              {selectedMed.adverseEffects.map((a, i) => (
                <li key={i} className="flex items-start space-x-1.5">
                  <span className="text-amber-400 mt-0.5">•</span>
                  <span>{a}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Dental & Jaw Precautions (MRONJ) */}
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            <span className="font-semibold text-cyan-300 block mb-0.5">Dental / MRONJ Recommendations:</span>
            <p className="text-slate-300 text-[11px] leading-relaxed">{selectedMed.dentalPrecautions}</p>
          </div>

          {/* Duration & Sequential Therapy */}
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            <span className="font-semibold text-slate-200 block mb-0.5">Duration & Follow-up Strategy:</span>
            <p className="text-slate-400 text-[11px] leading-relaxed">{selectedMed.durationRecommendation}</p>
          </div>
        </div>
      </div>
    </div>
  );
};
