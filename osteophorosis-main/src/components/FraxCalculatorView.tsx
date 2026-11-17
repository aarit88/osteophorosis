import React from 'react';
import { PatientProfile, FraxCalculationResult } from '../types/osteoporosis';
import { ShieldAlert, AlertTriangle, CheckCircle2, AlertOctagon, HelpCircle, Activity, Info } from 'lucide-react';

interface FraxCalculatorViewProps {
  patient: PatientProfile;
  onChangePatient: (updated: PatientProfile) => void;
  fraxResult: FraxCalculationResult;
  onNavigateToDexa: () => void;
  onNavigateToPharma: () => void;
}

export const FraxCalculatorView: React.FC<FraxCalculatorViewProps> = ({
  patient,
  onChangePatient,
  fraxResult,
  onNavigateToDexa,
  onNavigateToPharma,
}) => {
  const updateField = <K extends keyof PatientProfile>(key: K, value: PatientProfile[K]) => {
    onChangePatient({
      ...patient,
      [key]: value,
    });
  };

  // NOGG age curve helper points for graph (age 40 to 90)
  const agePoints = [40, 50, 60, 70, 80, 90];
  const thresholdPoints = agePoints.map((a) => ({
    age: a,
    threshold: Math.min(Math.max(6.0 + (a - 40) * 0.38, 7.0), 30.0),
  }));

  const getRiskBadge = () => {
    switch (fraxResult.riskCategory) {
      case 'very_high':
        return {
          label: 'VERY HIGH RISK',
          color: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
          icon: AlertOctagon,
          actionText: 'Urgent Anabolic Bone Therapy Indicated',
        };
      case 'high':
        return {
          label: 'HIGH RISK',
          color: 'bg-orange-500/20 text-orange-300 border-orange-500/40',
          icon: AlertTriangle,
          actionText: 'Pharmacological Treatment Recommended',
        };
      case 'intermediate':
        return {
          label: 'INTERMEDIATE RISK',
          color: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
          icon: HelpCircle,
          actionText: 'Measure BMD via DEXA to Recalculate Risk',
        };
      case 'low':
      default:
        return {
          label: 'LOW RISK',
          color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
          icon: CheckCircle2,
          actionText: 'Lifestyle, Calcium, Vitamin D & Reassurance',
        };
    }
  };

  const badge = getRiskBadge();
  const RiskIcon = badge.icon;

  return (
    <div className="space-y-6">
      {/* Top Banner / Risk Summary Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Gauges */}
        <div className="lg:col-span-2 rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-80 h-80 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

          <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
            <div>
              <div className="flex items-center space-x-2">
                <ShieldAlert className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-bold text-white">FRAX® 10-Year Fracture Risk Output</h2>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">Calculated using WHO/NOGG clinical risk coefficients</p>
            </div>

            <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border text-xs font-bold tracking-wide ${badge.color}`}>
              <RiskIcon className="w-4 h-4" />
              <span>{badge.label}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            {/* Major Osteoporotic Fracture Card */}
            <div className="rounded-xl bg-slate-950/70 border border-slate-800/80 p-4">
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-medium text-slate-400">Major Osteoporotic Fracture (10-Yr)</span>
                <span className="text-xs font-mono text-cyan-400">Target &lt; {fraxResult.interventionThreshold}%</span>
              </div>
              <div className="flex items-baseline space-x-2">
                <span className={`text-3xl sm:text-4xl font-extrabold ${
                  fraxResult.tenYearMajorFractureRisk >= fraxResult.interventionThreshold
                    ? 'text-rose-400'
                    : fraxResult.tenYearMajorFractureRisk >= 10
                    ? 'text-amber-400'
                    : 'text-emerald-400'
                }`}>
                  {fraxResult.tenYearMajorFractureRisk}%
                </span>
                <span className="text-xs text-slate-500">probability</span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-slate-800 h-2.5 rounded-full mt-3 overflow-hidden relative">
                <div
                  className={`h-full transition-all duration-500 rounded-full ${
                    fraxResult.tenYearMajorFractureRisk >= fraxResult.interventionThreshold
                      ? 'bg-rose-500'
                      : fraxResult.tenYearMajorFractureRisk >= 10
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.min(fraxResult.tenYearMajorFractureRisk * 2.5, 100)}%` }}
                />
                {/* Intervention line marker */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-yellow-400"
                  style={{ left: `${Math.min(fraxResult.interventionThreshold * 2.5, 100)}%` }}
                  title={`Intervention threshold: ${fraxResult.interventionThreshold}%`}
                />
              </div>

              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0%</span>
                <span className="text-amber-400">Threshold: {fraxResult.interventionThreshold}%</span>
                <span>40%+</span>
              </div>
            </div>

            {/* Hip Fracture Risk Card */}
            <div className="rounded-xl bg-slate-950/70 border border-slate-800/80 p-4">
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-medium text-slate-400">Hip Fracture (10-Yr)</span>
                <span className="text-xs font-mono text-rose-400">BHOF Threshold: 3.0%</span>
              </div>
              <div className="flex items-baseline space-x-2">
                <span className={`text-3xl sm:text-4xl font-extrabold ${
                  fraxResult.tenYearHipFractureRisk >= 3.0
                    ? 'text-rose-400'
                    : fraxResult.tenYearHipFractureRisk >= 1.5
                    ? 'text-amber-400'
                    : 'text-emerald-400'
                }`}>
                  {fraxResult.tenYearHipFractureRisk}%
                </span>
                <span className="text-xs text-slate-500">probability</span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-slate-800 h-2.5 rounded-full mt-3 overflow-hidden relative">
                <div
                  className={`h-full transition-all duration-500 rounded-full ${
                    fraxResult.tenYearHipFractureRisk >= 3.0
                      ? 'bg-rose-500'
                      : fraxResult.tenYearHipFractureRisk >= 1.5
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.min(fraxResult.tenYearHipFractureRisk * 10, 100)}%` }}
                />
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-rose-400"
                  style={{ left: `${Math.min(3.0 * 10, 100)}%` }}
                  title="Hip fracture high-risk threshold: 3.0%"
                />
              </div>

              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0%</span>
                <span className="text-rose-400">BHOF High: 3.0%</span>
                <span>10%+</span>
              </div>
            </div>
          </div>

          {/* Clinical Recommendation Box */}
          <div className="rounded-xl bg-slate-950 border border-slate-800 p-4">
            <div className="flex items-start space-x-3">
              <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="text-xs font-semibold text-cyan-300 uppercase tracking-wide">
                  Clinical Management Recommendation:
                </div>
                <p className="text-sm text-slate-200 font-medium">
                  {fraxResult.recommendation}
                </p>

                {fraxResult.rationale.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-slate-800/80">
                    <span className="text-xs text-slate-400 font-medium">Key Contributing Risk Factors:</span>
                    <ul className="mt-1 space-y-1">
                      {fraxResult.rationale.map((r, idx) => (
                        <li key={idx} className="text-xs text-slate-300 flex items-start space-x-1.5">
                          <span className="text-cyan-400 mt-0.5">•</span>
                          <span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2 pt-2 border-t border-slate-800">
              {fraxResult.riskCategory === 'intermediate' && !patient.hasBMD && (
                <button
                  onClick={onNavigateToDexa}
                  className="text-xs px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30 font-medium transition"
                >
                  Configure DEXA BMD Scan →
                </button>
              )}
              {(fraxResult.riskCategory === 'high' || fraxResult.riskCategory === 'very_high') && (
                <button
                  onClick={onNavigateToPharma}
                  className="text-xs px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/30 font-medium transition"
                >
                  View First-Line Pharmacotherapy Options →
                </button>
              )}
            </div>
          </div>
        </div>

        {/* NOGG Treatment Threshold Graph */}
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">NOGG Age vs. Risk Plot</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Patient position vs. age-specific treatment threshold (UK/US NOGG curves)
            </p>

            {/* SVG Visualizer */}
            <div className="bg-slate-950 rounded-xl p-3 border border-slate-800">
              <svg viewBox="0 0 260 160" className="w-full h-auto">
                {/* Background grid */}
                <line x1="30" y1="20" x2="250" y2="20" stroke="#1e293b" strokeDasharray="2,2" />
                <line x1="30" y1="60" x2="250" y2="60" stroke="#1e293b" strokeDasharray="2,2" />
                <line x1="30" y1="100" x2="250" y2="100" stroke="#1e293b" strokeDasharray="2,2" />
                <line x1="30" y1="140" x2="250" y2="140" stroke="#334155" />
                <line x1="30" y1="20" x2="30" y2="140" stroke="#334155" />

                {/* Y-axis labels (% MOF risk) */}
                <text x="25" y="24" fill="#64748b" fontSize="8" textAnchor="end">35%</text>
                <text x="25" y="64" fill="#64748b" fontSize="8" textAnchor="end">20%</text>
                <text x="25" y="104" fill="#64748b" fontSize="8" textAnchor="end">10%</text>
                <text x="25" y="142" fill="#64748b" fontSize="8" textAnchor="end">0%</text>

                {/* X-axis labels (Age) */}
                <text x="30" y="152" fill="#64748b" fontSize="8" textAnchor="middle">40</text>
                <text x="85" y="152" fill="#64748b" fontSize="8" textAnchor="middle">55</text>
                <text x="140" y="152" fill="#64748b" fontSize="8" textAnchor="middle">70</text>
                <text x="195" y="152" fill="#64748b" fontSize="8" textAnchor="middle">80</text>
                <text x="245" y="152" fill="#64748b" fontSize="8" textAnchor="middle">90</text>

                {/* High risk threshold area */}
                <path
                  d="M 30,120 L 74,115 L 118,105 L 162,90 L 206,75 L 250,55 L 250,20 L 30,20 Z"
                  fill="rgba(244, 63, 94, 0.08)"
                />

                {/* Treatment threshold curve (yellow line) */}
                <path
                  d="M 30,120 L 74,115 L 118,105 L 162,90 L 206,75 L 250,55"
                  fill="none"
                  stroke="#eab308"
                  strokeWidth="2"
                  strokeDasharray="4,2"
                />

                {/* Patient marker point */}
                {(() => {
                  const boundedAge = Math.min(Math.max(patient.age, 40), 90);
                  const xPos = 30 + ((boundedAge - 40) / 50) * 220;
                  const clampedMof = Math.min(Math.max(fraxResult.tenYearMajorFractureRisk, 0), 35);
                  const yPos = 140 - (clampedMof / 35) * 120;
                  const isAbove = fraxResult.tenYearMajorFractureRisk >= fraxResult.interventionThreshold;

                  return (
                    <g>
                      <circle
                        cx={xPos}
                        cy={yPos}
                        r="8"
                        fill={isAbove ? '#f43f5e' : '#10b981'}
                        opacity="0.25"
                      />
                      <circle
                        cx={xPos}
                        cy={yPos}
                        r="4"
                        fill={isAbove ? '#f43f5e' : '#10b981'}
                        stroke="#ffffff"
                        strokeWidth="1.5"
                      />
                      <text
                        x={xPos}
                        y={Math.max(yPos - 8, 14)}
                        fill="#ffffff"
                        fontSize="8"
                        fontWeight="bold"
                        textAnchor="middle"
                      >
                        Patient ({fraxResult.tenYearMajorFractureRisk}%)
                      </text>
                    </g>
                  );
                })()}
              </svg>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-xs space-y-1.5 text-slate-400">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-0.5 bg-yellow-400 inline-block" />
              <span>NOGG Intervention Threshold Line</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80 inline-block" />
              <span>Treatment Indicated (Above Line or Prior Fracture)</span>
            </div>
          </div>
        </div>
      </div>

      {/* FRAX Questionnaire Form */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex items-center justify-between mb-6 border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-base font-bold text-white">Patient Profile & Clinical Variables</h3>
            <p className="text-xs text-slate-400">Modify patient factors to dynamically re-evaluate fracture vulnerability</p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 px-3 py-1 rounded-md border border-cyan-900">
            BMI: {fraxResult.bmi} kg/m²
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Demographic: Age */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Age (years)
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="number"
                min="40"
                max="90"
                value={patient.age}
                onChange={(e) => updateField('age', Math.min(Math.max(Number(e.target.value) || 40, 40), 90))}
                className="w-24 px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white font-mono text-sm focus:border-cyan-500 focus:outline-none"
              />
              <input
                type="range"
                min="40"
                max="90"
                value={patient.age}
                onChange={(e) => updateField('age', Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
            </div>
            <span className="text-[11px] text-slate-500 mt-1 block">Valid range: 40 - 90</span>
          </div>

          {/* Gender */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Sex Assigned at Birth
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => updateField('gender', 'female')}
                className={`py-2 px-3 rounded-lg text-xs font-semibold border transition ${
                  patient.gender === 'female'
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                }`}
              >
                Female
              </button>
              <button
                type="button"
                onClick={() => updateField('gender', 'male')}
                className={`py-2 px-3 rounded-lg text-xs font-semibold border transition ${
                  patient.gender === 'male'
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                }`}
              >
                Male
              </button>
            </div>
          </div>

          {/* Weight */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Weight (kg)
            </label>
            <input
              type="number"
              min="30"
              max="160"
              value={patient.weightKg}
              onChange={(e) => updateField('weightKg', Math.max(Number(e.target.value) || 30, 25))}
              className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white font-mono text-sm focus:border-cyan-500 focus:outline-none"
            />
            <span className="text-[11px] text-slate-500 mt-1 block">Low weight (&lt;57kg) elevates hip risk</span>
          </div>

          {/* Height */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Height (cm)
            </label>
            <input
              type="number"
              min="120"
              max="215"
              value={patient.heightCm}
              onChange={(e) => updateField('heightCm', Math.max(Number(e.target.value) || 120, 100))}
              className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white font-mono text-sm focus:border-cyan-500 focus:outline-none"
            />
            <span className="text-[11px] text-slate-500 mt-1 block">Calculated BMI: {fraxResult.bmi}</span>
          </div>
        </div>

        {/* Clinical Binary Risk Checkboxes */}
        <div className="mt-8 pt-6 border-t border-slate-800">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Validated Clinical Risk Factors
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Previous Fracture */}
            <label className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition ${
              patient.previousFracture ? 'bg-cyan-950/40 border-cyan-700/60' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}>
              <input
                type="checkbox"
                checked={patient.previousFracture}
                onChange={(e) => updateField('previousFracture', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Prior Fragility Fracture</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">Fracture occurring spontaneously or from fall standing height</span>
              </div>
            </label>

            {/* Parent Hip Fracture */}
            <label className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition ${
              patient.parentHipFracture ? 'bg-cyan-950/40 border-cyan-700/60' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}>
              <input
                type="checkbox"
                checked={patient.parentHipFracture}
                onChange={(e) => updateField('parentHipFracture', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Parental Hip Fracture</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">Mother or father sustained fractured hip</span>
              </div>
            </label>

            {/* Glucocorticoids */}
            <label className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition ${
              patient.glucocorticoids ? 'bg-cyan-950/40 border-cyan-700/60' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}>
              <input
                type="checkbox"
                checked={patient.glucocorticoids}
                onChange={(e) => updateField('glucocorticoids', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Oral Glucocorticoids</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">≥5 mg/day prednisolone for ≥3 months (past or current)</span>
              </div>
            </label>

            {/* Current Smoker */}
            <label className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition ${
              patient.currentSmoker ? 'bg-cyan-950/40 border-cyan-700/60' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}>
              <input
                type="checkbox"
                checked={patient.currentSmoker}
                onChange={(e) => updateField('currentSmoker', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Current Tobacco Smoker</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">Active daily cigarette or tobacco smoking</span>
              </div>
            </label>

            {/* Rheumatoid Arthritis */}
            <label className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition ${
              patient.rheumatoidArthritis ? 'bg-cyan-950/40 border-cyan-700/60' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}>
              <input
                type="checkbox"
                checked={patient.rheumatoidArthritis}
                onChange={(e) => updateField('rheumatoidArthritis', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Rheumatoid Arthritis</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">Confirmed diagnosis of inflammatory rheumatoid disease</span>
              </div>
            </label>

            {/* Secondary Osteoporosis */}
            <label className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition ${
              patient.secondaryOsteoporosis ? 'bg-cyan-950/40 border-cyan-700/60' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}>
              <input
                type="checkbox"
                checked={patient.secondaryOsteoporosis}
                onChange={(e) => updateField('secondaryOsteoporosis', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Secondary Osteoporosis</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">Type 1 DM, hypogonadism, malabsorption, hyperthyroidism</span>
              </div>
            </label>

            {/* Alcohol >= 3 units */}
            <label className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition ${
              patient.alcoholThreeOrMore ? 'bg-cyan-950/40 border-cyan-700/60' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
            }`}>
              <input
                type="checkbox"
                checked={patient.alcoholThreeOrMore}
                onChange={(e) => updateField('alcoholThreeOrMore', e.target.checked)}
                className="mt-1 w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <div>
                <span className="text-xs font-semibold text-slate-200 block">Alcohol ≥ 3 Units / Day</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">Approx. 3 standard drinks (1 pint beer or 1 large glass wine)</span>
              </div>
            </label>
          </div>
        </div>

        {/* Femoral Neck BMD Inclusion Slider */}
        <div className="mt-8 pt-6 border-t border-slate-800 bg-slate-950/50 p-5 rounded-xl border border-slate-800/80">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="hasBmdToggle"
                checked={patient.hasBMD}
                onChange={(e) => updateField('hasBMD', e.target.checked)}
                className="w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500 cursor-pointer"
              />
              <label htmlFor="hasBmdToggle" className="cursor-pointer">
                <span className="text-sm font-bold text-white block">Include Femoral Neck BMD (T-Score)</span>
                <span className="text-xs text-slate-400 block">Improves FRAX calibration accuracy from dual-energy X-ray absorptiometry</span>
              </label>
            </div>

            {patient.hasBMD && (
              <div className="flex items-center space-x-3">
                <span className="text-xs text-slate-400">T-Score:</span>
                <span className={`px-2.5 py-1 rounded-md font-mono text-sm font-bold ${
                  patient.femoralNeckTScore <= -2.5
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                    : patient.femoralNeckTScore <= -1.0
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                }`}>
                  {patient.femoralNeckTScore > 0 ? `+${patient.femoralNeckTScore.toFixed(1)}` : patient.femoralNeckTScore.toFixed(1)}
                </span>
              </div>
            )}
          </div>

          {patient.hasBMD && (
            <div className="space-y-3 pt-2">
              <input
                type="range"
                min="-4.5"
                max="1.5"
                step="0.1"
                value={patient.femoralNeckTScore}
                onChange={(e) => updateField('femoralNeckTScore', parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-400">
                <span className="text-rose-400">-4.5 (Severe Osteoporosis)</span>
                <span className="text-amber-400">-2.5 (Osteoporosis Threshold)</span>
                <span className="text-yellow-400">-1.0 (Osteopenia)</span>
                <span className="text-emerald-400">+1.5 (High Density)</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
