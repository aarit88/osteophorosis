import React, { useState } from 'react';
import { Bone, Layers, Eye, Sliders, ShieldCheck, AlertCircle } from 'lucide-react';

export const ImagingSimulatorView: React.FC = () => {
  const [boneState, setBoneState] = useState<'healthy' | 'osteopenia' | 'osteoporosis' | 'severe'>('osteoporosis');
  const [genantGrade, setGenantGrade] = useState<number>(2); // 0, 1, 2, 3
  const [fractureType, setFractureType] = useState<'wedge' | 'biconcave' | 'crush'>('wedge');
  const [showMeasurements, setShowMeasurements] = useState<boolean>(true);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);

  // Microstructural quantitative metrics based on state
  const metrics = {
    healthy: {
      bvTv: '24.2%', // Bone Volume / Total Volume
      tbTh: '165 µm', // Trabecular Thickness
      tbSp: '620 µm', // Trabecular Separation
      connD: '6.8 /mm³', // Connectivity Density
      corticalThick: '1.8 mm',
      status: 'Intact interconnected trabecular plates',
      color: '#10b981',
    },
    osteopenia: {
      bvTv: '17.5%',
      tbTh: '135 µm',
      tbSp: '790 µm',
      connD: '4.5 /mm³',
      corticalThick: '1.3 mm',
      status: 'Early trabecular thinning and plate-to-rod conversion',
      color: '#f59e0b',
    },
    osteoporosis: {
      bvTv: '11.8%',
      tbTh: '98 µm',
      tbSp: '1,040 µm',
      connD: '2.1 /mm³',
      corticalThick: '0.8 mm',
      status: 'Marked rod perforation, fenestration, loss of connectivity',
      color: '#f97316',
    },
    severe: {
      bvTv: '7.2%',
      tbTh: '75 µm',
      tbSp: '1,420 µm',
      connD: '0.9 /mm³',
      corticalThick: '0.4 mm',
      status: 'Severe structural fragmentation; microcallus and microfractures',
      color: '#ef4444',
    },
  }[boneState];

  // Genant Grade reduction specs
  const genantSpecs = [
    { grade: 0, label: 'Grade 0: Normal', reduction: '< 20%', desc: 'Normal vertebral body height and contour.' },
    { grade: 1, label: 'Grade 1: Mild Deformity', reduction: '20 - 25%', desc: 'Mild reduction in anterior, middle, or posterior height.' },
    { grade: 2, label: 'Grade 2: Moderate Deformity', reduction: '25 - 40%', desc: 'Moderate compression; significant structural mechanical weakening.' },
    { grade: 3, label: 'Grade 3: Severe Deformity', reduction: '> 40%', desc: 'Severe vertebral collapse; frequent adjacent level fracture cascade.' },
  ][genantGrade];

  // Calculate coordinates for deformed vertebra in lateral spine diagram
  // Baseline vertebra width = 90, height = 50
  // Wedge: anterior (left) compressed
  // Biconcave: middle compressed
  // Crush: both anterior, middle, posterior compressed
  const getVertebraShape = () => {
    const w = 110;
    const hNormal = 55;
    const reductionFactor = [0, 0.22, 0.35, 0.52][genantGrade];

    let anteriorH = hNormal;
    let middleH = hNormal;
    let posteriorH = hNormal;

    if (fractureType === 'wedge') {
      anteriorH = hNormal * (1 - reductionFactor);
      middleH = hNormal * (1 - reductionFactor * 0.5);
    } else if (fractureType === 'biconcave') {
      middleH = hNormal * (1 - reductionFactor);
    } else {
      // crush
      anteriorH = hNormal * (1 - reductionFactor);
      middleH = hNormal * (1 - reductionFactor);
      posteriorH = hNormal * (1 - reductionFactor * 0.7);
    }

    return { w, anteriorH, middleH, posteriorH, hNormal };
  };

  const vShape = getVertebraShape();

  return (
    <div className="space-y-6">
      {/* Micro-CT & Trabecular Architecture Section */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <Bone className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">Trabecular Microarchitecture & Micro-CT Simulation</h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              3D Cancellous bone volume fraction, trabecular connectivity, and cortical porosity
            </p>
          </div>

          {/* Bone State Buttons */}
          <div className="flex rounded-xl bg-slate-950 p-1 border border-slate-800">
            {(['healthy', 'osteopenia', 'osteoporosis', 'severe'] as const).map((st) => (
              <button
                key={st}
                onClick={() => setBoneState(st)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold capitalize transition ${
                  boneState === st
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {st}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          {/* Micro-CT Trabecular Network Canvas/SVG */}
          <div className="lg:col-span-7 bg-slate-950 rounded-xl p-4 border border-slate-800 relative">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="font-mono">Micro-CT Cross Section (1.5 x 1.5 cm)</span>
              <span className="font-semibold text-cyan-400">Cortical Shell & Cancellous Core</span>
            </div>

            <svg viewBox="0 0 340 220" className="w-full h-auto rounded-lg">
              {/* Dark bone marrow background */}
              <rect x="0" y="0" width="340" height="220" fill="#0b1120" />

              {/* Cortical outer bone shell */}
              {(() => {
                const cThickness = {
                  healthy: 16,
                  osteopenia: 11,
                  osteoporosis: 6,
                  severe: 3,
                }[boneState];

                return (
                  <g>
                    {/* Top cortical wall */}
                    <rect x="10" y="10" width="320" height={cThickness} rx="3" fill="#cbd5e1" opacity="0.9" />
                    {/* Bottom cortical wall */}
                    <rect x="10" y={210 - cThickness} width="320" height={cThickness} rx="3" fill="#cbd5e1" opacity="0.9" />
                    {/* Left cortical wall */}
                    <rect x="10" y="10" width={cThickness} height="200" rx="3" fill="#cbd5e1" opacity="0.9" />
                    {/* Right cortical wall */}
                    <rect x={330 - cThickness} y="10" width={cThickness} height="200" rx="3" fill="#cbd5e1" opacity="0.9" />
                  </g>
                );
              })()}

              {/* Trabecular struts and plates network */}
              {(() => {
                const strutCount = { healthy: 60, osteopenia: 42, osteoporosis: 24, severe: 12 }[boneState];
                const strokeW = { healthy: 4.5, osteopenia: 3.0, osteoporosis: 1.8, severe: 1.0 }[boneState];
                const opacity = { healthy: 0.9, osteopenia: 0.8, osteoporosis: 0.7, severe: 0.5 }[boneState];

                // Seeded pattern points for trabecular lattices
                const lattice = [
                  [40, 45, 90, 70], [90, 70, 140, 50], [140, 50, 195, 75], [195, 75, 250, 40], [250, 40, 300, 60],
                  [50, 95, 105, 110], [105, 110, 160, 90], [160, 90, 220, 120], [220, 120, 280, 100],
                  [45, 150, 100, 160], [100, 160, 160, 140], [160, 140, 210, 170], [210, 170, 270, 150], [270, 150, 310, 165],
                  [90, 70, 105, 110], [140, 50, 160, 90], [195, 75, 220, 120], [250, 40, 280, 100],
                  [105, 110, 100, 160], [160, 90, 160, 140], [220, 120, 210, 170], [280, 100, 270, 150],
                  [70, 30, 90, 70], [165, 30, 160, 90], [240, 30, 250, 40],
                  [80, 190, 100, 160], [170, 190, 160, 140], [230, 190, 210, 170],
                  [50, 95, 45, 150], [300, 60, 280, 100], [280, 100, 310, 165],
                ];

                const visibleLattice = lattice.slice(0, strutCount);

                return (
                  <g>
                    {visibleLattice.map(([x1, y1, x2, y2], idx) => (
                      <line
                        key={idx}
                        x1={x1}
                        y1={y1}
                        x2={x2}
                        y2={y2}
                        stroke="#e2e8f0"
                        strokeWidth={strokeW}
                        strokeLinecap="round"
                        opacity={opacity}
                      />
                    ))}

                    {/* Nodes / Intersection plates */}
                    {visibleLattice.map(([x1, y1], idx) => (
                      <circle
                        key={`node-${idx}`}
                        cx={x1}
                        cy={y1}
                        r={strokeW * 1.3}
                        fill="#cbd5e1"
                        opacity={opacity}
                      />
                    ))}
                  </g>
                );
              })()}

              {/* Status overlay label */}
              <text x="25" y="200" fill={metrics.color} fontSize="11" fontWeight="bold">
                {metrics.status}
              </text>
            </svg>
          </div>

          {/* Quantitative Microstructural Metrics Table */}
          <div className="lg:col-span-5 space-y-3">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Histomorphometric & Micro-CT Parameters
            </h3>

            <div className="space-y-2">
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                <div>
                  <span className="text-xs font-medium text-slate-300 block">Bone Volume Fraction (BV/TV)</span>
                  <span className="text-[10px] text-slate-500">Normal young adult: 22 - 28%</span>
                </div>
                <span className="font-mono text-base font-bold text-white">{metrics.bvTv}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                <div>
                  <span className="text-xs font-medium text-slate-300 block">Trabecular Thickness (Tb.Th)</span>
                  <span className="text-[10px] text-slate-500">Thickness of individual struts</span>
                </div>
                <span className="font-mono text-base font-bold text-white">{metrics.tbTh}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                <div>
                  <span className="text-xs font-medium text-slate-300 block">Trabecular Separation (Tb.Sp)</span>
                  <span className="text-[10px] text-slate-500">Inter-trabecular marrow gap</span>
                </div>
                <span className="font-mono text-base font-bold text-white">{metrics.tbSp}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                <div>
                  <span className="text-xs font-medium text-slate-300 block">Connectivity Density (Conn.D)</span>
                  <span className="text-[10px] text-slate-500">Topological 3D branching index</span>
                </div>
                <span className="font-mono text-base font-bold text-white">{metrics.connD}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                <div>
                  <span className="text-xs font-medium text-slate-300 block">Cortical Shell Thickness</span>
                  <span className="text-[10px] text-slate-500">Peripheral load-bearing rim</span>
                </div>
                <span className="font-mono text-base font-bold text-white">{metrics.corticalThick}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Genant Semi-Quantitative (SQ) Vertebral Fracture Classification */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <Eye className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">Genant Semi-Quantitative (SQ) Vertebral Fracture Grading</h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Lateral thoracolumbar spine radiograph grading for anterior wedge, biconcave, and crush fractures
            </p>
          </div>

          {/* Grade selection buttons */}
          <div className="flex rounded-xl bg-slate-950 p-1 border border-slate-800">
            {[0, 1, 2, 3].map((g) => (
              <button
                key={g}
                onClick={() => setGenantGrade(g)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  genantGrade === g
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/50'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Grade {g}
              </button>
            ))}
          </div>
        </div>

        {/* Fracture Type Selection & Measurement Toggles */}
        <div className="flex flex-wrap items-center justify-between gap-3 mb-6 bg-slate-950/70 p-3 rounded-xl border border-slate-800">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-medium text-slate-400">Deformity Pattern:</span>
            {(['wedge', 'biconcave', 'crush'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setFractureType(t)}
                className={`px-2.5 py-1 rounded-md text-xs font-semibold capitalize transition ${
                  fractureType === t
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'bg-slate-900 text-slate-400 border border-slate-800 hover:text-slate-200'
                }`}
              >
                {t === 'wedge' ? 'Anterior Wedge' : t === 'biconcave' ? 'Biconcave (Fish)' : 'Crush (Compression)'}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2 cursor-pointer text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showMeasurements}
                onChange={(e) => setShowMeasurements(e.target.checked)}
                className="w-3.5 h-3.5 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <span>Height Calipers (Ha, Hm, Hp)</span>
            </label>

            <label className="flex items-center space-x-2 cursor-pointer text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
                className="w-3.5 h-3.5 rounded text-cyan-500 bg-slate-900 border-slate-700 accent-cyan-500"
              />
              <span>BMD Radiodensity Map</span>
            </label>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          {/* Lateral Spine Diagram */}
          <div className="lg:col-span-7 bg-slate-950 rounded-xl p-4 border border-slate-800">
            <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
              <span className="font-mono">Lateral Radiograph Projection (T11 - L2)</span>
              <span className="font-bold text-rose-400">{genantSpecs.label}</span>
            </div>

            <svg viewBox="0 0 320 280" className="w-full h-auto">
              <rect width="320" height="280" fill="#030712" />

              {/* T11 Normal Vertebra */}
              <g transform="translate(100, 20)">
                <rect x="0" y="0" width="110" height="45" rx="5" fill="#334155" stroke="#94a3b8" strokeWidth="1.5" />
                <text x="55" y="27" fill="#cbd5e1" fontSize="10" fontWeight="bold" textAnchor="middle">T11 (Normal)</text>
              </g>

              {/* Intervertebral Disc 1 */}
              <rect x="105" y="68" width="100" height="8" rx="2" fill="#0284c7" opacity="0.4" />

              {/* T12 Fractured Vertebra (Interactive) */}
              <g transform="translate(100, 80)">
                {/* Deformed shape polygon */}
                {(() => {
                  const { w, anteriorH, middleH, posteriorH } = vShape;
                  // Polygon points:
                  // Top left: (0, 0)
                  // Top middle: (w/2, dip)
                  // Top right: (w, 0)
                  // Bottom right: (w, posteriorH)
                  // Bottom middle: (w/2, middleH)
                  // Bottom left: (0, anteriorH)
                  let path = '';
                  if (fractureType === 'wedge') {
                    // Anterior wedge: left height is smaller
                    path = `M 0,${50 - anteriorH} L ${w},0 L ${w},50 L 0,50 Z`;
                  } else if (fractureType === 'biconcave') {
                    // Middle endplate depression
                    const dip = (50 - middleH) / 2;
                    path = `M 0,0 Q ${w/2},${dip} ${w},0 L ${w},50 Q ${w/2},${50 - dip} 0,50 Z`;
                  } else {
                    // Crush: both top and bottom compressed
                    path = `M 0,${50 - anteriorH} L ${w},${50 - posteriorH} L ${w},50 L 0,50 Z`;
                  }

                  return (
                    <g>
                      <path
                        d={path}
                        fill={showHeatmap ? 'url(#heatGrad)' : genantGrade > 0 ? 'rgba(244, 63, 94, 0.25)' : '#334155'}
                        stroke={genantGrade > 0 ? '#f43f5e' : '#94a3b8'}
                        strokeWidth="2"
                      />
                      <text x="55" y="30" fill="#ffffff" fontSize="10" fontWeight="bold" textAnchor="middle">
                        T12 ({genantGrade === 0 ? 'Intact' : `Grade ${genantGrade} Fx`})
                      </text>

                      {/* Measurement caliper arrows */}
                      {showMeasurements && (
                        <g>
                          {/* Anterior Height (Ha) */}
                          <line x1="-12" y1={50 - anteriorH} x2="-12" y2="50" stroke="#38bdf8" strokeWidth="1.5" />
                          <circle cx="-12" cy={50 - anteriorH} r="2" fill="#38bdf8" />
                          <circle cx="-12" cy="50" r="2" fill="#38bdf8" />
                          <text x="-16" y={50 - anteriorH / 2 + 3} fill="#38bdf8" fontSize="8" textAnchor="end">
                            Ha: {Math.round(anteriorH * 0.55)}mm
                          </text>

                          {/* Posterior Height (Hp) */}
                          <line x1={w + 12} y1="0" x2={w + 12} y2="50" stroke="#38bdf8" strokeWidth="1.5" />
                          <circle cx={w + 12} cy="0" r="2" fill="#38bdf8" />
                          <circle cx={w + 12} cy="50" r="2" fill="#38bdf8" />
                          <text x={w + 16} y="28" fill="#38bdf8" fontSize="8" textAnchor="start">
                            Hp: {Math.round(posteriorH * 0.55)}mm
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })()}
              </g>

              {/* Intervertebral Disc 2 */}
              <rect x="105" y="136" width="100" height="8" rx="2" fill="#0284c7" opacity="0.4" />

              {/* L1 Normal Vertebra */}
              <g transform="translate(100, 148)">
                <rect x="0" y="0" width="110" height="48" rx="5" fill="#334155" stroke="#94a3b8" strokeWidth="1.5" />
                <text x="55" y="28" fill="#cbd5e1" fontSize="10" fontWeight="bold" textAnchor="middle">L1 (Normal)</text>
              </g>

              {/* Intervertebral Disc 3 */}
              <rect x="105" y="200" width="100" height="8" rx="2" fill="#0284c7" opacity="0.4" />

              {/* L2 Normal Vertebra */}
              <g transform="translate(100, 212)">
                <rect x="0" y="0" width="110" height="48" rx="5" fill="#334155" stroke="#94a3b8" strokeWidth="1.5" />
                <text x="55" y="28" fill="#cbd5e1" fontSize="10" fontWeight="bold" textAnchor="middle">L2 (Normal)</text>
              </g>

              {/* SVG Gradient definition for False-Color Radiodensity Map */}
              <defs>
                <linearGradient id="heatGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#10b981" stopOpacity="0.8" />
                </linearGradient>
              </defs>
            </svg>
          </div>

          {/* Diagnostic Interpretation and Clinical Significance */}
          <div className="lg:col-span-5 space-y-4">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-xs font-semibold text-slate-400 block mb-1">Genant Semi-Quantitative Status:</span>
              <div className="text-base font-bold text-white flex items-center space-x-2">
                <span>{genantSpecs.label}</span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40">
                  {genantSpecs.reduction} reduction
                </span>
              </div>
              <p className="text-xs text-slate-300 mt-2">
                {genantSpecs.desc}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
              <div className="font-semibold text-cyan-300 flex items-center space-x-1.5">
                <AlertCircle className="w-4 h-4" />
                <span>Clinical Cascade Warning:</span>
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                Even a single Grade 1 or 2 vertebral fracture increases the risk of subsequent vertebral fractures by <strong className="text-white">4 to 5-fold</strong> over the next 12 months (the "fracture cascade").
              </p>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Up to 60-70% of osteoporotic vertebral fractures remain clinically silent / painless or mistaken for benign back strain.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
