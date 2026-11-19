import React, { useState, useEffect } from 'react';
import { RefreshCw, Play, Pause, RotateCcw, Zap, Info, ShieldCheck } from 'lucide-react';

export const BoneRemodelingExplorer: React.FC = () => {
  const [currentPhase, setCurrentPhase] = useState<number>(1); // 0 to 4
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [scenario, setScenario] = useState<'healthy' | 'postmenopausal' | 'bisphosphonate' | 'anabolic'>('postmenopausal');

  const phases = [
    {
      id: 0,
      name: '1. Quiescence & Mechanical Sensing',
      duration: 'Ongoing',
      description: 'Osteocytes detect mechanical microdamage via fluid shear stress in canaliculi and release RANKL and sclerostin signals.',
      osteoclastsActive: false,
      osteoblastsActive: false,
    },
    {
      id: 1,
      name: '2. Activation & Osteoclast Resorption',
      duration: '2 - 4 Weeks',
      description: 'Multinucleated osteoclasts bind to bone, seal a resorption pit (Howship lacuna), and secrete hydrochloric acid and Cathepsin K to dissolve mineral and collagen.',
      osteoclastsActive: true,
      osteoblastsActive: false,
    },
    {
      id: 2,
      name: '3. Reversal Phase',
      duration: '1 - 2 Weeks',
      description: 'Osteoclasts undergo apoptosis. Mononuclear cells smooth the resorption cavity and prepare the cement line for osteoblast migration.',
      osteoclastsActive: false,
      osteoblastsActive: false,
    },
    {
      id: 3,
      name: '4. Osteoblast Formation & Osteoid Secretion',
      duration: '3 - 4 Months',
      description: 'Teams of osteoblasts synthesize type I collagen and bone matrix proteins (osteoid), refilling the excavated lacuna.',
      osteoclastsActive: false,
      osteoblastsActive: true,
    },
    {
      id: 4,
      name: '5. Mineralization & Matrix Maturation',
      duration: 'Several Months',
      description: 'Calcium and phosphate precipitate as hydroxyapatite crystals. Remaining osteoblasts either become embedded as osteocytes or differentiate into lining cells.',
      osteoclastsActive: false,
      osteoblastsActive: false,
    },
  ];

  // Auto-play animation timer
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentPhase((prev) => (prev + 1) % phases.length);
    }, 4500);
    return () => clearInterval(interval);
  }, [isPlaying, phases.length]);

  const activePhaseInfo = phases[currentPhase];

  // Net bone balance calculation based on scenario
  const scenarioConfig = {
    healthy: {
      title: 'Normal Premenopausal Adult',
      resorptionDepth: 40,
      formationHeight: 40,
      netBalance: 'Zero Net Deficit (Coupled 1:1)',
      balanceColor: 'text-emerald-400',
      ranklOpgRatio: 'Balanced (OPG decoy inhibits excess RANKL)',
      estrogen: 'Normal physiological',
    },
    postmenopausal: {
      title: 'Estrogen Deficiency (Postmenopausal)',
      resorptionDepth: 65,
      formationHeight: 38,
      netBalance: '-27 µm Net Deficit per Remodeling Cycle',
      balanceColor: 'text-rose-400',
      ranklOpgRatio: 'High RANKL / Low OPG (Increased osteoclast survival)',
      estrogen: 'Severely Depleted',
    },
    bisphosphonate: {
      title: 'Bisphosphonate / Denosumab Therapy',
      resorptionDepth: 20,
      formationHeight: 25,
      netBalance: '+5 µm Net Gain / Stabilization',
      balanceColor: 'text-cyan-400',
      ranklOpgRatio: 'Suppressed osteoclast resorption',
      estrogen: 'Deficient, but pharmacologically rescued',
    },
    anabolic: {
      title: 'Anabolic Therapy (Teriparatide / Romosozumab)',
      resorptionDepth: 35,
      formationHeight: 65,
      netBalance: '+30 µm Marked Net Bone Apposition',
      balanceColor: 'text-teal-300',
      ranklOpgRatio: 'Wnt-beta catenin pathway activated',
      estrogen: 'Direct stimulation of osteoblastogenesis',
    },
  }[scenario];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center space-x-2">
              <RefreshCw className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">Bone Multicellular Unit (BMU) Remodeling Explorer</h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Molecular balance: Osteoclasts (resorption) vs. Osteoblasts (formation) across disease and therapy states
            </p>
          </div>

          {/* Animation controls */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-md transition"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isPlaying ? 'Pause' : 'Play Simulation'}</span>
            </button>
            <button
              onClick={() => setCurrentPhase(0)}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 border border-slate-700 transition"
              title="Reset to Phase 1"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Clinical Scenario Switcher */}
        <div className="flex flex-wrap gap-2 pt-2 border-t border-slate-800">
          {(['healthy', 'postmenopausal', 'bisphosphonate', 'anabolic'] as const).map((sc) => (
            <button
              key={sc}
              onClick={() => setScenario(sc)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                scenario === sc
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50'
                  : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200'
              }`}
            >
              {sc === 'healthy' ? 'Healthy Adult' : sc === 'postmenopausal' ? 'Estrogen Deficient (Osteoporosis)' : sc === 'bisphosphonate' ? 'On Antiresorptive' : 'On Anabolic Agent'}
            </button>
          ))}
        </div>
      </div>

      {/* Main Simulation Stage & Step Indicators */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Interactive BMU Animation Stage */}
        <div className="lg:col-span-8 rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-slate-400 mb-3">
              <span className="font-bold text-cyan-300">{activePhaseInfo.name}</span>
              <span className="font-mono text-slate-400">Biological Duration: {activePhaseInfo.duration}</span>
            </div>

            {/* Canvas/SVG Remodeling Unit */}
            <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 relative overflow-hidden">
              <svg viewBox="0 0 460 220" className="w-full h-auto">
                {/* Background bone matrix (mineralized lamellar bone) */}
                <rect x="0" y="100" width="460" height="120" fill="#1e293b" />
                {/* Bone lamellae striations */}
                <line x1="0" y1="130" x2="460" y2="130" stroke="#334155" strokeDasharray="6,4" />
                <line x1="0" y1="160" x2="460" y2="160" stroke="#334155" strokeDasharray="6,4" />
                <line x1="0" y1="190" x2="460" y2="190" stroke="#334155" strokeDasharray="6,4" />

                {/* Embedded Osteocytes inside mineralized matrix */}
                <g>
                  <circle cx="60" cy="140" r="5" fill="#0284c7" />
                  <circle cx="120" cy="175" r="5" fill="#0284c7" />
                  <circle cx="340" cy="145" r="5" fill="#0284c7" />
                  <circle cx="410" cy="180" r="5" fill="#0284c7" />
                  <text x="60" y="155" fill="#94a3b8" fontSize="7" textAnchor="middle">Osteocyte</text>
                </g>

                {/* Phase-specific bone surface cavity and cellular machinery */}
                {(() => {
                  const resDepth = scenarioConfig.resorptionDepth;
                  const formH = scenarioConfig.formationHeight;

                  if (currentPhase === 0) {
                    // Quiescent flat surface with bone lining cells
                    return (
                      <g>
                        <line x1="0" y1="100" x2="460" y2="100" stroke="#94a3b8" strokeWidth="3" />
                        {/* Bone lining cells */}
                        {[40, 100, 160, 220, 280, 340, 400].map((x) => (
                          <ellipse key={x} cx={x} cy="96" rx="14" ry="4" fill="#38bdf8" opacity="0.8" />
                        ))}
                        <text x="230" y="75" fill="#38bdf8" fontSize="10" textAnchor="middle" fontWeight="bold">
                          Flat Bone Lining Cells (Quiescent Surface)
                        </text>
                      </g>
                    );
                  }

                  if (currentPhase === 1) {
                    // Resorption pit with active Osteoclast
                    return (
                      <g>
                        {/* Resorption Howship Lacuna cavity */}
                        <path
                          d={`M 130,100 Q 230,${100 + resDepth} 330,100 L 460,100 L 460,100 L 0,100 Z`}
                          fill="#0f172a"
                        />
                        {/* Active multinucleated osteoclast */}
                        <path
                          d="M 170,95 Q 230,45 290,95 Q 260,105 230,105 Q 200,105 170,95 Z"
                          fill="#ef4444"
                          stroke="#f87171"
                          strokeWidth="2"
                        />
                        {/* Multiple nuclei inside osteoclast */}
                        <circle cx="205" cy="80" r="4" fill="#7f1d1d" />
                        <circle cx="225" cy="74" r="4" fill="#7f1d1d" />
                        <circle cx="245" cy="79" r="4" fill="#7f1d1d" />
                        <circle cx="225" cy="88" r="4" fill="#7f1d1d" />

                        {/* Ruffled border & acid secretion */}
                        <path
                          d="M 180,98 L 185,104 L 190,98 L 195,104 L 200,98 L 205,104 L 210,98 L 215,104 L 220,98 L 225,104 L 230,98 L 235,104 L 240,98 L 245,104 L 250,98 L 255,104 L 260,98 L 265,104 L 270,98 L 275,104 L 280,98"
                          fill="none"
                          stroke="#fbbf24"
                          strokeWidth="1.5"
                        />
                        <text x="230" y="40" fill="#f87171" fontSize="10" fontWeight="bold" textAnchor="middle">
                          Active Multinucleated Osteoclast (HCl + Cathepsin K)
                        </text>
                      </g>
                    );
                  }

                  if (currentPhase === 2) {
                    // Reversal phase: cleared pit, macrophages smoothing cement line
                    return (
                      <g>
                        <path
                          d={`M 130,100 Q 230,${100 + resDepth} 330,100 L 330,100`}
                          fill="none"
                          stroke="#f59e0b"
                          strokeWidth="3"
                        />
                        {/* Reversal macrophage cells */}
                        <circle cx="200" cy={100 + resDepth * 0.7} r="7" fill="#fbbf24" />
                        <circle cx="230" cy={100 + resDepth * 0.8} r="8" fill="#fbbf24" />
                        <circle cx="260" cy={100 + resDepth * 0.7} r="7" fill="#fbbf24" />
                        <text x="230" y="70" fill="#fbbf24" fontSize="10" fontWeight="bold" textAnchor="middle">
                          Reversal Mononuclear Cells (Laying Cement Line)
                        </text>
                      </g>
                    );
                  }

                  if (currentPhase === 3) {
                    // Formation: osteoblasts secreting osteoid
                    return (
                      <g>
                        {/* Resorption base cavity */}
                        <path
                          d={`M 130,100 Q 230,${100 + resDepth} 330,100`}
                          fill="none"
                          stroke="#64748b"
                          strokeWidth="2"
                          strokeDasharray="2,2"
                        />
                        {/* Unmineralized osteoid layer (pink/orange) */}
                        <path
                          d={`M 130,100 Q 230,${100 + resDepth} 330,100 Q 230,${100 + resDepth - formH} 130,100 Z`}
                          fill="#f472b6"
                          opacity="0.6"
                        />
                        {/* Team of cuboidal Osteoblasts on top */}
                        {[160, 185, 210, 235, 260, 285].map((x) => (
                          <rect
                            key={x}
                            x={x}
                            y={100 + resDepth - formH - 14}
                            width="16"
                            height="14"
                            rx="3"
                            fill="#10b981"
                            stroke="#34d399"
                            strokeWidth="1"
                          />
                        ))}
                        <text x="230" y="55" fill="#34d399" fontSize="10" fontWeight="bold" textAnchor="middle">
                          Cuboidal Osteoblasts Secreting Osteoid (Collagen Type I)
                        </text>
                      </g>
                    );
                  }

                  // Phase 4: Mineralization
                  return (
                    <g>
                      {/* Remodeled new bone (mineralized) */}
                      <path
                        d={`M 130,100 Q 230,${100 + resDepth} 330,100 Q 230,${100 + resDepth - formH} 130,100 Z`}
                        fill="#38bdf8"
                        opacity="0.7"
                      />
                      {/* Newly formed embedded osteocyte */}
                      <circle cx="230" cy={100 + resDepth - formH / 2} r="5" fill="#0284c7" />
                      <line x1="0" y1="100" x2="460" y2="100" stroke="#94a3b8" strokeWidth="2" />
                      <text x="230" y="65" fill="#38bdf8" fontSize="10" fontWeight="bold" textAnchor="middle">
                        Hydroxyapatite Mineralization Complete
                      </text>
                    </g>
                  );
                })()}

                {/* Net Balance Annotation */}
                <rect x="10" y="10" width="220" height="26" rx="6" fill="#030712" stroke="#334155" strokeWidth="1" />
                <text x="20" y="27" fill="#ffffff" fontSize="9" fontWeight="bold">
                  BMU Status: {scenarioConfig.title}
                </text>
              </svg>
            </div>
          </div>

          {/* Phase Sequence Buttons */}
          <div className="grid grid-cols-5 gap-2 mt-4 pt-3 border-t border-slate-800">
            {phases.map((p, idx) => (
              <button
                key={p.id}
                onClick={() => {
                  setCurrentPhase(idx);
                  setIsPlaying(false);
                }}
                className={`p-2 rounded-xl text-center text-xs transition border ${
                  currentPhase === idx
                    ? 'bg-cyan-950/60 border-cyan-500/60 text-cyan-300'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                <div className="font-bold">Step {idx + 1}</div>
                <div className="text-[10px] truncate mt-0.5">{p.name.split('. ')[1]}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Biological Mechanism & Scenario Details */}
        <div className="lg:col-span-4 rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <span className="text-xs text-slate-400 uppercase tracking-wider block">Remodeling Balance Outcome</span>
            <div className={`text-base font-bold mt-1 ${scenarioConfig.balanceColor}`}>
              {scenarioConfig.netBalance}
            </div>
          </div>

          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 block mb-0.5 font-medium">RANKL / OPG Signaling Ratio:</span>
              <p className="text-slate-200 font-semibold">{scenarioConfig.ranklOpgRatio}</p>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 block mb-0.5 font-medium">Estrogen / Endocrine State:</span>
              <p className="text-slate-200 font-semibold">{scenarioConfig.estrogen}</p>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
              <span className="text-cyan-300 font-semibold block flex items-center space-x-1.5">
                <Info className="w-4 h-4" />
                <span>Phase Biological Description:</span>
              </span>
              <p className="text-slate-300 leading-relaxed text-[11px]">
                {activePhaseInfo.description}
              </p>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-cyan-950/30 border border-cyan-800/40 text-[11px] text-cyan-200 leading-relaxed">
            <strong>Physiological Pearl:</strong> At any given time, ~1 to 2 million BMUs are actively remodeling bone across the adult human skeleton, renewing ~10% of total skeletal mass per year.
          </div>
        </div>
      </div>
    </div>
  );
};
