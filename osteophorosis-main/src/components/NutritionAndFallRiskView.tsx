import React, { useState } from 'react';
import { PatientProfile } from '../types/osteoporosis';
import { FOOD_CALCIUM_DATABASE } from '../data/mockData';
import { Dumbbell, Plus, Trash2, Clock, Play, Square, RotateCcw, AlertTriangle, CheckCircle2, ShieldCheck, Heart } from 'lucide-react';

interface NutritionAndFallRiskViewProps {
  patient: PatientProfile;
}

export const NutritionAndFallRiskView: React.FC<NutritionAndFallRiskViewProps> = ({ patient }) => {
  // Selected foods in dietary calcium tracker
  const [selectedItems, setSelectedItems] = useState<{ id: string; quantity: number }[]>([
    { id: 'milk_whole', quantity: 1 },
    { id: 'cheese_cheddar', quantity: 1 },
    { id: 'yogurt_greek', quantity: 1 },
  ]);

  // Timed Up and Go (TUG) stopwatch state
  const [tugSeconds, setTugSeconds] = useState<number>(0);
  const [isTugRunning, setIsTugRunning] = useState<boolean>(false);
  const [tugIntervalId, setTugIntervalId] = useState<any>(null);

  // Serum 25(OH)D slider
  const [serumVitD, setSerumVitD] = useState<number>(24); // ng/mL

  // Recommended Daily Allowance (RDA) calculation
  // Females > 50 or Males > 70: 1,200 mg. Others: 1,000 mg.
  const isHighCaRequirement = (patient.gender === 'female' && patient.age >= 50) || patient.age >= 71;
  const targetCalciumMg = isHighCaRequirement ? 1200 : 1000;
  const targetVitDIU = patient.age >= 70 ? 800 : 600; // minimum; clinical goal usually 1000-2000 IU

  // Calculate current intake from items + non-dairy baseline background (~250mg)
  const backgroundCalciumMg = 250;
  const calculatedFoodCalcium = selectedItems.reduce((acc, curr) => {
    const item = FOOD_CALCIUM_DATABASE.find((f) => f.id === curr.id);
    return acc + (item ? item.calciumMg * curr.quantity : 0);
  }, 0);

  const totalCalciumIntake = calculatedFoodCalcium + backgroundCalciumMg;
  const calciumDeficit = Math.max(0, targetCalciumMg - totalCalciumIntake);

  // Add food item
  const addItem = (id: string) => {
    const existing = selectedItems.find((i) => i.id === id);
    if (existing) {
      setSelectedItems(selectedItems.map((i) => (i.id === id ? { ...i, quantity: i.quantity + 1 } : i)));
    } else {
      setSelectedItems([...selectedItems, { id, quantity: 1 }]);
    }
  };

  const removeItem = (id: string) => {
    setSelectedItems(selectedItems.filter((i) => i.id !== id));
  };

  // Stopwatch handlers
  const startTug = () => {
    if (isTugRunning) return;
    setIsTugRunning(true);
    const startTimestamp = Date.now() - tugSeconds * 1000;
    const interval = setInterval(() => {
      setTugSeconds(parseFloat(((Date.now() - startTimestamp) / 1000).toFixed(1)));
    }, 100);
    setTugIntervalId(interval);
  };

  const stopTug = () => {
    if (!isTugRunning) return;
    setIsTugRunning(false);
    clearInterval(tugIntervalId);
  };

  const resetTug = () => {
    setIsTugRunning(false);
    clearInterval(tugIntervalId);
    setTugSeconds(0);
  };

  return (
    <div className="space-y-6">
      {/* Calcium & Vitamin D Section */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <Heart className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">Dietary Calcium & Vitamin D Optimizer</h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Personalized for {patient.gender === 'female' ? 'Female' : 'Male'}, Age {patient.age} (Target: {targetCalciumMg} mg Ca/day, {targetVitDIU}-2000 IU Vit D3/day)
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <div className="text-right">
              <span className="text-[11px] text-slate-400 block">Total Est. Daily Intake:</span>
              <span className={`font-mono text-base font-bold ${
                totalCalciumIntake >= targetCalciumMg ? 'text-emerald-400' : 'text-amber-400'
              }`}>
                {totalCalciumIntake} mg / day
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Intake Status & Supplement Calculator */}
          <div className="lg:col-span-5 space-y-4">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400">Target Calcium Allowance:</span>
                <span className="font-mono font-bold text-white">{targetCalciumMg} mg</span>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 rounded-full ${
                    totalCalciumIntake >= targetCalciumMg ? 'bg-emerald-500' : 'bg-cyan-500'
                  }`}
                  style={{ width: `${Math.min((totalCalciumIntake / targetCalciumMg) * 100, 100)}%` }}
                />
              </div>

              <div className="flex justify-between text-xs text-slate-400 pt-1">
                <span>Dietary + Baseline: {totalCalciumIntake} mg</span>
                <span>{Math.round((totalCalciumIntake / targetCalciumMg) * 100)}% of goal</span>
              </div>

              {/* Supplement Recommendation */}
              <div className="mt-3 pt-3 border-t border-slate-800/80">
                {calciumDeficit > 0 ? (
                  <div className="space-y-1">
                    <span className="text-xs font-semibold text-amber-300 block">
                      Recommended Supplemental Calcium: ~{Math.round(calciumDeficit / 100) * 100} mg/day
                    </span>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Split into doses &le; 500 mg for optimal intestinal absorption. Use Calcium Citrate if taking PPI/H2RA or elderly.
                    </p>
                  </div>
                ) : (
                  <div className="text-xs text-emerald-300 flex items-center space-x-1.5 font-medium">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Target met primarily via dietary sources! No routine Ca supplement needed.</span>
                  </div>
                )}
              </div>
            </div>

            {/* Serum 25(OH)D Evaluation */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs font-semibold text-slate-300">Serum 25-Hydroxyvitamin D:</span>
                <span className="font-mono text-sm font-bold text-cyan-400">{serumVitD} ng/mL</span>
              </div>

              <input
                type="range"
                min="8"
                max="65"
                value={serumVitD}
                onChange={(e) => setSerumVitD(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />

              <div className="flex justify-between text-[10px] text-slate-500">
                <span className="text-rose-400">&lt; 20 (Deficient)</span>
                <span className="text-amber-400">20-29 (Insufficient)</span>
                <span className="text-emerald-400">30-50 (Optimal Target)</span>
                <span className="text-slate-400">&gt; 50</span>
              </div>

              <div className="text-xs pt-1">
                {serumVitD < 20 ? (
                  <span className="text-rose-400 font-semibold block">
                    Severe Deficiency: Treat with Ergocalciferol / Cholecalciferol 50,000 IU weekly for 8 weeks.
                  </span>
                ) : serumVitD < 30 ? (
                  <span className="text-amber-400 font-semibold block">
                    Insufficiency: Supplement with Vitamin D3 1,000 - 2,000 IU daily.
                  </span>
                ) : (
                  <span className="text-emerald-400 font-semibold block">
                    Optimal Range: Maintain with 800 - 1,000 IU daily maintenance.
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Quick Dietary Food Item Selector & Current Log */}
          <div className="lg:col-span-7 space-y-4">
            <div>
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                Quick Add High-Calcium Foods
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {FOOD_CALCIUM_DATABASE.slice(0, 9).map((f) => (
                  <button
                    key={f.id}
                    onClick={() => addItem(f.id)}
                    className="p-2 rounded-lg bg-slate-950 hover:bg-cyan-950/40 border border-slate-800 hover:border-cyan-800/60 text-left transition flex items-center justify-between"
                  >
                    <div className="truncate pr-1">
                      <span className="text-xs font-semibold text-slate-200 block truncate">{f.name}</span>
                      <span className="text-[10px] text-cyan-400 font-mono">+{f.calciumMg} mg</span>
                    </div>
                    <Plus className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  </button>
                ))}
              </div>
            </div>

            {/* Current Logged Items Table */}
            <div className="rounded-xl bg-slate-950 border border-slate-800 overflow-hidden">
              <div className="px-3 py-2 bg-slate-900 border-b border-slate-800 text-xs font-semibold text-slate-300 flex justify-between items-center">
                <span>Active Daily Diet Log</span>
                <span className="text-[11px] text-slate-400">+250 mg default non-dairy food baseline</span>
              </div>
              <div className="divide-y divide-slate-800/60 max-h-48 overflow-y-auto">
                {selectedItems.map((item) => {
                  const details = FOOD_CALCIUM_DATABASE.find((f) => f.id === item.id);
                  if (!details) return null;
                  return (
                    <div key={item.id} className="px-3 py-2 flex items-center justify-between text-xs">
                      <div>
                        <span className="text-slate-200 font-medium">{details.name}</span>
                        <span className="text-slate-500 text-[10px] ml-2">({details.serving} x {item.quantity})</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className="font-mono text-cyan-400 font-semibold">{details.calciumMg * item.quantity} mg</span>
                        <button
                          onClick={() => removeItem(item.id)}
                          className="text-slate-500 hover:text-rose-400 transition"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Fall Risk & Exercise Prescription Section */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <Dumbbell className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">Fall Risk Stratification (TUG Test) & Exercise Prescription</h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Timed Up and Go physical functional assessment and spine-safe physical training guidelines
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* TUG Test Interactive Stopwatch */}
          <div className="lg:col-span-5 p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center space-x-2 text-xs font-semibold text-slate-300 mb-1">
                <Clock className="w-4 h-4 text-cyan-400" />
                <span>Timed Up and Go (TUG) Stopwatch</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Patient stands from standard armchair, walks 3 meters (10 ft) at normal pace, turns, walks back, and sits.
              </p>

              {/* Digital Timer Display */}
              <div className="text-center py-6 bg-slate-900/80 rounded-xl border border-slate-800 mt-3">
                <span className="font-mono text-4xl sm:text-5xl font-extrabold text-white tracking-wider">
                  {tugSeconds.toFixed(1)}
                  <span className="text-lg text-slate-400 font-normal ml-1">sec</span>
                </span>
              </div>

              {/* Controls */}
              <div className="flex justify-center items-center space-x-3 mt-3">
                {!isTugRunning ? (
                  <button
                    onClick={startTug}
                    className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs shadow-md transition"
                  >
                    <Play className="w-4 h-4" />
                    <span>Start Test</span>
                  </button>
                ) : (
                  <button
                    onClick={stopTug}
                    className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs shadow-md transition"
                  >
                    <Square className="w-4 h-4" />
                    <span>Stop Test</span>
                  </button>
                )}

                <button
                  onClick={resetTug}
                  className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs border border-slate-700 transition"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span>Reset</span>
                </button>
              </div>
            </div>

            {/* Test Interpretation */}
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs">
              <span className="text-slate-400 block mb-1">Functional Fall Risk Category:</span>
              {tugSeconds === 0 ? (
                <span className="text-slate-400">Execute stopwatch to calculate patient fall risk.</span>
              ) : tugSeconds < 10 ? (
                <div className="text-emerald-400 font-bold flex items-center space-x-1.5">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Normal Mobility (&lt; 10s) — Low Fall Vulnerability</span>
                </div>
              ) : tugSeconds <= 12 ? (
                <div className="text-amber-400 font-bold flex items-center space-x-1.5">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Borderline / Moderate Risk (10 - 12s) — Consider Balance Training</span>
                </div>
              ) : (
                <div className="text-rose-400 font-bold flex items-center space-x-1.5">
                  <AlertTriangle className="w-4 h-4" />
                  <span>High Fall Risk (&gt; 12s) — Physical Therapy & Assistive Device Evaluation Indicated</span>
                </div>
              )}
            </div>
          </div>

          {/* Exercise Prescription & Contraindications */}
          <div className="lg:col-span-7 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Safe / Evidence-Based Exercises */}
            <div className="p-4 rounded-xl bg-slate-950 border border-emerald-900/40 space-y-3">
              <div className="flex items-center space-x-2 text-emerald-400 text-xs font-bold uppercase tracking-wider">
                <ShieldCheck className="w-4 h-4" />
                <span>Recommended Exercises (Bone-Loading)</span>
              </div>
              <ul className="space-y-2 text-xs text-slate-300">
                <li className="flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold mt-0.5">•</span>
                  <span><strong>Progressive Resistance Training:</strong> Weighted squats, leg press, lat pull-downs to load femoral neck and spine.</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold mt-0.5">•</span>
                  <span><strong>Weight-Bearing Aerobics:</strong> Brisk walking, stair climbing, hiking (30 min 4-5x weekly).</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold mt-0.5">•</span>
                  <span><strong>Balance & Proprioception:</strong> Tai Chi, single-leg stance, tandem walking to prevent falls.</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold mt-0.5">•</span>
                  <span><strong>Back Extensor Strengthening:</strong> Prone chest lifts, bird-dog to counteract kyphosis.</span>
                </li>
              </ul>
            </div>

            {/* Contraindicated Movements */}
            <div className="p-4 rounded-xl bg-slate-950 border border-rose-900/40 space-y-3">
              <div className="flex items-center space-x-2 text-rose-400 text-xs font-bold uppercase tracking-wider">
                <AlertTriangle className="w-4 h-4" />
                <span>Contraindicated Movements (Spine Risk)</span>
              </div>
              <ul className="space-y-2 text-xs text-slate-300">
                <li className="flex items-start space-x-2">
                  <span className="text-rose-400 font-bold mt-0.5">•</span>
                  <span><strong>End-Range Forward Spinal Flexion:</strong> Traditional sit-ups, crunches, and toe-touching create anterior crush force on vertebrae.</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-rose-400 font-bold mt-0.5">•</span>
                  <span><strong>Rapid Loaded Spinal Twisting:</strong> Full-force golf or tennis swings with excessive axial torque.</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-rose-400 font-bold mt-0.5">•</span>
                  <span><strong>High-Impact Jumping in Established Osteoporosis:</strong> High jump, trampoline, horseback riding if T &le; -2.5.</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-rose-400 font-bold mt-0.5">•</span>
                  <span><strong>Lifting with Bent Back:</strong> Always hinge at the hips and bend knees; keep weights close to torso.</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
