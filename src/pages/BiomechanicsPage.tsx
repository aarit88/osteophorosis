import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Compass,
  Activity,
  AlertTriangle,
  RotateCw,
  ShieldAlert,
  Sliders,
  CheckCircle2,
  FileText,
} from 'lucide-react';
import { runBiomechanicsGeometry, runBiomechanicsMirels } from '../api/client';
import { BiomechanicsGeometryResponse, BiomechanicsMirelsResponse } from '../api/types';

export const BiomechanicsPage: React.FC = () => {
  // Geometry State
  const [phantomType, setPhantomType] = useState<string>('intact_femur');
  const [voxelSize, setVoxelSize] = useState<number>(0.5);
  const [geomData, setGeomData] = useState<BiomechanicsGeometryResponse | null>(null);
  const [geomLoading, setGeomLoading] = useState<boolean>(false);
  const [geomError, setGeomError] = useState<string | null>(null);

  // Mirels State
  const [site, setSite] = useState<number>(2); // lower extremity
  const [pain, setPain] = useState<number>(2); // moderate
  const [nature, setNature] = useState<number>(3); // lytic
  const [size, setSize] = useState<number>(2); // 1/3 - 2/3
  const [mirelsData, setMirelsData] = useState<BiomechanicsMirelsResponse | null>(null);
  const [mirelsLoading, setMirelsLoading] = useState<boolean>(false);
  const [mirelsError, setMirelsError] = useState<string | null>(null);

  const fetchGeometry = async (type = phantomType, sizeVal = voxelSize) => {
    setGeomLoading(true);
    setGeomError(null);
    try {
      const res = await runBiomechanicsGeometry({
        phantom_type: type,
        voxel_size_mm: sizeVal,
      });
      setGeomData(res);
    } catch (err: any) {
      setGeomError(err.message || 'Failed to compute cross-sectional geometry.');
    } finally {
      setGeomLoading(false);
    }
  };

  const fetchMirels = async (s = site, p = pain, n = nature, sz = size) => {
    setMirelsLoading(true);
    setMirelsError(null);
    try {
      const res = await runBiomechanicsMirels({
        site: s,
        pain: p,
        radiographic_nature: n,
        lesion_size: sz,
      });
      setMirelsData(res);
    } catch (err: any) {
      setMirelsError(err.message || 'Failed to calculate Mirels score.');
    } finally {
      setMirelsLoading(false);
    }
  };

  useEffect(() => {
    fetchGeometry();
    fetchMirels();
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5 space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-teal-950/70 border border-teal-800 text-teal-300 text-xs font-semibold">
          <Compass className="w-3.5 h-3.5" />
          <span>Stage 28 Engine: Biomechanics & Mirels Scoring</span>
        </div>
        <h1 className="text-3xl font-black text-white tracking-tight">
          Biomechanical Geometry & Conventional Mirels Engine
        </h1>
        <p className="text-slate-400 text-sm max-w-3xl">
          Integrates deterministic cross-sectional second moment calculation (I_xx, I_yy, J) on synthetic cortical phantoms with conventional Mirels (1989) clinical fracture risk scoring logic.
        </p>
      </div>

      {/* Geometry Section */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-850 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <Compass className="w-5 h-5 text-teal-400" />
              <span>1. Cross-Sectional Geometry Engine</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Computes geometric area moments of inertia and torsion resistance on synthetic cross-sections.
            </p>
          </div>
          <button
            type="button"
            onClick={() => fetchGeometry(phantomType, voxelSize)}
            disabled={geomLoading}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white font-bold text-xs disabled:opacity-50 transition cursor-pointer self-start sm:self-auto"
          >
            {geomLoading ? <RotateCw className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
            <span>Calculate Geometry</span>
          </button>
        </div>

        {/* Geometry Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Synthetic Bone Phantom (Cross-Section)
            </label>
            <select
              value={phantomType}
              onChange={(e) => {
                setPhantomType(e.target.value);
                fetchGeometry(e.target.value, voxelSize);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="intact_femur">Intact Diaphyseal Femur Baseline</option>
              <option value="mild_osteopenia">Mild Osteopenia (Endosteal Thinning)</option>
              <option value="severe_osteopenia">Severe Osteopenia (Marked Cortical Loss)</option>
              <option value="small_lytic_defect">Small Lytic Defect (&lt; 1/3 circumference)</option>
              <option value="med_lytic_defect">Moderate Lytic Defect (1/3 - 2/3 circumference)</option>
              <option value="large_lytic_defect">Large Lytic Defect (&gt; 2/3 circumference)</option>
              <option value="sclerotic_rim">Sclerotic Rim Lesion (Dense Margin)</option>
              <option value="cortical_defect_33">Cortical Defect 33% Arc Loss</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Pixel / Voxel Resolution (mm)
            </label>
            <select
              value={voxelSize}
              onChange={(e) => {
                const val = Number(e.target.value);
                setVoxelSize(val);
                fetchGeometry(phantomType, val);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value={0.5}>0.5 mm / pixel (High Precision)</option>
              <option value={1.0}>1.0 mm / pixel (Standard)</option>
            </select>
          </div>
        </div>

        {geomError && (
          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
            {geomError}
          </div>
        )}

        {geomData && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Cortical Area</span>
                <span className="text-lg font-mono font-bold text-white mt-1 block">
                  {geomData.area_mm2.toFixed(1)} <span className="text-xs text-slate-400 font-normal">mm²</span>
                </span>
                <span className="text-[10px] text-teal-300 block mt-1">
                  Retention: {(geomData.area_retention_ratio * 100).toFixed(1)}%
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Bending Inertia I_xx</span>
                <span className="text-lg font-mono font-bold text-white mt-1 block">
                  {geomData.I_xx_mm4.toLocaleString(undefined, { maximumFractionDigits: 0 })}{' '}
                  <span className="text-xs text-slate-400 font-normal">mm⁴</span>
                </span>
                <span className="text-[10px] text-teal-300 block mt-1">
                  Retention: {(geomData.bending_retention_ratio_x * 100).toFixed(1)}%
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Bending Inertia I_yy</span>
                <span className="text-lg font-mono font-bold text-white mt-1 block">
                  {geomData.I_yy_mm4.toLocaleString(undefined, { maximumFractionDigits: 0 })}{' '}
                  <span className="text-xs text-slate-400 font-normal">mm⁴</span>
                </span>
                <span className="text-[10px] text-teal-300 block mt-1">
                  Retention: {(geomData.bending_retention_ratio_y * 100).toFixed(1)}%
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Polar Torsion J</span>
                <span className="text-lg font-mono font-bold text-teal-300 mt-1 block">
                  {geomData.polar_J_mm4.toLocaleString(undefined, { maximumFractionDigits: 0 })}{' '}
                  <span className="text-xs text-slate-400 font-normal">mm⁴</span>
                </span>
                <span className="text-[10px] text-teal-300 block mt-1">
                  Retention: {(geomData.torsion_retention_ratio * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-amber-300 flex items-start space-x-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 text-amber-400 mt-0.5" />
              <span>
                <strong>Biomechanical Notice:</strong> {geomData.warning}
              </span>
            </div>
          </motion.div>
        )}
      </div>

      {/* Conventional Mirels Scoring Section */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-850 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <FileText className="w-5 h-5 text-amber-400" />
              <span>2. Conventional Mirels Clinical Scoring System (1989)</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Standard 4-parameter clinical rubric for pathologic fracture risk assessment in metastatic long bone disease.
            </p>
          </div>
          <button
            type="button"
            onClick={() => fetchMirels(site, pain, nature, size)}
            disabled={mirelsLoading}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs disabled:opacity-50 transition cursor-pointer self-start sm:self-auto"
          >
            {mirelsLoading ? <RotateCw className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
            <span>Calculate Mirels</span>
          </button>
        </div>

        {/* Mirels 4 Parameters Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Anatomic Site
            </label>
            <select
              value={site}
              onChange={(e) => {
                const val = Number(e.target.value);
                setSite(val);
                fetchMirels(val, pain, nature, size);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              <option value={1}>Upper Extremity (1 pt)</option>
              <option value={2}>Lower Extremity (2 pts)</option>
              <option value={3}>Peritrochanteric Femur (3 pts)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Pain Quality
            </label>
            <select
              value={pain}
              onChange={(e) => {
                const val = Number(e.target.value);
                setPain(val);
                fetchMirels(site, val, nature, size);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              <option value={1}>Mild / Non-weightbearing (1 pt)</option>
              <option value={2}>Moderate (2 pts)</option>
              <option value={3}>Functional / Weightbearing (3 pts)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Radiographic Nature
            </label>
            <select
              value={nature}
              onChange={(e) => {
                const val = Number(e.target.value);
                setNature(val);
                fetchMirels(site, pain, val, size);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              <option value={1}>Blastic (1 pt)</option>
              <option value={2}>Mixed (2 pts)</option>
              <option value={3}>Lytic (3 pts)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Lesion Size (Cortical Diameter)
            </label>
            <select
              value={size}
              onChange={(e) => {
                const val = Number(e.target.value);
                setSize(val);
                fetchMirels(site, pain, nature, val);
              }}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              <option value={1}>&lt; 1/3 Cortical Width (1 pt)</option>
              <option value={2}>1/3 - 2/3 Cortical Width (2 pts)</option>
              <option value={3}>&gt; 2/3 Cortical Width (3 pts)</option>
            </select>
          </div>
        </div>

        {mirelsError && (
          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
            {mirelsError}
          </div>
        )}

        {mirelsData && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-xs">Total Mirels Score</span>
                <span className="text-3xl font-black text-amber-400 mt-1 block">
                  {mirelsData.total_score} <span className="text-xs text-slate-500 font-normal">/ 12</span>
                </span>
                <span className="text-[11px] text-slate-400 block mt-1">
                  Tier: <strong className="text-white">{mirelsData.recommendation_tier}</strong>
                </span>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-xs">Empirical Fracture Risk</span>
                <span className="text-3xl font-black text-rose-400 mt-1 block">
                  {mirelsData.fracture_risk_percent}%
                </span>
                <span className="text-[11px] text-slate-400 block mt-1">
                  Based on Mirels 1989 cohort rate
                </span>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-xs">Prophylactic Fixation</span>
                <span className={`text-base font-bold mt-1 block ${
                  mirelsData.prophylactic_fixation_indicated ? 'text-rose-400' : 'text-emerald-400'
                }`}>
                  {mirelsData.prophylactic_fixation_indicated ? 'Indicated (Score ≥ 9)' : 'Observation / Radiotherapy'}
                </span>
                <span className="text-[11px] text-slate-400 block mt-1">
                  Threshold: Score ≥ 9 prompts surgical consultation
                </span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2">
              <span className="font-bold text-slate-200 block">Clinical Guidance:</span>
              <p className="text-slate-300 leading-relaxed">{mirelsData.clinical_guidance}</p>
              <div className="pt-2 border-t border-slate-850 text-[11px] text-slate-400 italic">
                {mirelsData.disclaimer}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default BiomechanicsPage;
