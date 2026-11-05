import React from 'react';
import { motion } from 'framer-motion';
import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
  ResponsiveContainer,
} from 'recharts';
import { FractureRiskResult } from '../api/types';
import { ShieldAlert, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface FractureGaugeProps {
  fractureRisk: FractureRiskResult;
}

const LEVEL_CONFIG = {
  low: {
    color: '#10B981',
    label: 'Low Structural Vulnerability',
    badgeClass: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    icon: CheckCircle2,
  },
  medium: {
    color: '#F59E0B',
    label: 'Moderate Fracture Risk',
    badgeClass: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    icon: AlertTriangle,
  },
  high: {
    color: '#EF4444',
    label: 'High Impending Fracture Vulnerability',
    badgeClass: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
    icon: ShieldAlert,
  },
};

export const FractureGauge: React.FC<FractureGaugeProps> = ({ fractureRisk }) => {
  const config = LEVEL_CONFIG[fractureRisk.level] || LEVEL_CONFIG.medium;
  const RiskIcon = config.icon;
  const isHigh = fractureRisk.level === 'high';

  const data = [
    {
      name: 'Fracture Risk',
      value: fractureRisk.percent,
      fill: config.color,
    },
  ];

  return (
    <div
      className="flex flex-col items-center justify-center relative p-2"
      role="img"
      aria-label={`Fracture risk gauge showing ${fractureRisk.percent}% probability, rated as ${fractureRisk.level} risk`}
    >
      <div className="relative w-48 h-48 flex items-center justify-center">
        {/* Pulsing glow ring for high risk */}
        {isHigh && (
          <motion.div
            animate={{ scale: [0.95, 1.1, 0.95], opacity: [0.2, 0.5, 0.2] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
            className="absolute inset-0 rounded-full bg-rose-500/20 blur-md pointer-events-none"
          />
        )}

        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="72%"
            outerRadius="100%"
            barSize={14}
            data={data}
            startAngle={225}
            endAngle={-45}
          >
            <PolarAngleAxis
              type="number"
              domain={[0, 100]}
              angleAxisId={0}
              tick={false}
            />
            <RadialBar
              background={{ fill: 'rgba(255, 255, 255, 0.07)' }}
              dataKey="value"
              cornerRadius={8}
              isAnimationActive={true}
              animationDuration={1200}
            />
          </RadialBarChart>
        </ResponsiveContainer>

        {/* Center Percentage Display */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex items-baseline space-x-0.5"
          >
            <span
              className="text-4xl font-extrabold font-mono tracking-tight"
              style={{ color: config.color }}
            >
              {fractureRisk.percent}
            </span>
            <span className="text-sm font-semibold text-slate-400">%</span>
          </motion.div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mt-0.5">
            Fracture Index
          </span>
        </div>
      </div>

      {/* Risk Badge with subtle pulse */}
      <div
        className={`mt-2 flex items-center space-x-1.5 px-3 py-1 rounded-full border text-xs font-bold ${
          config.badgeClass
        } ${isHigh ? 'animate-pulse' : ''}`}
      >
        <RiskIcon className="w-3.5 h-3.5 shrink-0" />
        <span>{config.label}</span>
      </div>
    </div>
  );
};
