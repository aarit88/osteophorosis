import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { ClassProbabilities } from '../api/types';

interface ProbabilityChartProps {
  probabilities: ClassProbabilities;
}

const CLASS_CONFIG = {
  metastasis: { label: 'Bone Metastasis', color: '#EF4444' },
  paget: { label: "Paget's Disease", color: '#F59E0B' },
  oi: { label: 'Osteogenesis Imp. (OI)', color: '#8B5CF6' },
  healthy: { label: 'Healthy Bone', color: '#10B981' },
};

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ probabilities }) => {
  const chartData = [
    {
      key: 'metastasis',
      name: CLASS_CONFIG.metastasis.label,
      probability: Math.round(probabilities.metastasis * 1000) / 10,
      color: CLASS_CONFIG.metastasis.color,
    },
    {
      key: 'paget',
      name: CLASS_CONFIG.paget.label,
      probability: Math.round(probabilities.paget * 1000) / 10,
      color: CLASS_CONFIG.paget.color,
    },
    {
      key: 'oi',
      name: CLASS_CONFIG.oi.label,
      probability: Math.round(probabilities.oi * 1000) / 10,
      color: CLASS_CONFIG.oi.color,
    },
    {
      key: 'healthy',
      name: CLASS_CONFIG.healthy.label,
      probability: Math.round(probabilities.healthy * 1000) / 10,
      color: CLASS_CONFIG.healthy.color,
    },
  ];

  return (
    <div
      className="w-full h-56"
      role="img"
      aria-label="Horizontal bar chart displaying predictive probability distribution across four pathology classes"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 8, right: 28, left: 24, bottom: 8 }}
        >
          <XAxis
            type="number"
            domain={[0, 100]}
            unit="%"
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#94a3b8"
            fontSize={12}
            width={140}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                  <div className="bg-slate-900 border border-slate-700 p-2.5 rounded-xl shadow-xl text-xs">
                    <span className="font-bold text-white block">{data.name}</span>
                    <span className="font-mono text-cyan-300 font-semibold">{data.probability}%</span>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar
            dataKey="probability"
            radius={[0, 6, 6, 0]}
            isAnimationActive={true}
            animationDuration={900}
            animationEasing="ease-out"
          >
            {chartData.map((entry) => (
              <Cell key={entry.key} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
