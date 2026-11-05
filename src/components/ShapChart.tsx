import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { ShapFeature } from '../api/types';

interface ShapChartProps {
  shapFeatures: ShapFeature[];
}

export const ShapChart: React.FC<ShapChartProps> = ({ shapFeatures }) => {
  const chartData = shapFeatures.map((feat) => ({
    name: feat.name,
    value: feat.value,
    absVal: Math.abs(feat.value),
    color: feat.value >= 0 ? '#10B981' : '#EF4444',
  }));

  return (
    <div
      className="w-full h-64"
      role="img"
      aria-label="Diverging bar chart showing SHAP feature attributions for radiomics parameters"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 8, right: 24, left: 24, bottom: 8 }}
        >
          <XAxis
            type="number"
            domain={[-0.8, 0.8]}
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#94a3b8"
            fontSize={11}
            width={150}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine x={0} stroke="#475569" strokeWidth={1.5} />
          <Tooltip
            cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                const isPositive = data.value >= 0;
                return (
                  <div className="bg-slate-900 border border-slate-700 p-2.5 rounded-xl shadow-xl text-xs space-y-1">
                    <span className="font-bold text-white block">{data.name}</span>
                    <div className="flex items-center space-x-1.5 font-mono">
                      <span className="text-slate-400">SHAP Value:</span>
                      <span className={`font-semibold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {data.value > 0 ? `+${data.value}` : data.value}
                      </span>
                    </div>
                    <span className="text-[10px] text-slate-400 block">
                      {isPositive ? 'Pushes prediction toward top class' : 'Counteracts / lowers confidence in top class'}
                    </span>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar
            dataKey="value"
            isAnimationActive={true}
            animationDuration={1000}
            radius={4}
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.color}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
