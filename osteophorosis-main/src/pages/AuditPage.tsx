import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { FeedbackRecord, FeedbackStats } from '../api/types';
import { getFeedbackList, getFeedbackStats } from '../api/client';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Scale, CheckCircle2, XCircle, HelpCircle, ShieldCheck, RefreshCw, Filter, FileText } from 'lucide-react';

const COLORS = {
  agree: '#10B981',
  disagree: '#EF4444',
  unsure: '#F59E0B',
};

export const AuditPage: React.FC = () => {
  const [records, setRecords] = useState<FeedbackRecord[]>([]);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [filterDecision, setFilterDecision] = useState<string>('all');

  const loadData = async () => {
    setIsLoading(true);
    try {
      const [listData, statsData] = await Promise.all([getFeedbackList(), getFeedbackStats()]);
      setRecords(listData);
      setStats(statsData);
    } catch (err) {
      console.error('Failed to load audit data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const filteredRecords = filterDecision === 'all'
    ? records
    : records.filter((r) => r.decision === filterDecision);

  const pieData = stats
    ? [
        { name: 'Agree', value: stats.agree, color: COLORS.agree },
        { name: 'Disagree', value: stats.disagree, color: COLORS.disagree },
        { name: 'Unsure', value: stats.unsure, color: COLORS.unsure },
      ].filter((d) => d.value > 0)
    : [];

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6 px-4">
      {/* Header & Regulatory Anchor */}
      <div className="space-y-3 border-b border-slate-800 pb-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center border border-cyan-500/40">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                Clinician Audit Trail & Feedback Ledger
              </h1>
              <p className="text-xs text-slate-400">
                EU AI Act Article 14 (Human Oversight) & FDA Good Machine Learning Practice (GMLP)
              </p>
            </div>
          </div>

          <button
            onClick={loadData}
            disabled={isLoading}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh Ledger</span>
          </button>
        </div>

        {/* Regulatory callout */}
        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 leading-relaxed font-sans">
          <strong>Regulatory Mandate:</strong> Article 14 of the EU AI Act specifies that high-risk AI medical devices must be designed with interfaces enabling healthcare professionals to monitor operation, identify anomalies, and intervene or reject automated predictions.
        </div>
      </div>

      {/* Aggregate Statistics Row */}
      {stats && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          {/* Metrics summary cards */}
          <div className="lg:col-span-8 grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800">
              <span className="text-[11px] text-slate-400 font-medium block">Total Reviews</span>
              <span className="text-2xl font-mono font-bold text-white mt-1 block">{stats.total}</span>
              <span className="text-[10px] text-slate-500">Recorded cases</span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900 border border-emerald-500/30">
              <span className="text-[11px] text-emerald-400 font-medium block">Clinician Concurrence</span>
              <span className="text-2xl font-mono font-bold text-emerald-300 mt-1 block">{stats.agree}</span>
              <span className="text-[10px] text-slate-500">
                {stats.total > 0 ? `${Math.round((stats.agree / stats.total) * 100)}% agreement` : '0%'}
              </span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900 border border-rose-500/30">
              <span className="text-[11px] text-rose-400 font-medium block">Disagreements</span>
              <span className="text-2xl font-mono font-bold text-rose-300 mt-1 block">{stats.disagree}</span>
              <span className="text-[10px] text-slate-500">
                {stats.total > 0 ? `${Math.round((stats.disagree / stats.total) * 100)}% discordance` : '0%'}
              </span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900 border border-amber-500/30">
              <span className="text-[11px] text-amber-400 font-medium block">Equivocal / Unsure</span>
              <span className="text-2xl font-mono font-bold text-amber-300 mt-1 block">{stats.unsure}</span>
              <span className="text-[10px] text-slate-500">Referred to panel</span>
            </div>
          </div>

          {/* Animated Donut Chart */}
          <div className="lg:col-span-4 p-4 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-between">
            <div className="w-28 h-28 relative">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    innerRadius={32}
                    outerRadius={48}
                    paddingAngle={3}
                    dataKey="value"
                    isAnimationActive={true}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="text-xs space-y-1.5 flex-1 pl-4">
              <div className="flex items-center space-x-1.5 text-emerald-400">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <span>Agree: {stats.agree}</span>
              </div>
              <div className="flex items-center space-x-1.5 text-rose-400">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                <span>Disagree: {stats.disagree}</span>
              </div>
              <div className="flex items-center space-x-1.5 text-amber-400">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                <span>Unsure: {stats.unsure}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filter and Table */}
      <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <h2 className="text-sm font-bold text-white">Logged Feedback Records</h2>

          {/* Decision filter buttons */}
          <div className="flex items-center space-x-1 text-xs">
            {['all', 'agree', 'disagree', 'unsure'].map((f) => (
              <button
                key={f}
                onClick={() => setFilterDecision(f)}
                className={`px-3 py-1 rounded-lg font-semibold uppercase text-[10px] transition ${
                  filterDecision === f
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {/* Table of records */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                <th className="py-2.5 px-3">Audit ID</th>
                <th className="py-2.5 px-3">Request ID</th>
                <th className="py-2.5 px-3">Predicted Class</th>
                <th className="py-2.5 px-3">Clinician Decision</th>
                <th className="py-2.5 px-3">Reason / Observation</th>
                <th className="py-2.5 px-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredRecords.map((item, idx) => {
                const isAgree = item.decision === 'agree';
                const isDisagree = item.decision === 'disagree';
                return (
                  <motion.tr
                    key={item.feedbackId || idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2, delay: idx * 0.03 }}
                    className="hover:bg-slate-800/40 transition"
                  >
                    <td className="py-3 px-3 text-cyan-400 font-semibold">{item.feedbackId}</td>
                    <td className="py-3 px-3 text-slate-300">{item.requestId}</td>
                    <td className="py-3 px-3 uppercase text-slate-200">{item.predictedClass}</td>
                    <td className="py-3 px-3">
                      <span
                        className={`inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold border uppercase ${
                          isAgree
                            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                            : isDisagree
                            ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                            : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                        }`}
                      >
                        {isAgree ? <CheckCircle2 className="w-3 h-3" /> : isDisagree ? <XCircle className="w-3 h-3" /> : <HelpCircle className="w-3 h-3" />}
                        <span>{item.decision}</span>
                      </span>
                    </td>
                    <td className="py-3 px-3 font-sans text-slate-300 max-w-xs truncate">
                      {item.reasonCode && <span className="font-semibold text-amber-300 mr-1">[{item.reasonCode}]</span>}
                      {item.reasonText || 'No additional text'}
                    </td>
                    <td className="py-3 px-3 text-slate-500 text-[11px]">
                      {new Date(item.timestamp).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
