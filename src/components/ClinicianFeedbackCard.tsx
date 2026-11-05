import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, X, HelpCircle, Send, CheckCircle2, ShieldCheck, Scale, ArrowRight } from 'lucide-react';
import { ClinicianDecision, FeedbackReasonCode, BonePathologyClass } from '../api/types';
import { submitFeedback, logTelemetryEvent } from '../api/client';
import { Link } from 'react-router-dom';

interface ClinicianFeedbackCardProps {
  requestId: string;
  predictedClass: BonePathologyClass;
}

const REASONS: Array<{ code: FeedbackReasonCode; label: string }> = [
  { code: 'artifact', label: 'Radiographic artifact / motion blur' },
  { code: 'insufficient_context', label: 'Insufficient clinical context / missing lateral view' },
  { code: 'missed_lesion', label: 'Model missed focal cortical/trabecular lesion' },
  { code: 'class_confusion', label: 'Class confusion (e.g. Sclerotic Paget vs Blastic Metastasis)' },
  { code: 'other', label: 'Other clinical or anatomical consideration' },
];

export const ClinicianFeedbackCard: React.FC<ClinicianFeedbackCardProps> = ({ requestId, predictedClass }) => {
  const [decision, setDecision] = useState<ClinicianDecision | null>(null);
  const [reasonCode, setReasonCode] = useState<FeedbackReasonCode>('insufficient_context');
  const [reasonText, setReasonText] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submittedId, setSubmittedId] = useState<string | null>(null);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!decision) return;

    setIsSubmitting(true);
    try {
      const res = await submitFeedback({
        requestId,
        decision,
        predictedClass,
        reasonCode: decision !== 'agree' ? reasonCode : undefined,
        reasonText: decision !== 'agree' ? reasonText : undefined,
        timestamp: new Date().toISOString(),
      });

      setSubmittedId(res.feedbackId);
      logTelemetryEvent({
        type: 'clinician_feedback_submitted',
        requestId,
        metadata: { decision, reasonCode },
      });
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="p-6 rounded-3xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <Scale className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-sm font-bold text-white">Clinician-in-the-Loop Verification</h3>
            <span className="text-[10px] text-slate-400 font-mono">
              EU AI Act Art. 14 Human Oversight & FDA GMLP Audit Log
            </span>
          </div>
        </div>

        <Link
          to="/audit"
          className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 font-semibold"
        >
          <span>View Audit Trail</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      <AnimatePresence mode="wait">
        {submittedId ? (
          /* Submission success state */
          <motion.div
            key="success"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="p-4 rounded-2xl bg-emerald-950/40 border border-emerald-500/50 flex items-center justify-between text-xs"
          >
            <div className="flex items-center space-x-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
              <div>
                <span className="font-bold text-white block">Clinical Feedback Recorded in Audit Ledger</span>
                <span className="text-[11px] text-emerald-300/80 font-mono">Feedback ID: {submittedId}</span>
              </div>
            </div>
            <Link
              to="/audit"
              className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition"
            >
              Inspect Audit
            </Link>
          </motion.div>
        ) : (
          /* Decision Input Form */
          <motion.form key="form" onSubmit={handleSubmit} className="space-y-4">
            <div className="text-xs text-slate-300">
              Do you concur with the model's primary classification of <strong className="text-white uppercase font-mono">{predictedClass}</strong>?
            </div>

            {/* Three Big Decision Buttons */}
            <div className="grid grid-cols-3 gap-3">
              {/* Agree */}
              <button
                type="button"
                onClick={() => setDecision('agree')}
                className={`py-3 px-3 rounded-2xl border font-bold text-xs sm:text-sm flex flex-col items-center justify-center space-y-1 transition ${
                  decision === 'agree'
                    ? 'bg-emerald-500/25 border-emerald-500 text-emerald-300 shadow-lg shadow-emerald-500/10'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Agree</span>
              </button>

              {/* Disagree */}
              <button
                type="button"
                onClick={() => setDecision('disagree')}
                className={`py-3 px-3 rounded-2xl border font-bold text-xs sm:text-sm flex flex-col items-center justify-center space-y-1 transition ${
                  decision === 'disagree'
                    ? 'bg-rose-500/25 border-rose-500 text-rose-300 shadow-lg shadow-rose-500/10'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <X className="w-5 h-5 text-rose-400" />
                <span>Disagree</span>
              </button>

              {/* Unsure */}
              <button
                type="button"
                onClick={() => setDecision('unsure')}
                className={`py-3 px-3 rounded-2xl border font-bold text-xs sm:text-sm flex flex-col items-center justify-center space-y-1 transition ${
                  decision === 'unsure'
                    ? 'bg-amber-500/25 border-amber-500 text-amber-300 shadow-lg shadow-amber-500/10'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <HelpCircle className="w-5 h-5 text-amber-400" />
                <span>Unsure</span>
              </button>
            </div>

            {/* Optional Reason Form (Shown when Disagree or Unsure) */}
            {(decision === 'disagree' || decision === 'unsure') && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-3 pt-2 border-t border-slate-800"
              >
                <div>
                  <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                    Primary Clinical Rationale for Disagreement / Uncertainty:
                  </label>
                  <select
                    value={reasonCode}
                    onChange={(e) => setReasonCode(e.target.value as FeedbackReasonCode)}
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white focus:border-cyan-500 focus:outline-none"
                  >
                    {REASONS.map((r) => (
                      <option key={r.code} value={r.code}>
                        {r.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                    Additional Observations (Optional):
                  </label>
                  <textarea
                    rows={2}
                    value={reasonText}
                    onChange={(e) => setReasonText(e.target.value)}
                    placeholder="e.g. Sclerotic margin suggests longstanding indolent process..."
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                  />
                </div>
              </motion.div>
            )}

            {/* Submit button */}
            <div className="pt-2">
              <button
                type="submit"
                disabled={!decision || isSubmitting}
                className={`w-full py-2.5 px-4 rounded-xl font-bold text-xs flex items-center justify-center space-x-1.5 transition ${
                  decision && !isSubmitting
                    ? 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-600/20 cursor-pointer'
                    : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                }`}
              >
                <Send className="w-3.5 h-3.5" />
                <span>{isSubmitting ? 'Recording Audit…' : 'Submit Review to Audit Log'}</span>
              </button>
            </div>
          </motion.form>
        )}
      </AnimatePresence>
    </div>
  );
};
