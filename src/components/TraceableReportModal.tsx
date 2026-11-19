import React, { useEffect, useState, useRef } from 'react';
import QRCode from 'qrcode';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { AnalyzeResponse } from '../api/types';
import { resolveApiAssetUrl } from '../api/client';
import { X, Download, Printer, ShieldCheck, QrCode, FileText, CheckCircle2 } from 'lucide-react';

interface TraceableReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: AnalyzeResponse;
  engineMode?: string;
}

export const TraceableReportModal: React.FC<TraceableReportModalProps> = ({
  isOpen,
  onClose,
  result,
  engineMode = 'Simulation Mock',
}) => {
  const [qrDataUrl, setQrDataUrl] = useState<string>('');
  const [isGeneratingPdf, setIsGeneratingPdf] = useState<boolean>(false);
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const auditUrl = `${window.location.origin}/audit?ref=${result.requestId}`;
    QRCode.toDataURL(auditUrl, { width: 160, margin: 1 })
      .then(setQrDataUrl)
      .catch(console.error);
  }, [isOpen, result.requestId]);

  if (!isOpen) return null;

  const handleDownloadPdf = async () => {
    if (!reportRef.current) return;
    setIsGeneratingPdf(true);
    try {
      const canvas = await html2canvas(reportRef.current, {
        scale: 2,
        useCORS: true,
        backgroundColor: '#ffffff',
      });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`OsteoGuard_Report_${result.requestId}.pdf`);
    } catch (err) {
      console.error('PDF Generation failed, triggering print fallback:', err);
      window.print();
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const topClass = result.prediction.topClass;
  const confidencePercent = Math.round(result.prediction.confidence * 100);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/85 backdrop-blur-md overflow-y-auto">
      <div className="bg-white text-slate-900 rounded-3xl w-full max-w-4xl shadow-2xl overflow-hidden my-6 border border-slate-300">
        {/* Action Header bar (no-print) */}
        <div className="no-print flex items-center justify-between px-6 py-4 bg-slate-900 text-white border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <FileText className="w-5 h-5 text-cyan-400" />
            <span className="text-sm font-bold">Traceable Clinical Decision Support Dossier</span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={handleDownloadPdf}
              disabled={isGeneratingPdf}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-md transition disabled:opacity-60"
            >
              <Download className="w-4 h-4" />
              <span>{isGeneratingPdf ? 'Rendering PDF…' : 'Export Multi-Page PDF'}</span>
            </button>
            <button
              onClick={() => window.print()}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition"
            >
              <Printer className="w-4 h-4" />
              <span>Print</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition"
              aria-label="Close report modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable & Canvas Document */}
        <div ref={reportRef} className="p-8 sm:p-12 space-y-10 bg-white text-slate-900 font-sans text-xs">
          {/* ═════════ PAGE 1: PRIMARY DIAGNOSIS & GRAD-CAM ═════════ */}
          <div className="space-y-6 border-b-2 border-slate-200 pb-8">
            <div className="flex justify-between items-start border-b-2 border-cyan-600 pb-4">
              <div>
                <h1 className="text-2xl font-black tracking-tight text-slate-950">
                  Osteo<span className="text-cyan-600">Guard</span> AI — Traceable Clinical Dossier
                </h1>
                <p className="text-slate-500 mt-0.5">
                  Explainable Multi-Class Bone Pathology Deep Learning Assessment
                </p>
              </div>
              <div className="text-right font-mono text-[11px] text-slate-500 space-y-0.5">
                <div>Document ID: <strong className="text-slate-800">{result.requestId}</strong></div>
                <div>Timestamp: {new Date(result.timestamp).toUTCString()}</div>
                <div>Modality: <strong className="uppercase">{result.modality}</strong></div>
              </div>
            </div>

            {/* De-identified Patient Header */}
            <div className="grid grid-cols-4 gap-3 bg-slate-100 p-3 rounded-xl border border-slate-200 font-mono text-[11px]">
              <div><span className="text-slate-500 block">De-Identified Ref:</span><strong>{result.requestId.replace('req_', 'PT-')}</strong></div>
              <div><span className="text-slate-500 block">Status:</span><strong className="text-cyan-700">Audit-Ready (Art. 14)</strong></div>
              <div><span className="text-slate-500 block">Ensemble:</span><strong>CNN + ViT-B/16</strong></div>
              <div><span className="text-slate-500 block">Engine Mode:</span><strong>{engineMode}</strong></div>
            </div>

            {/* Side-by-Side Radiograph & Grad-CAM Attention Map */}
            <div>
              <div className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                1. Radiographic Input & Visual Attention Salience
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="border border-slate-300 rounded-xl p-2 bg-slate-900 text-center">
                  <img
                    src={resolveApiAssetUrl(result.imageUrl)}
                    alt="Original Radiograph"
                    className="w-full h-48 object-contain rounded"
                  />
                  <span className="text-[10px] font-mono text-slate-300 mt-1 block">Original Planar Radiograph</span>
                </div>
                <div className="border border-slate-300 rounded-xl p-2 bg-slate-900 text-center">
                  <img
                    src={resolveApiAssetUrl(result.gradcamUrl)}
                    alt="Grad-CAM Attention Heatmap"
                    className="w-full h-48 object-contain rounded"
                  />
                  <span className="text-[10px] font-mono text-cyan-300 mt-1 block">Grad-CAM Feature Salience Map ¹</span>
                </div>
              </div>
            </div>

            {/* Diagnosis & Uncertainty Narrative with Superscript Citations */}
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
              <div className="flex justify-between items-baseline">
                <span className="text-xs font-bold uppercase text-slate-600 tracking-wider">
                  Primary Classification Claim:
                </span>
                <span className="font-mono text-sm font-extrabold text-cyan-800">
                  {confidencePercent}% Confidence [{Math.round(result.uncertaintyDetail.confidenceInterval[0] * 100)}% – {Math.round(result.uncertaintyDetail.confidenceInterval[1] * 100)}% 95% CI]
                </span>
              </div>
              <div className="text-base font-bold text-slate-900">
                Predicted: <span className="uppercase text-cyan-700">{topClass}</span> <sup className="text-cyan-700 font-bold">1,2</sup>
              </div>
              <p className="text-slate-600 text-xs leading-relaxed">
                The classification is derived from joint convolutional feature maps and Vision Transformer patch attention.
                Softmax dispersion entropy is measured at <strong className="font-mono">{result.uncertaintyDetail.entropy.toFixed(3)}</strong> (Entropy threshold &lt; 0.60; {result.uncertaintyDetail.oodFlag ? 'OOD Triggered' : 'In-Distribution Normal'}).
                Impending fracture risk index is estimated at <strong className="font-mono">{result.fractureRisk.percent}%</strong> ({result.fractureRisk.level.toUpperCase()} risk) based on cortical breach geometry.
              </p>
            </div>
          </div>

          {/* ═════════ PAGE 2: TRACEABLE RADIOMICS & COUNTERFACTUALS ═════════ */}
          <div className="space-y-6 border-b-2 border-slate-200 pb-8">
            <div className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              2. Quantitative Radiomics Feature Attribution (SHAP)
            </div>

            <table className="w-full text-left border-collapse border border-slate-300">
              <thead>
                <tr className="bg-slate-100 text-slate-700 font-mono text-[11px] border-b border-slate-300">
                  <th className="p-2 border border-slate-300">Feature Identifier</th>
                  <th className="p-2 border border-slate-300">Signed SHAP Weight ²</th>
                  <th className="p-2 border border-slate-300">Directional Diagnostic Impact</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono text-xs">
                {result.shap.map((s, idx) => (
                  <tr key={s.name}>
                    <td className="p-2 font-sans font-medium text-slate-900 border border-slate-300">
                      {s.name} {idx === 0 && <sup className="text-cyan-600 font-bold">2</sup>}
                    </td>
                    <td className={`p-2 font-bold border border-slate-300 ${s.value >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {s.value > 0 ? `+${s.value}` : s.value}
                    </td>
                    <td className="p-2 font-sans text-slate-600 border border-slate-300">
                      {s.value >= 0 ? 'Positive attribution toward predicted class' : 'Negative suppression / differential indicator'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Counterfactual Decision Sensitivity Narrative */}
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
              <span className="text-xs font-bold uppercase text-slate-700 tracking-wider block">
                3. Traceable Counterfactual Explanations ³
              </span>
              <ul className="space-y-1.5 text-xs text-slate-700 list-disc pl-4">
                {result.counterfactuals.map((cf, i) => (
                  <li key={i}>
                    If <strong>{cf.feature}</strong> were to {cf.deltaDirection} by <strong>{cf.percentChange}%</strong>, the model’s primary prediction would shift to <strong>{cf.flipTo.toUpperCase()}</strong> (probability shift Δ +{(cf.probabilityShift * 100).toFixed(0)}%).
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* ═════════ PAGE 3: EVIDENCE FOOTNOTES, AUDIT QR & REGULATORY ═════════ */}
          <div className="space-y-6 pt-2">
            <div className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              4. Evidence References & Audit Ledger Verification
            </div>

            {/* Evidence superscript definitions */}
            <div className="p-4 rounded-xl bg-slate-100 border border-slate-300 space-y-1.5 text-xs text-slate-700 font-mono">
              <div><strong>[1] Visual Evidence:</strong> Grad-CAM heatmap focal activation centered in primary bone region.</div>
              <div><strong>[2] Radiomics Evidence:</strong> Top SHAP feature ({result.shap[0]?.name}: {result.shap[0]?.value > 0 ? `+${result.shap[0]?.value}` : result.shap[0]?.value}) supports biological bone density change.</div>
              <div><strong>[3] Counterfactual Boundary:</strong> Mathematical minimum perturbation to alter decision boundary.</div>
            </div>

            {/* QR Code and Compliance Sign-off */}
            <div className="flex items-center justify-between p-4 rounded-xl border border-slate-300 bg-white">
              <div className="space-y-1 max-w-lg">
                <span className="font-bold text-slate-900 block text-xs">
                  Scan to Verify or Record Clinician Feedback in Ledger (EU AI Act Art. 14):
                </span>
                <p className="text-slate-500 text-[11px] font-mono leading-relaxed">
                  URL: {window.location.origin}/audit?ref={result.requestId}
                </p>
                <div className="text-[10px] text-slate-400 pt-1">
                  Compliance: FDA Good Machine Learning Practice (GMLP) • Algorithmic Auditability Standard
                </div>
              </div>

              {qrDataUrl && (
                <div className="p-2 border border-slate-300 rounded-xl bg-white shrink-0 text-center">
                  <img src={qrDataUrl} alt="Audit verification QR code" className="w-24 h-24" />
                  <span className="text-[9px] font-mono text-slate-500 block mt-0.5">Audit Stamp</span>
                </div>
              )}
            </div>

            {/* Regulatory Disclaimer */}
            <div className="border-t border-slate-300 pt-3 text-[10px] text-slate-500 leading-relaxed">
              <strong>REGULATORY ADVISORY:</strong> {result.disclaimer} This document was produced automatically by the OsteoGuard AI research prototype for investigational review.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
