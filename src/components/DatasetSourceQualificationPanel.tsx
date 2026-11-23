import React, { useState } from 'react';
import { addSourceEvidence, createAcquisitionPlan, createSourceQualification, getOnboardingSources, getSourceQualificationReports, getSourceQualifications, linkQualificationStage38, OnboardingSource, QualifiedSource, reviewSourceQualification, transitionAcquisitionPlan, updateSourceAssessment } from '../api/client';

const button = 'rounded bg-cyan-900 px-3 py-1.5 text-xs text-cyan-100 disabled:opacity-50';
const exampleRecord = `{
  "source_id": "source-unique-id", "source_name": "Candidate source name", "original_url": "https://example.org/dataset",
  "repository_or_institution": "Repository or institution", "modalities": [], "available_labels": [],
  "label_mapping": {}, "known_limitations": ["Evidence must be reviewed before qualification."],
  "qualification_status": "DISCOVERED"
}`;
const examplePlan = `{
  "source_id": "candidate-source-id", "method": "Describe the authorized request route",
  "access_request_instructions": "Document official steps; this UI does not submit requests",
  "required_institutional_approval": [], "license_requirements": [],
  "transfer_method": "Describe institution-approved secure transfer", "storage_location_reference": "approved storage reference",
  "expected_package_structure": "Document expected folder structure", "expected_modalities": ["ct"],
  "expected_labels": ["document the source labels"], "expected_manifest": "Stage 33 manifest version",
  "expected_provenance": ["source citation"], "verification_requirements": ["Stage 38 package verification"],
  "responsible_researcher_reference": "institutionally assigned reviewer reference"
}`;
const reviewTransitions: Record<string, string[]> = {
  DISCOVERED: ['INFORMATION_VERIFIED', 'REJECTED', 'ARCHIVED'],
  INFORMATION_VERIFIED: ['LICENSE_REVIEW_PENDING', 'ACCESS_RESTRICTED', 'ACCESS_REQUEST_REQUIRED', 'GOVERNANCE_REVIEW_REQUIRED', 'QUALIFIED_FOR_REQUEST', 'REJECTED', 'ARCHIVED'],
  LICENSE_REVIEW_PENDING: ['INFORMATION_VERIFIED', 'ACCESS_RESTRICTED', 'ACCESS_REQUEST_REQUIRED', 'GOVERNANCE_REVIEW_REQUIRED', 'QUALIFIED_FOR_REQUEST', 'REJECTED', 'ARCHIVED'],
  ACCESS_RESTRICTED: ['GOVERNANCE_REVIEW_REQUIRED', 'REJECTED', 'ARCHIVED'],
  ACCESS_REQUEST_REQUIRED: ['QUALIFIED_FOR_REQUEST', 'GOVERNANCE_REVIEW_REQUIRED', 'ACCESS_RESTRICTED', 'REJECTED', 'ARCHIVED'],
  GOVERNANCE_REVIEW_REQUIRED: ['QUALIFIED_FOR_REQUEST', 'QUALIFIED_FOR_ACQUISITION', 'ACCESS_RESTRICTED', 'REJECTED', 'ARCHIVED'],
  QUALIFIED_FOR_REQUEST: ['GOVERNANCE_REVIEW_REQUIRED', 'QUALIFIED_FOR_ACQUISITION', 'ACCESS_RESTRICTED', 'REJECTED', 'ARCHIVED'],
  QUALIFIED_FOR_ACQUISITION: ['GOVERNANCE_REVIEW_REQUIRED', 'ACCESS_RESTRICTED', 'REJECTED', 'ARCHIVED'],
  REJECTED: ['ARCHIVED'], ARCHIVED: [],
};

export const DatasetSourceQualificationPanel: React.FC = () => {
  const [token, setToken] = useState('');
  const [sources, setSources] = useState<QualifiedSource[]>([]);
  const [coverage, setCoverage] = useState<Record<string, unknown> | null>(null);
  const [feasibility, setFeasibility] = useState<Record<string, unknown> | null>(null);
  const [plans, setPlans] = useState<Array<Record<string, unknown>>>([]);
  const [stage38Sources, setStage38Sources] = useState<OnboardingSource[]>([]);
  const [planJson, setPlanJson] = useState(examplePlan);
  const [handoffSource, setHandoffSource] = useState('');
  const [handoffTarget, setHandoffTarget] = useState('');
  const [handoffConfirmed, setHandoffConfirmed] = useState(false);
  const [decisionStatuses, setDecisionStatuses] = useState<Record<string, string>>({});
  const [reviewConfirmations, setReviewConfirmations] = useState<Record<string, boolean>>({});
  const [recordJson, setRecordJson] = useState(exampleRecord);
  const [evidenceJson, setEvidenceJson] = useState('');
  const [assessmentJson, setAssessmentJson] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    setBusy(true); setStatus('');
    try {
      const [candidateResult, reports, onboarding] = await Promise.all([getSourceQualifications(token), getSourceQualificationReports(token), getOnboardingSources(token)]);
      setSources(candidateResult.sources); setCoverage(reports.coverage); setFeasibility(reports.feasibility); setPlans(reports.plans); setStage38Sources(onboarding.sources);
      setStatus('Candidate registry loaded. Candidate coverage is not an acquired cohort.');
    } catch (error) { setStatus(error instanceof Error ? error.message : 'Qualification registry request failed.'); }
    finally { setBusy(false); }
  };
  const createPlan = async () => {
    setBusy(true); setStatus('');
    try { await createAcquisitionPlan(token, JSON.parse(planJson) as Record<string, unknown>); setStatus('Acquisition plan saved. It did not submit an access request or grant permission.'); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Acquisition plan was not saved.'); setBusy(false); }
  };
  const handoff = async () => {
    if (!handoffConfirmed) { setStatus('Confirm that the selected Stage 38 registration refers to this qualified source.'); return; }
    setBusy(true); setStatus('');
    try { await linkQualificationStage38(token, handoffSource, handoffTarget, 'Reviewer explicitly confirmed this existing Stage 38 source reference and modality.'); setHandoffConfirmed(false); setStatus('Linked to the existing Stage 38 registration. Package access, verification, and staging remain separate gates.'); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Stage 38 link was not created.'); setBusy(false); }
  };
  const markPlanReady = async (planId: string) => {
    setBusy(true); setStatus('');
    try { await transitionAcquisitionPlan(token, planId, 'READY_FOR_EXTERNAL_REQUEST', 'Authorized operator reviewed the acquisition checklist.'); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Plan status was not changed.'); setBusy(false); }
  };
  const requestPlanReview = async (planId: string) => {
    setBusy(true); setStatus('');
    try { await transitionAcquisitionPlan(token, planId, 'REVIEW_REQUIRED', 'Submitted for authorized review.'); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Plan status was not changed.'); setBusy(false); }
  };
  const create = async () => {
    setBusy(true); setStatus('');
    try { await createSourceQualification(token, JSON.parse(recordJson) as Record<string, unknown>); setStatus('Source recorded as DISCOVERED. No access approval or data acquisition was granted.'); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Source registration failed.'); setBusy(false); }
  };
  const addEvidence = async (source: QualifiedSource) => {
    setBusy(true); setStatus('');
    try { await addSourceEvidence(token, source.source_id, JSON.parse(evidenceJson) as Record<string, unknown>); setEvidenceJson(''); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Evidence was not recorded.'); setBusy(false); }
  };
  const review = async (source: QualifiedSource) => {
    const nextStatus = decisionStatuses[source.source_id] || 'INFORMATION_VERIFIED';
    const evidenceIds = source.evidence.filter((item) => item.kind !== 'UNVERIFIED_ASSUMPTION').map((item) => item.evidence_id);
    if (!evidenceIds.length) { setStatus('Add traceable evidence before requesting a reviewer decision.'); return; }
    if (!reviewConfirmations[source.source_id]) { setStatus('Explicitly confirm the selected evidence-backed review decision.'); return; }
    setBusy(true); setStatus('');
    try { await reviewSourceQualification(token, source.source_id, { status: nextStatus, evidence_ids: evidenceIds, explicit_confirmation: true, notes: 'Reviewed against the cited evidence; independent governance and access approval remain required.' }); setReviewConfirmations((old) => ({ ...old, [source.source_id]: false })); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Qualification review failed.'); setBusy(false); }
  };
  const updateAssessment = async (source: QualifiedSource) => {
    setBusy(true); setStatus('');
    try { await updateSourceAssessment(token, source.source_id, JSON.parse(assessmentJson) as Record<string, unknown>); setAssessmentJson(''); await refresh(); }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Evidence-linked assessment update failed.'); setBusy(false); }
  };

  return <section className="space-y-4 rounded-2xl border border-cyan-900/70 bg-slate-900/60 p-5" aria-label="Dataset source qualification and acquisition readiness">
    <div><span className="text-[10px] font-bold uppercase tracking-wider text-cyan-300">Stage 40 · Source qualification</span>
      <h2 className="mt-1 text-lg font-bold text-white">Candidate sources and acquisition planning</h2>
      <p className="mt-1 text-xs text-slate-400">Evidence and reviewer status are tracked separately from access permission, Stage 38 package verification, and Stage 39 preparation. No source record represents a clinical cohort.</p>
    </div>
    <div className="flex flex-wrap gap-2">
      <label className="sr-only" htmlFor="qualification-reviewer-token">Reviewer bearer token</label>
      <input id="qualification-reviewer-token" type="password" autoComplete="off" value={token} onChange={(event) => setToken(event.target.value)} placeholder="Reviewer bearer token (kept in memory)" className="min-w-64 flex-1 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white" />
      <button className={button} disabled={busy || !token} onClick={refresh}>Load qualification records</button>
    </div>
    <details className="rounded border border-slate-800 bg-slate-950/60 p-3"><summary className="cursor-pointer text-xs font-semibold text-slate-200">Register a candidate source as DISCOVERED</summary>
      <p className="my-2 text-xs text-amber-300">Use only source facts that can be cited. Do not enter credentials, patient data, approvals, or assumed access terms.</p>
      <textarea aria-label="Candidate source JSON" className="min-h-44 w-full rounded border border-slate-700 bg-slate-950 p-3 font-mono text-[11px] text-slate-200" value={recordJson} onChange={(event) => setRecordJson(event.target.value)} />
      <button className={`${button} mt-2`} disabled={busy || !token} onClick={create}>Register candidate</button>
    </details>
    {status && <p role="status" className="text-xs text-amber-300">{status}</p>}
    <div className="grid gap-3 lg:grid-cols-2">
      <article className="rounded border border-slate-800 bg-slate-950/60 p-3"><h3 className="text-xs font-bold text-white">Candidate class/modality coverage</h3><p className="mt-1 text-[11px] text-amber-300">Theoretical source coverage only · actual patient count: none reported</p><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-[10px] text-slate-400">{coverage ? JSON.stringify(coverage, null, 2) : 'Load the reviewer registry to view evidence-backed candidate coverage.'}</pre></article>
      <article className="rounded border border-slate-800 bg-slate-950/60 p-3"><h3 className="text-xs font-bold text-white">Cohort feasibility</h3><p className="mt-1 text-[11px] text-amber-300">Feasibility is not a patient cohort size or readiness decision.</p><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-[10px] text-slate-400">{feasibility ? JSON.stringify(feasibility, null, 2) : 'No feasibility report loaded.'}</pre></article>
    </div>
    <div className="space-y-3">{sources.map((source) => <article key={source.source_id} className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
      <div className="flex flex-wrap justify-between gap-2"><div><h3 className="text-sm font-semibold text-white">{source.source_name}</h3><a href={source.original_url} target="_blank" rel="noreferrer" className="text-[11px] text-cyan-300 underline">Source documentation</a><p className="text-[11px] text-slate-500">{source.source_id} · Stage 38: {source.stage38_source_id ?? 'not linked'}</p></div><span className="h-fit rounded border border-amber-900 px-2 py-1 text-xs text-amber-200">{source.qualification_status}</span></div>
      <p className="mt-2 text-xs text-slate-300">Candidate class interests (not label mappings): {source.target_class_interest?.join(', ') || 'none'} · Modalities: {source.modalities.join(', ') || 'unknown'} · License: {source.license_verification_status} · Access request: {source.access_request_status} · Patient grouping: {source.patient_grouping_feasibility}</p>
      {source.known_limitations.length > 0 && <p className="mt-2 text-xs text-amber-300">Limitations: {source.known_limitations.join('; ')}</p>}
      <ul className="mt-2 list-disc pl-5 text-[11px] text-slate-400">{source.evidence.map((item) => <li key={item.evidence_id}>{item.kind}: {item.statement} (<a href={item.reference_url} target="_blank" rel="noreferrer" className="text-cyan-300 underline">evidence</a>)</li>)}</ul>
      <details className="mt-3"><summary className="cursor-pointer text-xs text-slate-300">Add evidence / record review</summary><textarea aria-label={`Evidence JSON ${source.source_id}`} className="mt-2 min-h-24 w-full rounded border border-slate-700 bg-slate-950 p-2 font-mono text-[10px] text-slate-200" placeholder={'{"kind":"PUBLISHED_CLAIM","statement":"...","reference_url":"https://..."}'} value={evidenceJson} onChange={(event) => setEvidenceJson(event.target.value)} /><div className="mt-2 flex flex-wrap gap-2"><button className={button} disabled={busy || !token || !evidenceJson} onClick={() => addEvidence(source)}>Add evidence</button><label className="text-xs text-slate-300">Review decision<select aria-label={`Review decision ${source.source_id}`} className="ml-2 rounded border border-slate-700 bg-slate-950 px-2 py-1 text-white" value={decisionStatuses[source.source_id] || (reviewTransitions[source.qualification_status]?.[0] ?? '')} onChange={(event) => setDecisionStatuses((old) => ({ ...old, [source.source_id]: event.target.value }))}>{(reviewTransitions[source.qualification_status] || []).map((value) => <option key={value} value={value}>{value}</option>)}</select></label><label className="inline-flex items-center gap-1 text-xs text-slate-300"><input type="checkbox" checked={Boolean(reviewConfirmations[source.source_id])} onChange={(event) => setReviewConfirmations((old) => ({ ...old, [source.source_id]: event.target.checked }))} />I reviewed the cited evidence and confirm this decision.</label><button className={button} disabled={busy || !token || !source.evidence.length || !(reviewTransitions[source.qualification_status] || []).length} onClick={() => review(source)}>Record reviewer decision</button></div><textarea aria-label={`Assessment JSON ${source.source_id}`} className="mt-2 min-h-20 w-full rounded border border-slate-700 bg-slate-950 p-2 font-mono text-[10px] text-slate-200" placeholder={'{"evidence_ids":["evidence-id"],"explicit_confirmation":true,"notes":"Reviewer checked evidence","license_verification_status":"PENDING"}'} value={assessmentJson} onChange={(event) => setAssessmentJson(event.target.value)} /><button className={`${button} mt-2`} disabled={busy || !token || !assessmentJson} onClick={() => updateAssessment(source)}>Update evidence-linked licensing / access assessment</button></details>
    </article>)}{sources.length === 0 && <p className="text-xs text-slate-500">No source records loaded. Public source findings are documented in the Stage 40 qualification guide and must be registered and reviewed before status changes.</p>}</div>
    <div className="rounded border border-slate-800 bg-slate-950/60 p-3"><h3 className="text-xs font-bold text-white">Acquisition plans</h3>{plans.length ? <ul className="mt-2 space-y-1 text-xs text-slate-300">{plans.map((plan) => <li key={String(plan.plan_id)}>{String(plan.plan_id)} · {String(plan.source_id)} · {String(plan.status)} · Stage 38 link: {String(plan.stage38_source_id ?? 'not linked')}</li>)}</ul> : <p className="mt-1 text-xs text-slate-500">No acquisition plans recorded.</p>}</div>
    <details className="rounded border border-slate-800 bg-slate-950/60 p-3"><summary className="cursor-pointer text-xs font-semibold text-slate-200">Create acquisition plan</summary><p className="my-2 text-xs text-amber-300">Plans document external work only. Do not claim an approval that has not been granted.</p><textarea aria-label="Acquisition plan JSON" className="min-h-44 w-full rounded border border-slate-700 bg-slate-950 p-3 font-mono text-[10px] text-slate-200" value={planJson} onChange={(event) => setPlanJson(event.target.value)} /><button className={`${button} mt-2`} disabled={busy || !token} onClick={createPlan}>Save acquisition plan</button></details>
    {plans.filter((plan) => plan.status === 'DRAFT').map((plan) => <button key={String(plan.plan_id)} className={button} disabled={busy || !token} onClick={() => requestPlanReview(String(plan.plan_id))}>Submit {String(plan.plan_id)} for review</button>)}
    {plans.filter((plan) => plan.status === 'REVIEW_REQUIRED').map((plan) => <button key={String(plan.plan_id)} className={button} disabled={busy || !token || Boolean((plan.blockers as string[] | undefined)?.length)} onClick={() => markPlanReady(String(plan.plan_id))}>Mark {String(plan.plan_id)} ready for external request</button>)}
    <div className="flex flex-wrap items-center gap-2 rounded border border-slate-800 bg-slate-950/60 p-3"><label className="sr-only" htmlFor="qualification-handoff-source">Qualified source for Stage 38 handoff</label><select id="qualification-handoff-source" className="min-w-48 rounded border border-slate-700 bg-slate-950 px-2 py-2 text-xs text-white" value={handoffSource} onChange={(event) => setHandoffSource(event.target.value)}><option value="">Select acquisition-qualified source</option>{sources.filter((source) => source.qualification_status === 'QUALIFIED_FOR_ACQUISITION').map((source) => <option key={source.source_id} value={source.source_id}>{source.source_name}</option>)}</select><label className="sr-only" htmlFor="qualification-handoff-target">Existing Stage 38 source</label><select id="qualification-handoff-target" className="min-w-48 rounded border border-slate-700 bg-slate-950 px-2 py-2 text-xs text-white" value={handoffTarget} onChange={(event) => setHandoffTarget(event.target.value)}><option value="">Select existing Stage 38 registration</option>{stage38Sources.map((source) => <option key={source.source_id} value={source.source_id}>{source.source_id} · {source.acquisition_status}</option>)}</select><label className="inline-flex items-center gap-1 text-xs text-slate-300"><input type="checkbox" checked={handoffConfirmed} onChange={(event) => setHandoffConfirmed(event.target.checked)} />I confirm the Stage 38 source refers to this candidate.</label><button className={button} disabled={busy || !token || !handoffSource || !handoffTarget || !handoffConfirmed} onClick={handoff}>Link existing Stage 38 source</button></div>
  </section>;
};

export default DatasetSourceQualificationPanel;
