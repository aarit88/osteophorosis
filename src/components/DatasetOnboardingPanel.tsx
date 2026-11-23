import React, { useState } from 'react';
import { getOnboardingSources, OnboardingSource, registerOnboardingSource, stageOnboardingSource, transitionOnboardingSource, verifyOnboardingSource } from '../api/client';

const inputClass = 'w-full rounded bg-slate-950 border border-slate-700 px-3 py-2 text-sm text-white';

export const DatasetOnboardingPanel: React.FC = () => {
  const [token, setToken] = useState('');
  const [sources, setSources] = useState<OnboardingSource[]>([]);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({ source_id: '', name: '', source_reference: '', description: '', version: 'v1', intended_use: '', license_or_terms: '', data_use_reference: '', ethics_reference: '' });

  const refresh = async () => {
    setBusy(true); setMessage('');
    try { setSources((await getOnboardingSources(token)).sources); setMessage('Registry loaded.'); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'Registry request failed.'); }
    finally { setBusy(false); }
  };

  const register = async (event: React.FormEvent) => {
    event.preventDefault(); setBusy(true); setMessage('');
    try {
      await registerOnboardingSource(token, { ...form, license_or_terms: form.license_or_terms || null,
        data_use_reference: form.data_use_reference || null, ethics_reference: form.ethics_reference || null,
        source_type: 'INSTITUTIONAL', modalities: ['xray'],
        target_categories: ['SECONDARY_BONE_METASTASIS'], acquisition_method: 'Operator documented; verify before acquisition',
        access_restriction: 'RESTRICTED', intended_use: form.intended_use });
      setForm({ source_id: '', name: '', source_reference: '', description: '', version: 'v1', intended_use: '', license_or_terms: '', data_use_reference: '', ethics_reference: '' });
      await refresh();
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Registration failed.'); }
    finally { setBusy(false); }
  };

  const changeStatus = async (source: OnboardingSource, status: string) => {
    const reason = window.prompt(`Reason for ${status}:`);
    if (!reason?.trim()) return;
    const authorization_reference = status === 'ACCESS_APPROVED' ? window.prompt('External authorization reference:') : undefined;
    setBusy(true); setMessage('');
    try {
      await transitionOnboardingSource(token, source.source_id, { status, reason,
        authorization_reference: authorization_reference || undefined });
      await refresh();
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Lifecycle update failed.'); }
    finally { setBusy(false); }
  };

  const stage = async (source: OnboardingSource) => {
    setBusy(true); setMessage('');
    try { await stageOnboardingSource(token, source.source_id); await refresh(); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'Staging failed.'); }
    finally { setBusy(false); }
  };

  const verify = async (source: OnboardingSource) => {
    setBusy(true); setMessage('');
    try {
      const report = await verifyOnboardingSource(token, source.source_id);
      if (!report.valid) { setMessage(`Verification blocked: ${report.errors.join('; ')}`); return; }
      await transitionOnboardingSource(token, source.source_id, { status: 'VERIFIED', reason: 'Package validator passed; clinical validity is not established.' });
      await refresh();
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Package verification failed.'); }
    finally { setBusy(false); }
  };

  return <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-5" aria-label="Dataset onboarding registry">
    <div>
      <h2 className="text-lg font-bold text-white">Stage 38 · Dataset source onboarding</h2>
      <p className="mt-1 text-xs text-slate-400">Registration, acquisition, integrity verification and Stage 33 staging are separate states. A local hash does not prove authenticity, clinical validity, ethics approval, or training authorization.</p>
    </div>
    <div className="flex flex-col gap-2 sm:flex-row">
      <input aria-label="Reviewer bearer token" className={inputClass} type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} placeholder="Authorized reviewer token (memory only)" />
      <button disabled={!token || busy} onClick={refresh} className="rounded bg-cyan-800 px-4 py-2 text-sm text-white disabled:opacity-50">Load registry</button>
    </div>
    {token && <form onSubmit={register} className="grid gap-2 rounded-lg border border-slate-800 p-3 md:grid-cols-3" aria-label="Register source">
      {(['source_id', 'name', 'source_reference', 'description', 'version', 'intended_use'] as const).map((field) => <input key={field} required aria-label={field.replace(/_/g, ' ')} className={inputClass} value={form[field]} onChange={(e) => setForm({ ...form, [field]: e.target.value })} placeholder={field.replace(/_/g, ' ')} />)}
      {(['license_or_terms', 'data_use_reference', 'ethics_reference'] as const).map((field) => <input key={field} aria-label={field.replace(/_/g, ' ')} className={inputClass} value={form[field]} onChange={(e) => setForm({ ...form, [field]: e.target.value })} placeholder={`${field.replace(/_/g, ' ')} (if documented)`} />)}
      <button disabled={busy} className="rounded bg-slate-700 px-4 py-2 text-sm text-white disabled:opacity-50">Register restricted source</button>
    </form>}
    {message && <p role="status" className="text-xs text-amber-300">{message}</p>}
    <div className="space-y-3">
      {sources.map((source) => <article key={source.source_id} className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
        <div className="flex flex-wrap items-start justify-between gap-2"><div><h3 className="font-semibold text-white">{source.name}</h3><p className="text-xs text-slate-400">{source.source_id} · {source.version}</p></div><span className="rounded border border-amber-800 px-2 py-1 text-xs text-amber-300">{source.acquisition_status}</span></div>
        <div className="mt-3 grid gap-2 text-xs text-slate-300 sm:grid-cols-2 lg:grid-cols-4">
          <div>Verification: <b>{source.verification_status}</b></div><div>Integrity: <b>{source.integrity_status}</b></div>
          <div>Staging: <b>{source.staging_status}</b></div><div>Governance approved: <b>{source.governance_approved ? 'Yes' : 'No'}</b></div>
        </div>
        {source.package && <div className="mt-2 text-xs text-slate-400">{source.package.valid ? `Inventory SHA-256: ${source.package.inventory_sha256}` : `Validation blockers: ${source.package.errors.join('; ')}`} · Clinical validity: {source.package.clinical_validity} · Training authorized: {String(source.package.training_authorized)}</div>}
        {source.acquisition_status === 'REGISTERED' && <button disabled={busy || !token} onClick={() => changeStatus(source, 'ACCESS_PENDING')} className="mt-3 rounded bg-slate-700 px-3 py-1.5 text-xs text-white">Record access request</button>}
        {source.acquisition_status === 'ACCESS_PENDING' && <button disabled={busy || !token} onClick={() => changeStatus(source, 'ACCESS_APPROVED')} className="mt-3 rounded bg-slate-700 px-3 py-1.5 text-xs text-white">Record external authorization</button>}
        {source.acquisition_status === 'ACCESS_APPROVED' && <button disabled={busy || !token} onClick={() => changeStatus(source, 'ACQUIRED')} className="mt-3 rounded bg-slate-700 px-3 py-1.5 text-xs text-white">Record package receipt</button>}
        {source.acquisition_status === 'ACQUIRED' && <button disabled={busy || !token} onClick={() => changeStatus(source, 'VERIFICATION_PENDING')} className="mt-3 rounded bg-slate-700 px-3 py-1.5 text-xs text-white">Begin verification</button>}
        {source.acquisition_status === 'VERIFICATION_PENDING' && <button disabled={busy || !token} onClick={() => verify(source)} className="mt-3 rounded bg-cyan-800 px-3 py-1.5 text-xs text-white">Verify package</button>}
        {source.acquisition_status === 'VERIFIED' && source.staging_status !== 'STAGED_FOR_STAGE33_INTAKE' && <button disabled={busy || !token} onClick={() => stage(source)} className="mt-3 rounded bg-emerald-800 px-3 py-1.5 text-xs text-white">Stage for Stage 33 intake</button>}
      </article>)}
      {sources.length === 0 && <p className="text-xs text-slate-500">No authorized registry entries loaded.</p>}
    </div>
  </section>;
};

export default DatasetOnboardingPanel;
