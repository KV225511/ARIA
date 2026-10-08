import { useEffect, useRef, useState } from 'react';
import { apiFetch, newId, responseData } from '../api';


function UploadSlot({ kind, title, hint, document, onReady }) {
  const inputRef = useRef(null);
  const [status, setStatus] = useState(document?.status || 'idle');
  const [name, setName] = useState(document?.original_filename || '');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!document) return;
    setStatus(document.status);
    setName(document.original_filename);
  }, [document]);

  const poll = async (id) => {
    for (let attempt = 0; attempt < 60; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const data = await responseData(await apiFetch(`/api/documents/${id}`));
      setStatus(data.status);
      if (data.status === 'ready') {
        onReady(data);
        return;
      }
      if (data.status === 'failed') {
        setError(data.error_code || 'Document processing failed.');
        return;
      }
    }
    setError('Document processing is taking longer than expected. You can return later.');
  };

  const upload = async (file) => {
    if (!file) return;
    setError('');
    setName(file.name);
    setStatus('uploading');
    const form = new FormData();
    form.append('kind', kind);
    form.append('file', file);
    try {
      const data = await responseData(await apiFetch('/api/documents', {
        method: 'POST',
        headers: { 'Idempotency-Key': newId() },
        body: form,
      }));
      setStatus(data.status);
      await poll(data.id);
    } catch (uploadError) {
      setStatus('failed');
      setError(uploadError.message);
    }
  };

  return (
    <div className="document-field">
      <div className="field-heading"><label>{title}</label><span>PDF · 10 MB max</span></div>
      <button type="button" className={`file-dropzone ${status !== 'idle' ? 'has-file' : ''}`} onClick={() => inputRef.current?.click()}>
        <span className="file-copy">
          <strong>{name || 'Choose a PDF'}</strong>
          <small>{error || (status === 'idle' ? hint : status.replaceAll('_', ' '))}</small>
        </span>
        <span className="file-action">{status === 'processing' || status === 'uploading' ? 'Working…' : 'Browse'}</span>
      </button>
      <input ref={inputRef} type="file" accept="application/pdf,.pdf" hidden onChange={(event) => upload(event.target.files[0])} />
    </div>
  );
}


function Dashboard({ user, onOpenInterview, onLogout }) {
  const [resume, setResume] = useState(null);
  const [jd, setJd] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadHistory = async () => {
    try {
      const data = await responseData(await apiFetch('/api/interviews'));
      setHistory(data.items);
    } catch (historyError) {
      setError(historyError.message);
    }
  };

  useEffect(() => { loadHistory(); }, []);

  const waitForInterview = async (id) => {
    for (let attempt = 0; attempt < 90; attempt += 1) {
      const data = await responseData(await apiFetch(`/api/interviews/${id}`));
      if (['ready', 'active', 'paused'].includes(data.status)) {
        onOpenInterview(id);
        return;
      }
      if (data.status === 'failed') throw new Error(data.last_error_code || 'Interview preparation failed.');
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    throw new Error('Interview preparation is taking longer than expected. It remains in your history.');
  };

  const prepare = async (event) => {
    event.preventDefault();
    if (!resume || !jd) return;
    setLoading(true);
    setError('');
    try {
      if (user.consent_version !== user.required_consent_version) {
        await responseData(await apiFetch('/api/me/consent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ version: user.required_consent_version }),
        }));
        user.consent_version = user.required_consent_version;
      }
      const interview = await responseData(await apiFetch('/api/interviews', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Idempotency-Key': newId() },
        body: JSON.stringify({
          resume_document_id: resume.id,
          jd_document_id: jd.id,
          requested_role: 'backend_developer',
        }),
      }));
      await waitForInterview(interview.id);
    } catch (prepareError) {
      setError(prepareError.message);
      loadHistory();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-page">
      <header className="site-header dashboard-header">
        <a className="brand" href="/" aria-label="ARIA home"><span className="brand-mark">A</span><span>ARIA</span></a>
        <div className="account-actions"><span>{user.display_name || user.email}</span><button type="button" onClick={onLogout}>Sign out</button></div>
      </header>
      <div className="dashboard-grid">
        <section className="setup-card">
          <div className="card-heading"><div><p className="section-kicker">New session</p><h2>Build the interview</h2></div><span className="step-label">2 documents</span></div>
          <form onSubmit={prepare}>
            <UploadSlot kind="job_description" title="Job description" hint="Role, responsibilities, and requirements" document={jd} onReady={setJd} />
            <UploadSlot kind="resume" title="Candidate résumé" hint="Your background and experience" document={resume} onReady={setResume} />
            <p className="privacy-note">Files are private to your account. Audio is transcribed and then discarded.</p>
            {error && <p className="form-error" role="alert">{error}</p>}
            <button className="primary-button" disabled={!resume || !jd || loading}>
              <span>{loading ? 'Preparing interview' : 'Prepare interview'}</span><span aria-hidden="true">→</span>
            </button>
          </form>
        </section>
        <section className="history-panel">
          <div className="card-heading"><div><p className="section-kicker">Your account</p><h2>Interview history</h2></div><span className="step-label">{history.length} sessions</span></div>
          {history.length === 0 ? <p className="empty-history">Completed and paused interviews will appear here.</p> : (
            <div className="history-list">
              {history.map((item) => (
                <button type="button" key={item.id} onClick={() => onOpenInterview(item.id)} disabled={item.status === 'preparing' || item.status === 'failed'}>
                  <span><strong>{item.inferred_role || item.requested_role}</strong><small>{new Date(item.created_at).toLocaleString()}</small></span>
                  <span className={`history-status ${item.status}`}>{item.status} · {item.accepted_answer_count} answers</span>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default Dashboard;
