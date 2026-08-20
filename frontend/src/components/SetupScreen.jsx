import { useRef, useState } from 'react';

const FileIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M7.75 3.75h5.5l4 4v12.5h-9.5a2 2 0 0 1-2-2V5.75a2 2 0 0 1 2-2Z" />
    <path d="M13.25 3.75v4h4M8.75 12h5.5M8.75 15.5h4" />
  </svg>
);

const ArrowIcon = () => (
  <svg viewBox="0 0 20 20" aria-hidden="true">
    <path d="M4 10h12M11.5 5.5 16 10l-4.5 4.5" />
  </svg>
);

function DocumentField({ label, hint, file, onChange }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const acceptFile = (selectedFile) => {
    if (selectedFile?.type === 'application/pdf' || selectedFile?.name.toLowerCase().endsWith('.pdf')) {
      onChange(selectedFile);
    }
  };

  return (
    <div className="document-field">
      <div className="field-heading">
        <label>{label}</label>
        <span>PDF</span>
      </div>
      <button
        type="button"
        className={`file-dropzone ${file ? 'has-file' : ''} ${dragActive ? 'is-dragging' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragActive(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragActive(false);
          acceptFile(event.dataTransfer.files[0]);
        }}
      >
        <span className="file-icon"><FileIcon /></span>
        <span className="file-copy">
          <strong>{file ? file.name : 'Choose a file or drop it here'}</strong>
          <small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready` : hint}</small>
        </span>
        <span className="file-action">{file ? 'Replace' : 'Browse'}</span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => acceptFile(event.target.files[0])}
        hidden
      />
    </div>
  );
}

function SetupScreen({ onStartSession }) {
  const [jdFile, setJdFile] = useState(null);
  const [resumeFile, setResumeFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!jdFile || !resumeFile) return;

    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('job_description', jdFile);
      formData.append('resume', resumeFile);
      formData.append('role_name', 'backend_developer');

      // POST to FastAPI backend
      const res = await fetch('http://localhost:8000/api/start-session', {
        method: 'POST',
        // Note: Do NOT set Content-Type header manually when using FormData, 
        // the browser sets it automatically with the correct boundary.
        body: formData
      });
      
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.session_id) {
        onStartSession(data.session_id);
      } else {
        setError(data.detail || 'We could not prepare this interview. Check both PDFs and try again.');
      }
    } catch (error) {
      console.error('Failed to start session', error);
      setError('ARIA could not reach the local interview service. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="setup-page">
      <header className="site-header">
        <a className="brand" href="/" aria-label="ARIA home">
          <span className="brand-mark">A</span>
          <span>ARIA</span>
        </a>
        <div className="header-meta">
          <span className="status-dot" />
          Private local mode
        </div>
      </header>

      <div className="setup-grid">
        <section className="setup-intro" aria-labelledby="setup-title">
          <p className="eyebrow">Adaptive interview system</p>
          <h1 id="setup-title">Every question should earn its place.</h1>
          <p className="intro-copy">
            ARIA builds a focused interview from the role and the candidate—then adapts in real time as evidence develops.
          </p>

          <div className="process-list" aria-label="How ARIA prepares an interview">
            <div className="process-item">
              <span>01</span>
              <div>
                <strong>Map the role</strong>
                <p>Extract the skills and dependencies that matter.</p>
              </div>
            </div>
            <div className="process-item">
              <span>02</span>
              <div>
                <strong>Understand the candidate</strong>
                <p>Ground questions in relevant claims and experience.</p>
              </div>
            </div>
            <div className="process-item">
              <span>03</span>
              <div>
                <strong>Adapt with evidence</strong>
                <p>Probe, validate, and change depth as the session unfolds.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="setup-card" aria-labelledby="form-title">
          <div className="card-heading">
            <div>
              <p className="section-kicker">New session</p>
              <h2 id="form-title">Build the interview</h2>
            </div>
            <span className="step-label">2 documents</span>
          </div>

          <form onSubmit={handleSubmit}>
            <DocumentField
              label="Job description"
              hint="The role, responsibilities, and requirements"
              file={jdFile}
              onChange={setJdFile}
            />
            <DocumentField
              label="Candidate résumé"
              hint="The candidate’s background and experience"
              file={resumeFile}
              onChange={setResumeFile}
            />

            {error && <div className="form-error" role="alert">{error}</div>}

            <button
              type="submit"
              className="primary-button"
              disabled={loading || !jdFile || !resumeFile}
            >
              <span>{loading ? 'Preparing interview' : 'Prepare interview'}</span>
              {loading ? <span className="button-spinner" aria-hidden="true" /> : <ArrowIcon />}
            </button>
            <p className="privacy-note">Documents stay on this device and are used only for this session.</p>
          </form>
        </section>
      </div>

      <footer className="setup-footer">
        <span>Autonomous Reinforcement-Based Interview Agent</span>
        <span>Private · Local · Evidence-led</span>
      </footer>
    </div>
  );
}

export default SetupScreen;
