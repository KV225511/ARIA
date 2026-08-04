import React, { useState } from 'react';

function SetupScreen({ onStartSession }) {
  const [jdFile, setJdFile] = useState(null);
  const [resumeFile, setResumeFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!jdFile || !resumeFile) return;

    setLoading(true);
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
      
      const data = await res.json();
      if (data.session_id) {
        onStartSession(data.session_id);
      } else {
        alert(data.detail || 'Failed to start session');
      }
    } catch (error) {
      console.error('Failed to start session', error);
      alert('Failed to connect to backend. Is uvicorn running?');
    }
    setLoading(false);
  };

  return (
    <div className="glass-panel" style={{ width: '600px', maxWidth: '90%' }}>
      <h1 style={{ marginBottom: '1.5rem', textAlign: 'center' }}>ARIA Initialization</h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', textAlign: 'center' }}>
        Upload the target Job Description and Candidate Resume (PDFs) to generate the dynamic skill ontology.
      </p>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Job Description (PDF)</label>
          <input 
            type="file"
            accept="application/pdf"
            className="input-field" 
            onChange={(e) => setJdFile(e.target.files[0])}
          />
        </div>

        <div style={{ marginBottom: '2rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Candidate Resume (PDF)</label>
          <input 
            type="file"
            accept="application/pdf"
            className="input-field" 
            onChange={(e) => setResumeFile(e.target.files[0])}
          />
        </div>

        <button 
          type="submit" 
          className="btn" 
          style={{ width: '100%' }}
          disabled={loading || !jdFile || !resumeFile}
        >
          {loading ? 'Parsing PDFs & Generating Ontology...' : 'Start Interview'}
        </button>
      </form>
    </div>
  );
}

export default SetupScreen;
