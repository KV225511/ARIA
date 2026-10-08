import { useEffect, useState } from 'react';
import { apiFetch, loadCsrf, responseData } from './api';
import Dashboard from './components/Dashboard';
import InterviewScreen from './components/InterviewScreen';
import SignInScreen from './components/SignInScreen';
import './index.css';

function App() {
  const [user, setUser] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    apiFetch('/api/auth/me').then(responseData)
      .then(async (account) => {
        setUser(account);
        await loadCsrf();
      })
      .catch((authError) => {
        if (!authError.message.toLowerCase().includes('authentication')) setError(authError.message);
      })
      .finally(() => setLoading(false));
  }, []);

  const logout = async () => {
    try { await apiFetch('/api/auth/logout', { method: 'POST' }); } finally {
      setUser(null);
      setSessionId(null);
    }
  };

  if (loading) return <main className="app-shell"><div className="app-loading">Loading ARIA…</div></main>;
  if (!user) return <main className="app-shell"><SignInScreen error={error} /></main>;

  return (
    <main className="app-shell">
      {!sessionId ? (
        <Dashboard user={user} onOpenInterview={setSessionId} onLogout={logout} />
      ) : (
        <InterviewScreen
          sessionId={sessionId}
          onEndSession={() => setSessionId(null)}
        />
      )}
    </main>
  );
}

export default App;
