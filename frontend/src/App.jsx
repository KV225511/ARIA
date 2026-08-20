import { useState } from 'react';
import SetupScreen from './components/SetupScreen';
import InterviewScreen from './components/InterviewScreen';
import './index.css';

function App() {
  const [sessionId, setSessionId] = useState(null);

  return (
    <main className="app-shell">
      {!sessionId ? (
        <SetupScreen onStartSession={setSessionId} />
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
