import React, { useState } from 'react';
import SetupScreen from './components/SetupScreen';
import InterviewScreen from './components/InterviewScreen';
import './index.css';

function App() {
  const [sessionId, setSessionId] = useState(null);

  return (
    <>
      {!sessionId ? (
        <SetupScreen onStartSession={setSessionId} />
      ) : (
        <InterviewScreen sessionId={sessionId} />
      )}
    </>
  );
}

export default App;
