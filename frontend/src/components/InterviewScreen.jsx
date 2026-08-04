import React, { useEffect, useState, useRef } from 'react';

function InterviewScreen({ sessionId }) {
  const [ws, setWs] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [candidateInput, setCandidateInput] = useState('');
  
  // Real implementation would use mediaRecorder for audio/video chunks
  // For MVP, we will simulate candidate speaking via text input

  useEffect(() => {
    // Connect to WebSocket
    const socket = new WebSocket(`ws://localhost:8000/ws/interview/${sessionId}`);
    
    socket.onopen = () => {
      console.log('Connected to ARIA Orchestrator');
    };
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'aria_question') {
        setIsSpeaking(true);
        setMessages(prev => [...prev, { sender: 'ARIA', text: data.text, action: data.action }]);
        
        // Simulate TTS speaking duration
        setTimeout(() => setIsSpeaking(false), 3000);
      }
    };
    
    socket.onclose = () => console.log('Disconnected');
    
    setWs(socket);
    
    return () => socket.close();
  }, [sessionId]);

  const handleSendAnswer = (e) => {
    e.preventDefault();
    if (!candidateInput.trim() || !ws) return;
    
    setMessages(prev => [...prev, { sender: 'Candidate', text: candidateInput }]);
    
    // Send to backend
    ws.send(JSON.stringify({
      type: 'candidate_answer',
      text: candidateInput
    }));
    
    setCandidateInput('');
  };

  return (
    <div style={{ width: '100%', maxWidth: '1000px', padding: '2rem' }}>
      
      {/* Top Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <h2 style={{ color: 'var(--accent)' }}>ARIA Interview Session</h2>
        <div style={{ 
          background: ws ? 'rgba(0, 255, 204, 0.1)' : 'rgba(255, 77, 77, 0.1)',
          color: ws ? 'var(--accent)' : 'var(--error)',
          padding: '8px 16px',
          borderRadius: '20px',
          fontSize: '0.9rem',
          fontWeight: 600
        }}>
          {ws ? 'Live Connection Active' : 'Disconnected'}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '2rem', height: '600px' }}>
        
        {/* Left Panel: Avatar & Visuals */}
        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          
          <div className={`avatar-ring ${isSpeaking ? 'speaking' : ''}`}>
            {isSpeaking ? '🗣️' : '🧠'}
          </div>
          
          <h3 style={{ marginTop: '2rem', color: isSpeaking ? 'var(--accent)' : 'var(--text-secondary)' }}>
            {isSpeaking ? 'ARIA is speaking...' : 'ARIA is listening...'}
          </h3>
          
          <div style={{ marginTop: 'auto', width: '100%', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', height: '150px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
            [Webcam Feed Placeholder]
          </div>
        </div>

        {/* Right Panel: Transcript & Interaction */}
        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1rem', borderBottom: '1px solid var(--panel-border)', paddingBottom: '0.5rem' }}>Live Transcript</h3>
          
          <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {messages.map((msg, idx) => (
              <div key={idx} style={{ 
                alignSelf: msg.sender === 'Candidate' ? 'flex-end' : 'flex-start',
                background: msg.sender === 'Candidate' ? 'rgba(255,255,255,0.1)' : 'rgba(0, 255, 204, 0.1)',
                padding: '12px 16px',
                borderRadius: '12px',
                maxWidth: '85%',
                border: msg.sender === 'ARIA' ? '1px solid var(--accent)' : 'none'
              }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  {msg.sender} {msg.action && <span style={{ color: 'var(--accent)' }}>[{msg.action}]</span>}
                </div>
                <div>{msg.text}</div>
              </div>
            ))}
          </div>

          <form onSubmit={handleSendAnswer} style={{ display: 'flex', gap: '1rem' }}>
            <input 
              type="text" 
              className="input-field" 
              style={{ marginBottom: 0, flex: 1 }}
              placeholder="Simulate speech to text answer..."
              value={candidateInput}
              onChange={(e) => setCandidateInput(e.target.value)}
              disabled={isSpeaking}
            />
            <button type="submit" className="btn" disabled={isSpeaking || !candidateInput.trim()}>
              Send
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}

export default InterviewScreen;
