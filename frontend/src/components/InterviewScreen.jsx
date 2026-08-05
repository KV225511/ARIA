import React, { useEffect, useState, useRef } from 'react';

function InterviewScreen({ sessionId }) {
  const [ws, setWs] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [candidateInput, setCandidateInput] = useState('');
  const [debugPrompt, setDebugPrompt] = useState('');
  
  // Media states and refs
  const [stream, setStream] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Interaction refs for tap-to-record
  const mouseDownTime = useRef(0);
  const isToggledRef = useRef(false);
  const activeStreamRef = useRef(null);

  // Initialize media devices
  useEffect(() => {
    let isMounted = true;
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then(s => {
        if (!isMounted) {
          s.getTracks().forEach(track => track.stop());
          return;
        }
        if (activeStreamRef.current) {
          activeStreamRef.current.getTracks().forEach(track => track.stop());
        }
        activeStreamRef.current = s;
        setStream(s);
      })
      .catch(err => console.error("Error accessing media devices:", err));
      
    return () => {
      isMounted = false;
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach(track => track.stop());
        activeStreamRef.current = null;
      }
    };
  }, []);

  // Sync video element with stream
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

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
        
        // Native Browser TTS
        const utterance = new SpeechSynthesisUtterance(data.text);
        utterance.rate = 1.05;
        utterance.pitch = 1.1;
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = () => setIsSpeaking(false);
        
        // Cancel any currently playing speech before starting new one
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
        
      } else if (data.type === 'transcription_result') {
        setMessages(prev => [...prev, { sender: 'Candidate', text: data.text }]);
      } else if (data.type === 'prompt_debug') {
        setDebugPrompt(data.prompt);
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

  const startRecording = () => {
    if (!stream) return;
    setIsRecording(true);
    audioChunksRef.current = [];
    
    try {
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      
      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
          const base64Audio = reader.result;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
              type: 'candidate_audio',
              audio_base64: base64Audio
            }));
          }
        };
      };
      
      mediaRecorderRef.current = recorder;
      recorder.start();
    } catch (err) {
      console.error("Failed to start MediaRecorder:", err);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const handlePointerDown = (e) => {
    if (e.button !== undefined && e.button !== 0) return; // Only left click
    
    if (isRecording) {
      if (isToggledRef.current) {
        stopRecording();
        isToggledRef.current = false;
        mouseDownTime.current = 0; // Prevent pointer up from triggering
      }
    } else {
      mouseDownTime.current = Date.now();
      startRecording();
    }
  };

  const handlePointerUp = () => {
    if (mouseDownTime.current === 0) return;
    
    const duration = Date.now() - mouseDownTime.current;
    if (duration < 300) {
      // Quick tap -> toggle mode on
      isToggledRef.current = true;
    } else {
      // Long press -> turn off when released
      stopRecording();
      isToggledRef.current = false;
    }
    mouseDownTime.current = 0;
  };

  const handlePointerLeave = () => {
    if (isRecording && !isToggledRef.current) {
      stopRecording();
      isToggledRef.current = false;
    }
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
          
          <h3 style={{ marginTop: '2rem', color: isSpeaking || isRecording ? 'var(--accent)' : 'var(--text-secondary)' }}>
            {isRecording ? '🎤 You are speaking...' : (isSpeaking ? 'ARIA is speaking...' : 'ARIA is listening...')}
          </h3>
          
          <div style={{ marginTop: 'auto', width: '100%', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', overflow: 'hidden' }}>
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} 
            />
          </div>
          
          {debugPrompt && (
            <div style={{ marginTop: '1rem', width: '100%', background: 'rgba(0,0,0,0.5)', padding: '1rem', borderRadius: '12px', fontSize: '0.75rem', color: '#00ffcc', overflowY: 'auto', maxHeight: '150px', textAlign: 'left', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '4px', color: 'white' }}>LLM Prompt Sent:</div>
              {debugPrompt}
            </div>
          )}
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

          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <button 
              className="btn" 
              style={{ 
                flex: 1, 
                backgroundColor: isRecording ? 'var(--error)' : 'var(--accent)',
                animation: isRecording ? 'pulse 1.5s infinite' : 'none',
                userSelect: 'none',
                touchAction: 'none'
              }}
              onPointerDown={handlePointerDown}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerLeave}
              disabled={isSpeaking || !stream}
            >
              {isRecording ? (isToggledRef.current ? '🎙️ Recording... (Tap to Stop)' : '🎙️ Recording... (Release to Send)') : 'Hold or Tap to Speak'}
            </button>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Or type:
            </div>
          </div>
          
          <form onSubmit={handleSendAnswer} style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
            <input 
              type="text" 
              className="input-field" 
              style={{ marginBottom: 0, flex: 1, opacity: isRecording ? 0.5 : 1 }}
              placeholder={isRecording ? "Recording audio..." : "Simulate speech to text answer..."}
              value={candidateInput}
              onChange={(e) => setCandidateInput(e.target.value)}
              disabled={isSpeaking || isRecording}
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
