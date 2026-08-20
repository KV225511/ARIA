import { useEffect, useMemo, useRef, useState } from 'react';

const MicIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="8.5" y="3.5" width="7" height="11" rx="3.5" />
    <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M8.5 21h7" />
  </svg>
);

const SendIcon = () => (
  <svg viewBox="0 0 20 20" aria-hidden="true">
    <path d="m4 10 12-6-3.5 12-2.25-4.25L4 10Z" />
    <path d="m10.25 11.75 2.5-2.5" />
  </svg>
);

const CameraIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="3.5" y="6.5" width="13" height="11" rx="2" />
    <path d="m16.5 10 4-2v8l-4-2" />
  </svg>
);

function SpeakingBars() {
  return (
    <span className="speaking-bars" aria-hidden="true">
      {Array.from({ length: 7 }, (_, index) => <i key={index} />)}
    </span>
  );
}

function InterviewScreen({ sessionId, onEndSession }) {
  const [connectionState, setConnectionState] = useState('connecting');
  const [messages, setMessages] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPreparing, setIsPreparing] = useState(true);
  const [candidateInput, setCandidateInput] = useState('');
  const [debugPrompt, setDebugPrompt] = useState('');
  const [stream, setStream] = useState(null);
  const [mediaState, setMediaState] = useState('requesting');
  const [recordingMode, setRecordingMode] = useState('idle');

  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const transcriptRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const pointerDownTimeRef = useRef(0);
  const activeStreamRef = useRef(null);

  const isRecording = recordingMode !== 'idle';
  const ariaMessages = useMemo(
    () => messages.filter((message) => message.sender === 'ARIA'),
    [messages],
  );
  const currentQuestion = ariaMessages.at(-1);
  const questionNumber = ariaMessages.length;
  const currentFocus = currentQuestion?.action?.replaceAll('_', ' ') || 'Building skill map';

  useEffect(() => {
    let isMounted = true;

    if (!navigator.mediaDevices?.getUserMedia) {
      setMediaState('unavailable');
      return undefined;
    }

    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then((mediaStream) => {
        if (!isMounted) {
          mediaStream.getTracks().forEach((track) => track.stop());
          return;
        }
        activeStreamRef.current?.getTracks().forEach((track) => track.stop());
        activeStreamRef.current = mediaStream;
        setStream(mediaStream);
        setMediaState('ready');
      })
      .catch((error) => {
        console.error('Error accessing media devices:', error);
        setMediaState('blocked');
      });

    return () => {
      isMounted = false;
      activeStreamRef.current?.getTracks().forEach((track) => track.stop());
      activeStreamRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
  }, [stream]);

  useEffect(() => {
    const socket = new WebSocket(`ws://localhost:8000/ws/interview/${sessionId}`);
    wsRef.current = socket;
    setConnectionState('connecting');

    socket.onopen = () => setConnectionState('live');
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'aria_chunk') {
        setIsPreparing(true);
      } else if (data.type === 'aria_question') {
        setIsPreparing(false);
        setIsSpeaking(true);
        setMessages((previous) => [...previous, { sender: 'ARIA', text: data.text, action: data.action }]);

        const utterance = new SpeechSynthesisUtterance(data.text);
        utterance.rate = 1.02;
        utterance.pitch = 1;
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = () => setIsSpeaking(false);
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      } else if (data.type === 'transcription_result') {
        setMessages((previous) => [...previous, { sender: 'Candidate', text: data.text }]);
      } else if (data.type === 'prompt_debug') {
        setDebugPrompt(data.prompt);
      } else if (data.type === 'error') {
        setConnectionState('error');
        setIsPreparing(false);
      }
    };
    socket.onerror = () => {
      setConnectionState('error');
      setIsPreparing(false);
    };
    socket.onclose = () => setConnectionState((current) => (
      current === 'error' ? 'error' : 'disconnected'
    ));

    return () => {
      window.speechSynthesis.cancel();
      socket.close();
      wsRef.current = null;
    };
  }, [sessionId]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);

  const sendCandidateText = (text) => {
    const socket = wsRef.current;
    if (!text.trim() || socket?.readyState !== WebSocket.OPEN) return;
    setMessages((previous) => [...previous, { sender: 'Candidate', text: text.trim() }]);
    setIsPreparing(true);
    socket.send(JSON.stringify({ type: 'candidate_answer', text: text.trim() }));
  };

  const handleSendAnswer = (event) => {
    event.preventDefault();
    sendCandidateText(candidateInput);
    setCandidateInput('');
  };

  const startRecording = () => {
    if (!stream || isSpeaking || mediaRecorderRef.current?.state === 'recording') return;
    audioChunksRef.current = [];
    try {
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
          const socket = wsRef.current;
          if (socket?.readyState === WebSocket.OPEN) {
            setIsPreparing(true);
            socket.send(JSON.stringify({ type: 'candidate_audio', audio_base64: reader.result }));
          }
        };
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecordingMode('hold');
    } catch (error) {
      console.error('Failed to start MediaRecorder:', error);
      setRecordingMode('idle');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state !== 'inactive') mediaRecorderRef.current?.stop();
    setRecordingMode('idle');
  };

  const handlePointerDown = (event) => {
    if (event.button !== undefined && event.button !== 0) return;
    if (recordingMode === 'locked') {
      stopRecording();
      pointerDownTimeRef.current = 0;
      return;
    }
    if (recordingMode === 'idle') {
      pointerDownTimeRef.current = Date.now();
      startRecording();
    }
  };

  const handlePointerUp = () => {
    if (!pointerDownTimeRef.current) return;
    const duration = Date.now() - pointerDownTimeRef.current;
    if (duration < 300) setRecordingMode('locked');
    else stopRecording();
    pointerDownTimeRef.current = 0;
  };

  const handlePointerLeave = () => {
    if (recordingMode === 'hold') stopRecording();
    pointerDownTimeRef.current = 0;
  };

  const connectionLabel = {
    connecting: 'Connecting',
    live: 'Live',
    disconnected: 'Disconnected',
    error: 'Connection issue',
  }[connectionState];

  return (
    <div className="interview-page">
      <header className="interview-header">
        <div className="brand brand-inverse" aria-label="ARIA">
          <span className="brand-mark">A</span><span>ARIA</span>
        </div>
        <div className="interview-title">
          <span>Adaptive interview</span>
          <strong>{questionNumber ? `Question ${String(questionNumber).padStart(2, '0')}` : 'Preparing session'}</strong>
        </div>
        <div className="header-actions">
          <span className={`connection-pill ${connectionState}`}><i /> {connectionLabel}</span>
          <button type="button" className="end-button" onClick={onEndSession}>End session</button>
        </div>
      </header>

      <div className="interview-workspace">
        <aside className="candidate-rail">
          <section className="camera-card">
            <div className="camera-frame">
              {stream ? <video ref={videoRef} autoPlay playsInline muted /> : (
                <div className="camera-placeholder">
                  <CameraIcon />
                  <span>{mediaState === 'requesting' ? 'Starting camera' : 'Camera unavailable'}</span>
                </div>
              )}
              <div className="camera-overlay">
                <span><i /> Candidate</span>
                <span>{mediaState === 'ready' ? 'Camera on' : 'No video'}</span>
              </div>
            </div>
            <div className="camera-caption">
              <div>
                <span className="meta-label">Presence</span>
                <strong>{mediaState === 'ready' ? 'Audio and video ready' : 'Limited media access'}</strong>
              </div>
              <span className={`device-indicator ${mediaState}`}><MicIcon /></span>
            </div>
          </section>

          <section className="session-panel">
            <div className="panel-title-row">
              <span className="meta-label">Session</span><span className="secure-label">Local</span>
            </div>
            <dl className="session-details">
              <div><dt>Mode</dt><dd>Adaptive</dd></div>
              <div><dt>Input</dt><dd>Voice + text</dd></div>
              <div><dt>Session ID</dt><dd>{sessionId.slice(0, 8)}</dd></div>
            </dl>
          </section>
        </aside>

        <section className="interview-main">
          <div className="question-stage">
            <div className="question-meta">
              <span>{connectionState === 'error' || connectionState === 'disconnected' ? 'Connection unavailable' : isSpeaking ? 'ARIA is speaking' : isPreparing ? 'Reviewing evidence' : 'Your turn'}</span>
              {isSpeaking && <SpeakingBars />}
            </div>
            {currentQuestion ? <h1>{currentQuestion.text}</h1> : (
              <div className="question-loading" aria-label="Preparing the first question"><span /><span /><span /></div>
            )}
          </div>

          <div className="conversation-panel">
            <div className="conversation-heading">
              <div><span className="meta-label">Conversation</span><strong>Live transcript</strong></div>
              <span>{messages.length} entries</span>
            </div>
            <div className="transcript" ref={transcriptRef} aria-live="polite">
              {messages.length === 0 ? (
                <div className="empty-transcript"><span className="empty-line" /><p>The transcript will build here as the interview begins.</p></div>
              ) : messages.map((message, index) => (
                <article className={`message ${message.sender === 'Candidate' ? 'candidate-message' : 'aria-message'}`} key={`${message.sender}-${index}`}>
                  <div className="message-author">
                    <span>{message.sender === 'Candidate' ? 'You' : 'ARIA'}</span>
                    {message.action && <small>{message.action.replaceAll('_', ' ')}</small>}
                  </div>
                  <p>{message.text}</p>
                </article>
              ))}
            </div>

            <div className={`answer-composer ${isRecording ? 'is-recording' : ''}`}>
              <button
                type="button"
                className="record-button"
                onPointerDown={handlePointerDown}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerLeave}
                disabled={isSpeaking || mediaState !== 'ready' || connectionState !== 'live'}
                aria-label={isRecording ? 'Stop recording' : 'Record answer'}
              ><MicIcon /></button>
              <form onSubmit={handleSendAnswer}>
                <label htmlFor="candidate-answer">
                  {recordingMode === 'locked' ? 'Recording — tap the microphone to send' : recordingMode === 'hold' ? 'Recording — release to send' : 'Answer by voice or type below'}
                </label>
                <div className="text-entry">
                  <input
                    id="candidate-answer"
                    type="text"
                    placeholder="Type your answer…"
                    value={candidateInput}
                    onChange={(event) => setCandidateInput(event.target.value)}
                    disabled={isSpeaking || isRecording || connectionState !== 'live'}
                  />
                  <button type="submit" disabled={isSpeaking || isRecording || !candidateInput.trim() || connectionState !== 'live'} aria-label="Send answer"><SendIcon /></button>
                </div>
              </form>
            </div>
          </div>
        </section>

        <aside className="context-rail">
          <section className="focus-panel">
            <span className="meta-label">Current strategy</span>
            <div className="focus-index">{String(Math.max(questionNumber, 1)).padStart(2, '0')}</div>
            <strong>{currentFocus}</strong>
            <p>ARIA changes depth and direction as the evidence becomes clearer.</p>
          </section>
          <section className="signal-panel">
            <div className="panel-title-row"><span className="meta-label">System signals</span><span>Live</span></div>
            <ul>
              <li><span>Interview policy</span><i className="signal-ready" /></li>
              <li><span>Belief updates</span><i className="signal-ready" /></li>
              <li><span>Session privacy</span><i className="signal-ready" /></li>
            </ul>
          </section>
          {debugPrompt && (
            <details className="debug-panel"><summary>Developer trace</summary><pre>{debugPrompt}</pre></details>
          )}
        </aside>
      </div>
    </div>
  );
}

export default InterviewScreen;
