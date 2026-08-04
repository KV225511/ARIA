from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
import uuid
import json
import logging
from typing import Dict, Any
import fitz  # PyMuPDF

# Import ARIA Modules
from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.environment import ARIAInterviewEnv
from modules.module_08_llm.generator import LLMQuestionGenerator
from modules.module_09_tts.engine import TTSAvatarBaseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ARIA Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For MVP frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to hold active sessions
sessions: Dict[str, Dict[str, Any]] = {}

# Pre-load heavy singletons
try:
    llm_gen = LLMQuestionGenerator()
    tts_engine = TTSAvatarBaseline()
except Exception as e:
    logger.error(f"Failed to load singletons: {e}")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Helper to extract text from a PDF memory stream using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
    return text.strip()

@app.post("/api/start-session")
async def start_session(
    job_description: UploadFile = File(...),
    resume: UploadFile = File(...),
    role_name: str = Form("backend_developer")
):
    """
    Initializes a new interview session using uploaded PDFs. Runs Module 5.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Starting session {session_id} for role {role_name}")
    
    # Extract text from PDFs
    jd_bytes = await job_description.read()
    resume_bytes = await resume.read()
    
    jd_text = extract_text_from_pdf(jd_bytes)
    resume_text = extract_text_from_pdf(resume_bytes)
    
    if not jd_text or not resume_text:
        raise HTTPException(status_code=400, detail="Failed to extract text from the provided PDFs.")
    
    # 1. Initialize Ontology (Module 5)
    ontology = SkillOntologyGraph(role_name=role_name)
    adaptation_success = ontology.adapt_to_candidate(jd_text, resume_text)
    
    # 2. Initialize Belief State (Module 6)
    all_skills = ontology.get_all_skills()
    belief_updater = BeliefStateUpdater(all_skills)
    
    # Store session state
    sessions[session_id] = {
        "ontology": ontology,
        "belief": belief_updater,
        "history": [],
        "resume": resume_text, # Save text for Module 8 context
        "turn": 0
    }
    
    return {
        "session_id": session_id,
        "adaptation_success": adaptation_success,
        "skills_loaded": len(all_skills)
    }

@app.websocket("/ws/interview/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time interview interactions.
    The frontend sends text/audio/video markers, and ARIA responds.
    """
    await websocket.accept()
    
    if session_id not in sessions:
        await websocket.send_json({"type": "error", "message": "Invalid session ID"})
        await websocket.close()
        return
        
    session = sessions[session_id]
    
    # Start the interview loop by choosing the first action
    # For MVP, we hardcode the first action to 'probe_foundation'
    # In full production, this comes from Module 7 RL agent.
    try:
        first_action = "probe_foundation"
        # convert numpy arrays to lists for JSON serialization
        current_belief = {k: v.tolist() for k, v in session["belief"].beliefs.items()}
        
        # Module 8: Generate Question (Streaming)
        question_text = ""
        for chunk in llm_gen.generate_question_stream(
            action=first_action,
            belief_state=current_belief,
            resume=session["resume"],
            history=session["history"]
        ):
            question_text += chunk
            await websocket.send_json({
                "type": "aria_chunk",
                "text": chunk
            })
            
        # Send final completion event
        await websocket.send_json({
            "type": "aria_question",
            "text": question_text,
            "action": first_action
        })
        
        # Module 9: Speak it
        # tts_engine.speak(question_text) # Uncomment if you want actual computer audio
        
        # The main interview loop
        import base64
        import tempfile
        import os
        import subprocess
        from modules.module_01_stt.transcriber import transcribe_file
        
        while True:
            # Wait for candidate's answer from the UI
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            candidate_text = ""
            
            if payload.get("type") == "candidate_answer":
                candidate_text = payload.get("text", "")
                
            elif payload.get("type") == "candidate_audio":
                audio_b64 = payload.get("audio_base64", "")
                if audio_b64.startswith("data:audio/webm;base64,"):
                    audio_b64 = audio_b64.split(",")[1]
                    
                audio_bytes = base64.b64decode(audio_b64)
                
                # Save webm
                webm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"temp_{session_id}.webm")
                wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"temp_{session_id}.wav")
                
                with open(webm_path, "wb") as f:
                    f.write(audio_bytes)
                
                # Convert webm to wav using ffmpeg
                subprocess.run(["ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path, "-y"], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Transcribe
                try:
                    stt_result = await transcribe_file(wav_path)
                    candidate_text = stt_result.get("transcript", "")
                except Exception as e:
                    logger.error(f"Whisper transcription failed: {e}")
                    candidate_text = ""
                    
                # Cleanup
                if os.path.exists(webm_path): os.remove(webm_path)
                if os.path.exists(wav_path): os.remove(wav_path)
                
                # Inform UI of transcription
                await websocket.send_json({
                    "type": "transcription_result",
                    "text": candidate_text
                })
            
            if candidate_text:
                # Save to history
                session["history"].append({
                    "q": question_text,
                    "a": candidate_text
                })
                
                # Fake Perception Processing (Normally Modules 1-4)
                # Fake Belief Update (Module 6)
                # We'll just randomly update a skill to show progress
                skills = session["ontology"].get_all_skills()
                if skills:
                    session["belief"].update_belief(skills[0], semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
                
                # Next Action (Module 7 Fake)
                session["turn"] += 1
                next_action = "increase_difficulty" if session["turn"] % 2 == 0 else "probe_foundation"
                
                # Generate next question (Module 8) Streaming
                new_belief = {k: v.tolist() for k, v in session["belief"].beliefs.items()}
                question_text = ""
                for chunk in llm_gen.generate_question_stream(
                    action=next_action,
                    belief_state=new_belief,
                    resume=session["resume"],
                    history=session["history"]
                ):
                    question_text += chunk
                    await websocket.send_json({
                        "type": "aria_chunk",
                        "text": chunk
                    })
                
                # Send back to UI final state
                await websocket.send_json({
                    "type": "aria_question",
                    "text": question_text,
                    "action": next_action,
                    "belief_state": new_belief
                })
                
    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        try:
            await websocket.close()
        except:
            pass
