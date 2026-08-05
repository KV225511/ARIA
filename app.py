from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
import uuid
import json
import logging
from typing import Dict, Any
import fitz  # PyMuPDF
import base64
import tempfile
import os
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Import ARIA Modules
from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.environment import ARIAInterviewEnv
from modules.module_08_llm.generator import LLMQuestionGenerator
from modules.module_09_tts.engine import TTSAvatarBaseline
from modules.module_01_stt.transcriber import transcribe_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ARIA Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to hold active sessions
sessions: Dict[str, Dict[str, Any]] = {}

llm_gen = None
tts_engine = None

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
    
    # Offload PDF extraction to thread pool
    jd_text = await asyncio.to_thread(extract_text_from_pdf, jd_bytes)
    resume_text = await asyncio.to_thread(extract_text_from_pdf, resume_bytes)
    
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
        "role": getattr(ontology, "inferred_role", role_name),
        "experience": getattr(ontology, "inferred_experience", "Mid-Level"),
        "turn": 0
    }
    
    return {
        "session_id": session_id,
        "adaptation_success": adaptation_success,
        "skills_loaded": len(all_skills),
        "inferred_role": sessions[session_id]["role"],
        "inferred_experience": sessions[session_id]["experience"]
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
        
    if not llm_gen:
        await websocket.send_json({"type": "error", "message": "LLM Generator is offline"})
        await websocket.close()
        return

    session = sessions[session_id]
    
    try:
        first_action = "probe_foundation"
        current_belief = {k: v.tolist() for k, v in session["belief"].beliefs.items()}
        
        # Module 8: Generate Question (Streaming)
        question_text = ""
        async for chunk in llm_gen.generate_question_stream(
            action=first_action,
            belief_state=current_belief,
            resume=session["resume"],
            history=session["history"],
            role=session["role"],
            experience=session["experience"]
        ):
            if isinstance(chunk, dict) and chunk.get("type") == "prompt_debug":
                await websocket.send_json(chunk)
                continue
                
            question_text += chunk
            await websocket.send_json({
                "type": "aria_chunk",
                "text": chunk
            })
            
        await websocket.send_json({
            "type": "aria_question",
            "text": question_text,
            "action": first_action
        })
        
        # if tts_engine: tts_engine.speak(question_text)
        
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            candidate_text = ""
            
            if payload.get("type") == "candidate_answer":
                candidate_text = payload.get("text", "")
                
            elif payload.get("type") == "candidate_audio":
                audio_b64 = payload.get("audio_base64", "")
                if "base64," in audio_b64:
                    audio_b64 = audio_b64.split("base64,")[1]
                    
                audio_bytes = base64.b64decode(audio_b64)
                
                # Use a dedicated temp directory for audio files
                temp_dir = tempfile.gettempdir()
                webm_path = os.path.join(temp_dir, f"temp_{session_id}.webm")
                wav_path = os.path.join(temp_dir, f"temp_{session_id}.wav")
                
                with open(webm_path, "wb") as f:
                    f.write(audio_bytes)
                
                # Convert webm to wav using ffmpeg asynchronously
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path, "-y",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await process.communicate()
                
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
                
                await websocket.send_json({
                    "type": "transcription_result",
                    "text": candidate_text
                })
            
            if candidate_text:
                session["history"].append({
                    "q": question_text,
                    "a": candidate_text
                })
                
                skills = session["ontology"].get_all_skills()
                if skills:
                    session["belief"].update_belief(skills[0], semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
                
                session["turn"] += 1
                next_action = "increase_difficulty" if session["turn"] % 2 == 0 else "probe_foundation"
                
                new_belief = {k: v.tolist() for k, v in session["belief"].beliefs.items()}
                question_text = ""
                async for chunk in llm_gen.generate_question_stream(
                    action=next_action,
                    belief_state=new_belief,
                    resume=session["resume"],
                    history=session["history"],
                    role=session["role"],
                    experience=session["experience"]
                ):
                    if isinstance(chunk, dict) and chunk.get("type") == "prompt_debug":
                        await websocket.send_json(chunk)
                        continue
                        
                    question_text += chunk
                    await websocket.send_json({
                        "type": "aria_chunk",
                        "text": chunk
                    })
                
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
    finally:
        # Clean up session on disconnect to prevent memory leaks
        if session_id in sessions:
            logger.info(f"Cleaning up session: {session_id}")
            del sessions[session_id]
