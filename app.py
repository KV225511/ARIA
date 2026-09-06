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
from modules.module_05_ontology.grounding import (
    grounding_packet,
    normalize_generated_question,
    validate_grounded_question,
)
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
    if not adaptation_success or ontology.role_profile is None:
        raise HTTPException(
            status_code=422,
            detail="The job description could not be converted into a grounded role profile.",
        )
    
    # 2. Initialize Belief State (Module 6)
    all_skills = ontology.get_all_skills()
    belief_updater = BeliefStateUpdater(all_skills)
    
    # Store session state
    sessions[session_id] = {
        "ontology": ontology,
        "belief": belief_updater,
        "history": [],
        "resume": resume_text, # Save text for Module 8 context
        "jd": jd_text,
        "role_profile": ontology.role_profile,
        "current_target": None,
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


async def generate_grounded_session_question(session: dict, action: str) -> str:
    profile = session["role_profile"]
    skills = list(profile.skills)
    if not skills:
        raise RuntimeError("Session role profile has no assessable skills")
    current = session.get("current_target")
    candidates = []
    if current and action == "increase_difficulty":
        candidates = session["ontology"].get_advanced(current.canonical_name)
    elif current and action in {"decrease_difficulty", "probe_foundation"}:
        candidates = session["ontology"].get_prerequisites(current.canonical_name)
    elif current and action != "switch_topic":
        candidates = [current.canonical_name]
    if not candidates:
        candidates = [
            skill.canonical_name for skill in skills
            if current is None or skill.skill_id != current.skill_id
        ] or [current.canonical_name]
    target = min(
        (session["ontology"].get_skill_metadata(name) for name in candidates),
        key=lambda skill: (
            skill.selection_priority,
            session["belief"].get_evidence_count(skill.canonical_name),
            skill.skill_id,
        ),
    )
    context = grounding_packet(profile, target)
    belief = {k: v.tolist() for k, v in session["belief"].beliefs.items()}
    rejected: list[list[str]] = []
    for _ in range(3):
        correction = "; ".join(rejected[-1]) if rejected else None
        raw_question = await llm_gen.generate_question(
            action=action,
            belief_state=belief,
            resume=session["resume"],
            history=session["history"],
            role=session["role"],
            experience=session["experience"],
            target_skill=target.canonical_name,
            grounding_context=context,
            correction=correction,
        )
        question = normalize_generated_question(raw_question)
        result = validate_grounded_question(question, context, session["history"])
        if result["valid"]:
            session["current_target"] = target
            return question
        rejected.append(result["reasons"])
    raise RuntimeError(f"Question grounding failed: {rejected[-1]}")

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
        first_action = "switch_topic"
        question_text = await generate_grounded_session_question(session, first_action)
            
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
                
                if session.get("current_target"):
                    session["belief"].update_belief(
                        session["current_target"].canonical_name,
                        semantic_score=0.9,
                        cognitive_load="low",
                        behavior_score=0.9,
                    )
                
                session["turn"] += 1
                action_cycle = (
                    "probe_foundation",
                    "increase_difficulty",
                    "switch_topic",
                )
                next_action = action_cycle[(session["turn"] - 1) % len(action_cycle)]
                
                new_belief = {k: v.tolist() for k, v in session["belief"].beliefs.items()}
                question_text = await generate_grounded_session_question(
                    session, next_action
                )
                
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
