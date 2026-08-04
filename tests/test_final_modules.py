import pytest
import os
import json
from unittest.mock import patch, MagicMock

# Import the new modules
from modules.module_08_llm.generator import LLMQuestionGenerator
from modules.module_09_tts.engine import TTSAvatarBaseline
from modules.module_12_incongruence.detector import CrossModalIncongruenceDetector
from modules.module_13_fairness.auditor import InterviewFairnessAuditor
from modules.module_14_evaluation.report import ReportGenerator
from modules.module_15_feedback.logger import FeedbackLogger

# --- Test Module 8 (LLM Generator) ---
@patch('requests.post')
def test_llm_generator_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Could you explain Docker namespaces?"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    generator = LLMQuestionGenerator()
    q = generator.generate_question("probe_foundation", {"Docker": [0.8, 0.1, 0.1]}, "Resume text", [])
    assert q == "Could you explain Docker namespaces?"

def test_llm_generator_fallback():
    # Will fail connection since no Ollama is running locally during CI
    generator = LLMQuestionGenerator(ollama_host="http://localhost:99999")
    q = generator.generate_question("increase_difficulty", {}, "Resume", [])
    assert "Fallback Question" in q
    assert "increase_difficulty" in q

# --- Test Module 9 (TTS Baseline) ---
@patch('pyttsx3.init')
def test_tts_engine_initialization(mock_init):
    mock_engine = MagicMock()
    # Mock property return
    mock_voice = MagicMock()
    mock_voice.id = "test_voice"
    mock_engine.getProperty.return_value = [mock_voice, mock_voice]
    mock_init.return_value = mock_engine

    tts = TTSAvatarBaseline()
    tts.speak("Test")
    mock_engine.say.assert_called_with("Test")
    mock_engine.runAndWait.assert_called_once()
    assert tts.get_avatar_frame() is None

# --- Test Module 12 (Incongruence Detector) ---
def test_incongruence_bluffing():
    detector = CrossModalIncongruenceDetector(incongruence_threshold=0.4)
    # High confidence (0.9) but garbage answer (0.2)
    result = detector.detect(semantic_score=0.2, prosody_confidence=0.9)
    assert result["incongruence_flag"] is True
    assert result["magnitude"] == pytest.approx(0.7)

def test_incongruence_honest_struggle():
    detector = CrossModalIncongruenceDetector(incongruence_threshold=0.4)
    # Low confidence (0.2) and garbage answer (0.2) -> Honest lack of knowledge
    result = detector.detect(semantic_score=0.2, prosody_confidence=0.2)
    assert result["incongruence_flag"] is False
    assert result["magnitude"] == 0.0

# --- Test Module 13 (Fairness Auditor) ---
def test_fairness_auditor():
    auditor = InterviewFairnessAuditor()
    # Mock some female-adjacent high pitch
    auditor.log_turn({"pitch_f0_hz": 220, "speech_rate_syllables_per_sec": 4}, "probe_foundation")
    auditor.log_turn({"pitch_f0_hz": 210, "speech_rate_syllables_per_sec": 4}, "probe_foundation")
    # Mock some male-adjacent low pitch
    auditor.log_turn({"pitch_f0_hz": 110, "speech_rate_syllables_per_sec": 3}, "increase_difficulty")
    auditor.log_turn({"pitch_f0_hz": 100, "speech_rate_syllables_per_sec": 3}, "increase_difficulty")
    
    report = auditor.generate_audit_report()
    assert report["total_turns_audited"] == 4
    assert report["average_pitch_by_action_hz"]["increase_difficulty"] == 105.0
    assert report["average_pitch_by_action_hz"]["probe_foundation"] == 215.0
    assert report["bias_detected"] is True  # > 50Hz diff

# --- Test Module 14 (Report Generator) ---
@patch('requests.post')
def test_report_generator(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Strong candidate."}
    mock_post.return_value = mock_response

    generator = ReportGenerator()
    belief_state = {
        "Python": [0.1, 0.2, 0.7],  # Score: (0.2*0.5 + 0.7*1.0)*100 = 80
        "SQL": [0.0, 0.0, 1.0]      # Score: (0 + 1.0)*100 = 100
    }
    
    report = generator.generate_report(belief_state, {}, {}, {})
    assert report["candidate_score"] == 90.0
    assert report["recommendation"] == "Strong Hire"
    assert report["narrative_summary"] == "Strong candidate."

# --- Test Module 15 (Feedback Logger) ---
def test_feedback_logger():
    # Use in-memory or temp file db
    logger = FeedbackLogger(db_path="test_aria.db")
    logger.log_session("session_123", [{"state": 1, "action": 2}])
    logger.update_human_feedback("session_123", hired=True)
    
    # Verify directly via sqlite
    import sqlite3
    with sqlite3.connect(logger.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT human_feedback_reward FROM sessions WHERE session_id='session_123'")
        reward = cursor.fetchone()[0]
        
    assert reward == 1
    
    # Cleanup
    try:
        if os.path.exists(logger.db_path):
            os.remove(logger.db_path)
    except PermissionError:
        pass
