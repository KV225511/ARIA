"""
ARIA Live Multimodal Diagnostics & Calibration Tool

A standalone, plug-and-play script allowing users to test their real webcam and microphone
against ARIA's end-to-end multimodal processing pipeline (Modules 1 through 4).

Usage:
    python tools/run_live_calibration.py --duration 30
    python tools/run_live_calibration.py --duration 120 --name "Your Name"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# P1 — Initialize PyTorch CUDA runtime before TensorFlow (imported via DeepFace)
# to avoid Windows DLL collision on cudnnGetLibConfig.
if torch.cuda.is_available():
    try:
        torch.zeros(1).cuda()
    except Exception:
        pass

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules without modification
try:
    from modules.module_01_stt.transcriber import Transcriber
    from modules.module_01_stt.semantic_grader import SemanticGrader
    from modules.module_02_vision.vision_processor import VisionProcessor
    from modules.module_02_vision.temporal_au import TemporalAUTracker
    from modules.module_03_prosody.extractor import ProsodyExtractor
    from modules.module_04_fusion import MultimodalFusionEngine
except ImportError as exc:
    print(f"[!] Error importing core modules: {exc}")
    sys.exit(1)

# FIX C5 — Dissonance threshold raised from 0.35 to 0.65 based on observed
# real-world calibration sessions. Scores of 0.35 fired on every calm session,
# making the warning meaningless. 0.65 reflects genuine cross-modal conflict.
DISSONANCE_WARNING_THRESHOLD = 0.65


def record_live_stream(duration_sec: int, speaker_name: str = "User", sample_rate: int = 16000) -> tuple[np.ndarray, list[np.ndarray]]:
    """Records synchronized microphone audio and webcam frames."""
    audio_frames = []
    video_frames = []

    has_cv2 = False
    has_sd = False

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        print("[!] OpenCV not installed or unavailable.")

    try:
        import sounddevice as sd
        has_sd = True
    except ImportError:
        print("[!] sounddevice not installed or unavailable.")

    print(f"\n[*] Preparing live capture for {duration_sec} seconds...")
    print("--- PROMPT TO READ ---")
    print(f'"Hello ARIA, my name is {speaker_name}. I am reviewing the system calibration today.')
    print('When faced with high pressure, I always stay calm, focused, and analytical."')
    print("----------------------\n")

    # Audio callback
    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
        audio_frames.append(indata.copy())

    cap = None
    stream = None

    if has_cv2:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[!] Could not open webcam (index 0). Will simulate video frames.")
            cap = None

    if has_sd:
        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=audio_callback)
            stream.start()
        except Exception as e:
            print(f"[!] Could not open audio input stream: {e}. Will simulate audio.")
            stream = None

    start_time = time.time()
    print("[*] RECORDING STARTED! Speak now...\n")

    while time.time() - start_time < duration_sec:
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                video_frames.append(frame)
        time.sleep(0.033)  # roughly 30 FPS buffer rate

    print("\n[*] Recording finished. Processing streams...")

    if stream is not None:
        stream.stop()
        stream.close()

    if cap is not None:
        cap.release()

    # Process Audio
    synthetic_audio = False
    if audio_frames:
        audio_arr = np.concatenate(audio_frames, axis=0).flatten()
    else:
        synthetic_audio = True
        print("[*] Generating synthetic audio signal for testing...")
        t = np.linspace(0, duration_sec, duration_sec * sample_rate, endpoint=False)
        audio_arr = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # FIX H7 — Print a clearly visible warning when synthetic audio is used.
    if synthetic_audio:
        print("\n" + "!" * 70)
        print("  !!! WARNING: SYNTHETIC AUDIO IN USE — RESULTS ARE NOT REAL !!!")
        print("  !!! Microphone failed to capture. STT/prosody results are fake. !!!")
        print("!" * 70 + "\n")

    # Process Video
    if not video_frames:
        print("[*] Generating synthetic blank video frame for testing...")
        if has_cv2:
            import cv2
            synth_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(synth_frame, "ARIA Test", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video_frames = [synth_frame] * 10

    return audio_arr, video_frames


def print_scorecard(stt_res: Dict[str, Any], vision_res: Dict[str, Any], prosody_res: Dict[str, Any], fusion_res: Dict[str, Any], au_res: Dict[str, Any] | None = None, grade_res: Dict[str, Any] | None = None):
    """Renders a formatted diagnostic scorecard for the user."""
    print("\n" + "="*70)
    print("                ARIA LIVE SESSION CALIBRATION SCORECARD                ")
    print("="*70)

    # 1. Speech-to-Text
    print("\n[ Module 1 : Speech-to-Text (STT) ]")
    safe_transcript = str(stt_res.get('transcript', '')).encode('ascii', 'replace').decode('ascii')
    print(f"  -> Transcript: \"{safe_transcript}\"")
    print(f"  -> STT Confidence: {stt_res.get('confidence', 0.0)*100:.1f}%")
    print(f"  -> Acoustic Response Latency: {stt_res.get('response_latency_ms', 0.0):.1f} ms")
    if grade_res:
        print(f"  -> Automated Rubric Grade: {grade_res.get('final_grade', 'N/A')} (Similarity: {grade_res.get('similarity_score', 0.0)*100:.1f}% | Keywords: {grade_res.get('keyword_coverage', 0.0)*100:.1f}%)")

    # 2. Vision
    print("\n[ Module 2 : Vision & Facial Analysis ]")
    print(f"  -> Static Dominant Trait: {vision_res.get('dominant_emotion', vision_res.get('emotion_label', 'blank')).upper()} (Conf: {vision_res.get('vision_confidence', 0.0)*100:.1f}%)")
    if au_res:
        print(f"  -> SOTA Temporal AU Trait: {str(au_res.get('temporal_emotion_prediction', 'blank')).upper()} (Dynamic Conf: {au_res.get('temporal_confidence', 0.0)*100:.1f}%)")
        print(f"  -> Micro-Expression Velocity: {au_res.get('au_velocity_mean', 0.0):.4f}")
    print(f"  -> Eye Contact Score: {vision_res.get('eye_contact_score', 0.0)*100:.1f}%")
    # FIX M6 — vision_processor outputs 'blink_rate', not 'blink_rate_per_min'
    print(f"  -> Blink Rate: {vision_res.get('blink_rate', vision_res.get('blink_rate_per_min', 0.0)):.1f} blinks/min")

    # 3. Prosody
    print("\n[ Module 3 : Prosody & Vocal Acoustic Analysis ]")
    print(f"  -> Mean Pitch (F0): {prosody_res.get('pitch_mean', 0.0):.1f} Hz")
    print(f"  -> Speech Rate: {prosody_res.get('speech_rate', 0.0):.2f} words/sec")
    print(f"  -> Speech Energy: {prosody_res.get('energy_mean', 0.0):.4f}")
    wavlm_emb = prosody_res.get("wavlm_embedding", [])
    print(f"  -> Self-Supervised WavLM Vector: Extracted ({len(wavlm_emb)} dimensions)")

    # 4. Multimodal Fusion
    print("\n[ Module 4 : Multimodal Fusion Engine ]")
    weights = fusion_res.get("modality_weights", {})
    print(f"  -> Dynamic Attention Weights: Text={weights.get('text', 0.0)*100:.1f}%, Vision={weights.get('vision', 0.0)*100:.1f}%, Audio={weights.get('prosody', 0.0)*100:.1f}%")
    dissonance = fusion_res.get("cross_modal_dissonance", 0.0)
    print(f"  -> Cross-Modal Dissonance Score: {dissonance:.4f}")
    # FIX C5 — raised threshold from 0.35 to DISSONANCE_WARNING_THRESHOLD (0.65)
    if dissonance > DISSONANCE_WARNING_THRESHOLD:
        print("  -> [!] WARNING: High cross-modal contradiction detected (verbal claims conflict with non-verbal cues)!")
    elif dissonance > 0.45:
        print("  -> [~] MODERATE: Some cross-modal tension detected — minor non-verbal inconsistency.")
    else:
        print("  -> [OK] Non-verbal delivery is congruent with verbal semantic content.")

    fused_vec = fusion_res.get("fused_vector", [])
    print(f"  -> Unified Representation Output: {len(fused_vec)}-dim vector generated.")
    print("="*70 + "\n")


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="ARIA Live Session Diagnostics Tool")
    parser.add_argument("--duration", type=int, default=15, help="Recording duration in seconds (e.g. 30 or 120)")
    # FIX M5 — accept speaker name as CLI argument instead of hardcoding personal name
    parser.add_argument("--name", type=str, default="Candidate", help="Your name for the session prompt")
    args = parser.parse_args()

    print("[*] Initializing ARIA Modules 1-4 & SOTA Extensions...")
    stt_module = Transcriber()
    vision_module = VisionProcessor()
    prosody_module = ProsodyExtractor()
    fusion_module = MultimodalFusionEngine()
    au_tracker = TemporalAUTracker()
    grader = SemanticGrader()

    audio_arr, video_frames = record_live_stream(args.duration, speaker_name=args.name)

    print("[*] Running Module 1 (Faster-Whisper STT)...")
    stt_res = stt_module.transcribe_sync(audio_arr)
    grade_res = grader.grade_response(
        stt_res.get("transcript", ""),
        "When faced with high pressure, I always stay calm, focused and analytical.",
        required_keywords=["calm", "focused", "analytical"]
    )

    print("[*] Running Module 2 (MediaPipe + DeepFace Vision)...")
    vision_module.start_turn(0.0)
    for idx, frame in enumerate(video_frames[::3]):  # process every 3rd frame for speed
        vision_module.process_frame(frame, timestamp_ms=idx * 100.0)
    vision_res = vision_module.summarize_turn(turn_duration_ms=args.duration * 1000.0)
    # FIX C4 — use public property instead of accessing private _turn_frames directly
    au_res = au_tracker.extract_temporal_features(vision_module.get_turn_frames())

    print("[*] Running Module 3 (openSMILE + WavLM Prosody)...")
    prosody_res = prosody_module.extract(
        audio_clip=audio_arr,
        word_timestamps=stt_res.get("word_timestamps"),
        response_latency_ms=stt_res.get("response_latency_ms"),
    )

    print("[*] Running Module 4 (Dynamic Attention Fusion Engine)...")
    fusion_res = fusion_module.fuse_turn(
        candidate_id="live_test_user",
        turn_id=1,
        stt_result=stt_res,
        semantic_features={"semantic_similarity": grade_res.get("similarity_score", 0.85), "confidence": stt_res.get("confidence", 0.8)},
        vision_summary=vision_res,
        prosody_features=prosody_res,
    )

    print_scorecard(stt_res, vision_res, prosody_res, fusion_res, au_res=au_res, grade_res=grade_res)

    # Close vision executor cleanly — FIX H5
    vision_module.close()


if __name__ == "__main__":
    main()
