"""
ARIA Live Multimodal Diagnostics & Calibration Tool

A standalone, plug-and-play script allowing users to test their real webcam and microphone
against ARIA's end-to-end multimodal processing pipeline (Modules 1 through 4).

Usage:
    python tools/run_live_calibration.py --duration 30
    python tools/run_live_calibration.py --duration 120
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
    from modules.module_02_vision.vision_processor import VisionProcessor
    from modules.module_03_prosody.extractor import ProsodyExtractor
    from modules.module_04_fusion import MultimodalFusionEngine
except ImportError as exc:
    print(f"[!] Error importing core modules: {exc}")
    sys.exit(1)


def record_live_stream(duration_sec: int, sample_rate: int = 16000) -> tuple[np.ndarray, list[np.ndarray]]:
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
    print('"Hello ARIA, my name is Krissh. I am reviewing the system calibration today.')
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
    if audio_frames:
        audio_arr = np.concatenate(audio_frames, axis=0).flatten()
    else:
        print("[*] Generating synthetic audio signal for testing...")
        t = np.linspace(0, duration_sec, duration_sec * sample_rate, endpoint=False)
        audio_arr = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # Process Video
    if not video_frames:
        print("[*] Generating synthetic blank video frame for testing...")
        if has_cv2:
            import cv2
            synth_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(synth_frame, "ARIA Test", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video_frames = [synth_frame] * 10

    return audio_arr, video_frames


def print_scorecard(stt_res: Dict[str, Any], vision_res: Dict[str, Any], prosody_res: Dict[str, Any], fusion_res: Dict[str, Any]):
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

    # 2. Vision
    print("\n[ Module 2 : Vision & Facial Analysis ]")
    print(f"  -> Dominant Trait: {vision_res.get('dominant_emotion', 'blank').upper()}")
    print(f"  -> Visual Confidence: {vision_res.get('vision_confidence', 0.0)*100:.1f}%")
    print(f"  -> Eye Contact Score: {vision_res.get('eye_contact_score', 0.0)*100:.1f}%")
    print(f"  -> Blink Rate: {vision_res.get('blink_rate_per_min', 0.0):.1f} blinks/min")

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
    if dissonance > 0.35:
        print("  -> [!] WARNING: High cross-modal contradiction detected (verbal claims conflict with non-verbal cues)!")
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
    args = parser.parse_args()

    print("[*] Initializing ARIA Modules 1-4...")
    stt_module = Transcriber()
    vision_module = VisionProcessor()
    prosody_module = ProsodyExtractor()
    fusion_module = MultimodalFusionEngine()

    audio_arr, video_frames = record_live_stream(args.duration)

    print("[*] Running Module 1 (Faster-Whisper STT)...")
    stt_res = stt_module.transcribe_sync(audio_arr)

    print("[*] Running Module 2 (MediaPipe + DeepFace Vision)...")
    vision_module.start_turn(0.0)
    for idx, frame in enumerate(video_frames[::3]):  # process every 3rd frame for speed
        vision_module.process_frame(frame, timestamp_ms=idx * 100.0)
    vision_res = vision_module.summarize_turn(turn_duration_ms=args.duration * 1000.0)

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
        semantic_features={"semantic_similarity": 0.85, "confidence": stt_res.get("confidence", 0.8)},
        vision_summary=vision_res,
        prosody_features=prosody_res,
    )

    print_scorecard(stt_res, vision_res, prosody_res, fusion_res)


if __name__ == "__main__":
    main()
