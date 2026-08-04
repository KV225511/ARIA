import pyttsx3
import logging
import threading

logger = logging.getLogger(__name__)

class TTSAvatarBaseline:
    def __init__(self):
        """
        Initializes the baseline TTS engine using pyttsx3.
        This runs locally without any heavy ML dependencies.
        """
        self.engine = pyttsx3.init()
        # Set a professional voice if available (usually index 1 is female on Windows)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
            
        self.engine.setProperty('rate', 160) # Natural speaking rate
        self.engine.setProperty('volume', 1.0)
        self.lock = threading.Lock()
        
    def speak(self, text: str):
        """
        Synthesizes speech from text and plays it through the speakers.
        In a real WebRTC architecture, this would output to a virtual audio cable or stream.
        """
        if not text:
            return
            
        # Run in a thread-safe way so we don't block the async pipeline
        with self.lock:
            try:
                logger.info(f"Speaking: '{text}'")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS Engine failed: {e}")
                
    def get_avatar_frame(self):
        """
        Placeholder for SadTalker avatar integration.
        Currently returns None since we are using the baseline offline TTS.
        """
        return None
