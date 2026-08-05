import sqlite3
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FeedbackLogger:
    def __init__(self, db_path="aria_trajectories.db"):
        """
        Initializes the SQLite database for logging interview trajectories.
        This forms the basis of the MLOps pipeline for RL fine-tuning (Phase 3).
        """
        # Save in the same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(current_dir, db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    human_feedback_reward INTEGER DEFAULT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    turn_index INTEGER,
                    action TEXT,
                    reward REAL,
                    is_done BOOLEAN,
                    info_gain REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            ''')
            conn.commit()

    def log_session(self, session_id: str, trajectory: list):
        """
        Logs a completed interview trajectory to the database.
        
        Args:
            session_id: Unique identifier for the interview.
            trajectory: List of turn dictionaries.
        """
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions (session_id, timestamp)
                    VALUES (?, ?)
                ''', (session_id, datetime.now().isoformat()))
                
                # Insert normalized turns
                for i, turn in enumerate(trajectory):
                    action = turn.get("action")
                    if isinstance(action, dict):
                        action = json.dumps(action)
                        
                    reward = float(turn.get("reward", 0.0))
                    is_done = bool(turn.get("done", False))
                    info_gain = float(turn.get("info", {}).get("info_gain", 0.0))
                    
                    cursor.execute('''
                        INSERT INTO turns (session_id, turn_index, action, reward, is_done, info_gain)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (session_id, i, str(action), reward, is_done, info_gain))
                    
                conn.commit()
            logger.info(f"Successfully logged trajectory for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log session to SQLite: {e}")

    def update_human_feedback(self, session_id: str, hired: bool):
        """
        Applies a terminal reward based on human HR feedback.
        +1 for Hire, -1 for Reject.
        This triggers the dataset readiness for RL IQL offline retraining.
        """
        reward = 1 if hired else -1
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions 
                    SET human_feedback_reward = ? 
                    WHERE session_id = ?
                ''', (reward, session_id))
                conn.commit()
            logger.info(f"Updated session {session_id} with terminal reward {reward}")
        except Exception as e:
            logger.error(f"Failed to update human feedback: {e}")
