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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    trajectory_json TEXT,
                    human_feedback_reward INTEGER DEFAULT NULL
                )
            ''')
            conn.commit()

    def log_session(self, session_id: str, trajectory: list):
        """
        Logs a completed interview trajectory to the database.
        
        Args:
            session_id: Unique identifier for the interview.
            trajectory: List of turn dictionaries containing (state, action, reward, next_state).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions (session_id, timestamp, trajectory_json)
                    VALUES (?, ?, ?)
                ''', (session_id, datetime.now().isoformat(), json.dumps(trajectory)))
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
            with sqlite3.connect(self.db_path) as conn:
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
