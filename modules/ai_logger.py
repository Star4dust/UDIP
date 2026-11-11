"""
UDIP v3 — modules/ai_logger.py
Monitoring and debug logging for AI interactions.
"""

import json
import time
from pathlib import Path
from modules.core import DATA_DIR, log

AI_LOG_FILE = DATA_DIR / "ai_debug_log.json"

def log_ai_interaction(prompt: str, response: str, metadata: dict = None):
    """
    Log an AI interaction for monitoring and debugging.
    """
    try:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt[:500] + ("..." if len(prompt) > 500 else ""),
            "response_length": len(response),
            "metadata": metadata or {}
        }
        
        logs = []
        if AI_LOG_FILE.exists():
            try:
                logs = json.loads(AI_LOG_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        
        logs.append(entry)
        # Keep only last 1000 interactions
        logs = logs[-1000:]
        
        AI_LOG_FILE.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to log AI interaction: {e}")

def get_ai_logs():
    """Retrieve all AI interaction logs."""
    if AI_LOG_FILE.exists():
        try:
            return json.loads(AI_LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def log_ai_feedback(msg_idx: int, feedback_type: str):
    """Log user feedback (like/dislike) for a specific message."""
    try:
        feedback_file = DATA_DIR / "ai_feedback.json"
        feedbacks = []
        if feedback_file.exists():
            feedbacks = json.loads(feedback_file.read_text())
        
        feedbacks.append({
            "timestamp": time.time(),
            "msg_idx": msg_idx,
            "type": feedback_type
        })
        feedback_file.write_text(json.dumps(feedbacks, indent=2))
    except Exception as e:
        log.error(f"Feedback logging failed: {e}")
