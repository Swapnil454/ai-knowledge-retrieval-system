import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

LOG_FILE = "data/query_logs.json"


def log_query(question: str, answer: str, confidence: float = None, sources: list = None):
    """
    Log a query and its answer for analytics.
    
    Args:
        question: The user's question
        answer: The generated answer
        confidence: Optional confidence score
        sources: Optional list of source information
    """
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        data = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "sources": sources
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
        logger.debug(f"Query logged: {question[:50]}...")
        
    except Exception as e:
        logger.error(f"Error logging query: {e}")


def get_logs(limit: int = 100) -> list:
    """
    Retrieve logged queries.
    
    Args:
        limit: Maximum number of logs to return
        
    Returns:
        List of log entries (most recent first)
    """
    try:
        if not os.path.exists(LOG_FILE):
            return []
            
        logs = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
                    
        return logs[-limit:][::-1]  # Most recent first
        
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return []


def clear_logs():
    """Clear all logged queries."""
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            logger.info("Query logs cleared")
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")