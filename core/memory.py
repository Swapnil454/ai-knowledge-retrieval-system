from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Store and manage conversation history for the RAG system.
    Supports message tracking and history management.
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of messages to retain
        """
        self.history = []
        self.max_history = max_history

    def add(self, role: str, message: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            message: The message content
        """
        if len(self.history) >= self.max_history:
            self.history.pop(0)  # Remove oldest
            
        self.history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    def get(self, last_n: int = None) -> list:
        """
        Get conversation history.
        
        Args:
            last_n: Number of recent messages to return (None for all)
            
        Returns:
            List of message dictionaries
        """
        if last_n is not None:
            return self.history[-last_n:]
        return self.history

    def get_context(self, last_n: int = 5) -> str:
        """
        Get recent conversation as context string.
        
        Args:
            last_n: Number of recent messages to include
            
        Returns:
            Formatted conversation string
        """
        recent = self.get(last_n)
        context_parts = []
        
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['message']}")
            
        return "\n".join(context_parts)

    def clear(self):
        """Clear all conversation history."""
        self.history = []
        logger.info("Conversation history cleared")

    def __len__(self):
        return len(self.history)