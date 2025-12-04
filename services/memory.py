"""
Conversation memory for session persistence.
Stores and retrieves conversation history.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config import Config, logger


class ConversationMemory:
    """Manages conversation history with session persistence."""
    
    def __init__(self, session_id: Optional[str] = None, config: Optional[Config] = None):
        self.config = config or Config()
        self.session_id = session_id or self._generate_session_id()
        self.history: List[Dict[str, Any]] = []
        self.max_items = self.config.MAX_MEMORY_ITEMS
        self._session_dir = self.config.SESSION_DIR
        
        os.makedirs(self._session_dir, exist_ok=True)
        self._load_session()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _session_path(self) -> str:
        """Get path to session file."""
        return os.path.join(self._session_dir, f"{self.session_id}.json")
    
    def _load_session(self):
        """Load existing session if available."""
        path = self._session_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                logger.debug(f"Loaded session {self.session_id}")
            except Exception as e:
                logger.warning(f"Could not load session: {e}")
    
    def _save_session(self):
        """Save session to disk."""
        try:
            with open(self._session_path(), "w") as f:
                json.dump({
                    "session_id": self.session_id,
                    "history": self.history,
                    "updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save session: {e}")
    
    def add(self, query: str, answer: str, metadata: Optional[Dict] = None):
        """Add query-answer pair to history."""
        entry = {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(entry)
        
        # Trim if too long
        if len(self.history) > self.max_items:
            self.history = self.history[-self.max_items:]
        
        self._save_session()
    
    def get_context(self, last_n: int = 3) -> str:
        """Get recent conversation context as string."""
        if not self.history:
            return ""
        
        recent = self.history[-last_n:]
        lines = []
        for entry in recent:
            lines.append(f"Q: {entry['query']}")
            lines.append(f"A: {entry['answer'][:200]}...")
        
        return "\n".join(lines)
    
    def get_last_answer(self) -> Optional[str]:
        """Get the last answer."""
        if self.history:
            return self.history[-1].get("answer")
        return None
    
    def get_last_transcript_ids(self) -> List[str]:
        """Get transcript IDs from last answer."""
        if self.history:
            return self.history[-1].get("metadata", {}).get("transcript_ids", [])
        return []
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        self._save_session()
