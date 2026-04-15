"""
Causal AI - Application Logic
This module provides the core application logic for the RAG system.
Used by server.py for the HTML/CSS/JS frontend.
"""

import uuid
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RAG system
try:
    from rag_system import RAGOrchestrator
    from config import Config
    RAG_AVAILABLE = True
except ImportError:
    try:
        from finalAgent.rag_system import RAGOrchestrator
        from finalAgent.config import Config
        RAG_AVAILABLE = True
    except ImportError as e:
        print(f"Import error: {e}")
        RAG_AVAILABLE = False


# Session storage
SESSIONS_DIR = Path(__file__).parent / "chat_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


def save_session(session_id: str, messages: list, title: str = None):
    """Save chat session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    auto_title = "New Chat"
    if messages:
        auto_title = messages[0]["content"][:40]
        if len(messages[0]["content"]) > 40:
            auto_title += "..."
    
    data = {
        "id": session_id,
        "title": title or auto_title,
        "messages": messages,
        "updated_at": datetime.now().isoformat()
    }
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)


def load_session(session_id: str) -> dict:
    """Load chat session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            return json.load(f)
    return None


def get_all_sessions() -> list:
    """Get all sessions sorted by date."""
    sessions = []
    for file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                sessions.append(json.load(f))
        except:
            pass
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def delete_session(session_id: str):
    """Delete a session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()


def new_session_id() -> str:
    """Generate new session ID."""
    return str(uuid.uuid4())[:8]


def get_response(question: str, session_id: str, fast_mode: bool = True, 
                 thinking_mode: bool = False, include_proof: bool = False) -> dict:
    """Get RAG system response."""
    if not RAG_AVAILABLE:
        return {
            "answer": "⚠️ RAG system not available. Please check the imports.",
            "error": True
        }
    
    try:
        config = Config()
        config.FAST_MODE = fast_mode
        config.THINKING_MODE = thinking_mode
        
        rag = RAGOrchestrator(config=config)
        result = rag.answer(
            question,
            session_id=session_id,
            include_proof=include_proof,
            thinking_mode=thinking_mode
        )
        
        # Build response dict with all metadata
        response_data = {
            "answer": result.answer,
            "quality_score": getattr(result, 'quality_score', None),
            "doc_count": len(result.documents) if result.documents else 0,
            "transcript_ids": getattr(result, 'transcript_ids', []),
            "multihop": result.metadata.get("multihop", False) if result.metadata else False,
            "thinking_mode": result.metadata.get("thinking_mode", False) if result.metadata else False,
            "evidence": getattr(result, 'evidence', None)
        }
        
        # Add thinking mode specific metadata
        if result.metadata and result.metadata.get("thinking_mode"):
            response_data["winning_model"] = result.metadata.get("winning_model")
            response_data["consensus_score"] = result.metadata.get("consensus_score")
            response_data["model_scores"] = result.metadata.get("model_scores", {})
        
        return response_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "error": True
        }


def process_query(question: str, session_id: str = None, fast_mode: bool = True,
                  thinking_mode: bool = False, include_proof: bool = False) -> dict:
    """
    Process a user query and return the response.
    This is the main entry point for the HTML/CSS/JS frontend.
    """
    if not session_id:
        session_id = new_session_id()
    
    # Load existing session messages
    session_data = load_session(session_id)
    messages = session_data.get("messages", []) if session_data else []
    
    # Add user message
    messages.append({
        "role": "user",
        "content": question
    })
    
    # Get response
    response = get_response(
        question, 
        session_id, 
        fast_mode=fast_mode,
        thinking_mode=thinking_mode,
        include_proof=include_proof
    )
    
    # Add assistant response
    messages.append({
        "role": "assistant",
        "content": response["answer"],
        "metadata": response
    })
    
    # Save session
    save_session(session_id, messages)
    
    return {
        "session_id": session_id,
        "response": response,
        "messages": messages
    }


# Export functions for use by server.py
__all__ = [
    'process_query',
    'get_response',
    'save_session',
    'load_session',
    'get_all_sessions',
    'delete_session',
    'new_session_id',
    'RAG_AVAILABLE'
]
