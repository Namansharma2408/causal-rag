"""
FastAPI Backend for Causal AI
Serves the frontend and handles RAG queries.
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import json
from datetime import datetime

# Import RAG system
try:
    from finalAgent.rag_system import RAGOrchestrator
    from finalAgent.config import Config
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG system not available: {e}")
    RAG_AVAILABLE = False

app = FastAPI(title="Causal AI", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
SESSIONS_DIR = Path(__file__).parent / "chat_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# Pydantic models
class ChatSettings(BaseModel):
    fast_mode: bool = True
    thinking_mode: bool = False
    include_proof: bool = False


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    settings: Optional[ChatSettings] = None


class SessionUpdate(BaseModel):
    messages: List[Dict[str, Any]]


def save_session(session_id: str, messages: list):
    """Save session to file."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    title = "New Chat"
    if messages:
        title = messages[0].get("content", "")[:40]
        if len(messages[0].get("content", "")) > 40:
            title += "..."
    
    data = {
        "id": session_id,
        "title": title,
        "messages": messages,
        "updated_at": datetime.now().isoformat()
    }
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)


def load_session(session_id: str) -> dict:
    """Load session from file."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            return json.load(f)
    return None


def get_all_sessions() -> list:
    """Get all sessions."""
    sessions = []
    for file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                sessions.append(json.load(f))
        except:
            pass
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def delete_session_file(session_id: str):
    """Delete a session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "frontend.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat messages."""
    session_id = request.session_id or str(uuid.uuid4())[:8]
    settings = request.settings or ChatSettings()
    
    if not RAG_AVAILABLE:
        return {
            "answer": "⚠️ RAG system not available. Please check server logs.",
            "error": True
        }
    
    try:
        config = Config()
        config.FAST_MODE = settings.fast_mode
        config.THINKING_MODE = settings.thinking_mode
        
        rag = RAGOrchestrator(config=config)
        result = rag.answer(
            request.message,
            session_id=session_id,
            include_proof=settings.include_proof,
            thinking_mode=settings.thinking_mode
        )
        
        response = {
            "answer": result.answer,
            "quality_score": getattr(result, 'quality_score', None),
            "doc_count": len(result.documents) if result.documents else 0,
            "multihop": result.metadata.get("multihop", False) if result.metadata else False,
            "thinking_mode": result.metadata.get("thinking_mode", False) if result.metadata else False,
            # Include source documents for citations
            "sources": [],
        }
        
        # Add source information for citations
        if result.documents:
            seen_transcripts = set()
            for doc in result.documents:
                tid = doc.transcript_id
                if tid and tid not in seen_transcripts:
                    seen_transcripts.add(tid)
                    response["sources"].append({
                        "transcript_id": tid,
                        "snippet": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "score": round(doc.score, 3) if doc.score else None
                    })
        
        if result.metadata and result.metadata.get("thinking_mode"):
            response["winning_model"] = result.metadata.get("winning_model")
            response["consensus_score"] = result.metadata.get("consensus_score")
            response["model_scores"] = result.metadata.get("model_scores", {})
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error: {str(e)}",
            "error": True
        }


@app.get("/api/sessions")
async def list_sessions():
    """List all chat sessions."""
    return get_all_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session."""
    session = load_session(session_id)
    if session:
        return session
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/sessions/{session_id}")
async def update_session(session_id: str, data: SessionUpdate):
    """Update a session."""
    save_session(session_id, data.messages)
    return {"success": True}


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a session."""
    delete_session_file(session_id)
    return {"success": True}


@app.get("/api/transcript/{transcript_id}")
async def get_transcript(transcript_id: str):
    """Fetch transcript content for citation modal."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        from finalAgent.services.transcripts import TranscriptManager
        tm = TranscriptManager()
        doc = tm.get_transcript(transcript_id)
        tm.close()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Transcript not found")
        
        # Get conversation (field is 'conversation' not 'turns')
        conversation = doc.get("conversation", doc.get("turns", []))
        formatted_turns = []
        speakers = set()
        
        for i, turn in enumerate(conversation):
            speaker = turn.get("speaker", "Unknown")
            speakers.add(speaker)
            formatted_turns.append({
                "index": i,
                "speaker": speaker,
                "text": turn.get("text", "")
            })
        
        return {
            "transcript_id": transcript_id,
            "turns": formatted_turns,
            "metadata": {
                "total_turns": len(formatted_turns),
                "speakers": list(speakers),
                "domain": doc.get("domain", ""),
                "intent": doc.get("intent", ""),
                "reason_for_call": doc.get("reason_for_call", ""),
                "time_of_interaction": doc.get("time_of_interaction", "")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting Causal AI Server...")
    print("Open http://localhost:5000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=5000)
