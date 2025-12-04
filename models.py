"""
Data models for the RAG system.
Defines core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class QueryType(Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


@dataclass
class Document:
    """Represents a retrieved document/span."""
    id: str
    content: str
    transcript_id: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "transcript_id": self.transcript_id,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class Query:
    """Represents a user query with metadata."""
    text: str
    query_type: QueryType = QueryType.FACTUAL
    context: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "query_type": self.query_type.value,
            "context": self.context,
            "session_id": self.session_id
        }


@dataclass
class AgentResponse:
    """Response from an agent in the pipeline."""
    agent_name: str
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    documents: List[Document] = field(default_factory=list)
    transcript_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "result": self.result,
            "metadata": self.metadata,
            "documents": [d.to_dict() for d in self.documents],
            "transcript_ids": self.transcript_ids
        }


@dataclass
class RAGResult:
    """Complete result from the RAG system."""
    answer: str
    query: str
    documents: List[Document] = field(default_factory=list)
    transcript_ids: List[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    quality_feedback: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "query": self.query,
            "documents": [d.to_dict() for d in self.documents],
            "transcript_ids": self.transcript_ids,
            "quality_score": self.quality_score,
            "quality_feedback": self.quality_feedback,
            "evidence": self.evidence,
            "metadata": self.metadata
        }
