"""
FinalAgent - Production-Ready Multi-Agent RAG System
=====================================================

A modular RAG (Retrieval-Augmented Generation) system with:
- Hybrid search (Vector + BM25)
- Multi-agent architecture
- MongoDB integration
- Evidence extraction and verification

Usage:
    from finalAgent import RAGSystem
    
    rag = RAGSystem()
    result = rag.answer("What is the refund policy?")
    evidence = rag.get_evidence()
"""

from .config import Config
from .rag_system import RAGSystem
from .api import (
    answer_question,
    answer_with_proof,
    get_evidence,
    get_conversation
)

__version__ = "1.0.0"
__all__ = [
    "Config",
    "RAGSystem",
    "answer_question",
    "answer_with_proof", 
    "get_evidence",
    "get_conversation"
]
