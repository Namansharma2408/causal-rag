"""Causal AI RAG package.

Production-ready multi-agent RAG with:
- hybrid retrieval (vector + keyword)
- transcript-backed evidence verification
- optional thinking mode with multi-model consensus
"""

from .config import Config
from .ragSystem import RAGSystem
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
