"""
Configuration module for the RAG system.
Handles environment variables, constants, and logging setup.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Central configuration for the RAG system."""
    
    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME: str = "conversations_db"
    COLLECTION_NAME: str = "spans"
    TRANSCRIPTS_DB: str = "full_transcripts"
    TRANSCRIPTS_COLLECTION: str = "transcripts"
    
    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    FAST_MODEL: str = "codellama:7b"
    QUALITY_MODEL: str = "qwen2.5-coder:7b"
    
    # Search settings
    DEFAULT_TOP_K: int = 5
    DEFAULT_TOP_CLUSTERS: int = 5
    HYBRID_ALPHA: float = 0.6  # Weight for vector search (1-alpha for BM25)
    MAX_CONTEXT_LENGTH: int = 8000
    
    # Embedding settings
    EMBEDDING_DIM: int = 768
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Session settings
    SESSION_DIR: str = "sessions"
    MAX_MEMORY_ITEMS: int = 10
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure and return the logger."""
    log_level = level or Config.LOG_LEVEL
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("FinalAgent")
    
    # Reduce noise from external libraries
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger


# Global logger instance
logger = setup_logging()
