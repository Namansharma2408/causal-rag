
import os
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    
    # MongoDB settings
    MONGODB_HOST: str = os.getenv("MONGODB_HOST", "localhost")
    MONGODB_PORT: int = int(os.getenv("MONGODB_PORT", "27017"))
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DB_NAME: str = "conversations_db"
    COLLECTION_NAME: str = "spans"
    TRANSCRIPTS_DB: str = "full_transcripts"
    TRANSCRIPTS_COLLECTION: str = "transcripts"
    
    # LLM Provider settings (ollama, openai, gemini)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    
    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    FAST_MODEL: str = "codellama:7b"
    QUALITY_MODEL: str = "qwen2.5-coder:7b"
    
    # OpenAI settings (optional)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Gemini settings (optional)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    
    # Search settings
    DEFAULT_TOP_K: int = 5
    DEFAULT_TOP_CLUSTERS: int = 5
    HYBRID_ALPHA: float = 0.6  # Weight for vector search (1-alpha for BM25)
    MAX_CONTEXT_LENGTH: int = 8000
    
    # Embedding settings
    EMBEDDING_DIM: int = 768
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Session settings
    SESSION_DIR: str = "chatSessions"
    MAX_MEMORY_ITEMS: int = 10
    
    # Performance settings
    FAST_MODE: bool = os.getenv("FAST_MODE", "true").lower() == "true"  # Skip reranking, use fast model
    PARALLEL_SUBQUERIES: bool = True  # Process sub-queries in parallel
    MAX_PARALLEL_QUERIES: int = 3  # Max concurrent sub-query processing
    SKIP_RERANK_THRESHOLD: int = 3  # Skip reranking if docs <= this
    
    # Thinking Mode settings
    THINKING_MODE: bool = os.getenv("THINKING_MODE", "false").lower() == "true"  # Multi-model consensus
    THINKING_MODELS: list = None  # Will be set in __post_init__
    
    def __post_init__(self):
        running_in_docker = os.path.exists("/.dockerenv")

        # Local-run normalization: docker service hostnames do not resolve on host machine
        if not running_in_docker:
            if self.MONGODB_HOST == "mongodb":
                self.MONGODB_HOST = "localhost"

            if self.OLLAMA_BASE_URL.startswith("http://ollama:") or self.OLLAMA_BASE_URL.startswith("https://ollama:"):
                self.OLLAMA_BASE_URL = self.OLLAMA_BASE_URL.replace("ollama", "localhost", 1)

            if "host.docker.internal" in self.OLLAMA_BASE_URL:
                self.OLLAMA_BASE_URL = self.OLLAMA_BASE_URL.replace("host.docker.internal", "localhost")

        if not self.MONGODB_URI:
            self.MONGODB_URI = f"mongodb://{self.MONGODB_HOST}:{self.MONGODB_PORT}"

        # If URI is explicitly set with Docker service host, normalize for local run
        if not running_in_docker and "mongodb://mongodb:" in self.MONGODB_URI:
            self.MONGODB_URI = self.MONGODB_URI.replace("mongodb://mongodb:", "mongodb://localhost:")

        if self.THINKING_MODELS is None:
            self.THINKING_MODELS = [
                "deepseek-r1:14b",      # Deep reasoning model
                "phi3:14b",              # Microsoft's powerful model  
                "qwen2.5-coder:7b",      # Coding specialist
                "codellama:7b",          # Meta's code model
            ]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    log_level = level or Config.LOG_LEVEL
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("CausalRAG")
    
    # Reduce noise from external libraries
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger


# Global logger instance
logger = setup_logging()
