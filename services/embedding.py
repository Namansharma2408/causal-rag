"""
Embedding service using Ollama's nomic-embed-text model.
Provides text embedding with caching for performance.
"""

import hashlib
import httpx
from typing import List, Dict, Optional
from functools import lru_cache

from ..config import Config, logger


class EmbeddingService:
    """Generate embeddings using Ollama."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.base_url = self.config.OLLAMA_BASE_URL
        self.model = self.config.EMBEDDING_MODEL
        self._cache: Dict[str, List[float]] = {}
        self._client = httpx.Client(timeout=60.0)
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            self._cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * self.config.EMBEDDING_DIM
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
    
    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self._client.close()
        except:
            pass
