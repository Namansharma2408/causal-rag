"""
Embedding service using Ollama's nomic-embed-text model.
"""

import hashlib
import httpx #type:ignore
from typing import List, Dict, Optional
from functools import lru_cache
from time import perf_counter

from ..config import Config, logger


class EmbeddingService:
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.base_url = self.config.OLLAMA_BASE_URL
        self.model = self.config.EMBEDDING_MODEL
        self._cache: Dict[str, List[float]] = {}
        self._client = httpx.Client(timeout=60.0)
        
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed(self, text: str) -> List[float]:
        t0 = perf_counter()
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._cache:
            logger.info(f"[TRACE][embedding][cache-hit] len={len(text)} took={perf_counter()-t0:.3f}s")
            return self._cache[cache_key]
        
        try:
            logger.info(f"[TRACE][embedding][request] model={self.model} url={self.base_url}/api/embeddings len={len(text)}")
            response = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            self._cache[cache_key] = embedding
            logger.info(f"[TRACE][embedding][done] dim={len(embedding)} took={perf_counter()-t0:.3f}s")
            return embedding
        except Exception as e:
            logger.error(f"[TRACE][embedding][error] took={perf_counter()-t0:.3f}s error={e}")
            return [0.0] * self.config.EMBEDDING_DIM
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]
    
    def clear_cache(self):
        self._cache.clear()
    
    def __del__(self):
        try:
            self._client.close()
        except:
            pass
