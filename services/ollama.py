"""
Ollama LLM service for text generation.
Provides fast and quality model interfaces.
"""

import httpx
from typing import Optional

from ..config import Config, logger


class OllamaLLM:
    """LLM service using Ollama models."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.base_url = self.config.OLLAMA_BASE_URL
        self._client = httpx.Client(timeout=120.0)
    
    def _generate(self, prompt: str, model: str) -> str:
        """Internal generation method."""
        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                }
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ""
    
    def generate_fast(self, prompt: str) -> str:
        """Generate using fast model (codellama:7b).
        
        Use for: routing, classification, reranking, quick decisions.
        """
        return self._generate(prompt, self.config.FAST_MODEL)
    
    def generate_quality(self, prompt: str) -> str:
        """Generate using quality model (qwen2.5-coder:7b).
        
        Use for: final answers, extraction, quality checks.
        """
        return self._generate(prompt, self.config.QUALITY_MODEL)
    
    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self._client.close()
        except:
            pass
