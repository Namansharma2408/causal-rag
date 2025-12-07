"""
Unified LLM Provider Interface

This module provides a clean abstraction for multiple LLM providers:
- Ollama (local models)
- OpenAI (ChatGPT)
- Google (Gemini)

All LLM calls go through the unified `generate()` function for:
- Consistent interface across providers
- Easy provider switching via config
- Centralized error handling
- API key management
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

# Lazy imports to avoid dependency issues
httpx = None
openai = None
google_genai = None


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    model: str
    provider: LLMProvider
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return self.text


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""
    
    provider: LLMProvider
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            model: Model to use (provider-specific)
            temperature: Creativity (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options
            
        Returns:
            LLMResponse with generated text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available and configured."""
        pass


class OllamaLLM(BaseLLM):
    """Ollama local LLM provider."""
    
    provider = LLMProvider.OLLAMA
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5-coder:7b",
        timeout: float = 120.0
    ):
        global httpx
        if httpx is None:
            import httpx as _httpx
            httpx = _httpx
            
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client
    
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using Ollama."""
        model = model or self.default_model
        client = self._get_client()
        
        try:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": options
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                text=data.get("response", "").strip(),
                model=model,
                provider=self.provider,
                tokens_used=data.get("eval_count"),
                metadata={"done": data.get("done", False)}
            )
        except Exception as e:
            return LLMResponse(
                text=f"Error: {str(e)}",
                model=model,
                provider=self.provider,
                metadata={"error": str(e)}
            )
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            client = self._get_client()
            response = client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def __del__(self):
        """Cleanup HTTP client."""
        if self._client:
            try:
                self._client.close()
            except:
                pass


class OpenAILLM(BaseLLM):
    """OpenAI ChatGPT provider."""
    
    provider = LLMProvider.OPENAI
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_model: str = "gpt-3.5-turbo",
        organization: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            global openai
            if openai is None:
                try:
                    import openai as _openai
                    openai = _openai
                except ImportError:
                    raise ImportError("openai package not installed. Run: pip install openai")
            
            self._client = openai.OpenAI(
                api_key=self.api_key,
                organization=self.organization
            )
        return self._client
    
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using OpenAI ChatGPT."""
        model = model or self.default_model
        
        if not self.api_key:
            return LLMResponse(
                text="Error: OpenAI API key not configured",
                model=model,
                provider=self.provider,
                metadata={"error": "API key missing"}
            )
        
        try:
            client = self._get_client()
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens or 4096
            )
            
            return LLMResponse(
                text=response.choices[0].message.content.strip(),
                model=model,
                provider=self.provider,
                tokens_used=response.usage.total_tokens if response.usage else None,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
        except Exception as e:
            return LLMResponse(
                text=f"Error: {str(e)}",
                model=model,
                provider=self.provider,
                metadata={"error": str(e)}
            )
    
    def is_available(self) -> bool:
        """Check if OpenAI API is configured."""
        return bool(self.api_key)


class GeminiLLM(BaseLLM):
    """Google Gemini provider."""
    
    provider = LLMProvider.GEMINI
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_model: str = "gemini-1.5-flash"
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.default_model = default_model
        self._client = None
    
    def _get_client(self, model: str):
        """Get Gemini model client."""
        global google_genai
        if google_genai is None:
            try:
                import google.generativeai as _genai
                google_genai = _genai
                google_genai.configure(api_key=self.api_key)
            except ImportError:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        return google_genai.GenerativeModel(model)
    
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using Google Gemini."""
        model = model or self.default_model
        
        if not self.api_key:
            return LLMResponse(
                text="Error: Google API key not configured",
                model=model,
                provider=self.provider,
                metadata={"error": "API key missing"}
            )
        
        try:
            client = self._get_client(model)
            
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            response = client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return LLMResponse(
                text=response.text.strip(),
                model=model,
                provider=self.provider,
                metadata={"finish_reason": "completed"}
            )
        except Exception as e:
            return LLMResponse(
                text=f"Error: {str(e)}",
                model=model,
                provider=self.provider,
                metadata={"error": str(e)}
            )
    
    def is_available(self) -> bool:
        """Check if Gemini API is configured."""
        return bool(self.api_key)


class UnifiedLLM:
    """
    Unified LLM interface that manages multiple providers.
    
    This is the main class that should be used throughout the application.
    It automatically selects the best available provider or uses the configured one.
    """
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5-coder:7b",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-1.5-flash",
        fallback_order: Optional[List[LLMProvider]] = None
    ):
        """
        Initialize the unified LLM interface.
        
        Args:
            provider: Preferred provider (auto-selects if None)
            ollama_url: Ollama server URL
            ollama_model: Default Ollama model
            openai_api_key: OpenAI API key
            openai_model: Default OpenAI model
            gemini_api_key: Google Gemini API key
            gemini_model: Default Gemini model
            fallback_order: Order of providers to try
        """
        self.preferred_provider = provider
        self.fallback_order = fallback_order or [
            LLMProvider.OLLAMA,
            LLMProvider.OPENAI,
            LLMProvider.GEMINI
        ]
        
        # Initialize all providers
        self.providers: Dict[LLMProvider, BaseLLM] = {
            LLMProvider.OLLAMA: OllamaLLM(
                base_url=ollama_url,
                default_model=ollama_model
            ),
            LLMProvider.OPENAI: OpenAILLM(
                api_key=openai_api_key,
                default_model=openai_model
            ),
            LLMProvider.GEMINI: GeminiLLM(
                api_key=gemini_api_key,
                default_model=gemini_model
            )
        }
        
        # Track the active provider
        self._active_provider: Optional[LLMProvider] = None
    
    def _get_provider(self, provider: Optional[LLMProvider] = None) -> BaseLLM:
        """Get the appropriate provider instance."""
        # Use specified provider
        if provider:
            return self.providers[provider]
        
        # Use preferred provider if set
        if self.preferred_provider:
            return self.providers[self.preferred_provider]
        
        # Auto-select based on availability
        for p in self.fallback_order:
            if self.providers[p].is_available():
                self._active_provider = p
                return self.providers[p]
        
        # Default to Ollama even if not available (will return error)
        return self.providers[LLMProvider.OLLAMA]
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the configured LLM provider.
        
        This is the main function that should be called throughout the application.
        
        Args:
            prompt: The input prompt
            model: Model to use (provider-specific, uses default if None)
            temperature: Creativity level (0.0-1.0)
            max_tokens: Maximum tokens to generate
            provider: Override the default provider for this call
            **kwargs: Additional provider-specific options
            
        Returns:
            Generated text as string
            
        Example:
            llm = UnifiedLLM()
            response = llm.generate("What is the capital of France?")
            print(response)  # "Paris"
        """
        llm = self._get_provider(provider)
        response = llm.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.text
    
    def generate_with_metadata(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text and return full response with metadata.
        
        Same as generate() but returns LLMResponse object with additional info.
        """
        llm = self._get_provider(provider)
        return llm.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        return [p for p in self.providers if self.providers[p].is_available()]
    
    def get_active_provider(self) -> Optional[LLMProvider]:
        """Get the currently active provider."""
        return self._active_provider or self.preferred_provider
    
    # Convenience methods for common use cases
    def generate_fast(self, prompt: str, **kwargs) -> str:
        """Generate using fast settings (higher temperature, lower quality)."""
        return self.generate(prompt, temperature=0.1, **kwargs)
    
    def generate_quality(self, prompt: str, **kwargs) -> str:
        """Generate using quality settings (lower temperature, higher quality)."""
        return self.generate(prompt, temperature=0.0, **kwargs)
    
    def generate_creative(self, prompt: str, **kwargs) -> str:
        """Generate using creative settings (higher temperature)."""
        return self.generate(prompt, temperature=0.8, **kwargs)


# Global instance for simple usage
_default_llm: Optional[UnifiedLLM] = None


def get_llm(
    provider: Optional[LLMProvider] = None,
    **kwargs
) -> UnifiedLLM:
    """
    Get the global LLM instance or create a new one.
    
    This is the recommended way to get an LLM instance for most use cases.
    Uses settings from config.py by default.
    
    Args:
        provider: Preferred provider (overrides config)
        **kwargs: Additional configuration options
        
    Returns:
        UnifiedLLM instance
        
    Example:
        from services.llm_provider import get_llm
        
        llm = get_llm()
        response = llm.generate("Hello, world!")
    """
    global _default_llm
    
    if _default_llm is None or provider is not None:
        # Import config here to avoid circular imports
        from ..config import Config
        config = Config()
        
        # Determine provider from config if not specified
        if provider is None:
            provider_str = config.LLM_PROVIDER.lower()
            if provider_str == "openai":
                provider = LLMProvider.OPENAI
            elif provider_str == "gemini":
                provider = LLMProvider.GEMINI
            else:
                provider = LLMProvider.OLLAMA
        
        # Set up default kwargs from config
        default_kwargs = {
            "ollama_url": config.OLLAMA_BASE_URL,
            "ollama_model": config.QUALITY_MODEL,
            "openai_api_key": config.OPENAI_API_KEY,
            "openai_model": config.OPENAI_MODEL,
            "gemini_api_key": config.GEMINI_API_KEY,
            "gemini_model": config.GEMINI_MODEL,
        }
        default_kwargs.update(kwargs)
        
        _default_llm = UnifiedLLM(provider=provider, **default_kwargs)
    
    return _default_llm


def reset_llm():
    """Reset the global LLM instance. Useful for config changes."""
    global _default_llm
    _default_llm = None


def generate(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    provider: Optional[LLMProvider] = None,
    **kwargs
) -> str:
    """
    Standalone function to generate text.
    
    This is the simplest way to call an LLM without managing instances.
    
    Args:
        prompt: The input prompt
        model: Model to use
        temperature: Creativity level
        provider: Provider to use
        **kwargs: Additional options
        
    Returns:
        Generated text
        
    Example:
        from services.llm_provider import generate
        
        response = generate("What is 2 + 2?")
        print(response)  # "4"
    """
    llm = get_llm(provider=provider)
    return llm.generate(prompt, model=model, temperature=temperature, **kwargs)
