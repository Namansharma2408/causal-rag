"""
Base agent class for all agents in the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..models import Query, AgentResponse
from ..config import logger


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger
    
    @abstractmethod
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Process a query and return response.
        
        Args:
            query: The query to process
            context: Optional context from previous agents
            
        Returns:
            AgentResponse with results
        """
        pass
    
    def log(self, message: str, level: str = "info"):
        """Log a message with agent name prefix."""
        msg = f"[{self.name}] {message}"
        getattr(self.logger, level)(msg)
