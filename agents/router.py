"""
Router agent - Classifies query type and determines processing strategy.
"""

from typing import Optional, Any

from .base import BaseAgent
from ..models import Query, QueryType, AgentResponse
from ..services.llm_provider import get_llm, UnifiedLLM


class RouterAgent(BaseAgent):
    """Routes queries based on type classification."""
    
    def __init__(self, llm: Optional[UnifiedLLM] = None):
        super().__init__("Router")
        self.llm = llm or get_llm()
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Classify query type."""
        prompt = f"""Classify this query type. Reply with ONE word only:
FACTUAL - asking for facts/information
PROCEDURAL - asking how to do something
COMPARATIVE - comparing options
ANALYTICAL - needs analysis/reasoning
CONVERSATIONAL - chitchat/greetings

Query: {query.text}

Type:"""
        
        response = self.llm.generate_fast(prompt)
        
        # Parse response
        query_type = QueryType.FACTUAL  # default
        response_upper = response.upper().strip()
        
        for qt in QueryType:
            if qt.value.upper() in response_upper:
                query_type = qt
                break
        
        self.log(f"Classified as: {query_type.value}")
        
        return AgentResponse(
            agent_name=self.name,
            result=query_type,
            metadata={"raw_response": response}
        )
