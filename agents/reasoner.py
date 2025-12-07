"""
Reasoner agent - Generates final answers from extracted information.
"""

from typing import Optional, Any, List

from .base import BaseAgent
from ..models import Query, Document, AgentResponse
from ..services.llm_provider import get_llm, UnifiedLLM


class ReasonerAgent(BaseAgent):
    """Generates answers using extracted information."""
    
    def __init__(self, llm: Optional[UnifiedLLM] = None):
        super().__init__("Reasoner")
        self.llm = llm or get_llm()
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Generate answer from context."""
        extracted_info = context.get("extracted", "") if isinstance(context, dict) else str(context)
        documents = context.get("documents", []) if isinstance(context, dict) else []
        conversation_context = context.get("conversation", "") if isinstance(context, dict) else ""
        
        prompt = f"""You are an expert analyst answering questions about customer service interactions and behavioral patterns.

Question: {query.text}

Extracted Information and Insights:
{extracted_info}

{f'Previous conversation context:{chr(10)}{conversation_context}' if conversation_context else ''}

Based on the extracted information, provide a comprehensive and direct answer to the question. Include:
- Specific patterns, causes, or factors identified
- Examples from the data when relevant
- Actionable insights if applicable

Answer:"""
        
        answer = self.llm.generate_quality(prompt)
        
        # Get transcript IDs
        transcript_ids = list(set(
            doc.transcript_id for doc in documents 
            if hasattr(doc, 'transcript_id') and doc.transcript_id
        ))
        
        self.log(f"Generated {len(answer)} char answer")
        
        return AgentResponse(
            agent_name=self.name,
            result=answer,
            documents=documents,
            transcript_ids=transcript_ids,
            metadata={"query": query.text}
        )
