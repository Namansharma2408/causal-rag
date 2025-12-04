"""
Reranker agent - Reorders documents by relevance using LLM.
"""

from typing import Optional, Any, List

from .base import BaseAgent
from ..models import Query, Document, AgentResponse
from ..services.ollama import OllamaLLM


class RerankerAgent(BaseAgent):
    """Reranks documents using LLM scoring."""
    
    def __init__(self, llm: Optional[OllamaLLM] = None):
        super().__init__("Reranker")
        self.llm = llm or OllamaLLM()
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Rerank documents by relevance."""
        documents: List[Document] = context if context else []
        
        if not documents:
            return AgentResponse(
                agent_name=self.name,
                result=[],
                documents=[]
            )
        
        # Score each document
        scored = []
        for doc in documents:
            score = self._score_document(query.text, doc.content)
            scored.append((doc, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Update document scores
        reranked = []
        for doc, score in scored:
            doc.score = score
            reranked.append(doc)
        
        # Extract transcript IDs (in reranked order)
        transcript_ids = []
        for doc in reranked:
            if doc.transcript_id and doc.transcript_id not in transcript_ids:
                transcript_ids.append(doc.transcript_id)
        
        self.log(f"Reranked {len(reranked)} documents")
        self.log(f"Top transcript IDs: {transcript_ids[:5]}")
        
        return AgentResponse(
            agent_name=self.name,
            result=reranked,
            documents=reranked,
            transcript_ids=transcript_ids
        )
    
    def _score_document(self, query: str, content: str) -> float:
        """Score a document's relevance to query."""
        prompt = f"""Rate relevance 0-10 (just the number):
Query: {query}
Document: {content[:500]}

Score:"""
        
        try:
            response = self.llm.generate_fast(prompt)
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                return min(float(numbers[0]) / 10.0, 1.0)
        except:
            pass
        
        return 0.5  # default score
