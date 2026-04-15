"""
Extractor agent - Extracts relevant information from documents.
"""

from typing import Optional, Any, List
from time import perf_counter

from .base import BaseAgent
from ..models import Query, Document, AgentResponse
from ..services.llmProvider import get_llm, UnifiedLLM


class ExtractorAgent(BaseAgent):
    """Extracts key information from documents."""
    
    def __init__(self, llm: Optional[UnifiedLLM] = None):
        super().__init__("Extractor")
        self.llm = llm or get_llm()
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Extract relevant information from documents."""
        t0 = perf_counter()
        print(f"[TRACE][ENTER][ExtractorAgent.process] q_len={len(query.text)}")
        documents: List[Document] = context if context else []
        
        if not documents:
            print(f"[TRACE][EXIT][ExtractorAgent.process] docs=0 took={perf_counter()-t0:.3f}s")
            return AgentResponse(
                agent_name=self.name,
                result="",
                metadata={"extracted": False}
            )
        
        # Combine document contents with rich metadata
        doc_texts = []
        for i, doc in enumerate(documents[:5], 1):
            meta = doc.metadata or {}
            doc_info = f"[Document {i}]"
            if meta.get("event_type"):
                doc_info += f"\nEvent Type: {meta['event_type']}"
            if meta.get("causal_motif"):
                doc_info += f"\nCausal Pattern: {meta['causal_motif']}"
            if meta.get("trigger_phrase"):
                doc_info += f"\nTrigger: {meta['trigger_phrase']}"
            if meta.get("explanation"):
                doc_info += f"\nExplanation: {meta['explanation']}"
            doc_info += f"\nConversation:\n{doc.content}"
            doc_texts.append(doc_info)
        
        combined = "\n\n---\n\n".join(doc_texts)
        
        prompt = f"""You are analyzing customer service conversation data to answer questions about customer behavior patterns.

Question: {query.text}

Below are relevant conversation excerpts with their metadata (event types, causal patterns, triggers, and explanations):

{combined}

Based on these conversation excerpts and their analysis, extract and synthesize the key information that answers the question. Focus on:
1. Patterns and common themes across conversations
2. Specific causes, triggers, and factors mentioned
3. Customer behaviors and agent responses

Key Information and Insights:"""
        
        extracted = self.llm.generate_quality(prompt)
        
        self.log(f"Extracted {len(extracted)} chars")
        print(
            f"[TRACE][EXIT][ExtractorAgent.process] docs={len(documents)} "
            f"extracted_len={len(extracted)} took={perf_counter()-t0:.3f}s"
        )
        
        return AgentResponse(
            agent_name=self.name,
            result=extracted,
            documents=documents,
            metadata={"doc_count": len(documents)}
        )
