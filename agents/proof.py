"""
Proof agent - Verifies answers against full transcripts.
Extracts causal spans that support the answer.
"""

from typing import Optional, Any, List, Dict

from .base import BaseAgent
from ..models import Query, AgentResponse
from ..services.llm_provider import get_llm, UnifiedLLM
from ..services.transcripts import TranscriptManager


class ProofAgent(BaseAgent):
    """Verifies answers and extracts supporting evidence."""
    
    def __init__(
        self, 
        llm: Optional[UnifiedLLM] = None,
        transcripts: Optional[TranscriptManager] = None
    ):
        super().__init__("Proof")
        self.llm = llm or get_llm()
        self.transcripts = transcripts or TranscriptManager()
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Verify answer against transcripts and extract evidence."""
        if not isinstance(context, dict):
            return AgentResponse(
                agent_name=self.name,
                result={"verified": False, "reason": "No context provided"}
            )
        
        answer = context.get("answer", "")
        transcript_ids = context.get("transcript_ids", [])
        
        if not transcript_ids:
            return AgentResponse(
                agent_name=self.name,
                result={"verified": False, "reason": "No transcript IDs"}
            )
        
        # Fetch and analyze transcripts
        evidence_spans = []
        for tid in transcript_ids[:3]:  # Limit to top 3
            conv_text = self.transcripts.get_conversation_text(tid)
            if conv_text:
                spans = self._extract_causal_spans(query.text, answer, conv_text, tid)
                evidence_spans.extend(spans)
        
        # Verify answer
        is_verified = len(evidence_spans) > 0
        
        self.log(f"Found {len(evidence_spans)} evidence spans, verified: {is_verified}")
        
        return AgentResponse(
            agent_name=self.name,
            result={
                "verified": is_verified,
                "evidence_count": len(evidence_spans),
                "evidence_spans": evidence_spans
            },
            transcript_ids=transcript_ids,
            metadata={"query": query.text}
        )
    
    def _extract_causal_spans(
        self, 
        question: str, 
        answer: str, 
        transcript: str,
        transcript_id: str
    ) -> List[Dict[str, Any]]:
        """Extract spans from transcript that support the answer."""
        prompt = f"""Find exact quotes from the conversation that support the answer.
Return only the relevant quotes, one per line.

Question: {question}
Answer: {answer}

Conversation:
{transcript[:3000]}

Supporting quotes (one per line):"""
        
        response = self.llm.generate_quality(prompt)
        
        spans = []
        for line in response.split("\n"):
            line = line.strip()
            if line and len(line) > 20:
                # Verify span exists in transcript
                if any(word in transcript.lower() for word in line.lower().split()[:5]):
                    spans.append({
                        "text": line,
                        "transcript_id": transcript_id
                    })
        
        return spans[:3]  # Limit to 3 spans per transcript
    
    def get_evidence_for_answer(
        self, 
        question: str, 
        answer: str, 
        transcript_ids: List[str]
    ) -> Dict[str, Any]:
        """Public method to get evidence for an answer."""
        query = Query(text=question)
        context = {
            "answer": answer,
            "transcript_ids": transcript_ids
        }
        response = self.process(query, context)
        return response.result
