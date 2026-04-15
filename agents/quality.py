"""
Quality checker agent - Evaluates answer quality.
"""

from typing import Optional, Any
import re

from .base import BaseAgent
from ..models import Query, AgentResponse
from ..services.llm_provider import get_llm, UnifiedLLM


class QualityCheckerAgent(BaseAgent):
    """Evaluates answer quality and provides feedback."""
    
    def __init__(self, llm: Optional[UnifiedLLM] = None):
        super().__init__("QualityChecker")
        self.llm = llm or get_llm()
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Evaluate answer quality."""
        if not isinstance(context, dict):
            return AgentResponse(
                agent_name=self.name,
                result={"score": 0, "feedback": "No context"}
            )
        
        answer = context.get("answer", "")
        documents = context.get("documents", [])
        
        if not answer:
            return AgentResponse(
                agent_name=self.name,
                result={"score": 0, "feedback": "No answer to evaluate"}
            )
        
        # Build context summary
        doc_summary = ""
        if documents:
            doc_texts = [d.content[:200] for d in documents[:3]]
            doc_summary = "\n".join(doc_texts)
        
        prompt = f"""Rate this answer 0-100 and give brief feedback.

Question: {query.text}
Answer: {answer}
Source context: {doc_summary[:500]}

Format: SCORE: [number]
FEEDBACK: [one sentence]"""
        
        response = self.llm.generate_quality(prompt)
        
        # Parse score
        score = 70  # default
        feedback = "Quality assessed"
        
        score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
        if score_match:
            score = min(int(score_match.group(1)), 100)
        
        feedback_match = re.search(r'FEEDBACK:\s*(.+)', response, re.IGNORECASE)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        
        self.log(f"Quality score: {score}/100")
        
        return AgentResponse(
            agent_name=self.name,
            result={"score": score, "feedback": feedback},
            metadata={"raw_response": response}
        )
