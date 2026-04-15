from typing import Optional, Dict, Any, List

from .ragSystem import RAGSystem

# Global instance for simple API usage
_default_system: Optional[RAGSystem] = None


def _get_system(session_id: Optional[str] = None) -> RAGSystem:
    global _default_system
    
    if _default_system is None or (session_id and session_id != _default_system.session_id):
        _default_system = RAGSystem(session_id=session_id)
    
    return _default_system


def answer_question(
    question: str, 
    session_id: Optional[str] = None
) -> str:
    """Answer a question.
    
    Args:
        question: The question to answer
        session_id: Optional session ID for conversation continuity
        
    Returns:
        Answer string
        
    Example:
        >>> answer = answer_question("What is your refund policy?")
        >>> print(answer)
    """
    system = _get_system(session_id)
    return system.answer(question)


def answer_with_proof(
    question: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Answer a question with evidence verification.
    
    Args:
        question: The question to answer
        session_id: Optional session ID
        
    Returns:
        Dict with answer, evidence, quality score
        
    Example:
        >>> result = answer_with_proof("How do I cancel my order?")
        >>> print(result["answer"])
        >>> print(result["evidence"])
    """
    system = _get_system(session_id)
    answer = system.answer(question, include_proof=True)
    result = system.get_last_result()
    
    return {
        "answer": answer,
        "evidence": result.evidence if result else None,
        "quality_score": result.quality_score if result else None,
        "transcript_ids": result.transcript_ids if result else [],
    }


def get_evidence(session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get evidence for the last answer.
    
    Args:
        session_id: Optional session ID
        
    Returns:
        Evidence dict or None
        
    Example:
        >>> answer_question("What is the return policy?")
        >>> evidence = get_evidence()
        >>> for span in evidence.get("evidence_spans", []):
        ...     print(span["text"])
    """
    system = _get_system(session_id)
    return system.get_evidence()


def get_conversation(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get conversation history.
    
    Args:
        session_id: Optional session ID
        
    Returns:
        List of conversation entries
        
    Example:
        >>> history = get_conversation()
        >>> for entry in history:
        ...     print(f"Q: {entry['query']}")
        ...     print(f"A: {entry['answer']}")
    """
    system = _get_system(session_id)
    return system.get_conversation()


def clear_conversation(session_id: Optional[str] = None):
    system = _get_system(session_id)
    system.clear_history()


def close():
    global _default_system
    if _default_system:
        _default_system.close()
        _default_system = None
