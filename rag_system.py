"""
RAG Orchestrator - Coordinates the multi-agent pipeline.
"""

from typing import Optional, List, Dict, Any

from .config import Config, logger
from .models import Query, QueryType, Document, RAGResult
from .services import (
    EmbeddingService, 
    OllamaLLM, 
    MongoDBManager, 
    ConversationMemory,
    TranscriptManager
)
from .agents import (
    RouterAgent,
    RetrieverAgent,
    RerankerAgent,
    ExtractorAgent,
    ReasonerAgent,
    ProofAgent,
    QualityCheckerAgent
)


class RAGOrchestrator:
    """Orchestrates the multi-agent RAG pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize services
        self.embedding = EmbeddingService(self.config)
        self.llm = OllamaLLM(self.config)
        self.mongodb = MongoDBManager(self.config)
        self.transcripts = TranscriptManager(self.config)
        
        # Initialize agents
        self.router = RouterAgent(self.llm)
        self.retriever = RetrieverAgent(self.mongodb, self.embedding, self.config)
        self.reranker = RerankerAgent(self.llm)
        self.extractor = ExtractorAgent(self.llm)
        self.reasoner = ReasonerAgent(self.llm)
        self.proof = ProofAgent(self.llm, self.transcripts)
        self.quality = QualityCheckerAgent(self.llm)
    
    def answer(
        self, 
        question: str,
        session_id: Optional[str] = None,
        include_proof: bool = False
    ) -> RAGResult:
        """Process a question through the RAG pipeline.
        
        Args:
            question: User question
            session_id: Optional session ID for conversation memory
            include_proof: Whether to include evidence verification
            
        Returns:
            RAGResult with answer and metadata
        """
        query = Query(text=question, session_id=session_id)
        
        # 1. Route query
        route_result = self.router.process(query)
        query.query_type = route_result.result
        
        # 2. Retrieve documents
        retrieve_result = self.retriever.process(query)
        documents = retrieve_result.documents
        
        # 3. Rerank documents
        rerank_result = self.reranker.process(query, documents)
        reranked_docs = rerank_result.documents
        transcript_ids = rerank_result.transcript_ids
        
        # 4. Extract information
        extract_result = self.extractor.process(query, reranked_docs)
        extracted = extract_result.result
        
        # 5. Generate answer
        reason_context = {
            "extracted": extracted,
            "documents": reranked_docs
        }
        reason_result = self.reasoner.process(query, reason_context)
        answer = reason_result.result
        
        # 6. Check quality
        quality_context = {
            "answer": answer,
            "documents": reranked_docs
        }
        quality_result = self.quality.process(query, quality_context)
        quality_score = quality_result.result.get("score", 0)
        quality_feedback = quality_result.result.get("feedback", "")
        
        # 7. Optional proof verification
        evidence = None
        if include_proof and transcript_ids:
            proof_context = {
                "answer": answer,
                "transcript_ids": transcript_ids
            }
            proof_result = self.proof.process(query, proof_context)
            evidence = proof_result.result
        
        return RAGResult(
            answer=answer,
            query=question,
            documents=reranked_docs,
            transcript_ids=transcript_ids,
            quality_score=quality_score,
            quality_feedback=quality_feedback,
            evidence=evidence,
            metadata={
                "query_type": query.query_type.value,
                "doc_count": len(reranked_docs)
            }
        )
    
    def close(self):
        """Cleanup resources."""
        self.mongodb.close()
        self.transcripts.close()


class RAGSystem:
    """High-level RAG system with session management."""
    
    def __init__(self, session_id: Optional[str] = None, config: Optional[Config] = None):
        self.config = config or Config()
        self.orchestrator = RAGOrchestrator(self.config)
        self.memory = ConversationMemory(session_id, self.config)
        self._last_result: Optional[RAGResult] = None
    
    @property
    def session_id(self) -> str:
        return self.memory.session_id
    
    def answer(self, question: str, include_proof: bool = False) -> str:
        """Answer a question.
        
        Args:
            question: User question
            include_proof: Whether to verify with full transcripts
            
        Returns:
            Answer string
        """
        # Add conversation context
        context = self.memory.get_context()
        
        # Get answer
        result = self.orchestrator.answer(
            question=question,
            session_id=self.session_id,
            include_proof=include_proof
        )
        
        # Store in memory
        self.memory.add(
            query=question,
            answer=result.answer,
            metadata={
                "transcript_ids": result.transcript_ids,
                "quality_score": result.quality_score
            }
        )
        
        self._last_result = result
        return result.answer
    
    def get_last_result(self) -> Optional[RAGResult]:
        """Get the full result from last query."""
        return self._last_result
    
    def get_evidence(self) -> Optional[Dict[str, Any]]:
        """Get evidence for last answer."""
        if not self._last_result:
            return None
        
        if self._last_result.evidence:
            return self._last_result.evidence
        
        # Fetch evidence if not already present
        transcript_ids = self._last_result.transcript_ids
        if not transcript_ids:
            return None
        
        query = Query(text=self._last_result.query)
        proof_context = {
            "answer": self._last_result.answer,
            "transcript_ids": transcript_ids
        }
        proof_result = self.orchestrator.proof.process(query, proof_context)
        self._last_result.evidence = proof_result.result
        
        return proof_result.result
    
    def get_conversation(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.memory.history
    
    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()
    
    def close(self):
        """Cleanup resources."""
        self.orchestrator.close()
