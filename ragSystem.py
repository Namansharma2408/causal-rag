from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from .config import Config, logger
from .models import Query, QueryType, Document, RAGResult
from .services import (
    EmbeddingService,
    MongoDBManager, 
    ConversationMemory,
    TranscriptManager,
    get_llm,
    UnifiedLLM
)
from .agents import (
    RouterAgent,
    RetrieverAgent,
    RerankerAgent,
    ExtractorAgent,
    ReasonerAgent,
    ProofAgent,
    QualityCheckerAgent,
    DecomposerAgent,
    ThinkingAgent
)


class RAGOrchestrator:
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize services
        self.embedding = EmbeddingService(self.config)
        self.llm = get_llm()  # Use unified LLM provider
        self.mongodb = MongoDBManager(self.config)
        self.transcripts = TranscriptManager(self.config)
        
        # Initialize agents (they will use get_llm() internally)
        self.decomposer = DecomposerAgent(self.llm)  # Multi-hop decomposition
        self.router = RouterAgent(self.llm)
        self.retriever = RetrieverAgent(self.mongodb, self.embedding, self.config)
        self.reranker = RerankerAgent(self.llm)
        self.extractor = ExtractorAgent(self.llm)
        self.reasoner = ReasonerAgent(self.llm)
        self.proof = ProofAgent(self.llm, self.transcripts)
        self.quality = QualityCheckerAgent(self.llm)
        self.thinking = ThinkingAgent(self.llm)  # Multi-model consensus
    
    def answer(
        self, 
        question: str,
        session_id: Optional[str] = None,
        include_proof: bool = False,
        thinking_mode: bool = None
    ) -> RAGResult:
        """Process a question through the RAG pipeline.
        
        Supports:
        - Simple queries: Processed directly through the pipeline
        - Complex queries: Decomposed into sub-queries (multi-hop)
        - Thinking mode: Multi-model consensus with peer review
        
        Args:
            question: User question
            session_id: Optional session ID for conversation memory
            include_proof: Whether to include evidence verification
            thinking_mode: Enable multi-model consensus (overrides config)
            
        Returns:
            RAGResult with answer and metadata
        """
        start = perf_counter()
        query = Query(text=question, session_id=session_id)
        logger.info(f"[TRACE][pipeline][start] session={session_id} q_len={len(question)}")
        use_thinking = thinking_mode if thinking_mode is not None else self.config.THINKING_MODE
        
        # Check for thinking mode
        if use_thinking:
            logger.info("[ThinkingMode] Using multi-model consensus")
            result = self._answer_thinking(query, include_proof)
            logger.info(f"[TRACE][pipeline][done] mode=thinking total={perf_counter()-start:.3f}s")
            return result
        
        # 0. Check if query needs decomposition (multi-hop)
        decompose_result = self.decomposer.process(query)
        decomposition = decompose_result.result
        
        if decomposition.needs_decomposition:
            # Multi-hop: Process each sub-query and synthesize
            logger.info(f"[MultiHop] Decomposing query into {len(decomposition.sub_queries)} sub-queries")
            result = self._answer_multihop(
                original_question=question,
                sub_queries=decomposition.sub_queries,
                session_id=session_id,
                include_proof=include_proof,
                complexity_score=decomposition.complexity_score
            )
            logger.info(f"[TRACE][pipeline][done] mode=multihop total={perf_counter()-start:.3f}s")
            return result
        else:
            # Simple query: Direct processing
            logger.info(f"[SingleHop] Processing simple query directly (complexity: {decomposition.complexity_score}/10)")
            result = self._answer_single(query, include_proof)
            logger.info(f"[TRACE][pipeline][done] mode=singlehop total={perf_counter()-start:.3f}s")
            return result
    
    def _answer_thinking(
        self,
        query: Query,
        include_proof: bool = False
    ) -> RAGResult:
        """
        Process query using Thinking Mode (multi-model consensus).
        
        Steps:
        1. Retrieve documents first
        2. Extract information
        3. Run 4 model personas in parallel
        4. Conduct anonymous peer reviews
        5. Select best answer and refine
        """
        # 1. Retrieve documents
        retrieve_result = self.retriever.process(query)
        documents = retrieve_result.documents
        transcript_ids = retrieve_result.transcript_ids
        
        # 2. Extract information
        extract_result = self.extractor.process(query, documents)
        extracted = extract_result.result
        
        # 3. Run thinking mode
        logger.info("[ThinkingMode] Running 4 model personas in parallel...")
        thinking_result = self.thinking.process(query, documents, extracted)
        
        logger.info(f"[ThinkingMode] Winner: {thinking_result.winning_model.value}")
        logger.info(f"[ThinkingMode] Consensus: {thinking_result.consensus_score}")
        logger.info(f"[ThinkingMode] Scores: {thinking_result.metadata.get('scores', {})}")
        
        # 4. Optional proof verification
        evidence = None
        if include_proof and transcript_ids:
            proof_context = {
                "answer": thinking_result.final_answer,
                "transcript_ids": transcript_ids
            }
            proof_result = self.proof.process(query, proof_context)
            evidence = proof_result.result
        
        return RAGResult(
            answer=thinking_result.final_answer,
            query=query.text,
            documents=documents,
            transcript_ids=transcript_ids,
            quality_score=int(thinking_result.consensus_score * 100),
            quality_feedback=thinking_result.refinement_notes,
            evidence=evidence,
            metadata={
                "query_type": "thinking",
                "doc_count": len(documents),
                "thinking_mode": True,
                "winning_model": thinking_result.winning_model.value,
                "consensus_score": thinking_result.consensus_score,
                "model_scores": thinking_result.metadata.get("scores", {}),
                "num_reviews": thinking_result.metadata.get("num_reviews", 0)
            }
        )
    
    def _answer_single(
        self, 
        query: Query, 
        include_proof: bool = False,
        fast_mode: bool = None
    ) -> RAGResult:
        """Process a single query through the standard pipeline.
        
        Args:
            query: Query object
            include_proof: Whether to include evidence verification
            fast_mode: Override config fast_mode setting
        """
        t_single = perf_counter()
        use_fast = fast_mode if fast_mode is not None else self.config.FAST_MODE
        
        # 1. Route query (skip in fast mode for sub-queries)
        if not use_fast:
            t_route = perf_counter()
            route_result = self.router.process(query)
            query.query_type = route_result.result
            logger.info(f"[TRACE][singlehop][router] {perf_counter()-t_route:.3f}s")
        else:
            query.query_type = QueryType.FACTUAL  # Default type
        
        # 2. Retrieve documents
        t_retrieve = perf_counter()
        retrieve_result = self.retriever.process(query)
        documents = retrieve_result.documents
        transcript_ids = retrieve_result.transcript_ids
        logger.info(f"[TRACE][singlehop][retriever] {perf_counter()-t_retrieve:.3f}s docs={len(documents)}")
        
        # 3. Rerank documents (skip in fast mode or if few docs)
        if not use_fast and len(documents) > self.config.SKIP_RERANK_THRESHOLD:
            t_rerank = perf_counter()
            rerank_result = self.reranker.process(query, documents)
            reranked_docs = rerank_result.documents
            transcript_ids = rerank_result.transcript_ids
            logger.info(f"[TRACE][singlehop][reranker] {perf_counter()-t_rerank:.3f}s docs={len(reranked_docs)}")
        else:
            reranked_docs = documents  # Use retrieval order
            if use_fast:
                logger.info(f"[FastMode] Skipping reranking")
        
        # 4. Extract information
        t_extract = perf_counter()
        extract_result = self.extractor.process(query, reranked_docs)
        extracted = extract_result.result
        logger.info(f"[TRACE][singlehop][extractor] {perf_counter()-t_extract:.3f}s")
        
        # 5. Generate answer
        reason_context = {
            "extracted": extracted,
            "documents": reranked_docs
        }
        t_reason = perf_counter()
        reason_result = self.reasoner.process(query, reason_context)
        answer = reason_result.result
        logger.info(f"[TRACE][singlehop][reasoner] {perf_counter()-t_reason:.3f}s answer_len={len(answer) if answer else 0}")
        
        # 6. Check quality (skip in fast mode)
        quality_score = 70  # Default
        quality_feedback = "Fast mode - quality check skipped"
        if not use_fast:
            t_quality = perf_counter()
            quality_context = {
                "answer": answer,
                "documents": reranked_docs
            }
            quality_result = self.quality.process(query, quality_context)
            quality_score = quality_result.result.get("score", 0)
            quality_feedback = quality_result.result.get("feedback", "")
            logger.info(f"[TRACE][singlehop][quality] {perf_counter()-t_quality:.3f}s score={quality_score}")
        
        # 7. Optional proof verification
        evidence = None
        if include_proof and transcript_ids:
            t_proof = perf_counter()
            proof_context = {
                "answer": answer,
                "transcript_ids": transcript_ids
            }
            proof_result = self.proof.process(query, proof_context)
            evidence = proof_result.result
            logger.info(f"[TRACE][singlehop][proof] {perf_counter()-t_proof:.3f}s")

        logger.info(f"[TRACE][singlehop][done] total={perf_counter()-t_single:.3f}s")
        
        return RAGResult(
            answer=answer,
            query=query.text,
            documents=reranked_docs,
            transcript_ids=transcript_ids,
            quality_score=quality_score,
            quality_feedback=quality_feedback,
            evidence=evidence,
            metadata={
                "query_type": query.query_type.value,
                "doc_count": len(reranked_docs),
                "multihop": False,
                "fast_mode": use_fast
            }
        )
    
    def _answer_multihop(
        self,
        original_question: str,
        sub_queries: List[str],
        session_id: Optional[str] = None,
        include_proof: bool = False,
        complexity_score: int = 0
    ) -> RAGResult:
        """
        Process complex query using multi-hop reasoning.
        
        OPTIMIZED:
        - Parallel processing of sub-queries using ThreadPoolExecutor
        - Fast mode for sub-queries (skip router, reranker, quality check)
        - Only synthesize and quality-check the final answer
        """
        sub_answers = []
        all_documents = []
        all_transcript_ids = []
        
        def process_subquery(sub_query_text: str, index: int) -> Dict[str, Any]:
            """Process a single sub-query (runs in parallel)."""
            logger.info(f"[MultiHop] Processing sub-query {index}: {sub_query_text[:50]}...")
            
            sub_query = Query(text=sub_query_text, session_id=session_id)
            
            # Fast retrieval (skip routing - use default type)
            sub_query.query_type = QueryType.FACTUAL
            
            # Retrieve
            retrieve_result = self.retriever.process(sub_query)
            documents = retrieve_result.documents
            transcript_ids = retrieve_result.transcript_ids
            
            # Skip reranking in multihop for speed - retrieval order is usually good
            # Extract directly from retrieved docs
            extract_result = self.extractor.process(sub_query, documents)
            extracted = extract_result.result
            
            return {
                "sub_query": sub_query_text,
                "extracted": extracted,
                "documents": documents,
                "transcript_ids": transcript_ids,
                "doc_count": len(documents)
            }
        
        # Process sub-queries in PARALLEL
        if self.config.PARALLEL_SUBQUERIES and len(sub_queries) > 1:
            logger.info(f"[MultiHop] Processing {len(sub_queries)} sub-queries in parallel")
            
            with ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_QUERIES) as executor:
                # Submit all sub-queries
                futures = {
                    executor.submit(process_subquery, sq, i): i 
                    for i, sq in enumerate(sub_queries, 1)
                }
                
                # Collect results as they complete
                results = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Sub-query {idx} failed: {e}")
                        results[idx] = {
                            "sub_query": sub_queries[idx-1],
                            "extracted": "",
                            "documents": [],
                            "transcript_ids": [],
                            "doc_count": 0
                        }
                
                # Sort by original order
                for i in sorted(results.keys()):
                    result = results[i]
                    sub_answers.append(result)
                    all_documents.extend(result["documents"])
                    all_transcript_ids.extend(result["transcript_ids"])
        else:
            # Sequential processing (fallback)
            for i, sub_query_text in enumerate(sub_queries, 1):
                result = process_subquery(sub_query_text, i)
                sub_answers.append(result)
                all_documents.extend(result["documents"])
                all_transcript_ids.extend(result["transcript_ids"])
        
        # Deduplicate transcript IDs
        all_transcript_ids = list(set(all_transcript_ids))
        
        # Synthesize final answer from all sub-answers
        final_answer = self._synthesize_answers(
            original_question=original_question,
            sub_answers=sub_answers
        )
        
        # Quality check on final answer only (not sub-queries)
        quality_score = 70  # Default
        quality_feedback = ""
        if not self.config.FAST_MODE:
            quality_context = {
                "answer": final_answer,
                "documents": all_documents[:10]
            }
            quality_result = self.quality.process(
                Query(text=original_question), 
                quality_context
            )
            quality_score = quality_result.result.get("score", 0)
            quality_feedback = quality_result.result.get("feedback", "")
        
        # Optional proof verification
        evidence = None
        if include_proof and all_transcript_ids:
            proof_context = {
                "answer": final_answer,
                "transcript_ids": all_transcript_ids[:5]
            }
            proof_result = self.proof.process(
                Query(text=original_question), 
                proof_context
            )
            evidence = proof_result.result
        
        return RAGResult(
            answer=final_answer,
            query=original_question,
            documents=all_documents[:10],
            transcript_ids=all_transcript_ids,
            quality_score=quality_score,
            quality_feedback=quality_feedback,
            evidence=evidence,
            metadata={
                "query_type": "multihop",
                "doc_count": len(all_documents),
                "multihop": True,
                "sub_query_count": len(sub_queries),
                "complexity_score": complexity_score,
                "sub_queries": sub_queries,
                "parallel": self.config.PARALLEL_SUBQUERIES
            }
        )
    
    def _synthesize_answers(
        self,
        original_question: str,
        sub_answers: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize sub-answers into a comprehensive final answer.
        """
        # Build context from all sub-answers
        sub_answer_text = ""
        for i, sa in enumerate(sub_answers, 1):
            sub_answer_text += f"\n--- Sub-question {i}: {sa['sub_query']} ---\n"
            sub_answer_text += f"{sa['extracted']}\n"
        
        prompt = f"""You are synthesizing answers from multiple sub-questions into one comprehensive response.

ORIGINAL QUESTION: {original_question}

INFORMATION FROM SUB-QUESTIONS:
{sub_answer_text}

INSTRUCTIONS:
1. Combine all the information into a single, coherent answer
2. Organize the response logically, grouping related points
3. Remove any redundancy between sub-answers
4. Ensure the final answer fully addresses the original question
5. Include specific examples and patterns from the data
6. Provide actionable insights where relevant

COMPREHENSIVE ANSWER:"""

        answer = self.llm.generate_quality(prompt)
        return answer
    
    def close(self):
        """Cleanup resources."""
        self.mongodb.close()
        self.transcripts.close()


class RAGSystem:
    
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
        return self._last_result
    
    def get_evidence(self) -> Optional[Dict[str, Any]]:
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
        return self.memory.history
    
    def clear_history(self):
        self.memory.clear()
    
    def close(self):
        self.orchestrator.close()
