"""
Retriever agent - Fetches relevant documents using hybrid search.
"""

from typing import Optional, Any, List
from time import perf_counter

from .base import BaseAgent
from ..models import Query, Document, AgentResponse
from ..services.embedding import EmbeddingService
from ..services.mongodb import MongoDBManager
from ..config import Config


class RetrieverAgent(BaseAgent):
    """Retrieves documents using hybrid search."""
    
    def __init__(
        self,
        mongodb: Optional[MongoDBManager] = None,
        embedding: Optional[EmbeddingService] = None,
        config: Optional[Config] = None
    ):
        super().__init__("Retriever")
        self.config = config or Config()
        self.mongodb = mongodb or MongoDBManager(self.config)
        self.embedding = embedding or EmbeddingService(self.config)
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Retrieve relevant documents."""
        t0 = perf_counter()
        print(f"[TRACE][ENTER][RetrieverAgent.process] q_len={len(query.text)}")

        # Generate query embedding
        query_embedding = self.embedding.embed(query.text)
        
        # Hybrid search
        documents = self.mongodb.hybrid_search(
            query_text=query.text,
            query_embedding=query_embedding,
            top_k=self.config.DEFAULT_TOP_K,
            top_clusters=self.config.DEFAULT_TOP_CLUSTERS,
            alpha=self.config.HYBRID_ALPHA
        )
        
        # Extract transcript IDs
        transcript_ids = list(set(d.transcript_id for d in documents if d.transcript_id))
        
        self.log(f"Retrieved {len(documents)} documents")
        self.log(f"Transcript IDs: {transcript_ids}")
        print(
            f"[TRACE][EXIT][RetrieverAgent.process] docs={len(documents)} "
            f"transcripts={len(transcript_ids)} took={perf_counter()-t0:.3f}s"
        )
        
        return AgentResponse(
            agent_name=self.name,
            result=documents,
            documents=documents,
            transcript_ids=transcript_ids,
            metadata={"query": query.text}
        )
