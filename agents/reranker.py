"""
Reranker agent - Reorders documents by relevance using Cross-Encoder and LLM.

Cross-Encoder provides more accurate semantic similarity scoring by processing
query and document together through a transformer model.
"""

from typing import Optional, Any, List, Tuple
import os

from .base import BaseAgent
from ..models import Query, Document, AgentResponse
from ..services.llm_provider import get_llm, UnifiedLLM

# Cross-encoder model options (from smallest to largest)
CROSS_ENCODER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",      # Fast, good quality (22M params)
    "cross-encoder/ms-marco-MiniLM-L-12-v2",     # Better quality (33M params)
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",    # Fastest (4.4M params)
    "BAAI/bge-reranker-base",                     # High quality Chinese/English
    "BAAI/bge-reranker-large",                    # Highest quality
]

# Default model - good balance of speed and quality
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Cross-Encoder based reranker using sentence-transformers."""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = None):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name: str = None):
        if self._initialized:
            return
            
        self.model_name = model_name or os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
        self._model = None
        self._available = None
        self._initialized = True
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return True
            
        try:
            from sentence_transformers import CrossEncoder
            print(f"[CrossEncoder] Loading model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            self._available = True
            print(f"[CrossEncoder] Model loaded successfully")
            return True
        except ImportError:
            print("[CrossEncoder] sentence-transformers not installed. Install with: pip install sentence-transformers")
            self._available = False
            return False
        except Exception as e:
            print(f"[CrossEncoder] Failed to load model: {e}")
            self._available = False
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if cross-encoder is available."""
        if self._available is None:
            self._load_model()
        return self._available
    
    def score(self, query: str, documents: List[str]) -> List[float]:
        """
        Score documents against query using cross-encoder.
        
        Args:
            query: The search query
            documents: List of document texts
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not self._load_model():
            return [0.5] * len(documents)
        
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Get scores from cross-encoder
        scores = self._model.predict(pairs)
        
        # Convert to list if numpy array
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        
        return scores
    
    def score_single(self, query: str, document: str) -> float:
        """Score a single document against query."""
        scores = self.score(query, [document])
        return scores[0] if scores else 0.5
    
    def rerank(self, query: str, documents: List[Tuple[Any, str]], top_k: int = None) -> List[Tuple[Any, float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of (item, text) tuples where text is used for scoring
            top_k: Number of top results to return (None = all)
            
        Returns:
            List of (item, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        # Extract texts for scoring
        texts = [text for _, text in documents]
        
        # Get scores
        scores = self.score(query, texts)
        
        # Combine items with scores
        scored = [(item, score) for (item, _), score in zip(documents, scores)]
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            scored = scored[:top_k]
        
        return scored


class RerankerAgent(BaseAgent):
    """
    Reranks documents using Cross-Encoder for accurate semantic similarity.
    Falls back to LLM-based scoring if cross-encoder is unavailable.
    """
    
    def __init__(
        self, 
        llm: Optional[UnifiedLLM] = None,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = None,
        cross_encoder_weight: float = 0.7,
        llm_weight: float = 0.3,
        use_hybrid_scoring: bool = False
    ):
        """
        Initialize the reranker agent.
        
        Args:
            llm: LLM instance for fallback/hybrid scoring
            use_cross_encoder: Whether to use cross-encoder (recommended)
            cross_encoder_model: Specific cross-encoder model to use
            cross_encoder_weight: Weight for cross-encoder score in hybrid mode
            llm_weight: Weight for LLM score in hybrid mode
            use_hybrid_scoring: Combine cross-encoder and LLM scores
        """
        super().__init__("Reranker")
        self.llm = llm or get_llm()
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_weight = cross_encoder_weight
        self.llm_weight = llm_weight
        self.use_hybrid_scoring = use_hybrid_scoring
        
        # Initialize cross-encoder
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker(cross_encoder_model)
        else:
            self.cross_encoder = None
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """Rerank documents by relevance using cross-encoder."""
        documents: List[Document] = context if context else []
        
        if not documents:
            return AgentResponse(
                agent_name=self.name,
                result=[],
                documents=[]
            )
        
        # Choose scoring method
        if self.use_cross_encoder and self.cross_encoder and self.cross_encoder.is_available:
            if self.use_hybrid_scoring:
                scored = self._hybrid_score(query.text, documents)
                self.log(f"Using hybrid scoring (cross-encoder + LLM)")
            else:
                scored = self._cross_encoder_score(query.text, documents)
                self.log(f"Using cross-encoder scoring")
        else:
            scored = self._llm_score(query.text, documents)
            self.log(f"Using LLM-based scoring (fallback)")
        
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
        self.log(f"Top scores: {[f'{s:.3f}' for _, s in scored[:5]]}")
        self.log(f"Top transcript IDs: {transcript_ids[:5]}")
        
        return AgentResponse(
            agent_name=self.name,
            result=reranked,
            documents=reranked,
            transcript_ids=transcript_ids
        )
    
    def _cross_encoder_score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Score documents using cross-encoder."""
        # Prepare texts for scoring
        texts = [doc.content[:1024] for doc in documents]  # Limit length for efficiency
        
        # Get scores from cross-encoder
        scores = self.cross_encoder.score(query, texts)
        
        # Normalize scores to 0-1 range using sigmoid-like normalization
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            normalized_scores = [0.5] * len(scores)
        
        return list(zip(documents, normalized_scores))
    
    def _llm_score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Score documents using LLM (fallback method)."""
        scored = []
        for doc in documents:
            score = self._score_document_llm(query, doc.content)
            scored.append((doc, score))
        return scored
    
    def _hybrid_score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Combine cross-encoder and LLM scores for best results."""
        # Get cross-encoder scores
        ce_scored = dict(self._cross_encoder_score(query, documents))
        
        # Get LLM scores  
        llm_scored = dict(self._llm_score(query, documents))
        
        # Combine scores
        combined = []
        for doc in documents:
            ce_score = ce_scored.get(doc, 0.5)
            llm_score = llm_scored.get(doc, 0.5)
            hybrid_score = (self.cross_encoder_weight * ce_score + 
                          self.llm_weight * llm_score)
            combined.append((doc, hybrid_score))
        
        return combined
    
    def _score_document_llm(self, query: str, content: str) -> float:
        """Score a document's relevance to query using LLM."""
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
    
    def batch_rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[Document]:
        """
        Convenience method for batch reranking.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Return only top k documents
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        query_obj = Query(text=query)
        response = self.process(query_obj, documents)
        
        reranked = response.documents or []
        
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
