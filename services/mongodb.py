"""
MongoDB service for document storage and retrieval.
Supports hybrid search with vector similarity and BM25.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pymongo import MongoClient
import numpy as np

from ..config import Config, logger
from ..models import Document


class MongoDBManager:
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._client = MongoClient(self.config.MONGODB_URI)
        self._db = self._client[self.config.DB_NAME]
        self._collection = self._db[self.config.COLLECTION_NAME]
        self._cluster_centroids: Optional[Dict[int, List[float]]] = None
        
    @property
    def collection(self):
        return self._collection
    
    def _load_cluster_centroids(self) -> Dict[int, List[float]]:
        if self._cluster_centroids is not None:
            return self._cluster_centroids
            
        centroids = {}
        try:
            pipeline = [
                {"$group": {
                    "_id": "$cluster_id",
                    "centroid": {"$first": "$centroid"}
                }}
            ]
            for doc in self._collection.aggregate(pipeline):
                if doc.get("centroid"):
                    centroids[doc["_id"]] = doc["centroid"]
        except Exception as e:
            logger.warning(f"Could not load centroids: {e}")
        
        self._cluster_centroids = centroids
        return centroids
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_np, b_np = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a_np), np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))
    
    def _find_top_clusters(
        self, 
        query_embedding: List[float], 
        top_n: int = 5
    ) -> List[int]:
        centroids = self._load_cluster_centroids()
        
        if not centroids:
            # Return all cluster IDs if centroids not available
            distinct = self._collection.distinct("cluster_id")
            return distinct[:top_n]
        
        similarities = [
            (cid, self._cosine_similarity(query_embedding, centroid))
            for cid, centroid in centroids.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in similarities[:top_n]]
    
    def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        cluster_ids: Optional[List[int]] = None
    ) -> List[Tuple[Dict, float]]:
        query = {}
        if cluster_ids:
            query["cluster_id"] = {"$in": cluster_ids}
        
        results = []
        for doc in self._collection.find(query).limit(top_k * 3):
            if "embedding" in doc:
                score = self._cosine_similarity(query_embedding, doc["embedding"])
                results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _bm25_search(
        self,
        query_text: str,
        top_k: int,
        cluster_ids: Optional[List[int]] = None
    ) -> List[Tuple[Dict, float]]:
        # Extract keywords
        keywords = re.findall(r'\w+', query_text.lower())
        if not keywords:
            return []
        
        # Build regex pattern
        pattern = "|".join(re.escape(k) for k in keywords)
        
        query = {"causal_span_text": {"$regex": pattern, "$options": "i"}}
        if cluster_ids:
            query["cluster_id"] = {"$in": cluster_ids}
        
        results = []
        for doc in self._collection.find(query).limit(top_k * 3):
            text = doc.get("causal_span_text", "").lower()
            score = sum(1 for k in keywords if k in text) / len(keywords)
            results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        top_clusters: int = 5,
        alpha: float = 0.6
    ) -> List[Document]:
        """Perform hybrid search combining vector and BM25.
        
        Args:
            query_text: Text query
            query_embedding: Query embedding vector
            top_k: Number of results to return
            top_clusters: Number of clusters to search
            alpha: Weight for vector search (1-alpha for BM25)
        
        Returns:
            List of Document objects
        """
        # Find relevant clusters
        cluster_ids = self._find_top_clusters(query_embedding, top_clusters)
        logger.debug(f"Searching clusters: {cluster_ids}")
        
        # Get results from both methods
        vector_results = self._vector_search(query_embedding, top_k * 2, cluster_ids)
        bm25_results = self._bm25_search(query_text, top_k * 2, cluster_ids)
        
        # Combine scores
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}
        
        for doc, score in vector_results:
            doc_id = str(doc["_id"])
            doc_scores[doc_id] = alpha * score
            doc_map[doc_id] = doc
        
        for doc, score in bm25_results:
            doc_id = str(doc["_id"])
            if doc_id in doc_scores:
                doc_scores[doc_id] += (1 - alpha) * score
            else:
                doc_scores[doc_id] = (1 - alpha) * score
                doc_map[doc_id] = doc
        
        # Sort and convert to Documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        documents = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = doc_map[doc_id]
            metadata = doc.get("metadata", {})
            documents.append(Document(
                id=doc_id,
                content=doc.get("causal_span_text", ""),
                transcript_id=doc.get("transcript_id", ""),
                score=score,
                metadata={
                    "cluster_id": doc.get("cluster_id"),
                    "domain": metadata.get("domain") if isinstance(metadata, dict) else None,
                    "intent": metadata.get("intent") if isinstance(metadata, dict) else None,
                    "event_type": doc.get("event_type"),
                    "trigger_phrase": doc.get("trigger_phrase"),
                    "causal_motif": doc.get("causal_motif"),
                    "explanation": doc.get("explanation")
                }
            ))
        
        return documents
    
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        from bson import ObjectId
        try:
            doc = self._collection.find_one({"_id": ObjectId(doc_id)})
            if doc:
                metadata = doc.get("metadata", {})
                return Document(
                    id=str(doc["_id"]),
                    content=doc.get("causal_span_text", ""),
                    transcript_id=doc.get("transcript_id", ""),
                    metadata={
                        "cluster_id": doc.get("cluster_id"),
                        "domain": metadata.get("domain") if isinstance(metadata, dict) else None,
                        "intent": metadata.get("intent") if isinstance(metadata, dict) else None,
                        "event_type": doc.get("event_type"),
                        "trigger_phrase": doc.get("trigger_phrase"),
                        "causal_motif": doc.get("causal_motif"),
                        "explanation": doc.get("explanation")
                    }
                )
        except Exception as e:
            logger.error(f"Error getting document: {e}")
        return None
    
    def close(self):
        self._client.close()
