"""
Hybrid retriever combining dense and sparse retrieval with re-ranking
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..embeddings.vector_store import VectorStore
from ..embeddings.embedding_service import ReRanker
from .bm25_index import BM25Index, EnhancedBM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that combines:
    - Dense vector search (semantic similarity)
    - Sparse BM25 search (keyword matching)
    - Cross-encoder re-ranking
    - Business-aware scoring (freshness, document type)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[ReRanker] = None,
        alpha: float = 0.6,  # Weight for dense search
        beta: float = 0.3,   # Weight for sparse search
        gamma: float = 0.1,  # Weight for freshness
        use_reranker: bool = True
    ):
        """
        Initialize hybrid retriever

        Args:
            vector_store: Vector store for dense retrieval
            bm25_index: BM25 index for sparse retrieval
            reranker: Cross-encoder re-ranker
            alpha: Weight for dense search score
            beta: Weight for sparse search score
            gamma: Weight for freshness score
            use_reranker: Whether to use re-ranking
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index or EnhancedBM25Index()
        self.reranker = reranker
        self.use_reranker = use_reranker

        # Score weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Document type boost factors
        self.doc_type_boosts = {
            "policy": 1.2,
            "sop": 1.2,
            "contract": 1.1,
            "report": 1.0,
            "meeting_notes": 0.9,
            "email": 0.8,
            "manual": 1.1,
            "export": 1.0
        }

    def retrieve(
        self,
        query: str,
        k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        user_roles: Optional[List[str]] = None,
        query_type: Optional[str] = None,
        include_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search

        Args:
            query: Query text
            k: Number of results to return
            filters: Metadata filters
            user_roles: User roles for ACL filtering
            query_type: Type of query for doc-type boosting
            include_scores: Whether to include detailed scores

        Returns:
            List of retrieved documents with scores
        """
        # Get more candidates for fusion
        candidate_k = k * 3

        # Dense retrieval
        dense_results = self._dense_search(query, candidate_k, filters, user_roles)

        # Sparse retrieval
        sparse_results = self._sparse_search(query, candidate_k, filters)

        # Fuse results
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=candidate_k
        )

        # Apply business logic boosts
        boosted_results = self._apply_business_boosts(fused_results, query_type)

        # Re-rank if enabled
        if self.use_reranker and self.reranker:
            final_results = self._rerank(query, boosted_results, k)
        else:
            final_results = boosted_results[:k]

        # Add final scores
        if include_scores:
            for i, result in enumerate(final_results):
                result["final_rank"] = i + 1

        return final_results

    def _dense_search(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        user_roles: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search"""
        if user_roles:
            results = self.vector_store.search_with_acl(
                query=query,
                user_roles=user_roles,
                k=k,
                additional_filters=filters
            )
        else:
            results = self.vector_store.search(
                query=query,
                k=k,
                filters=filters
            )

        # Normalize scores and add source tag
        for result in results:
            result["dense_score"] = result.get("similarity", 0)
            result["source"] = "dense"

        return results

    def _sparse_search(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform sparse BM25 search"""
        if isinstance(self.bm25_index, EnhancedBM25Index):
            results = self.bm25_index.search_enhanced(query, k)
        else:
            results = self.bm25_index.search_with_scores(query, k)

        # Normalize BM25 scores (typically 0-30 range)
        max_score = max((r.get("bm25_score", 0) for r in results), default=1)
        if max_score > 0:
            for result in results:
                result["sparse_score"] = result.get("bm25_score", 0) / max_score
                result["source"] = "sparse"

        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,  # RRF constant
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)

        RRF score = sum(1 / (k + rank_i)) for each result list
        """
        # Create a mapping from chunk_id to fused result
        fused_map: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result.get("chunk_id", str(rank))
            if chunk_id not in fused_map:
                fused_map[chunk_id] = result.copy()
                fused_map[chunk_id]["rrf_score"] = 0
                fused_map[chunk_id]["dense_rank"] = None
                fused_map[chunk_id]["sparse_rank"] = None

            fused_map[chunk_id]["rrf_score"] += 1 / (k + rank)
            fused_map[chunk_id]["dense_rank"] = rank
            fused_map[chunk_id]["dense_score"] = result.get("dense_score", result.get("similarity", 0))

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = result.get("chunk_id", str(rank))
            if chunk_id not in fused_map:
                fused_map[chunk_id] = result.copy()
                fused_map[chunk_id]["rrf_score"] = 0
                fused_map[chunk_id]["dense_rank"] = None
                fused_map[chunk_id]["sparse_rank"] = None

            fused_map[chunk_id]["rrf_score"] += 1 / (k + rank)
            fused_map[chunk_id]["sparse_rank"] = rank
            fused_map[chunk_id]["sparse_score"] = result.get("sparse_score", result.get("bm25_score", 0))

        # Sort by RRF score
        fused_results = list(fused_map.values())
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        if top_k:
            return fused_results[:top_k]

        return fused_results

    def _apply_business_boosts(
        self,
        results: List[Dict[str, Any]],
        query_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Apply business-aware score boosts"""
        now = datetime.utcnow()

        for result in results:
            metadata = result.get("metadata", {})
            boost = 1.0

            # Document type boost
            doc_type = metadata.get("doc_type", "report")
            boost *= self.doc_type_boosts.get(doc_type, 1.0)

            # Query-type specific boosts
            if query_type:
                if query_type == "policy" and doc_type in ["policy", "sop"]:
                    boost *= 1.5
                elif query_type == "metrics" and doc_type in ["report", "export"]:
                    boost *= 1.3
                elif query_type == "contract" and doc_type == "contract":
                    boost *= 1.5

            # Freshness boost (decay over time)
            # Assuming updated_at or created_at in metadata
            # For now, apply a small random boost
            freshness_score = 1.0
            result["freshness_score"] = freshness_score

            # Combine scores
            rrf_score = result.get("rrf_score", 0)
            dense_score = result.get("dense_score", 0)
            sparse_score = result.get("sparse_score", 0)

            # Weighted combination
            combined_score = (
                self.alpha * dense_score +
                self.beta * sparse_score +
                self.gamma * freshness_score
            ) * boost

            # If RRF score exists, blend it in
            if rrf_score > 0:
                combined_score = 0.7 * combined_score + 0.3 * rrf_score * 10

            result["combined_score"] = combined_score
            result["boost_applied"] = boost

        # Re-sort by combined score
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return results

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """Re-rank results using cross-encoder"""
        if not results:
            return []

        # Get texts for re-ranking
        texts = [r.get("chunk_text", "") for r in results]

        # Re-rank
        reranked = self.reranker.rerank(query, texts, top_k=k)

        # Build final results
        final_results = []
        for idx, score in reranked:
            result = results[idx].copy()
            result["rerank_score"] = float(score)
            final_results.append(result)

        return final_results

    def add_documents_to_bm25(self, documents: List[Dict[str, Any]]):
        """Add documents to the BM25 index"""
        self.bm25_index.add_documents(documents)

    def sync_bm25_with_vector_store(self):
        """
        Sync BM25 index with vector store
        Useful when documents are added directly to vector store
        """
        # This would require iterating over vector store
        # For now, documents should be added to both
        logger.warning("sync_bm25_with_vector_store not fully implemented")

    def detect_query_type(self, query: str) -> str:
        """Detect the type of query for doc-type boosting"""
        query_lower = query.lower()

        # Policy/SOP queries
        if any(term in query_lower for term in ["policy", "procedure", "how to", "process", "guideline"]):
            return "policy"

        # Metrics/Report queries
        if any(term in query_lower for term in ["report", "metrics", "numbers", "statistics", "data", "forecast"]):
            return "metrics"

        # Contract queries
        if any(term in query_lower for term in ["contract", "agreement", "sla", "terms", "penalty"]):
            return "contract"

        # Inventory queries
        if any(term in query_lower for term in ["stock", "inventory", "sku", "warehouse", "reorder"]):
            return "inventory"

        return "general"

    def retrieve_with_intent(
        self,
        query: str,
        k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        user_roles: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve with automatic query intent detection"""
        query_type = self.detect_query_type(query)

        return self.retrieve(
            query=query,
            k=k,
            filters=filters,
            user_roles=user_roles,
            query_type=query_type
        )
