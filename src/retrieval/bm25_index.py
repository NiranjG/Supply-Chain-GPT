"""
BM25 sparse index for keyword-based retrieval
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 sparse index for keyword/term-based retrieval
    Complements dense vector search for hybrid retrieval
    """

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._doc_id_to_idx: Dict[str, int] = {}

    def add_documents(self, documents: List[Dict[str, Any]], text_key: str = "chunk_text"):
        """
        Add documents to the BM25 index

        Args:
            documents: List of document dicts with text content
            text_key: Key for the text content in docs
        """
        for doc in documents:
            if text_key in doc:
                idx = len(self.documents)
                self.documents.append(doc)
                self._doc_id_to_idx[doc.get("chunk_id", str(idx))] = idx

                # Tokenize and add
                tokens = self._tokenize(doc[text_key])
                self.tokenized_docs.append(tokens)

        # Rebuild BM25 index
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"BM25 index built with {len(self.documents)} documents")

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search the BM25 index

        Args:
            query: Query text
            k: Number of results to return
            filters: Metadata filters to apply

        Returns:
            List of (index, score, document) tuples
        """
        if not self.bm25:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create (index, score) pairs
        indexed_scores = list(enumerate(scores))

        # Apply filters if provided
        if filters:
            indexed_scores = [
                (idx, score) for idx, score in indexed_scores
                if self._matches_filters(self.documents[idx], filters)
            ]

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k with documents
        results = []
        for idx, score in indexed_scores[:k]:
            if score > 0:  # Only include documents with positive scores
                results.append((idx, score, self.documents[idx]))

        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search and return documents with scores

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of document dicts with 'bm25_score' field
        """
        results = self.search(query, k)

        output = []
        for idx, score, doc in results:
            doc_copy = doc.copy()
            doc_copy["bm25_score"] = float(score)
            doc_copy["bm25_rank"] = len(output) + 1
            output.append(doc_copy)

        return output

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is not None:
            return self.documents[idx]
        return None

    def clear(self):
        """Clear the index"""
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        self._doc_id_to_idx = {}

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Split and filter
        tokens = text.split()

        # Remove very short tokens and stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "this",
            "that", "these", "those", "it", "its", "they", "them", "their",
            "we", "our", "you", "your", "he", "she", "him", "her", "his"
        }

        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]

        return tokens

    def _matches_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches all filters"""
        metadata = doc.get("metadata", doc)

        for key, value in filters.items():
            doc_value = metadata.get(key)

            if isinstance(value, list):
                # Check if doc value is in list
                if doc_value not in value:
                    return False
            elif doc_value != value:
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.tokenized_docs:
            return {"total_documents": 0}

        total_tokens = sum(len(doc) for doc in self.tokenized_docs)
        avg_tokens = total_tokens / len(self.tokenized_docs) if self.tokenized_docs else 0

        return {
            "total_documents": len(self.documents),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens
        }


class EnhancedBM25Index(BM25Index):
    """
    Enhanced BM25 with additional features:
    - Term frequency boosting for important fields
    - Phrase matching bonus
    - Numeric value matching
    """

    def __init__(self, field_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.field_weights = field_weights or {
            "chunk_text": 1.0,
            "doc_title": 2.0,
            "sku_ids": 3.0,
            "warehouse_ids": 3.0
        }

    def search_enhanced(
        self,
        query: str,
        k: int = 10,
        boost_exact_match: bool = True,
        boost_phrases: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with boosting

        Args:
            query: Query text
            k: Number of results
            boost_exact_match: Boost documents with exact query matches
            boost_phrases: Boost documents with phrase matches

        Returns:
            List of document dicts with scores
        """
        # Get base BM25 results
        base_results = self.search(query, k * 2)

        if not base_results:
            return []

        query_lower = query.lower()
        query_tokens = set(self._tokenize(query))

        scored_results = []
        for idx, base_score, doc in base_results:
            score = base_score

            text = doc.get("chunk_text", "").lower()
            metadata = doc.get("metadata", {})

            # Boost for exact query match
            if boost_exact_match and query_lower in text:
                score *= 1.5

            # Boost for phrase matches (consecutive tokens)
            if boost_phrases:
                words = query_lower.split()
                for i in range(len(words) - 1):
                    phrase = f"{words[i]} {words[i+1]}"
                    if phrase in text:
                        score *= 1.2

            # Boost for title match
            title = metadata.get("doc_title", "").lower()
            if any(token in title for token in query_tokens):
                score *= 1.3

            # Boost for SKU/warehouse match
            sku_ids = metadata.get("sku_ids", "").lower()
            warehouse_ids = metadata.get("warehouse_ids", "").lower()

            for token in query_tokens:
                if token in sku_ids:
                    score *= 2.0
                if token in warehouse_ids:
                    score *= 2.0

            doc_copy = doc.copy()
            doc_copy["enhanced_bm25_score"] = float(score)
            scored_results.append(doc_copy)

        # Sort by enhanced score
        scored_results.sort(key=lambda x: x["enhanced_bm25_score"], reverse=True)

        return scored_results[:k]
