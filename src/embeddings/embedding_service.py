"""
Embedding service for generating text embeddings
"""

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service

        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension"""
        if self._dimension is None:
            _ = self.model  # Load model to get dimension
        return self._dimension

    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed
            normalize: Whether to normalize the embedding

        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (same as embed_text but semantic distinction)
        Some models have different encoding for queries vs documents
        """
        return self.embed_text(query, normalize=True)

    def compute_similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def compute_similarities(
        self,
        query_embedding: Union[List[float], np.ndarray],
        embeddings: List[Union[List[float], np.ndarray]]
    ) -> List[float]:
        """
        Compute cosine similarities between a query and multiple embeddings

        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to compare against

        Returns:
            List of similarity scores
        """
        query_vec = np.array(query_embedding)
        emb_matrix = np.array(embeddings)

        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)

        # Compute all similarities at once
        similarities = np.dot(emb_norms, query_norm)

        return similarities.tolist()


class ReRanker:
    """
    Cross-encoder re-ranker for improving retrieval precision
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the re-ranker

        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading re-ranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Re-ranker model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Re-rank documents based on relevance to query

        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Create (index, score) pairs and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    def rerank_with_docs(
        self,
        query: str,
        documents: List[dict],
        text_key: str = "chunk_text",
        top_k: Optional[int] = None
    ) -> List[dict]:
        """
        Re-rank document dicts and return them with scores

        Args:
            query: Query text
            documents: List of document dicts
            text_key: Key for text content in docs
            top_k: Number of top results to return

        Returns:
            List of document dicts with added 'rerank_score'
        """
        if not documents:
            return []

        texts = [doc.get(text_key, "") for doc in documents]
        ranked = self.rerank(query, texts, top_k)

        result = []
        for idx, score in ranked:
            doc = documents[idx].copy()
            doc["rerank_score"] = float(score)
            result.append(doc)

        return result
