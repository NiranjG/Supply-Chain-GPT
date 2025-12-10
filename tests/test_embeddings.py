"""
Tests for embedding and vector store modules
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.embeddings.embedding_service import EmbeddingService, ReRanker
from src.embeddings.vector_store import VectorStore
from src.models.schemas import Chunk, ChunkMetadata, DocumentType, ACL


class TestEmbeddingService:
    """Tests for EmbeddingService class"""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance"""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")

    def test_embed_single_text(self, embedding_service):
        """Test embedding a single text"""
        text = "This is a test sentence for embedding."
        embedding = embedding_service.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_multiple_texts(self, embedding_service):
        """Test embedding multiple texts"""
        texts = [
            "First sentence about inventory.",
            "Second sentence about supply chain.",
            "Third sentence about warehouses."
        ]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)

    def test_embedding_dimension(self, embedding_service):
        """Test that embedding dimension is correct"""
        # all-MiniLM-L6-v2 has 384 dimensions
        text = "Test text"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_compute_similarity(self, embedding_service):
        """Test cosine similarity computation"""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]

        # Identical vectors should have similarity 1
        sim_same = embedding_service.compute_similarity(emb1, emb2)
        assert sim_same == pytest.approx(1.0, rel=1e-5)

        # Orthogonal vectors should have similarity 0
        sim_ortho = embedding_service.compute_similarity(emb1, emb3)
        assert sim_ortho == pytest.approx(0.0, rel=1e-5)

    def test_similar_texts_have_high_similarity(self, embedding_service):
        """Test that similar texts have higher similarity"""
        text1 = "The warehouse inventory is running low."
        text2 = "Warehouse stock levels are depleted."
        text3 = "I like to eat pizza on Fridays."

        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        emb3 = embedding_service.embed_text(text3)

        sim_similar = embedding_service.compute_similarity(emb1, emb2)
        sim_different = embedding_service.compute_similarity(emb1, emb3)

        assert sim_similar > sim_different

    def test_empty_text_handling(self, embedding_service):
        """Test handling of empty text"""
        embedding = embedding_service.embed_text("")
        assert isinstance(embedding, list)

    def test_batch_embedding_empty_list(self, embedding_service):
        """Test embedding empty list"""
        embeddings = embedding_service.embed_texts([])
        assert embeddings == []


class TestVectorStore:
    """Tests for VectorStore class"""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create temporary directory for vector store"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def vector_store(self, temp_persist_dir):
        """Create vector store instance"""
        return VectorStore(
            persist_directory=temp_persist_dir,
            collection_name="test_collection"
        )

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk"""
        return Chunk(
            chunk_id="test-chunk-001",
            chunk_text="This is a test chunk about inventory management and safety stock.",
            metadata=ChunkMetadata(
                doc_id="doc-001",
                source_uri="/test/doc.pdf",
                doc_title="Test Document",
                doc_type=DocumentType.POLICY,
                chunk_index=0,
                chunk_tokens=50,
                acl=ACL(roles=["viewer"])
            )
        )

    def test_add_single_chunk(self, vector_store, sample_chunk):
        """Test adding a single chunk"""
        count = vector_store.add_chunks([sample_chunk])

        assert count == 1
        stats = vector_store.get_collection_stats()
        assert stats["total_chunks"] == 1

    def test_add_multiple_chunks(self, vector_store):
        """Test adding multiple chunks"""
        chunks = []
        for i in range(5):
            chunk = Chunk(
                chunk_id=f"chunk-{i}",
                chunk_text=f"Test content number {i} about supply chain.",
                metadata=ChunkMetadata(
                    doc_id="doc-001",
                    source_uri="/test/doc.pdf",
                    doc_title="Test Doc",
                    doc_type=DocumentType.REPORT
                )
            )
            chunks.append(chunk)

        count = vector_store.add_chunks(chunks)

        assert count == 5
        stats = vector_store.get_collection_stats()
        assert stats["total_chunks"] == 5

    def test_search_returns_results(self, vector_store, sample_chunk):
        """Test that search returns relevant results"""
        vector_store.add_chunks([sample_chunk])

        results = vector_store.search("inventory management", k=5)

        assert len(results) >= 1
        assert results[0]["chunk_id"] == "test-chunk-001"
        assert "similarity" in results[0]

    def test_search_with_filters(self, vector_store):
        """Test search with metadata filters"""
        # Add chunks with different doc types
        chunks = [
            Chunk(
                chunk_id="policy-chunk",
                chunk_text="Policy about inventory management.",
                metadata=ChunkMetadata(
                    doc_id="doc-001",
                    source_uri="/policy.pdf",
                    doc_title="Policy",
                    doc_type=DocumentType.POLICY
                )
            ),
            Chunk(
                chunk_id="report-chunk",
                chunk_text="Report about inventory metrics.",
                metadata=ChunkMetadata(
                    doc_id="doc-002",
                    source_uri="/report.pdf",
                    doc_title="Report",
                    doc_type=DocumentType.REPORT
                )
            )
        ]
        vector_store.add_chunks(chunks)

        # Search with filter
        results = vector_store.search(
            "inventory",
            k=5,
            filters={"doc_type": "policy"}
        )

        # Should return only policy document
        assert len(results) >= 1

    def test_get_chunk_by_id(self, vector_store, sample_chunk):
        """Test retrieving chunk by ID"""
        vector_store.add_chunks([sample_chunk])

        chunk = vector_store.get_chunk("test-chunk-001")

        assert chunk is not None
        assert chunk["chunk_id"] == "test-chunk-001"
        assert "inventory management" in chunk["chunk_text"]

    def test_delete_document(self, vector_store):
        """Test deleting all chunks for a document"""
        chunks = [
            Chunk(
                chunk_id=f"doc1-chunk-{i}",
                chunk_text=f"Chunk {i} of document 1",
                metadata=ChunkMetadata(
                    doc_id="doc-001",
                    source_uri="/doc1.pdf",
                    doc_title="Doc 1"
                )
            )
            for i in range(3)
        ]
        vector_store.add_chunks(chunks)

        # Delete document
        deleted = vector_store.delete_document("doc-001")

        assert deleted == 3
        stats = vector_store.get_collection_stats()
        assert stats["total_chunks"] == 0

    def test_clear_collection(self, vector_store, sample_chunk):
        """Test clearing the entire collection"""
        vector_store.add_chunks([sample_chunk])
        vector_store.clear()

        stats = vector_store.get_collection_stats()
        assert stats["total_chunks"] == 0


class TestReRanker:
    """Tests for ReRanker class"""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance"""
        return ReRanker()

    def test_rerank_documents(self, reranker):
        """Test reranking documents"""
        query = "What is the safety stock for SKU-12345?"
        documents = [
            "The weather is nice today.",
            "Safety stock for SKU-12345 is 200 units.",
            "Paris is the capital of France.",
            "Reorder point calculation requires lead time data."
        ]

        ranked = reranker.rerank(query, documents, top_k=2)

        assert len(ranked) == 2
        # Most relevant document should be first
        assert ranked[0][0] == 1  # Index of safety stock document

    def test_rerank_with_docs(self, reranker):
        """Test reranking with document dicts"""
        query = "inventory policy"
        documents = [
            {"chunk_text": "Random unrelated content", "id": 1},
            {"chunk_text": "Inventory management policy details", "id": 2},
            {"chunk_text": "Weather forecast for tomorrow", "id": 3}
        ]

        ranked = reranker.rerank_with_docs(query, documents)

        assert len(ranked) == 3
        assert "rerank_score" in ranked[0]
        # Policy document should rank higher
        assert ranked[0]["id"] == 2

    def test_rerank_empty_list(self, reranker):
        """Test reranking empty document list"""
        ranked = reranker.rerank("test query", [])
        assert ranked == []
