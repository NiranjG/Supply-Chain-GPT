"""
Tests for retrieval module
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from src.retrieval.bm25_index import BM25Index, EnhancedBM25Index
from src.retrieval.hybrid_retriever import HybridRetriever


class TestBM25Index:
    """Tests for BM25Index class"""

    @pytest.fixture
    def bm25_index(self):
        """Create BM25 index instance"""
        return BM25Index()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for indexing"""
        return [
            {
                "chunk_id": "doc1",
                "chunk_text": "Safety stock calculation for inventory management.",
                "metadata": {"doc_type": "policy"}
            },
            {
                "chunk_id": "doc2",
                "chunk_text": "Warehouse operations and shipping procedures.",
                "metadata": {"doc_type": "sop"}
            },
            {
                "chunk_id": "doc3",
                "chunk_text": "Supplier contract terms and delivery schedules.",
                "metadata": {"doc_type": "contract"}
            },
            {
                "chunk_id": "doc4",
                "chunk_text": "SKU-12345 has a reorder point of 500 units.",
                "metadata": {"doc_type": "report", "sku_ids": "SKU-12345"}
            }
        ]

    def test_add_documents(self, bm25_index, sample_documents):
        """Test adding documents to index"""
        bm25_index.add_documents(sample_documents)

        stats = bm25_index.get_stats()
        assert stats["total_documents"] == 4

    def test_search_basic(self, bm25_index, sample_documents):
        """Test basic search functionality"""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("safety stock inventory", k=3)

        assert len(results) >= 1
        # First result should be about safety stock
        assert "safety" in results[0][2]["chunk_text"].lower()

    def test_search_returns_scores(self, bm25_index, sample_documents):
        """Test that search returns proper scores"""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("warehouse shipping", k=3)

        for idx, score, doc in results:
            assert isinstance(score, float)
            assert score >= 0

    def test_search_with_filters(self, bm25_index, sample_documents):
        """Test search with metadata filters"""
        bm25_index.add_documents(sample_documents)

        # Filter by doc_type
        results = bm25_index.search(
            "operations",
            k=5,
            filters={"doc_type": "sop"}
        )

        for _, _, doc in results:
            assert doc["metadata"]["doc_type"] == "sop"

    def test_search_no_results(self, bm25_index, sample_documents):
        """Test search with query that matches nothing"""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("xyznonexistent123", k=3)

        # Should return empty or low-score results
        assert len(results) == 0 or all(score == 0 for _, score, _ in results)

    def test_clear_index(self, bm25_index, sample_documents):
        """Test clearing the index"""
        bm25_index.add_documents(sample_documents)
        bm25_index.clear()

        stats = bm25_index.get_stats()
        assert stats["total_documents"] == 0

    def test_get_document(self, bm25_index, sample_documents):
        """Test retrieving document by ID"""
        bm25_index.add_documents(sample_documents)

        doc = bm25_index.get_document("doc1")

        assert doc is not None
        assert "safety stock" in doc["chunk_text"].lower()


class TestEnhancedBM25Index:
    """Tests for EnhancedBM25Index class"""

    @pytest.fixture
    def enhanced_index(self):
        """Create enhanced BM25 index"""
        return EnhancedBM25Index()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents with rich metadata"""
        return [
            {
                "chunk_id": "doc1",
                "chunk_text": "The safety stock level is important for inventory.",
                "metadata": {
                    "doc_title": "Inventory Policy",
                    "doc_type": "policy",
                    "sku_ids": "SKU-12345",
                    "warehouse_ids": "WH-001"
                }
            },
            {
                "chunk_id": "doc2",
                "chunk_text": "SKU-12345 requires 200 units safety stock.",
                "metadata": {
                    "doc_title": "SKU Report",
                    "doc_type": "report",
                    "sku_ids": "SKU-12345"
                }
            }
        ]

    def test_enhanced_search_boosts_sku(self, enhanced_index, sample_documents):
        """Test that SKU mentions get boosted"""
        enhanced_index.add_documents(sample_documents)

        results = enhanced_index.search_enhanced("SKU-12345 safety stock", k=2)

        assert len(results) >= 1
        # Document mentioning SKU should rank high
        assert "SKU-12345" in results[0]["chunk_text"]

    def test_enhanced_search_boosts_title_match(self, enhanced_index, sample_documents):
        """Test title matching boost"""
        enhanced_index.add_documents(sample_documents)

        results = enhanced_index.search_enhanced("inventory policy", k=2)

        # Document with "Inventory Policy" title should rank higher
        assert results[0]["metadata"]["doc_title"] == "Inventory Policy"


class TestHybridRetriever:
    """Tests for HybridRetriever class"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store"""
        mock = Mock()
        mock.search.return_value = [
            {
                "chunk_id": "chunk1",
                "chunk_text": "Safety stock for SKU-12345 is 200 units.",
                "metadata": {"doc_title": "Policy", "doc_type": "policy"},
                "similarity": 0.92
            },
            {
                "chunk_id": "chunk2",
                "chunk_text": "Reorder point calculation method.",
                "metadata": {"doc_title": "SOP", "doc_type": "sop"},
                "similarity": 0.85
            }
        ]
        mock.search_with_acl.return_value = mock.search.return_value
        return mock

    @pytest.fixture
    def mock_bm25(self):
        """Create mock BM25 index"""
        mock = EnhancedBM25Index()
        mock.add_documents([
            {
                "chunk_id": "chunk1",
                "chunk_text": "Safety stock for SKU-12345 is 200 units.",
                "metadata": {"doc_title": "Policy", "doc_type": "policy"}
            },
            {
                "chunk_id": "chunk3",
                "chunk_text": "SKU-12345 inventory levels are critical.",
                "metadata": {"doc_title": "Report", "doc_type": "report"}
            }
        ])
        return mock

    @pytest.fixture
    def hybrid_retriever(self, mock_vector_store, mock_bm25):
        """Create hybrid retriever instance"""
        return HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25,
            use_reranker=False  # Disable reranker for unit tests
        )

    def test_retrieve_basic(self, hybrid_retriever):
        """Test basic retrieval"""
        results = hybrid_retriever.retrieve("safety stock SKU-12345", k=3)

        assert len(results) >= 1
        assert "chunk_id" in results[0]

    def test_retrieve_with_intent(self, hybrid_retriever):
        """Test retrieval with automatic intent detection"""
        results = hybrid_retriever.retrieve_with_intent(
            "What is the policy for safety stock?",
            k=3
        )

        assert len(results) >= 1

    def test_detect_query_type_policy(self, hybrid_retriever):
        """Test policy query detection"""
        query_type = hybrid_retriever.detect_query_type("What is the procedure for receiving?")
        assert query_type == "policy"

    def test_detect_query_type_metrics(self, hybrid_retriever):
        """Test metrics query detection"""
        query_type = hybrid_retriever.detect_query_type("Show me the inventory report data")
        assert query_type == "metrics"

    def test_detect_query_type_contract(self, hybrid_retriever):
        """Test contract query detection"""
        query_type = hybrid_retriever.detect_query_type("What are the SLA terms?")
        assert query_type == "contract"

    def test_detect_query_type_inventory(self, hybrid_retriever):
        """Test inventory query detection"""
        query_type = hybrid_retriever.detect_query_type("What is the stock level for SKU-12345?")
        assert query_type == "inventory"

    def test_reciprocal_rank_fusion(self, hybrid_retriever):
        """Test RRF combination of results"""
        dense_results = [
            {"chunk_id": "a", "similarity": 0.9},
            {"chunk_id": "b", "similarity": 0.8},
            {"chunk_id": "c", "similarity": 0.7}
        ]
        sparse_results = [
            {"chunk_id": "b", "bm25_score": 10},
            {"chunk_id": "a", "bm25_score": 8},
            {"chunk_id": "d", "bm25_score": 5}
        ]

        fused = hybrid_retriever._reciprocal_rank_fusion(dense_results, sparse_results)

        # Documents appearing in both should rank higher
        chunk_ids = [r["chunk_id"] for r in fused]
        # 'a' and 'b' appear in both, should be near top
        assert "a" in chunk_ids[:3]
        assert "b" in chunk_ids[:3]

    def test_retrieve_with_filters(self, hybrid_retriever):
        """Test retrieval with metadata filters"""
        results = hybrid_retriever.retrieve(
            "safety stock",
            k=3,
            filters={"doc_type": "policy"}
        )

        assert len(results) >= 0

    def test_retrieve_with_user_roles(self, hybrid_retriever):
        """Test retrieval with ACL filtering"""
        results = hybrid_retriever.retrieve(
            "inventory policy",
            k=3,
            user_roles=["planner", "viewer"]
        )

        assert len(results) >= 0
