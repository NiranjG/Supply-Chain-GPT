"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json

from src.api.main import app
from src.models.schemas import QueryResponse, Citation, ForecastResponse


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator"""
    mock = Mock()
    mock.answer.return_value = QueryResponse(
        answer="Test answer based on documents.",
        citations=[
            Citation(
                doc_title="Test Doc",
                source_uri="/test/doc.pdf",
                chunk_text="Sample citation text",
                relevance_score=0.9
            )
        ],
        confidence=0.85,
        forecast_data=None,
        processing_time_ms=100.0,
        warning_badges=[]
    )
    return mock


class TestHealthEndpoints:
    """Tests for health check endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SupplyChainGPT"
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test basic health check"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestQueryEndpoints:
    """Tests for query/ask endpoints"""

    def test_ask_simple_endpoint(self, client):
        """Test simple ask endpoint"""
        with patch('src.api.dependencies.get_orchestrator') as mock_get:
            mock_orch = Mock()
            mock_orch.answer.return_value = QueryResponse(
                answer="Test answer",
                citations=[],
                confidence=0.8,
                processing_time_ms=50.0,
                warning_badges=[]
            )
            mock_get.return_value = mock_orch

            response = client.post(
                "/api/v1/ask/simple",
                params={"query": "What is the safety stock?", "user_id": "test"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data

    def test_ask_endpoint_with_full_request(self, client):
        """Test ask endpoint with full request body"""
        with patch('src.api.dependencies.get_orchestrator') as mock_get:
            mock_orch = Mock()
            mock_orch.answer.return_value = QueryResponse(
                answer="Detailed answer with citations.",
                citations=[
                    Citation(
                        doc_title="Policy",
                        source_uri="/policy.pdf",
                        chunk_text="Excerpt",
                        relevance_score=0.9
                    )
                ],
                confidence=0.9,
                processing_time_ms=150.0,
                warning_badges=[]
            )
            mock_get.return_value = mock_orch

            response = client.post(
                "/api/v1/ask",
                json={
                    "query": "What is the reorder point for SKU-12345?",
                    "user_context": {
                        "user_id": "test_user",
                        "tenant_id": "default",
                        "roles": ["planner"]
                    },
                    "top_k": 5,
                    "include_forecast": False
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "citations" in data
            assert "confidence" in data


class TestDocumentEndpoints:
    """Tests for document management endpoints"""

    def test_get_document_stats(self, client):
        """Test document stats endpoint"""
        with patch('src.api.dependencies.get_vector_store') as mock_get:
            mock_store = Mock()
            mock_store.get_collection_stats.return_value = {
                "collection_name": "supplychain_docs",
                "total_chunks": 100
            }
            mock_get.return_value = mock_store

            response = client.get("/api/v1/documents/stats")

            assert response.status_code == 200
            data = response.json()
            assert "total_chunks" in data

    def test_delete_document(self, client):
        """Test document deletion endpoint"""
        with patch('src.api.dependencies.get_vector_store') as mock_get:
            mock_store = Mock()
            mock_store.delete_document.return_value = 5
            mock_get.return_value = mock_store

            response = client.delete("/api/v1/documents/doc-123")

            assert response.status_code == 200
            data = response.json()
            assert data["doc_id"] == "doc-123"
            assert data["chunks_deleted"] == 5


class TestSearchEndpoints:
    """Tests for search endpoints"""

    def test_search_documents(self, client):
        """Test document search endpoint"""
        with patch('src.api.dependencies.get_retriever') as mock_get:
            mock_retriever = Mock()
            mock_retriever.retrieve_with_intent.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "Test content about inventory",
                    "metadata": {"doc_title": "Test", "doc_type": "report"},
                    "combined_score": 0.85,
                    "dense_score": 0.9,
                    "sparse_score": 0.7
                }
            ]
            mock_get.return_value = mock_retriever

            response = client.post(
                "/api/v1/search",
                params={"query": "inventory levels", "k": 5}
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert data["total_results"] >= 0


class TestForecastEndpoints:
    """Tests for forecast endpoints"""

    def test_get_forecast(self, client):
        """Test forecast endpoint"""
        with patch('src.api.dependencies.get_forecaster') as mock_fc, \
             patch('src.api.dependencies.get_safety_calculator') as mock_sc:

            mock_forecaster = Mock()
            mock_forecaster.forecast.return_value = {
                "sku_id": "SKU-001",
                "warehouse_id": None,
                "forecast": [
                    {"date": "2024-01-01", "predicted_demand": 100},
                    {"date": "2024-01-02", "predicted_demand": 105}
                ],
                "model_used": "xgboost",
                "mape": 8.5
            }
            mock_fc.return_value = mock_forecaster

            mock_calculator = Mock()
            mock_calculator.calculate.return_value = {
                "safety_stock": 200,
                "reorder_point": 500
            }
            mock_sc.return_value = mock_calculator

            response = client.post(
                "/api/v1/forecast",
                json={
                    "sku_id": "SKU-001",
                    "periods": 30,
                    "include_safety_stock": True
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["sku_id"] == "SKU-001"
            assert "forecast" in data

    def test_get_forecast_summary(self, client):
        """Test forecast summary endpoint"""
        with patch('src.api.dependencies.get_forecaster') as mock_get:
            mock_forecaster = Mock()
            mock_forecaster.forecast.return_value = {
                "sku_id": "SKU-001",
                "forecast": [{"date": "2024-01-01", "predicted_demand": 100}],
                "model_used": "xgboost"
            }
            mock_forecaster.forecast_to_text.return_value = "Forecast summary for SKU-001"
            mock_get.return_value = mock_forecaster

            response = client.get("/api/v1/forecast/SKU-001/summary")

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data


class TestFeedbackEndpoints:
    """Tests for feedback endpoints"""

    def test_submit_feedback(self, client):
        """Test feedback submission"""
        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "query-123",
                "helpful": True,
                "comment": "Very helpful response"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["query_id"] == "query-123"

    def test_submit_negative_feedback(self, client):
        """Test negative feedback submission"""
        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "query-456",
                "helpful": False,
                "error_type": "outdated",
                "comment": "Information was outdated"
            }
        )

        assert response.status_code == 200


class TestInputValidation:
    """Tests for input validation"""

    def test_ask_without_query(self, client):
        """Test ask endpoint without query"""
        response = client.post(
            "/api/v1/ask",
            json={
                "user_context": {
                    "user_id": "test"
                }
            }
        )

        assert response.status_code == 422  # Validation error

    def test_forecast_without_sku(self, client):
        """Test forecast endpoint without SKU"""
        response = client.post(
            "/api/v1/forecast",
            json={
                "periods": 30
            }
        )

        assert response.status_code == 422  # Validation error

    def test_search_with_invalid_k(self, client):
        """Test search with invalid k parameter"""
        with patch('src.api.dependencies.get_retriever') as mock_get:
            mock_retriever = Mock()
            mock_retriever.retrieve_with_intent.return_value = []
            mock_get.return_value = mock_retriever

            # k=0 should still work (returns empty)
            response = client.post(
                "/api/v1/search",
                params={"query": "test", "k": 0}
            )

            assert response.status_code == 200
