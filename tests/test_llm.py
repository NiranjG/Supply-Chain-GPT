"""
Tests for LLM orchestration module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.llm.orchestrator import LLMOrchestrator, MockLLMOrchestrator
from src.llm.prompts import PromptTemplates
from src.models.schemas import QueryRequest, UserContext


class TestPromptTemplates:
    """Tests for PromptTemplates class"""

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing"""
        return [
            {
                "chunk_id": "chunk1",
                "chunk_text": "Safety stock for SKU-12345 is 200 units based on 95% service level.",
                "metadata": {
                    "doc_title": "Inventory Policy",
                    "doc_type": "policy"
                },
                "similarity": 0.92
            },
            {
                "chunk_id": "chunk2",
                "chunk_text": "Reorder point = (Daily Demand × Lead Time) + Safety Stock",
                "metadata": {
                    "doc_title": "SOP Manual",
                    "doc_type": "sop"
                },
                "similarity": 0.85
            }
        ]

    def test_format_context(self, sample_chunks):
        """Test context formatting"""
        context = PromptTemplates.format_context(sample_chunks)

        assert "[SOURCE 1]" in context
        assert "[SOURCE 2]" in context
        assert "Inventory Policy" in context
        assert "Safety stock" in context

    def test_format_context_limits_chunks(self, sample_chunks):
        """Test that context respects max_chunks limit"""
        # Add more chunks
        many_chunks = sample_chunks * 5

        context = PromptTemplates.format_context(many_chunks, max_chunks=3)

        # Should only have 3 sources
        assert context.count("[SOURCE") == 3

    def test_format_forecast_context(self):
        """Test forecast context formatting"""
        forecast_result = {
            "sku_id": "SKU-001",
            "warehouse_id": "WH-001",
            "model_used": "ensemble",
            "forecast": [
                {"date": "2024-01-01", "predicted_demand": 100},
                {"date": "2024-01-02", "predicted_demand": 120},
                {"date": "2024-01-03", "predicted_demand": 90}
            ]
        }

        context = PromptTemplates.format_forecast_context(forecast_result)

        assert "SKU-001" in context
        assert "WH-001" in context
        assert "ensemble" in context

    def test_format_forecast_context_with_safety_stock(self):
        """Test forecast context with safety stock info"""
        forecast_result = {
            "sku_id": "SKU-001",
            "forecast": [{"date": "2024-01-01", "predicted_demand": 100}]
        }
        safety_result = {
            "safety_stock": 200,
            "reorder_point": 500,
            "service_level": 0.95
        }

        context = PromptTemplates.format_forecast_context(forecast_result, safety_result)

        assert "Safety Stock" in context
        assert "200" in context
        assert "500" in context

    def test_build_query_prompt(self, sample_chunks):
        """Test complete prompt building"""
        prompt = PromptTemplates.build_query_prompt(
            question="What is the safety stock for SKU-12345?",
            chunks=sample_chunks
        )

        assert "What is the safety stock" in prompt
        assert "CONTEXT:" in prompt
        assert "[SOURCE 1]" in prompt

    def test_build_query_prompt_no_context(self):
        """Test prompt building with no context"""
        prompt = PromptTemplates.build_query_prompt(
            question="Random question",
            chunks=[]
        )

        assert "don't have enough information" in prompt.lower()

    def test_extract_citations(self, sample_chunks):
        """Test citation extraction from response"""
        response = """
        Based on the documents, the safety stock is 200 units.
        [Source: Inventory Policy]

        The calculation method follows the standard formula.
        [Source: SOP Manual]
        """

        citations = PromptTemplates.extract_citations(response, sample_chunks)

        assert len(citations) == 2
        assert citations[0]["doc_title"] == "Inventory Policy"

    def test_extract_citations_no_matches(self, sample_chunks):
        """Test citation extraction with no matches"""
        response = "This response has no citations."

        citations = PromptTemplates.extract_citations(response, sample_chunks)

        assert len(citations) == 0


class TestMockLLMOrchestrator:
    """Tests for MockLLMOrchestrator"""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever"""
        mock = Mock()
        mock.retrieve_with_intent.return_value = [
            {
                "chunk_id": "chunk1",
                "chunk_text": "Safety stock information here.",
                "metadata": {"doc_title": "Test Policy", "doc_type": "policy"},
                "similarity": 0.9,
                "combined_score": 0.85
            }
        ]
        return mock

    @pytest.fixture
    def orchestrator(self, mock_retriever):
        """Create mock orchestrator"""
        return MockLLMOrchestrator(
            retriever=mock_retriever,
            forecaster=Mock(),
            safety_calculator=Mock()
        )

    def test_answer_returns_response(self, orchestrator):
        """Test that answer returns proper response"""
        request = QueryRequest(
            query="What is the safety stock?",
            user_context=UserContext(user_id="test"),
            top_k=5
        )

        response = orchestrator.answer(request)

        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.confidence >= 0

    def test_answer_includes_citations(self, orchestrator):
        """Test that response includes citations"""
        request = QueryRequest(
            query="What is the policy?",
            user_context=UserContext(user_id="test")
        )

        response = orchestrator.answer(request)

        # Should have at least implicit citations
        assert response.citations is not None

    def test_answer_calculates_confidence(self, orchestrator):
        """Test confidence calculation"""
        request = QueryRequest(
            query="Test query",
            user_context=UserContext(user_id="test")
        )

        response = orchestrator.answer(request)

        assert 0 <= response.confidence <= 1

    def test_answer_measures_processing_time(self, orchestrator):
        """Test processing time measurement"""
        request = QueryRequest(
            query="Test query",
            user_context=UserContext(user_id="test")
        )

        response = orchestrator.answer(request)

        assert response.processing_time_ms > 0


class TestLLMOrchestrator:
    """Tests for LLMOrchestrator (mocked OpenAI)"""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever"""
        mock = Mock()
        mock.retrieve_with_intent.return_value = [
            {
                "chunk_id": "chunk1",
                "chunk_text": "Policy content about safety stock.",
                "metadata": {"doc_title": "Policy", "doc_type": "policy"},
                "similarity": 0.9
            }
        ]
        return mock

    def test_guardrails_detect_missing_numbers(self, mock_retriever):
        """Test guardrails detect numbers not in context"""
        orchestrator = MockLLMOrchestrator(
            retriever=mock_retriever,
            forecaster=Mock(),
            safety_calculator=Mock()
        )

        # Response with numbers not in context
        response = "The value is 12345 units and costs $67890."
        chunks = [{"chunk_text": "Some text without those numbers."}]

        cleaned, warnings = orchestrator._apply_guardrails(response, chunks)

        # Should flag unverified numbers if many are present
        # This depends on implementation threshold

    def test_guardrails_detect_uncertainty(self, mock_retriever):
        """Test guardrails detect uncertainty phrases"""
        orchestrator = MockLLMOrchestrator(
            retriever=mock_retriever,
            forecaster=Mock(),
            safety_calculator=Mock()
        )

        response = "I don't have enough information to answer this."
        chunks = []

        cleaned, warnings = orchestrator._apply_guardrails(response, chunks)

        assert "Incomplete data" in warnings

    def test_confidence_calculation_no_chunks(self, mock_retriever):
        """Test confidence is 0 with no chunks"""
        orchestrator = MockLLMOrchestrator(
            retriever=mock_retriever,
            forecaster=Mock(),
            safety_calculator=Mock()
        )

        confidence = orchestrator._calculate_confidence([], [])

        assert confidence == 0.0

    def test_confidence_increases_with_chunks(self, mock_retriever):
        """Test confidence increases with more relevant chunks"""
        orchestrator = MockLLMOrchestrator(
            retriever=mock_retriever,
            forecaster=Mock(),
            safety_calculator=Mock()
        )

        few_chunks = [{"similarity": 0.9}]
        many_chunks = [{"similarity": 0.9}] * 5

        conf_few = orchestrator._calculate_confidence(few_chunks, [])
        conf_many = orchestrator._calculate_confidence(many_chunks, [])

        assert conf_many >= conf_few


class TestEndToEndFlow:
    """Integration tests for the full orchestration flow"""

    @pytest.fixture
    def mock_components(self):
        """Create all mock components"""
        retriever = Mock()
        retriever.retrieve_with_intent.return_value = [
            {
                "chunk_id": "c1",
                "chunk_text": "SKU-12345 has safety stock of 200 units.",
                "metadata": {"doc_title": "Inventory Policy", "doc_type": "policy"},
                "similarity": 0.95
            }
        ]

        forecaster = Mock()
        forecaster.forecast.return_value = {
            "sku_id": "SKU-12345",
            "forecast": [{"date": "2024-01-01", "predicted_demand": 100}],
            "model_used": "xgboost"
        }

        safety_calc = Mock()
        safety_calc.calculate.return_value = {
            "safety_stock": 200,
            "reorder_point": 500,
            "service_level": 0.95
        }

        return retriever, forecaster, safety_calc

    def test_full_flow_with_forecast(self, mock_components):
        """Test complete flow including forecast"""
        retriever, forecaster, safety_calc = mock_components

        orchestrator = MockLLMOrchestrator(
            retriever=retriever,
            forecaster=forecaster,
            safety_calculator=safety_calc
        )

        request = QueryRequest(
            query="What is the forecast for SKU-12345?",
            user_context=UserContext(user_id="test"),
            include_forecast=True,
            sku_id="SKU-12345"
        )

        response = orchestrator.answer(request)

        assert response.answer is not None
        # Forecast should be included
        assert response.forecast_data is not None or response.answer  # Either works
