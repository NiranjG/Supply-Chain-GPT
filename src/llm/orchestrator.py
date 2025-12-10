"""
LLM Orchestrator for SupplyChainGPT
"""

import logging
import time
from typing import Dict, Any, Optional, List

from ..models.schemas import (
    QueryRequest, QueryResponse, Citation, UserContext
)
from ..retrieval.hybrid_retriever import HybridRetriever
from ..forecasting.demand_forecaster import DemandForecaster
from ..forecasting.safety_stock import SafetyStockCalculator
from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Orchestrates the RAG pipeline:
    1. Query understanding
    2. Retrieval
    3. Context assembly
    4. LLM synthesis
    5. Citation extraction
    6. Guardrails
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        forecaster: Optional[DemandForecaster] = None,
        safety_calculator: Optional[SafetyStockCalculator] = None,
        llm_model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize orchestrator

        Args:
            retriever: Hybrid retriever for document search
            forecaster: Demand forecaster (optional)
            safety_calculator: Safety stock calculator (optional)
            llm_model: LLM model to use
            temperature: LLM temperature
            openai_api_key: OpenAI API key
        """
        self.retriever = retriever
        self.forecaster = forecaster or DemandForecaster()
        self.safety_calculator = safety_calculator or SafetyStockCalculator()
        self.llm_model = llm_model
        self.temperature = temperature
        self._openai_client = None
        self._openai_api_key = openai_api_key

    @property
    def openai_client(self):
        """Lazy load OpenAI client"""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self._openai_api_key)
        return self._openai_client

    def answer(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query and generate an answer

        Args:
            request: Query request with user context

        Returns:
            QueryResponse with answer, citations, and metadata
        """
        start_time = time.time()
        warning_badges = []

        try:
            # 1. Retrieve relevant chunks
            chunks = self._retrieve(request)

            if not chunks:
                warning_badges.append("Low evidence")

            # 2. Get forecast if requested
            forecast_result = None
            safety_result = None

            if request.include_forecast and request.sku_id:
                forecast_result, safety_result = self._get_forecast_data(
                    request.sku_id,
                    request.warehouse_id
                )

            # 3. Build prompt
            prompt = PromptTemplates.build_query_prompt(
                question=request.query,
                chunks=chunks,
                forecast_result=forecast_result,
                safety_stock_result=safety_result
            )

            # 4. Generate answer
            answer = self._generate_answer(prompt)

            # 5. Extract citations
            citations = self._extract_citations(answer, chunks)

            # 6. Apply guardrails
            answer, guardrail_warnings = self._apply_guardrails(answer, chunks)
            warning_badges.extend(guardrail_warnings)

            # 7. Calculate confidence
            confidence = self._calculate_confidence(chunks, citations)

            processing_time = (time.time() - start_time) * 1000

            return QueryResponse(
                answer=answer,
                citations=citations,
                confidence=confidence,
                forecast_data=forecast_result,
                processing_time_ms=processing_time,
                warning_badges=warning_badges
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = (time.time() - start_time) * 1000

            return QueryResponse(
                answer=f"I encountered an error processing your request: {str(e)}",
                citations=[],
                confidence=0.0,
                processing_time_ms=processing_time,
                warning_badges=["Error occurred"]
            )

    def _retrieve(self, request: QueryRequest) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks"""
        return self.retriever.retrieve_with_intent(
            query=request.query,
            k=request.top_k,
            filters=request.filters,
            user_roles=request.user_context.roles
        )

    def _get_forecast_data(
        self,
        sku_id: str,
        warehouse_id: Optional[str]
    ) -> tuple:
        """Get forecast and safety stock data"""
        import pandas as pd
        import numpy as np

        # Generate sample historical data (in production, this would come from database)
        # This is a placeholder - replace with actual data retrieval
        dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq='D')
        np.random.seed(hash(sku_id) % 2**32)
        base_demand = np.random.randint(50, 200)
        demand = base_demand + np.random.randn(90) * (base_demand * 0.2)
        demand = np.maximum(demand, 0)

        historical_data = pd.DataFrame({
            'ds': dates,
            'y': demand
        })

        # Get forecast
        forecast_result = self.forecaster.forecast(
            historical_data=historical_data,
            periods=30,
            sku_id=sku_id,
            warehouse_id=warehouse_id
        )

        # Calculate safety stock
        safety_result = self.safety_calculator.calculate(
            demand_data=historical_data,
            lead_time_days=7,
            service_level=0.95
        )

        return forecast_result, safety_result

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to a basic response
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Generate fallback response without LLM"""
        # Extract context from prompt
        if "CONTEXT:" in prompt:
            context_start = prompt.find("CONTEXT:") + 8
            context_end = prompt.find("USER QUESTION:")
            context = prompt[context_start:context_end].strip()

            if context:
                return f"Based on the available documents:\n\n{context[:500]}...\n\nPlease configure your OpenAI API key for more detailed responses."

        return "Unable to generate response. Please check your API configuration."

    def _extract_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Citation]:
        """Extract and format citations"""
        raw_citations = PromptTemplates.extract_citations(answer, chunks)

        citations = []
        for c in raw_citations:
            citations.append(Citation(
                doc_title=c["doc_title"],
                source_uri=c["source_uri"],
                chunk_text=c["chunk_text"],
                relevance_score=c["relevance_score"]
            ))

        # If no explicit citations found, include top chunks as implicit sources
        if not citations and chunks:
            for chunk in chunks[:3]:
                metadata = chunk.get("metadata", {})
                citations.append(Citation(
                    doc_title=metadata.get("doc_title", "Unknown"),
                    source_uri=metadata.get("source_uri", ""),
                    chunk_text=chunk.get("chunk_text", "")[:200],
                    relevance_score=chunk.get("similarity", chunk.get("combined_score", 0))
                ))

        return citations

    def _apply_guardrails(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> tuple:
        """Apply guardrails to the response"""
        warnings = []

        # Check for potential hallucinations (claims not in context)
        # This is a simplified check - production would use more sophisticated methods

        # Check if answer mentions specific numbers
        import re
        numbers_in_answer = re.findall(r'\b\d+(?:\.\d+)?\b', answer)

        # Combine all chunk text
        all_context = " ".join(c.get("chunk_text", "") for c in chunks)
        numbers_in_context = set(re.findall(r'\b\d+(?:\.\d+)?\b', all_context))

        # Flag numbers not in context (potential hallucination)
        suspicious_numbers = [n for n in numbers_in_answer if n not in numbers_in_context]
        if len(suspicious_numbers) > 3:
            warnings.append("Unverified numbers")

        # Check for uncertainty indicators
        uncertainty_phrases = [
            "i don't have", "not enough information", "cannot find",
            "no data", "unable to", "not available"
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            warnings.append("Incomplete data")

        # Check for outdated document warnings
        # (would check document dates in production)

        return answer, warnings

    def _calculate_confidence(
        self,
        chunks: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> float:
        """Calculate confidence score for the response"""
        if not chunks:
            return 0.0

        # Factors affecting confidence:
        # 1. Number of relevant chunks
        # 2. Average similarity score
        # 3. Number of citations

        chunk_count_score = min(len(chunks) / 5, 1.0) * 0.3

        avg_similarity = sum(
            c.get("similarity", c.get("combined_score", 0)) for c in chunks
        ) / len(chunks) if chunks else 0
        similarity_score = avg_similarity * 0.5

        citation_score = min(len(citations) / 3, 1.0) * 0.2

        confidence = chunk_count_score + similarity_score + citation_score

        return round(min(confidence, 1.0), 2)

    def answer_simple(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """Simplified answer method for quick queries"""
        request = QueryRequest(
            query=query,
            user_context=UserContext(user_id=user_id),
            top_k=5
        )
        response = self.answer(request)

        return {
            "answer": response.answer,
            "citations": [c.dict() for c in response.citations],
            "confidence": response.confidence,
            "processing_time_ms": response.processing_time_ms
        }


class MockLLMOrchestrator(LLMOrchestrator):
    """Mock orchestrator for testing without LLM API"""

    def _generate_answer(self, prompt: str) -> str:
        """Generate mock answer for testing"""
        # Extract question from prompt
        if "USER QUESTION:" in prompt:
            question_start = prompt.find("USER QUESTION:") + 14
            question_end = prompt.find("\n", question_start)
            question = prompt[question_start:question_end].strip()
        else:
            question = "Unknown question"

        # Extract first source title
        source_title = "Internal Document"
        if "[SOURCE 1]" in prompt:
            doc_start = prompt.find("Document:", prompt.find("[SOURCE 1]")) + 9
            doc_end = prompt.find("\n", doc_start)
            source_title = prompt[doc_start:doc_end].strip()

        return f"""Based on the available documentation, here's the answer to your question:

The information retrieved from our knowledge base indicates relevant details about "{question}".

[Source: {source_title}]

**Key Points:**
- Information has been retrieved from internal documents
- This is a mock response for testing purposes
- In production, this would be generated by the LLM

**Recommendations:**
1. Review the source documents for complete details
2. Contact the relevant department for clarification if needed

Please note: This response was generated using retrieved context from your organization's documents."""
