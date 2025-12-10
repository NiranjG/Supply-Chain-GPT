"""
FastAPI dependencies and service initialization
"""

import logging
from functools import lru_cache
from typing import Optional

from ..config import settings
from ..embeddings.vector_store import VectorStore
from ..embeddings.embedding_service import EmbeddingService, ReRanker
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.bm25_index import EnhancedBM25Index
from ..ingestion.pipeline import IngestionPipeline
from ..forecasting.demand_forecaster import DemandForecaster
from ..forecasting.safety_stock import SafetyStockCalculator
from ..llm.orchestrator import LLMOrchestrator, MockLLMOrchestrator

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Container for all services"""

    _instance: Optional["ServiceContainer"] = None

    def __init__(self):
        self._embedding_service: Optional[EmbeddingService] = None
        self._vector_store: Optional[VectorStore] = None
        self._bm25_index: Optional[EnhancedBM25Index] = None
        self._reranker: Optional[ReRanker] = None
        self._retriever: Optional[HybridRetriever] = None
        self._ingestion_pipeline: Optional[IngestionPipeline] = None
        self._forecaster: Optional[DemandForecaster] = None
        self._safety_calculator: Optional[SafetyStockCalculator] = None
        self._orchestrator: Optional[LLMOrchestrator] = None

    @classmethod
    def get_instance(cls) -> "ServiceContainer":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                model_name=settings.embedding_model
            )
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            settings.ensure_directories()
            self._vector_store = VectorStore(
                persist_directory=str(settings.chroma_persist_dir),
                collection_name="supplychain_docs",
                embedding_service=self.embedding_service
            )
        return self._vector_store

    @property
    def bm25_index(self) -> EnhancedBM25Index:
        if self._bm25_index is None:
            self._bm25_index = EnhancedBM25Index()
        return self._bm25_index

    @property
    def reranker(self) -> ReRanker:
        if self._reranker is None:
            self._reranker = ReRanker()
        return self._reranker

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                reranker=self.reranker,
                use_reranker=True
            )
        return self._retriever

    @property
    def ingestion_pipeline(self) -> IngestionPipeline:
        if self._ingestion_pipeline is None:
            self._ingestion_pipeline = IngestionPipeline()
        return self._ingestion_pipeline

    @property
    def forecaster(self) -> DemandForecaster:
        if self._forecaster is None:
            self._forecaster = DemandForecaster()
        return self._forecaster

    @property
    def safety_calculator(self) -> SafetyStockCalculator:
        if self._safety_calculator is None:
            self._safety_calculator = SafetyStockCalculator()
        return self._safety_calculator

    @property
    def orchestrator(self) -> LLMOrchestrator:
        if self._orchestrator is None:
            if settings.openai_api_key:
                self._orchestrator = LLMOrchestrator(
                    retriever=self.retriever,
                    forecaster=self.forecaster,
                    safety_calculator=self.safety_calculator,
                    llm_model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    openai_api_key=settings.openai_api_key
                )
            else:
                logger.warning("No OpenAI API key found, using mock orchestrator")
                self._orchestrator = MockLLMOrchestrator(
                    retriever=self.retriever,
                    forecaster=self.forecaster,
                    safety_calculator=self.safety_calculator
                )
        return self._orchestrator


@lru_cache()
def get_services() -> ServiceContainer:
    """Get the service container (cached)"""
    return ServiceContainer.get_instance()


def get_vector_store() -> VectorStore:
    """Dependency for vector store"""
    return get_services().vector_store


def get_retriever() -> HybridRetriever:
    """Dependency for retriever"""
    return get_services().retriever


def get_ingestion_pipeline() -> IngestionPipeline:
    """Dependency for ingestion pipeline"""
    return get_services().ingestion_pipeline


def get_orchestrator() -> LLMOrchestrator:
    """Dependency for LLM orchestrator"""
    return get_services().orchestrator


def get_forecaster() -> DemandForecaster:
    """Dependency for forecaster"""
    return get_services().forecaster


def get_safety_calculator() -> SafetyStockCalculator:
    """Dependency for safety stock calculator"""
    return get_services().safety_calculator
