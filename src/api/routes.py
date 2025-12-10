"""
API routes for SupplyChainGPT
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ..models.schemas import (
    QueryRequest, QueryResponse, UserContext,
    ForecastRequest, ForecastResponse,
    FeedbackRequest, DocumentType, IngestionStatus
)
from .dependencies import (
    get_orchestrator, get_vector_store, get_ingestion_pipeline,
    get_forecaster, get_safety_calculator, get_retriever
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Query Endpoints ==============

@router.post("/ask", response_model=QueryResponse, tags=["Query"])
async def ask_question(
    request: QueryRequest,
    orchestrator=Depends(get_orchestrator)
):
    """
    Ask a question and get an AI-powered answer with citations

    - Retrieves relevant documents using hybrid search
    - Generates answer using LLM with context
    - Returns citations and confidence score
    """
    try:
        response = orchestrator.answer(request)
        return response
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/simple", tags=["Query"])
async def ask_simple(
    query: str,
    user_id: str = "default",
    include_forecast: bool = False,
    sku_id: Optional[str] = None,
    orchestrator=Depends(get_orchestrator)
):
    """
    Simplified question endpoint for quick queries
    """
    request = QueryRequest(
        query=query,
        user_context=UserContext(user_id=user_id),
        include_forecast=include_forecast,
        sku_id=sku_id
    )
    response = orchestrator.answer(request)
    return {
        "answer": response.answer,
        "citations": [c.dict() for c in response.citations],
        "confidence": response.confidence,
        "processing_time_ms": response.processing_time_ms,
        "warning_badges": response.warning_badges
    }


# ============== Document Endpoints ==============

@router.post("/documents/upload", response_model=IngestionStatus, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None),
    pipeline=Depends(get_ingestion_pipeline),
    vector_store=Depends(get_vector_store),
    retriever=Depends(get_retriever)
):
    """
    Upload and ingest a document

    Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT
    """
    import time
    start_time = time.time()

    try:
        # Read file content
        content = await file.read()

        # Determine document type
        dtype = None
        if doc_type:
            try:
                dtype = DocumentType(doc_type)
            except ValueError:
                pass

        # Ingest document
        document = pipeline.ingest_bytes(
            content=content,
            filename=file.filename,
            doc_type=dtype
        )

        # Add to vector store
        chunks_added = vector_store.add_document(document)

        # Add to BM25 index
        bm25_docs = [
            {"chunk_id": c.chunk_id, "chunk_text": c.chunk_text, "metadata": c.metadata.dict()}
            for c in document.chunks
        ]
        retriever.add_documents_to_bm25(bm25_docs)

        processing_time = (time.time() - start_time) * 1000

        return IngestionStatus(
            doc_id=document.doc_id,
            source_uri=file.filename,
            status="success",
            chunks_created=chunks_added,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return IngestionStatus(
            doc_id="",
            source_uri=file.filename,
            status="error",
            error=str(e),
            processing_time_ms=(time.time() - start_time) * 1000
        )


@router.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(
    doc_id: str,
    vector_store=Depends(get_vector_store)
):
    """Delete a document and all its chunks"""
    chunks_deleted = vector_store.delete_document(doc_id)
    return {"doc_id": doc_id, "chunks_deleted": chunks_deleted}


@router.get("/documents/stats", tags=["Documents"])
async def get_document_stats(
    vector_store=Depends(get_vector_store)
):
    """Get statistics about indexed documents"""
    return vector_store.get_collection_stats()


# ============== Search Endpoints ==============

@router.post("/search", tags=["Search"])
async def search_documents(
    query: str,
    k: int = 8,
    doc_type: Optional[str] = None,
    retriever=Depends(get_retriever)
):
    """
    Search for relevant document chunks

    Returns chunks ranked by hybrid search score
    """
    filters = {}
    if doc_type:
        filters["doc_type"] = doc_type

    results = retriever.retrieve_with_intent(
        query=query,
        k=k,
        filters=filters if filters else None
    )

    return {
        "query": query,
        "results": [
            {
                "chunk_id": r.get("chunk_id"),
                "chunk_text": r.get("chunk_text", "")[:500],
                "doc_title": r.get("metadata", {}).get("doc_title", "Unknown"),
                "doc_type": r.get("metadata", {}).get("doc_type", "unknown"),
                "score": r.get("combined_score", r.get("similarity", 0)),
                "dense_score": r.get("dense_score", 0),
                "sparse_score": r.get("sparse_score", 0)
            }
            for r in results
        ],
        "total_results": len(results)
    }


# ============== Forecast Endpoints ==============

@router.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def get_forecast(
    request: ForecastRequest,
    forecaster=Depends(get_forecaster),
    safety_calc=Depends(get_safety_calculator)
):
    """
    Get demand forecast for a SKU

    Uses Prophet/XGBoost ensemble for predictions
    """
    import pandas as pd
    import numpy as np

    # Generate sample data (replace with actual data retrieval)
    np.random.seed(hash(request.sku_id) % 2**32)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq='D')
    base_demand = np.random.randint(50, 200)
    demand = base_demand + np.random.randn(90) * (base_demand * 0.2)
    demand = np.maximum(demand, 0)

    historical_data = pd.DataFrame({'ds': dates, 'y': demand})

    # Get forecast
    result = forecaster.forecast(
        historical_data=historical_data,
        periods=request.periods,
        sku_id=request.sku_id,
        warehouse_id=request.warehouse_id
    )

    # Calculate safety stock if requested
    safety_stock = None
    reorder_point = None

    if request.include_safety_stock:
        safety_result = safety_calc.calculate(
            demand_data=historical_data,
            lead_time_days=7
        )
        safety_stock = safety_result.get("safety_stock")
        reorder_point = safety_result.get("reorder_point")

    return ForecastResponse(
        sku_id=request.sku_id,
        warehouse_id=request.warehouse_id,
        forecast=result.get("forecast", []),
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        mape=result.get("mape"),
        model_used=result.get("model_used", "unknown")
    )


@router.get("/forecast/{sku_id}/summary", tags=["Forecast"])
async def get_forecast_summary(
    sku_id: str,
    periods: int = 30,
    forecaster=Depends(get_forecaster)
):
    """Get a text summary of the forecast"""
    import pandas as pd
    import numpy as np

    np.random.seed(hash(sku_id) % 2**32)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq='D')
    base_demand = np.random.randint(50, 200)
    demand = base_demand + np.random.randn(90) * (base_demand * 0.2)

    historical_data = pd.DataFrame({'ds': dates, 'y': np.maximum(demand, 0)})

    result = forecaster.forecast(
        historical_data=historical_data,
        periods=periods,
        sku_id=sku_id
    )

    summary = forecaster.forecast_to_text(result)

    return {"sku_id": sku_id, "summary": summary}


# ============== Feedback Endpoints ==============

@router.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a response

    Used for improving the system
    """
    logger.info(f"Feedback received for query {request.query_id}: helpful={request.helpful}")

    # In production, store this in a database
    return {"status": "received", "query_id": request.query_id}


# ============== Health Endpoints ==============

@router.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SupplyChainGPT"}


@router.get("/health/detailed", tags=["System"])
async def detailed_health_check(
    vector_store=Depends(get_vector_store)
):
    """Detailed health check with component status"""
    components = {
        "api": "healthy",
        "vector_store": "unknown",
        "embedding_model": "unknown"
    }

    try:
        stats = vector_store.get_collection_stats()
        components["vector_store"] = "healthy"
        components["total_chunks"] = stats.get("total_chunks", 0)
    except Exception as e:
        components["vector_store"] = f"unhealthy: {str(e)}"

    return {"status": "healthy", "components": components}
