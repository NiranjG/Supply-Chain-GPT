"""
Pydantic schemas for SupplyChainGPT
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class DocumentType(str, Enum):
    """Types of documents supported"""
    POLICY = "policy"
    SOP = "sop"
    CONTRACT = "contract"
    REPORT = "report"
    MEETING_NOTES = "meeting_notes"
    EMAIL = "email"
    MANUAL = "manual"
    EXPORT = "export"


class ACL(BaseModel):
    """Access Control List for documents"""
    tenants: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    users: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Document provenance tracking"""
    checksum: str
    parser: str
    ocr_confidence: Optional[float] = None


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk"""
    doc_id: str
    source_uri: str
    doc_title: str
    doc_type: DocumentType = DocumentType.REPORT
    chunk_index: int = 0
    chunk_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    effective_date_start: Optional[date] = None
    effective_date_end: Optional[date] = None
    sku_ids: Optional[List[str]] = None
    warehouse_ids: Optional[List[str]] = None
    supplier_ids: Optional[List[str]] = None
    region: Optional[str] = None
    version: str = "1.0"
    language: str = "en"
    acl: ACL = Field(default_factory=ACL)
    provenance: Optional[Provenance] = None


class Chunk(BaseModel):
    """A chunk of document text with metadata"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class Document(BaseModel):
    """A full document"""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_uri: str
    doc_title: str
    doc_type: DocumentType = DocumentType.REPORT
    content: str
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserContext(BaseModel):
    """User context for query filtering"""
    user_id: str
    tenant_id: str = "default"
    roles: List[str] = Field(default_factory=lambda: ["viewer"])


class QueryRequest(BaseModel):
    """Request for RAG query"""
    query: str
    user_context: UserContext
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 8
    include_forecast: bool = False
    sku_id: Optional[str] = None
    warehouse_id: Optional[str] = None


class Citation(BaseModel):
    """Citation for a response"""
    doc_title: str
    source_uri: str
    chunk_text: str
    relevance_score: float
    page_or_section: Optional[str] = None


class QueryResponse(BaseModel):
    """Response from RAG query"""
    answer: str
    citations: List[Citation]
    confidence: float
    forecast_data: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    warning_badges: List[str] = Field(default_factory=list)


class ForecastRequest(BaseModel):
    """Request for demand forecast"""
    sku_id: str
    warehouse_id: Optional[str] = None
    periods: int = 30
    include_safety_stock: bool = True


class ForecastResponse(BaseModel):
    """Response with forecast data"""
    sku_id: str
    warehouse_id: Optional[str]
    forecast: List[Dict[str, Any]]
    safety_stock: Optional[float] = None
    reorder_point: Optional[float] = None
    mape: Optional[float] = None
    model_used: str


class FeedbackRequest(BaseModel):
    """User feedback on response"""
    query_id: str
    helpful: bool
    error_type: Optional[str] = None
    comment: Optional[str] = None


class IngestionStatus(BaseModel):
    """Status of document ingestion"""
    doc_id: str
    source_uri: str
    status: str
    chunks_created: int = 0
    error: Optional[str] = None
    processing_time_ms: float = 0
