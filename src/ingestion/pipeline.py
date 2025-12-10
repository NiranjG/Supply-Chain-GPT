"""
Document ingestion pipeline for SupplyChainGPT
"""

import uuid
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .parsers import DocumentParser
from .chunker import TextChunker, ChunkConfig
from ..models.schemas import Document, Chunk, ChunkMetadata, DocumentType, ACL, Provenance

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline:
    1. Parse documents from various formats
    2. Chunk text semantically
    3. Extract metadata and entities
    4. Prepare for embedding and indexing
    """

    def __init__(
        self,
        chunk_config: Optional[ChunkConfig] = None,
        default_acl: Optional[ACL] = None
    ):
        self.parser = DocumentParser()
        self.chunker = TextChunker(chunk_config or ChunkConfig())
        self.default_acl = default_acl or ACL(roles=["viewer"])

    def ingest_file(
        self,
        file_path: Path,
        doc_type: Optional[DocumentType] = None,
        metadata_overrides: Optional[Dict[str, Any]] = None,
        acl: Optional[ACL] = None
    ) -> Document:
        """
        Ingest a single file

        Args:
            file_path: Path to the document
            doc_type: Type of document (auto-detected if not provided)
            metadata_overrides: Additional metadata to add
            acl: Access control list for this document

        Returns:
            Document object with chunks
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting file: {file_path}")

        # Parse document
        text, parse_metadata = self.parser.parse(file_path)

        # Auto-detect document type if not provided
        if doc_type is None:
            doc_type = self._infer_doc_type(file_path, text)

        # Extract entities
        entities = self.parser.extract_entities(text)

        # Create document ID
        doc_id = str(uuid.uuid4())

        # Chunk the document
        chunks = self._create_chunks(
            text=text,
            doc_id=doc_id,
            doc_type=doc_type,
            parse_metadata=parse_metadata,
            entities=entities,
            acl=acl or self.default_acl,
            metadata_overrides=metadata_overrides
        )

        # Create document object
        document = Document(
            doc_id=doc_id,
            source_uri=str(file_path),
            doc_title=parse_metadata.get("doc_title", file_path.stem),
            doc_type=doc_type,
            content=text,
            chunks=chunks,
            metadata={
                **parse_metadata,
                **entities,
                **(metadata_overrides or {})
            },
            created_at=datetime.utcnow()
        )

        logger.info(f"Created document {doc_id} with {len(chunks)} chunks")
        return document

    def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        doc_type: Optional[DocumentType] = None,
        metadata_overrides: Optional[Dict[str, Any]] = None,
        acl: Optional[ACL] = None
    ) -> Document:
        """Ingest document from bytes (for file uploads)"""
        logger.info(f"Ingesting uploaded file: {filename}")

        # Parse document
        text, parse_metadata = self.parser.parse_bytes(content, filename)

        # Auto-detect document type
        if doc_type is None:
            doc_type = self._infer_doc_type(Path(filename), text)

        # Extract entities
        entities = self.parser.extract_entities(text)

        # Create document ID
        doc_id = str(uuid.uuid4())

        # Chunk the document
        chunks = self._create_chunks(
            text=text,
            doc_id=doc_id,
            doc_type=doc_type,
            parse_metadata=parse_metadata,
            entities=entities,
            acl=acl or self.default_acl,
            metadata_overrides=metadata_overrides
        )

        # Create document object
        document = Document(
            doc_id=doc_id,
            source_uri=filename,
            doc_title=parse_metadata.get("doc_title", Path(filename).stem),
            doc_type=doc_type,
            content=text,
            chunks=chunks,
            metadata={
                **parse_metadata,
                **entities,
                **(metadata_overrides or {})
            },
            created_at=datetime.utcnow()
        )

        logger.info(f"Created document {doc_id} with {len(chunks)} chunks")
        return document

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        doc_type: Optional[DocumentType] = None,
        acl: Optional[ACL] = None
    ) -> List[Document]:
        """Ingest all supported documents from a directory"""
        directory = Path(directory)
        documents = []

        pattern = "**/*" if recursive else "*"
        supported_extensions = self.parser.supported_formats.keys()

        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    doc = self.ingest_file(file_path, doc_type=doc_type, acl=acl)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")

        logger.info(f"Ingested {len(documents)} documents from {directory}")
        return documents

    def _create_chunks(
        self,
        text: str,
        doc_id: str,
        doc_type: DocumentType,
        parse_metadata: Dict[str, Any],
        entities: Dict[str, Any],
        acl: ACL,
        metadata_overrides: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Create chunks from document text"""
        chunk_data = self.chunker.chunk(text, doc_type.value)

        chunks = []
        for idx, (chunk_text, token_count) in enumerate(chunk_data):
            chunk_id = str(uuid.uuid4())

            metadata = ChunkMetadata(
                doc_id=doc_id,
                source_uri=parse_metadata.get("source_uri", ""),
                doc_title=parse_metadata.get("doc_title", ""),
                doc_type=doc_type,
                chunk_index=idx,
                chunk_tokens=token_count,
                sku_ids=entities.get("sku_ids"),
                warehouse_ids=entities.get("warehouse_ids"),
                supplier_ids=entities.get("supplier_ids"),
                acl=acl,
                provenance=Provenance(
                    checksum=parse_metadata.get("checksum", ""),
                    parser=parse_metadata.get("parser", "unknown"),
                    ocr_confidence=parse_metadata.get("ocr_confidence")
                )
            )

            # Apply metadata overrides
            if metadata_overrides:
                for key, value in metadata_overrides.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)

            chunk = Chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks

    def _infer_doc_type(self, file_path: Path, text: str) -> DocumentType:
        """Infer document type from filename and content"""
        filename_lower = file_path.stem.lower()
        text_lower = text.lower()[:2000]  # Check first 2000 chars

        # Check filename patterns
        type_patterns = {
            DocumentType.POLICY: ["policy", "guideline", "rule"],
            DocumentType.SOP: ["sop", "procedure", "process", "workflow"],
            DocumentType.CONTRACT: ["contract", "agreement", "terms"],
            DocumentType.REPORT: ["report", "analysis", "summary"],
            DocumentType.MEETING_NOTES: ["meeting", "minutes", "notes"],
            DocumentType.EMAIL: ["email", "correspondence"],
            DocumentType.MANUAL: ["manual", "guide", "handbook"],
            DocumentType.EXPORT: ["export", "data", "inventory", "stock"],
        }

        for doc_type, patterns in type_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower or pattern in text_lower:
                    return doc_type

        # Check file extension
        if file_path.suffix.lower() in [".xlsx", ".csv"]:
            return DocumentType.EXPORT

        return DocumentType.REPORT


class PIIRedactor:
    """Redact PII and sensitive information from documents"""

    def __init__(self):
        import re
        self.patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "api_key": r"\b(?:api[_-]?key|token|secret)[=:\s]+['\"]?[\w-]{20,}['\"]?\b",
        }

    def redact(self, text: str, redact_types: Optional[List[str]] = None) -> str:
        """Redact sensitive information from text"""
        import re

        redact_types = redact_types or list(self.patterns.keys())

        for pii_type, pattern in self.patterns.items():
            if pii_type in redact_types:
                text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text, flags=re.IGNORECASE)

        return text
