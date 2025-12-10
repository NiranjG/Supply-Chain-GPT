"""
Tests for document ingestion module
"""

import pytest
from pathlib import Path
import tempfile

from src.ingestion.parsers import DocumentParser
from src.ingestion.chunker import TextChunker, ChunkConfig
from src.ingestion.pipeline import IngestionPipeline, PIIRedactor
from src.models.schemas import DocumentType


class TestDocumentParser:
    """Tests for DocumentParser class"""

    def test_parse_txt_file(self, temp_directory, sample_text):
        """Test parsing a text file"""
        # Create a temp text file
        txt_path = temp_directory / "test_doc.txt"
        txt_path.write_text(sample_text)

        parser = DocumentParser()
        text, metadata = parser.parse(txt_path)

        assert text is not None
        assert len(text) > 0
        assert "Inventory Policy" in text
        assert "SKU-12345" in text
        assert metadata["doc_title"] == "test_doc"
        assert metadata["file_type"] == ".txt"
        assert "checksum" in metadata

    def test_parse_csv_file(self, temp_directory, sample_csv_content):
        """Test parsing a CSV file"""
        csv_path = temp_directory / "inventory.csv"
        csv_path.write_text(sample_csv_content)

        parser = DocumentParser()
        text, metadata = parser.parse(csv_path)

        assert text is not None
        assert "SKU-12345" in text
        assert "WH-001" in text
        assert metadata["row_count"] == 5
        assert "sku_id" in metadata["columns"]

    def test_unsupported_format(self, temp_directory):
        """Test handling of unsupported file formats"""
        unsupported_path = temp_directory / "test.xyz"
        unsupported_path.write_text("some content")

        parser = DocumentParser()

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(unsupported_path)

    def test_file_not_found(self):
        """Test handling of non-existent files"""
        parser = DocumentParser()

        with pytest.raises(FileNotFoundError):
            parser.parse(Path("/nonexistent/file.txt"))

    def test_extract_entities(self, sample_text):
        """Test entity extraction from text"""
        parser = DocumentParser()
        entities = parser.extract_entities(sample_text)

        assert "sku_ids" in entities
        assert "warehouse_ids" in entities
        assert "supplier_ids" in entities
        assert "SKU-12345" in entities["sku_ids"]
        assert "WH-001" in entities["warehouse_ids"]
        assert "SUP-001" in entities["supplier_ids"]

    def test_clean_text(self):
        """Test text cleaning"""
        parser = DocumentParser()
        dirty_text = "Hello\n\n\n\nWorld   with   spaces"
        cleaned = parser._clean_text(dirty_text)

        assert "\n\n\n" not in cleaned
        assert "   " not in cleaned


class TestTextChunker:
    """Tests for TextChunker class"""

    def test_basic_chunking(self, sample_text):
        """Test basic text chunking"""
        chunker = TextChunker(ChunkConfig(max_tokens=100, overlap=20))
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 0
        for chunk_text, token_count in chunks:
            assert len(chunk_text) > 0
            assert token_count > 0

    def test_heading_based_chunking(self):
        """Test chunking by headings"""
        text = """# Main Title

        Introduction paragraph.

        ## Section 1

        Content of section 1 with details.

        ## Section 2

        Content of section 2 with more details.
        """

        chunker = TextChunker(ChunkConfig(max_tokens=500, min_chunk_size=10))
        chunks = chunker._chunk_by_headings(text)

        assert len(chunks) >= 1
        # Check that chunks contain heading context
        chunk_texts = [c[0] for c in chunks]
        assert any("#" in text for text in chunk_texts)

    def test_table_aware_chunking(self):
        """Test chunking that preserves tables"""
        text = """
        Introduction text.

        | Column 1 | Column 2 |
        |----------|----------|
        | Value 1  | Value 2  |
        | Value 3  | Value 4  |

        More text after the table.
        """

        chunker = TextChunker(ChunkConfig(max_tokens=500))
        chunks = chunker._chunk_tables_aware(text)

        assert len(chunks) >= 1
        # Find the table chunk
        table_chunks = [c for c in chunks if "|" in c[0]]
        assert len(table_chunks) >= 1

    def test_overlap_in_chunks(self):
        """Test that chunks have proper overlap"""
        # Create long text
        text = " ".join(["word"] * 500)

        chunker = TextChunker(ChunkConfig(max_tokens=100, overlap=20))
        chunks = chunker._chunk_sliding_window(text)

        # With overlap, chunks should share some content
        assert len(chunks) > 1

    def test_chunk_with_metadata(self, sample_text):
        """Test chunking with metadata attachment"""
        chunker = TextChunker()
        base_meta = {"doc_id": "test-001", "source": "test"}

        result = chunker.chunk_with_metadata(sample_text, base_metadata=base_meta)

        assert len(result) > 0
        for chunk_data in result:
            assert "chunk_text" in chunk_data
            assert "chunk_tokens" in chunk_data
            assert "chunk_index" in chunk_data
            assert "doc_id" in chunk_data


class TestIngestionPipeline:
    """Tests for IngestionPipeline class"""

    def test_ingest_text_file(self, temp_directory, sample_text):
        """Test ingesting a text file"""
        txt_path = temp_directory / "policy.txt"
        txt_path.write_text(sample_text)

        pipeline = IngestionPipeline()
        document = pipeline.ingest_file(txt_path, doc_type=DocumentType.POLICY)

        assert document.doc_id is not None
        assert document.doc_title == "policy"
        assert document.doc_type == DocumentType.POLICY
        assert len(document.chunks) > 0
        assert document.content is not None

    def test_ingest_with_acl(self, temp_directory, sample_text):
        """Test ingestion with ACL settings"""
        from src.models.schemas import ACL

        txt_path = temp_directory / "restricted.txt"
        txt_path.write_text(sample_text)

        acl = ACL(roles=["admin", "planner"], tenants=["acme"])
        pipeline = IngestionPipeline()
        document = pipeline.ingest_file(txt_path, acl=acl)

        # Check that chunks have ACL
        for chunk in document.chunks:
            assert "admin" in chunk.metadata.acl.roles
            assert "planner" in chunk.metadata.acl.roles

    def test_ingest_directory(self, temp_directory):
        """Test ingesting multiple files from directory"""
        # Create multiple files
        (temp_directory / "doc1.txt").write_text("Document 1 content")
        (temp_directory / "doc2.txt").write_text("Document 2 content")

        pipeline = IngestionPipeline()
        documents = pipeline.ingest_directory(temp_directory)

        assert len(documents) == 2

    def test_infer_document_type(self, temp_directory):
        """Test automatic document type inference"""
        policy_path = temp_directory / "inventory_policy.txt"
        policy_path.write_text("This policy defines inventory procedures...")

        pipeline = IngestionPipeline()
        doc_type = pipeline._infer_doc_type(policy_path, "This policy defines...")

        assert doc_type == DocumentType.POLICY

    def test_ingest_bytes(self):
        """Test ingesting from bytes"""
        content = b"This is test content for the document."
        filename = "test_document.txt"

        pipeline = IngestionPipeline()
        document = pipeline.ingest_bytes(content, filename)

        assert document.doc_id is not None
        assert document.source_uri == filename
        assert "test content" in document.content


class TestPIIRedactor:
    """Tests for PII redaction"""

    def test_email_redaction(self):
        """Test email address redaction"""
        redactor = PIIRedactor()
        text = "Contact john.doe@example.com for more info."
        redacted = redactor.redact(text)

        assert "john.doe@example.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_phone_redaction(self):
        """Test phone number redaction"""
        redactor = PIIRedactor()
        text = "Call us at 555-123-4567 or (555) 987-6543."
        redacted = redactor.redact(text)

        assert "555-123-4567" not in redacted
        assert "[REDACTED_PHONE]" in redacted

    def test_selective_redaction(self):
        """Test selective PII type redaction"""
        redactor = PIIRedactor()
        text = "Email: test@test.com, Phone: 555-123-4567"

        # Only redact emails
        redacted = redactor.redact(text, redact_types=["email"])

        assert "[REDACTED_EMAIL]" in redacted
        assert "555-123-4567" in redacted  # Phone should remain

    def test_no_pii(self):
        """Test text without PII"""
        redactor = PIIRedactor()
        text = "This is a clean document without any PII."
        redacted = redactor.redact(text)

        assert redacted == text
