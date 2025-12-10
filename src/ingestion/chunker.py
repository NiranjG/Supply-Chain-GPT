"""
Text chunking strategies for document processing
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for chunking"""
    max_tokens: int = 750
    overlap: int = 120
    min_chunk_size: int = 100
    preserve_tables: bool = True
    preserve_headings: bool = True


class TextChunker:
    """
    Semantic text chunker with support for:
    - Heading-based chunking for policies/SOPs
    - Table-aware chunking for reports
    - Overlap for context preservation
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def chunk(self, text: str, doc_type: str = "report") -> List[Tuple[str, int]]:
        """
        Chunk text based on document type

        Args:
            text: Text to chunk
            doc_type: Type of document (policy, sop, report, etc.)

        Returns:
            List of (chunk_text, token_count) tuples
        """
        if doc_type in ["policy", "sop", "manual"]:
            return self._chunk_by_headings(text)
        elif doc_type in ["report", "export"]:
            return self._chunk_tables_aware(text)
        else:
            return self._chunk_sliding_window(text)

    def _chunk_by_headings(self, text: str) -> List[Tuple[str, int]]:
        """Chunk by markdown headings (H1, H2, H3)"""
        chunks = []

        # Split by headings
        heading_pattern = r"(^#{1,3}\s+.+$)"
        parts = re.split(heading_pattern, text, flags=re.MULTILINE)

        current_chunk = ""
        current_heading = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if this is a heading
            if re.match(r"^#{1,3}\s+", part):
                # If we have accumulated content, save it
                if current_chunk and self._estimate_tokens(current_chunk) >= self.config.min_chunk_size:
                    chunks.append((current_chunk.strip(), self._estimate_tokens(current_chunk)))
                    current_chunk = ""

                current_heading = part
                current_chunk = current_heading + "\n\n"
            else:
                # Add content to current chunk
                potential_chunk = current_chunk + part

                if self._estimate_tokens(potential_chunk) > self.config.max_tokens:
                    # Need to split this content
                    if current_chunk and self._estimate_tokens(current_chunk) >= self.config.min_chunk_size:
                        chunks.append((current_chunk.strip(), self._estimate_tokens(current_chunk)))

                    # Split remaining content with overlap
                    sub_chunks = self._chunk_sliding_window(part)
                    for sub_chunk, token_count in sub_chunks:
                        # Prepend heading for context
                        if current_heading:
                            sub_chunk = f"{current_heading}\n\n{sub_chunk}"
                            token_count = self._estimate_tokens(sub_chunk)
                        chunks.append((sub_chunk, token_count))

                    current_chunk = current_heading + "\n\n" if current_heading else ""
                else:
                    current_chunk = potential_chunk + "\n\n"

        # Add final chunk
        if current_chunk.strip() and self._estimate_tokens(current_chunk) >= self.config.min_chunk_size:
            chunks.append((current_chunk.strip(), self._estimate_tokens(current_chunk)))

        return chunks if chunks else [(text, self._estimate_tokens(text))]

    def _chunk_tables_aware(self, text: str) -> List[Tuple[str, int]]:
        """Chunk while preserving tables as single units"""
        chunks = []

        # Split by tables (markdown tables)
        table_pattern = r"(\|.+\|(?:\n\|[-: |]+\|)?(?:\n\|.+\|)+)"
        parts = re.split(table_pattern, text)

        current_chunk = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if this is a table
            is_table = bool(re.match(r"^\|.+\|", part))

            if is_table:
                # Save current chunk if exists
                if current_chunk:
                    sub_chunks = self._chunk_sliding_window(current_chunk)
                    chunks.extend(sub_chunks)
                    current_chunk = ""

                # Table as its own chunk (even if larger than max)
                chunks.append((part, self._estimate_tokens(part)))
            else:
                current_chunk += part + "\n\n"

        # Process remaining content
        if current_chunk.strip():
            sub_chunks = self._chunk_sliding_window(current_chunk)
            chunks.extend(sub_chunks)

        return chunks if chunks else [(text, self._estimate_tokens(text))]

    def _chunk_sliding_window(self, text: str) -> List[Tuple[str, int]]:
        """Simple sliding window chunking by words with overlap"""
        words = text.split()

        if self._estimate_tokens(text) <= self.config.max_tokens:
            return [(text.strip(), self._estimate_tokens(text))]

        chunks = []
        step = self.config.max_tokens - self.config.overlap

        i = 0
        while i < len(words):
            # Take max_tokens worth of words
            chunk_words = words[i:i + self.config.max_tokens]
            chunk_text = " ".join(chunk_words)

            # Try to end at sentence boundary
            chunk_text = self._adjust_to_sentence_boundary(chunk_text, text, i, len(words))

            if chunk_text.strip():
                chunks.append((chunk_text.strip(), self._estimate_tokens(chunk_text)))

            i += step

            # Prevent infinite loop for very long words
            if i < len(words) and i <= 0:
                i = 1

        return chunks

    def _adjust_to_sentence_boundary(self, chunk: str, full_text: str, start_idx: int, total_words: int) -> str:
        """Try to end chunk at a sentence boundary"""
        # Look for sentence endings
        sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]

        for ending in sentence_endings:
            last_pos = chunk.rfind(ending)
            if last_pos > len(chunk) * 0.6:  # Only adjust if ending is in latter part
                return chunk[:last_pos + 1]

        return chunk

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation)
        More accurate would be to use tiktoken
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4

    def chunk_with_metadata(
        self,
        text: str,
        doc_type: str = "report",
        base_metadata: Optional[dict] = None
    ) -> List[dict]:
        """
        Chunk text and return chunks with metadata

        Returns:
            List of dicts with chunk_text, token_count, and chunk_index
        """
        chunks = self.chunk(text, doc_type)

        result = []
        for idx, (chunk_text, token_count) in enumerate(chunks):
            chunk_data = {
                "chunk_text": chunk_text,
                "chunk_tokens": token_count,
                "chunk_index": idx,
            }
            if base_metadata:
                chunk_data.update(base_metadata)
            result.append(chunk_data)

        return result


class SemanticChunker(TextChunker):
    """
    Advanced semantic chunker that uses embeddings
    to find optimal chunk boundaries
    """

    def __init__(self, config: Optional[ChunkConfig] = None, embedding_model=None):
        super().__init__(config)
        self.embedding_model = embedding_model

    def chunk_semantic(self, text: str) -> List[Tuple[str, int]]:
        """
        Chunk based on semantic similarity between sentences
        Falls back to sliding window if no embedding model
        """
        if self.embedding_model is None:
            return self._chunk_sliding_window(text)

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [(text, self._estimate_tokens(text))]

        # Get embeddings for each sentence
        embeddings = self.embedding_model.encode(sentences)

        # Find breakpoints based on similarity drop
        breakpoints = self._find_semantic_breakpoints(embeddings)

        # Create chunks from breakpoints
        chunks = []
        start = 0

        for bp in breakpoints:
            chunk_text = " ".join(sentences[start:bp])
            if self._estimate_tokens(chunk_text) >= self.config.min_chunk_size:
                chunks.append((chunk_text, self._estimate_tokens(chunk_text)))
            start = bp

        # Final chunk
        if start < len(sentences):
            chunk_text = " ".join(sentences[start:])
            if self._estimate_tokens(chunk_text) >= self.config.min_chunk_size:
                chunks.append((chunk_text, self._estimate_tokens(chunk_text)))

        return chunks if chunks else [(text, self._estimate_tokens(text))]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_semantic_breakpoints(self, embeddings, threshold: float = 0.5) -> List[int]:
        """Find indices where semantic similarity drops below threshold"""
        import numpy as np

        breakpoints = []
        current_tokens = 0

        for i in range(1, len(embeddings)):
            # Cosine similarity
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )

            # Track approximate tokens
            current_tokens += self.config.max_tokens // 10  # Rough estimate per sentence

            # Break if similarity drops and we have enough content
            if similarity < threshold and current_tokens >= self.config.min_chunk_size:
                breakpoints.append(i)
                current_tokens = 0

            # Also break if chunk is getting too large
            if current_tokens >= self.config.max_tokens:
                breakpoints.append(i)
                current_tokens = 0

        return breakpoints
