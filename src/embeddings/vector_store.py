"""
Vector store implementation using ChromaDB
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .embedding_service import EmbeddingService
from ..models.schemas import Chunk, Document

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store using ChromaDB for document storage and retrieval
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "supplychain_docs",
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize the vector store

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_service: Embedding service for generating embeddings
        """
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()

        # Initialize ChromaDB client
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized vector store with collection: {collection_name}")

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to the vector store

        Args:
            chunks: List of Chunk objects

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Prepare data
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.chunk_text)

            # Convert metadata to dict for ChromaDB
            meta = self._chunk_metadata_to_dict(chunk.metadata)
            metadatas.append(meta)

            # Generate embedding if not present
            if chunk.embedding:
                embeddings.append(chunk.embedding)
            else:
                embedding = self.embedding_service.embed_text(chunk.chunk_text)
                embeddings.append(embedding)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        logger.info(f"Added {len(chunks)} chunks to vector store")
        return len(chunks)

    def add_document(self, document: Document) -> int:
        """
        Add a document (all its chunks) to the vector store

        Args:
            document: Document object with chunks

        Returns:
            Number of chunks added
        """
        return self.add_chunks(document.chunks)

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add multiple documents to the vector store

        Args:
            documents: List of Document objects

        Returns:
            Total number of chunks added
        """
        total = 0
        for doc in documents:
            total += self.add_document(doc)
        return total

    def search(
        self,
        query: str,
        k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks

        Args:
            query: Query text
            k: Number of results to return
            filters: Metadata filters
            include_embeddings: Whether to include embeddings in results

        Returns:
            List of search results with chunk data and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Build where clause from filters
        where_clause = self._build_where_clause(filters) if filters else None

        # Query the collection
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
            include=include
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                result = {
                    "chunk_id": results["ids"][0][i],
                    "chunk_text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 1,
                }
                if include_embeddings and results.get("embeddings"):
                    result["embedding"] = results["embeddings"][0][i]
                formatted_results.append(result)

        return formatted_results

    def search_with_acl(
        self,
        query: str,
        user_roles: List[str],
        tenant_id: str = "default",
        k: int = 8,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with ACL filtering

        Args:
            query: Query text
            user_roles: User's roles for ACL filtering
            tenant_id: User's tenant ID
            k: Number of results
            additional_filters: Additional metadata filters

        Returns:
            List of search results accessible to the user
        """
        # Note: ChromaDB has limited filter support
        # For production, implement proper ACL filtering
        filters = additional_filters or {}

        # Add role filter if possible
        # ChromaDB doesn't support array contains well, so we filter post-query
        results = self.search(query, k=k * 2, filters=filters)

        # Post-filter by ACL
        filtered_results = []
        for result in results:
            meta = result.get("metadata", {})
            acl_roles = meta.get("acl_roles", "").split(",") if meta.get("acl_roles") else ["viewer"]

            # Check if user has access
            if "admin" in user_roles or any(role in user_roles for role in acl_roles):
                filtered_results.append(result)

            if len(filtered_results) >= k:
                break

        return filtered_results

    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document

        Args:
            doc_id: Document ID

        Returns:
            Number of chunks deleted
        """
        # Find all chunks for this document
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
            return len(results["ids"])

        return 0

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk data or None
        """
        results = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas", "embeddings"]
        )

        if results["ids"]:
            return {
                "chunk_id": results["ids"][0],
                "chunk_text": results["documents"][0] if results["documents"] else "",
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
                "embedding": results["embeddings"][0] if results.get("embeddings") else None
            }
        return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
        }

    def clear(self):
        """Clear all data from the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection: {self.collection_name}")

    def _chunk_metadata_to_dict(self, metadata) -> Dict[str, Any]:
        """Convert ChunkMetadata to a flat dict for ChromaDB"""
        meta_dict = {
            "doc_id": metadata.doc_id,
            "source_uri": metadata.source_uri,
            "doc_title": metadata.doc_title,
            "doc_type": metadata.doc_type.value if hasattr(metadata.doc_type, "value") else str(metadata.doc_type),
            "chunk_index": metadata.chunk_index,
            "chunk_tokens": metadata.chunk_tokens,
            "version": metadata.version,
            "language": metadata.language,
        }

        # Add optional fields
        if metadata.region:
            meta_dict["region"] = metadata.region

        # Flatten ACL for ChromaDB (limited filter support)
        if metadata.acl:
            meta_dict["acl_roles"] = ",".join(metadata.acl.roles) if metadata.acl.roles else "viewer"
            meta_dict["acl_tenants"] = ",".join(metadata.acl.tenants) if metadata.acl.tenants else "default"

        # Add entity lists as comma-separated strings
        if metadata.sku_ids:
            meta_dict["sku_ids"] = ",".join(metadata.sku_ids)
        if metadata.warehouse_ids:
            meta_dict["warehouse_ids"] = ",".join(metadata.warehouse_ids)
        if metadata.supplier_ids:
            meta_dict["supplier_ids"] = ",".join(metadata.supplier_ids)

        return meta_dict

    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters"""
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if isinstance(value, list):
                # OR condition for list values
                conditions.append({key: {"$in": value}})
            elif isinstance(value, dict):
                # Already a condition
                conditions.append({key: value})
            else:
                # Exact match
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}

        return None
