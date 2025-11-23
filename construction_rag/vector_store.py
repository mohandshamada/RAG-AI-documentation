"""
Vector Store for Construction RAG
==================================

Manages vector database storage and retrieval using ChromaDB.
Optimized for construction document search.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database manager using ChromaDB for efficient similarity search.

    Features:
    - Persistent storage
    - Metadata filtering
    - Batch operations for large documents
    - Efficient similarity search
    """

    def __init__(
        self,
        db_path: str,
        embedding_dimension: int,
        collection_name: str = "construction_docs"
    ):
        """
        Initialize vector store.

        Args:
            db_path: Path to store database
            embedding_dimension: Dimension of embedding vectors
            collection_name: Name of the collection
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name

        # Try to import ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings

            self.chromadb = chromadb
            self.chroma_available = True

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Construction documents RAG system"}
            )

            logger.info(f"Vector store initialized at {db_path}")

        except ImportError:
            logger.warning(
                "ChromaDB not installed. Vector store will use fallback mode. "
                "Install with: pip install chromadb"
            )
            self.chroma_available = False
            self.fallback_store = []

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_id: Optional[str] = None
    ) -> List[str]:
        """
        Add documents to vector store.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts for each chunk
            document_id: Identifier for the source document

        Returns:
            List of chunk IDs
        """
        if not texts:
            return []

        # Generate IDs
        if document_id:
            ids = [f"{document_id}_chunk_{i}" for i in range(len(texts))]
        else:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Add document_id to metadata
        for i, meta in enumerate(metadatas):
            meta["chunk_id"] = ids[i]
            if document_id:
                meta["document_id"] = document_id

        if self.chroma_available:
            try:
                # Convert metadata values to strings for ChromaDB compatibility
                processed_metadatas = []
                for meta in metadatas:
                    processed_meta = {}
                    for key, value in meta.items():
                        if isinstance(value, (list, dict)):
                            processed_meta[key] = json.dumps(value)
                        elif value is None:
                            processed_meta[key] = "null"
                        else:
                            processed_meta[key] = str(value)
                    processed_metadatas.append(processed_meta)

                # Add to ChromaDB in batches
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    end_idx = min(i + batch_size, len(texts))

                    self.collection.add(
                        ids=ids[i:end_idx],
                        embeddings=embeddings[i:end_idx],
                        documents=texts[i:end_idx],
                        metadatas=processed_metadatas[i:end_idx]
                    )

                logger.info(f"Added {len(texts)} documents to vector store")

            except Exception as e:
                logger.error(f"Error adding to ChromaDB: {str(e)}")
                raise
        else:
            # Fallback mode
            for i in range(len(texts)):
                self.fallback_store.append({
                    "id": ids[i],
                    "text": texts[i],
                    "embedding": embeddings[i],
                    "metadata": metadatas[i]
                })

            logger.info(f"Added {len(texts)} documents to fallback store")

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Filter results by metadata

        Returns:
            List of results with text, metadata, and similarity scores
        """
        if self.chroma_available:
            try:
                # Prepare where filter for ChromaDB
                where_filter = None
                if filter_metadata:
                    where_filter = {}
                    for key, value in filter_metadata.items():
                        if isinstance(value, (list, dict)):
                            where_filter[key] = json.dumps(value)
                        elif value is not None:
                            where_filter[key] = str(value)

                # Query ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter
                )

                # Format results
                formatted_results = []
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i].copy()

                    # Parse JSON strings back to objects
                    for key, value in metadata.items():
                        if value == "null":
                            metadata[key] = None
                        elif isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                            try:
                                metadata[key] = json.loads(value)
                            except:
                                pass

                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": metadata,
                        "score": results['distances'][0][i] if 'distances' in results else None
                    })

                return formatted_results

            except Exception as e:
                logger.error(f"Error searching ChromaDB: {str(e)}")
                return []
        else:
            # Fallback mode - simple cosine similarity
            import numpy as np

            query_vec = np.array(query_embedding)
            similarities = []

            for item in self.fallback_store:
                # Apply metadata filter
                if filter_metadata:
                    match = all(
                        item['metadata'].get(k) == v
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue

                # Calculate cosine similarity
                item_vec = np.array(item['embedding'])
                similarity = np.dot(query_vec, item_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(item_vec)
                )
                similarities.append({
                    "id": item['id'],
                    "text": item['text'],
                    "metadata": item['metadata'],
                    "score": float(1 - similarity)  # Convert to distance
                })

            # Sort by similarity
            similarities.sort(key=lambda x: x['score'])

            return similarities[:top_k]

    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful
        """
        if not ids:
            return True

        if self.chroma_available:
            try:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents")
                return True
            except Exception as e:
                logger.error(f"Error deleting from ChromaDB: {str(e)}")
                return False
        else:
            # Fallback mode
            self.fallback_store = [
                item for item in self.fallback_store
                if item['id'] not in ids
            ]
            logger.info(f"Deleted {len(ids)} documents from fallback store")
            return True

    def delete_by_metadata(self, filter_metadata: Dict[str, Any]) -> bool:
        """
        Delete documents by metadata filter.

        Args:
            filter_metadata: Metadata filter

        Returns:
            True if successful
        """
        if self.chroma_available:
            try:
                # ChromaDB doesn't have direct delete by metadata, so we query first
                results = self.collection.get(
                    where={k: str(v) for k, v in filter_metadata.items()}
                )

                if results['ids']:
                    self.collection.delete(ids=results['ids'])
                    logger.info(f"Deleted {len(results['ids'])} documents by metadata")

                return True
            except Exception as e:
                logger.error(f"Error deleting by metadata: {str(e)}")
                return False
        else:
            # Fallback mode
            initial_count = len(self.fallback_store)

            self.fallback_store = [
                item for item in self.fallback_store
                if not all(
                    item['metadata'].get(k) == v
                    for k, v in filter_metadata.items()
                )
            ]

            deleted_count = initial_count - len(self.fallback_store)
            logger.info(f"Deleted {deleted_count} documents by metadata (fallback)")

            return True

    def get_count(self) -> int:
        """
        Get total number of documents in store.

        Returns:
            Number of documents
        """
        if self.chroma_available:
            try:
                return self.collection.count()
            except:
                return 0
        else:
            return len(self.fallback_store)

    def reset(self):
        """Reset the vector store (delete all documents)."""
        if self.chroma_available:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Construction documents RAG system"}
                )
                logger.info("Vector store reset")
            except Exception as e:
                logger.error(f"Error resetting vector store: {str(e)}")
        else:
            self.fallback_store = []
            logger.info("Fallback store reset")
