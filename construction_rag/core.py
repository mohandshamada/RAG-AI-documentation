"""
Core Construction RAG System
=============================

Main class that orchestrates the entire RAG pipeline for construction documents.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

from .document_processor import DocumentProcessor
from .ifc_parser import IFCParser
from .vector_store import VectorStore
from .embeddings import EmbeddingHandler
from .query_engine import QueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstructionRAG:
    """
    Complete RAG system for construction engineering documents.

    Handles ingestion, processing, storage, and retrieval of:
    - PDF specifications
    - PDF drawings
    - IFC/BIM files

    Features:
    - Automatic document type detection
    - Chunking strategy for large files
    - Vector embeddings with multiple model support
    - Semantic search and retrieval
    - Construction domain-specific processing
    """

    def __init__(
        self,
        vector_db_path: str = "./construction_vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_gpu: bool = False,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4",
    ):
        """
        Initialize Construction RAG system.

        Args:
            vector_db_path: Path to store vector database
            embedding_model: Model name for embeddings (sentence-transformers, openai, etc.)
            chunk_size: Maximum size of text chunks (tokens/characters)
            chunk_overlap: Overlap between chunks for context preservation
            use_gpu: Whether to use GPU for embeddings (if available)
            llm_api_key: API key for LLM service (OpenAI, Anthropic, etc.)
            llm_model: LLM model to use for generation
        """
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info("Initializing Construction RAG System...")

        # Initialize components
        self.embedding_handler = EmbeddingHandler(
            model_name=embedding_model,
            use_gpu=use_gpu
        )

        self.vector_store = VectorStore(
            db_path=str(self.vector_db_path),
            embedding_dimension=self.embedding_handler.get_dimension()
        )

        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.ifc_parser = IFCParser()

        self.query_engine = QueryEngine(
            vector_store=self.vector_store,
            embedding_handler=self.embedding_handler,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        )

        self.ingested_documents = {}

        logger.info("Construction RAG System initialized successfully")

    def ingest_document(
        self,
        file_path: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a construction document into the RAG system.

        Args:
            file_path: Path to the document
            document_type: Type of document (specification, drawing, ifc, or auto-detect)
            metadata: Additional metadata to store with document

        Returns:
            Dictionary with ingestion results and statistics
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"Ingesting document: {file_path.name}")

        # Auto-detect document type if not provided
        if document_type is None:
            document_type = self._detect_document_type(file_path)

        # Prepare metadata
        doc_metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower(),
            "document_type": document_type,
            "ingestion_date": datetime.now().isoformat(),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        }

        if metadata:
            doc_metadata.update(metadata)

        # Process document based on type
        if file_path.suffix.lower() == '.ifc':
            chunks = self._process_ifc_file(file_path, doc_metadata)
        elif file_path.suffix.lower() == '.pdf':
            chunks = self._process_pdf_file(file_path, doc_metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_handler.generate_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        # Store in vector database
        logger.info("Storing in vector database...")
        doc_id = str(file_path)
        chunk_ids = self.vector_store.add_documents(
            texts=[chunk["text"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=[{**doc_metadata, **chunk.get("metadata", {})} for chunk in chunks],
            document_id=doc_id
        )

        # Track ingested document
        self.ingested_documents[doc_id] = {
            "file_name": file_path.name,
            "document_type": document_type,
            "num_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "metadata": doc_metadata,
            "ingestion_date": doc_metadata["ingestion_date"]
        }

        result = {
            "status": "success",
            "file_name": file_path.name,
            "document_id": doc_id,
            "document_type": document_type,
            "num_chunks": len(chunks),
            "file_size_mb": doc_metadata["file_size_mb"]
        }

        logger.info(f"Successfully ingested {file_path.name} ({len(chunks)} chunks)")

        return result

    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest all supported documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            file_extensions: List of file extensions to process (default: ['.pdf', '.ifc'])

        Returns:
            List of ingestion results for each document
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.ifc']

        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        results = []

        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    result = self.ingest_document(str(file_path))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path.name}: {str(e)}")
                    results.append({
                        "status": "failed",
                        "file_name": file_path.name,
                        "error": str(e)
                    })

        logger.info(f"Ingested {len([r for r in results if r['status'] == 'success'])} documents from {directory}")

        return results

    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a construction-related question.

        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            filter_metadata: Filter results by metadata (e.g., document_type, file_name)
            return_sources: Whether to include source documents in response

        Returns:
            Dictionary containing answer and optionally source information
        """
        logger.info(f"Processing query: {question[:100]}...")

        response = self.query_engine.query(
            question=question,
            top_k=top_k,
            filter_metadata=filter_metadata,
            return_sources=return_sources
        )

        return response

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search without LLM generation.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Filter results by metadata

        Returns:
            List of relevant document chunks with metadata and scores
        """
        query_embedding = self.embedding_handler.generate_embeddings([query])[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        return results

    def get_document_info(self, document_id: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Get information about ingested documents.

        Args:
            document_id: Specific document ID, or None for all documents

        Returns:
            Document information dictionary or list of all documents
        """
        if document_id:
            return self.ingested_documents.get(document_id, {})
        else:
            return list(self.ingested_documents.values())

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the RAG system.

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        if document_id not in self.ingested_documents:
            logger.warning(f"Document not found: {document_id}")
            return False

        doc_info = self.ingested_documents[document_id]
        chunk_ids = doc_info.get("chunk_ids", [])

        # Delete from vector store
        self.vector_store.delete_by_ids(chunk_ids)

        # Remove from tracking
        del self.ingested_documents[document_id]

        logger.info(f"Deleted document: {doc_info['file_name']}")

        return True

    def _detect_document_type(self, file_path: Path) -> str:
        """
        Auto-detect document type based on file extension and name.
        """
        file_name = file_path.name.lower()
        extension = file_path.suffix.lower()

        if extension == '.ifc':
            return 'ifc_model'

        # PDF heuristics
        if 'spec' in file_name or 'specification' in file_name:
            return 'specification'
        elif 'draw' in file_name or 'plan' in file_name or 'sheet' in file_name:
            return 'drawing'
        else:
            return 'document'

    def _process_pdf_file(self, file_path: Path, metadata: Dict) -> List[Dict]:
        """
        Process PDF file and return chunks.
        """
        return self.document_processor.process_pdf(str(file_path), metadata)

    def _process_ifc_file(self, file_path: Path, metadata: Dict) -> List[Dict]:
        """
        Process IFC file and return chunks.
        """
        return self.ifc_parser.parse(str(file_path), metadata)

    def save_state(self, save_path: Optional[str] = None):
        """
        Save the current state of the RAG system.

        Args:
            save_path: Path to save state (default: vector_db_path/state.json)
        """
        if save_path is None:
            save_path = self.vector_db_path / "state.json"

        import json

        state = {
            "ingested_documents": self.ingested_documents,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_handler.model_name,
            }
        }

        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved to {save_path}")

    def load_state(self, load_path: Optional[str] = None):
        """
        Load a previously saved state.

        Args:
            load_path: Path to load state from (default: vector_db_path/state.json)
        """
        if load_path is None:
            load_path = self.vector_db_path / "state.json"

        import json

        if not Path(load_path).exists():
            logger.warning(f"State file not found: {load_path}")
            return

        with open(load_path, 'r') as f:
            state = json.load(f)

        self.ingested_documents = state.get("ingested_documents", {})

        logger.info(f"State loaded from {load_path}")
