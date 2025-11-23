"""
Construction RAG System
=======================

A complete RAG (Retrieval-Augmented Generation) system designed for construction engineers.
Supports:
- PDF specifications and drawings
- IFC (Industry Foundation Classes) files for BIM data
- Large file processing with chunking
- Vector database storage and retrieval
- Embeddings generation
- Construction-specific document understanding

Usage:
    from construction_rag import ConstructionRAG

    rag = ConstructionRAG(
        vector_db_path="./vector_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Ingest documents
    rag.ingest_document("specifications.pdf")
    rag.ingest_document("drawing.pdf")
    rag.ingest_document("building_model.ifc")

    # Query
    response = rag.query("What are the concrete strength requirements?")
"""

from .core import ConstructionRAG
from .ifc_parser import IFCParser
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .embeddings import EmbeddingHandler
from .query_engine import QueryEngine

__version__ = "1.0.0"
__all__ = [
    "ConstructionRAG",
    "IFCParser",
    "DocumentProcessor",
    "VectorStore",
    "EmbeddingHandler",
    "QueryEngine",
]
