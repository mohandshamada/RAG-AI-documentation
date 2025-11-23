"""
Comprehensive Test Suite for Construction RAG System
=====================================================

Tests all components of the Construction RAG system.
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from construction_rag import (
    ConstructionRAG,
    DocumentProcessor,
    IFCParser,
    VectorStore,
    EmbeddingHandler,
    QueryEngine
)


class TestDocumentProcessor:
    """Test PDF document processing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

    def test_initialization(self):
        """Test document processor initialization."""
        assert self.processor.chunk_size == 500
        assert self.processor.chunk_overlap == 100

    def test_chunk_text_simple(self):
        """Test simple text chunking."""
        text = "This is a test. " * 100  # ~1600 characters
        chunks = self.processor._chunk_text(text, 500, 100)

        assert len(chunks) > 1
        assert all(len(chunk) <= 600 for chunk in chunks)  # Allow some overhead

    def test_chunk_text_with_headers(self):
        """Test chunking with markdown headers."""
        text = """
# Header 1
This is content for header 1.

## Header 2
This is content for header 2.

### Header 3
This is content for header 3.
        """
        chunks = self.processor._chunk_text(text, 500, 100)

        # Should preserve headers with content
        assert any('Header 1' in chunk for chunk in chunks)

    def test_split_on_paragraphs(self):
        """Test paragraph-based splitting."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = self.processor._split_on_paragraphs(text, 50, 10)

        assert len(chunks) > 0

    def test_split_on_sentences(self):
        """Test sentence-based splitting."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = self.processor._split_on_sentences(text, 30, 5)

        assert len(chunks) > 0


class TestIFCParser:
    """Test IFC file parsing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = IFCParser()

    def test_initialization(self):
        """Test IFC parser initialization."""
        assert self.parser is not None

    def test_fallback_parsing(self):
        """Test fallback text-based parsing."""
        # Create a temporary fake IFC file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ifc', delete=False) as f:
            f.write("ISO-10303-21;\n")
            f.write("HEADER;\n")
            f.write("FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');\n")
            f.write("ENDSEC;\n")
            f.write("DATA;\n")
            f.write("#1=IFCPROJECT('3vB2YO$MX4xv5uCqZZG05S',$,'Test Project',$,$,$,$,$,$);\n")
            f.write("#2=IFCWALL('2vB2YO$MX4xv5uCqZZG05T',$,'Wall-001',$,$,$,$,$,$);\n")
            f.write("ENDSEC;\n")
            f.write("END-ISO-10303-21;\n")
            temp_file = f.name

        try:
            chunks = self.parser._parse_fallback(temp_file, {})

            assert len(chunks) > 0
            assert any('IFC' in chunk['text'] for chunk in chunks)

        finally:
            Path(temp_file).unlink()


class TestEmbeddingHandler:
    """Test embedding generation."""

    def setup_method(self):
        """Setup test fixtures."""
        # Use fallback mode for testing (no dependencies required)
        self.handler = EmbeddingHandler(model_name="fallback-test")
        self.handler.provider = "fallback"
        self.handler.embedding_dimension = 384

    def test_initialization(self):
        """Test embedding handler initialization."""
        assert self.handler.embedding_dimension == 384

    def test_single_embedding(self):
        """Test generating single embedding."""
        text = "This is a test sentence."
        embedding = self.handler.generate_embeddings(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_batch_embeddings(self):
        """Test generating batch embeddings."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = self.handler.generate_embeddings(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_fallback_reproducibility(self):
        """Test that fallback embeddings are reproducible."""
        text = "Test reproducibility."
        emb1 = self.handler._generate_fallback([text])[0]
        emb2 = self.handler._generate_fallback([text])[0]

        assert emb1 == emb2


class TestVectorStore:
    """Test vector database operations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = VectorStore(
            db_path=self.temp_dir,
            embedding_dimension=384
        )

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test vector store initialization."""
        assert self.store.embedding_dimension == 384

    def test_add_documents(self):
        """Test adding documents."""
        texts = ["Document 1", "Document 2", "Document 3"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        metadatas = [
            {"source": "test1"},
            {"source": "test2"},
            {"source": "test3"}
        ]

        ids = self.store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            document_id="test_doc"
        )

        assert len(ids) == 3
        assert all("test_doc" in id for id in ids)

    def test_search(self):
        """Test similarity search."""
        # Add documents
        texts = ["Construction concrete", "Steel beams", "Foundation design"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        self.store.add_documents(texts=texts, embeddings=embeddings)

        # Search
        query_embedding = [0.15] * 384
        results = self.store.search(query_embedding=query_embedding, top_k=2)

        assert len(results) <= 2

    def test_get_count(self):
        """Test document count."""
        initial_count = self.store.get_count()

        texts = ["Doc 1", "Doc 2"]
        embeddings = [[0.1] * 384, [0.2] * 384]
        self.store.add_documents(texts=texts, embeddings=embeddings)

        new_count = self.store.get_count()
        assert new_count == initial_count + 2

    def test_delete_by_ids(self):
        """Test deleting documents by ID."""
        texts = ["Doc to delete"]
        embeddings = [[0.1] * 384]

        ids = self.store.add_documents(
            texts=texts,
            embeddings=embeddings,
            document_id="delete_test"
        )

        initial_count = self.store.get_count()
        success = self.store.delete_by_ids(ids)

        assert success
        assert self.store.get_count() == initial_count - 1


class TestQueryEngine:
    """Test query engine."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create minimal components
        self.vector_store = VectorStore(
            db_path=self.temp_dir,
            embedding_dimension=384
        )

        self.embedding_handler = EmbeddingHandler(model_name="fallback-test")
        self.embedding_handler.provider = "fallback"
        self.embedding_handler.embedding_dimension = 384

        self.query_engine = QueryEngine(
            vector_store=self.vector_store,
            embedding_handler=self.embedding_handler,
            llm_api_key=None  # No LLM for testing
        )

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test query engine initialization."""
        assert self.query_engine.llm_available == False

    def test_query_without_llm(self):
        """Test querying without LLM."""
        # Add some test documents
        texts = [
            "Concrete strength requirement is 3000 PSI.",
            "Steel beams must be Grade 50.",
            "Foundation depth is 8 feet."
        ]
        embeddings = self.embedding_handler.generate_embeddings(texts)
        metadatas = [
            {"file_name": "spec.pdf", "page": 1},
            {"file_name": "spec.pdf", "page": 2},
            {"file_name": "spec.pdf", "page": 3}
        ]

        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # Query
        response = self.query_engine.query(
            question="What is the concrete strength?",
            top_k=3
        )

        assert "answer" in response
        assert response["context_used"] == True

    def test_assemble_context(self):
        """Test context assembly."""
        results = [
            {
                "text": "Test content 1",
                "metadata": {"file_name": "test.pdf", "page": 1}
            },
            {
                "text": "Test content 2",
                "metadata": {"file_name": "test.pdf", "page": 2}
            }
        ]

        context = self.query_engine._assemble_context(results)

        assert "Test content 1" in context
        assert "Test content 2" in context
        assert "test.pdf" in context


class TestConstructionRAG:
    """Test main Construction RAG system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.rag = ConstructionRAG(
            vector_db_path=self.temp_dir,
            embedding_model="fallback-test",
            chunk_size=500,
            chunk_overlap=100
        )

        # Force fallback mode for testing
        self.rag.embedding_handler.provider = "fallback"
        self.rag.embedding_handler.embedding_dimension = 384

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test RAG system initialization."""
        assert self.rag.chunk_size == 500
        assert self.rag.chunk_overlap == 100

    def test_document_type_detection(self):
        """Test automatic document type detection."""
        spec_type = self.rag._detect_document_type(Path("concrete_specification.pdf"))
        assert spec_type == "specification"

        drawing_type = self.rag._detect_document_type(Path("floor_plan_drawing.pdf"))
        assert drawing_type == "drawing"

        ifc_type = self.rag._detect_document_type(Path("building.ifc"))
        assert ifc_type == "ifc_model"

    def test_search_empty_db(self):
        """Test searching empty database."""
        results = self.rag.search("test query", top_k=5)
        assert len(results) == 0

    def test_get_document_info_empty(self):
        """Test getting document info when empty."""
        docs = self.rag.get_document_info()
        assert len(docs) == 0

    def test_state_save_load(self):
        """Test saving and loading state."""
        state_file = Path(self.temp_dir) / "test_state.json"

        self.rag.save_state(str(state_file))
        assert state_file.exists()

        self.rag.load_state(str(state_file))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_chunking(self):
        """Test chunking empty text."""
        processor = DocumentProcessor()
        chunks = processor._chunk_text("", 500, 100)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_very_small_chunk_size(self):
        """Test with very small chunk size."""
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
        text = "This is a test sentence."
        chunks = processor._chunk_text(text, 10, 2)
        assert len(chunks) > 0

    def test_large_chunk_size(self):
        """Test with chunk size larger than text."""
        processor = DocumentProcessor(chunk_size=10000, chunk_overlap=100)
        text = "Short text."
        chunks = processor._chunk_text(text, 10000, 100)
        assert len(chunks) == 1

    def test_vector_store_empty_add(self):
        """Test adding empty document list."""
        temp_dir = tempfile.mkdtemp()
        try:
            store = VectorStore(db_path=temp_dir, embedding_dimension=384)
            ids = store.add_documents(texts=[], embeddings=[])
            assert len(ids) == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_embedding_handler_empty_text(self):
        """Test generating embeddings for empty text."""
        handler = EmbeddingHandler(model_name="fallback-test")
        handler.provider = "fallback"
        handler.embedding_dimension = 384

        embedding = handler.generate_embeddings("")
        assert len(embedding) == 384


class TestIntegration:
    """Integration tests for the complete system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.rag = ConstructionRAG(
            vector_db_path=self.temp_dir,
            embedding_model="fallback-test",
            chunk_size=200,
            chunk_overlap=50
        )
        # Force fallback mode
        self.rag.embedding_handler.provider = "fallback"
        self.rag.embedding_handler.embedding_dimension = 384

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_sample_pdf_and_ingest(self):
        """Test creating a sample PDF and ingesting it."""
        # Create a simple text file instead of PDF for testing
        test_file = Path(self.temp_dir) / "test_spec.txt"
        test_file.write_text("Concrete strength: 3000 PSI\nSteel grade: Grade 50")

        # We can't easily test PDF ingestion without actual PDF files
        # but we can test the workflow with our mock data

    def test_full_workflow(self):
        """Test complete workflow: ingest -> query -> delete."""
        # Simulate document ingestion by adding chunks directly
        texts = [
            "The concrete strength requirement is 4000 PSI for all structural elements.",
            "Steel reinforcement must be Grade 60 with proper cover.",
            "Foundation design requires soil bearing capacity of 3000 PSF."
        ]

        embeddings = self.rag.embedding_handler.generate_embeddings(texts)
        metadatas = [
            {
                "file_name": "test_spec.pdf",
                "page": i+1,
                "document_type": "specification"
            }
            for i in range(len(texts))
        ]

        doc_id = "test_document_1"
        chunk_ids = self.rag.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            document_id=doc_id
        )

        # Track in RAG system
        self.rag.ingested_documents[doc_id] = {
            "file_name": "test_spec.pdf",
            "document_type": "specification",
            "num_chunks": len(texts),
            "chunk_ids": chunk_ids
        }

        # Test query
        response = self.rag.query("What is the concrete strength?", top_k=3)
        assert "answer" in response

        # Test search
        results = self.rag.search("steel reinforcement", top_k=2)
        assert len(results) > 0

        # Test document info
        docs = self.rag.get_document_info()
        assert len(docs) == 1

        # Test delete
        success = self.rag.delete_document(doc_id)
        assert success

        docs_after = self.rag.get_document_info()
        assert len(docs_after) == 0


def run_tests():
    """Run all tests and report results."""
    print("="*80)
    print("Running Construction RAG System Tests")
    print("="*80)

    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])


if __name__ == "__main__":
    run_tests()
