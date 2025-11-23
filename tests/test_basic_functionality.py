"""
Basic Functionality Tests
==========================

Tests that can run without heavy dependencies.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    try:
        from construction_rag import (
            ConstructionRAG,
            DocumentProcessor,
            IFCParser,
            VectorStore,
            EmbeddingHandler,
            QueryEngine
        )
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_document_processor_basics():
    """Test document processor basic functionality."""
    from construction_rag.document_processor import DocumentProcessor

    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

    # Test initialization
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 100
    print("✓ Document processor initialized")

    # Test text chunking
    text = "This is a test. " * 100
    chunks = processor._chunk_text(text, 500, 100)
    assert len(chunks) > 0
    print(f"✓ Text chunking works ({len(chunks)} chunks created)")

    # Test splitting methods
    para_text = "Para 1.\n\nPara 2.\n\nPara 3."
    para_chunks = processor._split_on_paragraphs(para_text, 50, 10)
    assert len(para_chunks) > 0
    print(f"✓ Paragraph splitting works ({len(para_chunks)} chunks)")

    return True


def test_ifc_parser_basics():
    """Test IFC parser basic functionality."""
    from construction_rag.ifc_parser import IFCParser

    parser = IFCParser()
    assert parser is not None
    print("✓ IFC parser initialized")

    return True


def test_vector_store_basics():
    """Test vector store basic functionality."""
    import tempfile
    import shutil
    from construction_rag.vector_store import VectorStore

    temp_dir = tempfile.mkdtemp()

    try:
        store = VectorStore(db_path=temp_dir, embedding_dimension=384)
        assert store.embedding_dimension == 384
        print("✓ Vector store initialized")

        # Test add documents
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        metadatas = [{"source": f"test{i}"} for i in range(3)]

        ids = store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        assert len(ids) == 3
        print(f"✓ Added {len(ids)} documents to vector store")

        # Test count
        count = store.get_count()
        print(f"✓ Vector store count: {count}")

        # Test search
        query_emb = [0.15] * 384
        results = store.search(query_embedding=query_emb, top_k=2)
        print(f"✓ Search returned {len(results)} results")

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedding_handler_basics():
    """Test embedding handler basic functionality."""
    from construction_rag.embeddings import EmbeddingHandler

    # Test with fallback mode
    handler = EmbeddingHandler(model_name="test")
    handler.provider = "fallback"
    handler.embedding_dimension = 384

    print("✓ Embedding handler initialized (fallback mode)")

    # Test single embedding
    text = "This is a test."
    emb = handler.generate_embeddings(text)
    assert len(emb) == 384
    print(f"✓ Single embedding generated (dim={len(emb)})")

    # Test batch embeddings
    texts = ["Text 1", "Text 2", "Text 3"]
    embs = handler.generate_embeddings(texts)
    assert len(embs) == 3
    assert all(len(e) == 384 for e in embs)
    print(f"✓ Batch embeddings generated ({len(embs)} embeddings)")

    return True


def test_query_engine_basics():
    """Test query engine basic functionality."""
    import tempfile
    import shutil
    from construction_rag.query_engine import QueryEngine
    from construction_rag.vector_store import VectorStore
    from construction_rag.embeddings import EmbeddingHandler

    temp_dir = tempfile.mkdtemp()

    try:
        # Create components
        vector_store = VectorStore(db_path=temp_dir, embedding_dimension=384)
        embedding_handler = EmbeddingHandler(model_name="test")
        embedding_handler.provider = "fallback"
        embedding_handler.embedding_dimension = 384

        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_handler=embedding_handler,
            llm_api_key=None
        )

        print("✓ Query engine initialized")

        # Add test documents
        texts = [
            "Concrete strength is 4000 PSI.",
            "Steel grade is Grade 50.",
            "Foundation depth is 8 feet."
        ]
        embeddings = embedding_handler.generate_embeddings(texts)
        metadatas = [{"file_name": "test.pdf", "page": i+1} for i in range(3)]

        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # Test query
        response = query_engine.query("What is the concrete strength?", top_k=2)
        assert "answer" in response
        print("✓ Query executed successfully")

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_construction_rag_basics():
    """Test main Construction RAG system."""
    import tempfile
    import shutil
    from construction_rag import ConstructionRAG

    temp_dir = tempfile.mkdtemp()

    try:
        rag = ConstructionRAG(
            vector_db_path=temp_dir,
            embedding_model="test",
            chunk_size=500,
            chunk_overlap=100
        )

        # Force fallback mode
        rag.embedding_handler.provider = "fallback"
        rag.embedding_handler.embedding_dimension = 384

        print("✓ Construction RAG system initialized")

        # Test document type detection
        assert rag._detect_document_type(Path("spec.pdf")) == "specification"
        assert rag._detect_document_type(Path("drawing.pdf")) == "drawing"
        assert rag._detect_document_type(Path("model.ifc")) == "ifc_model"
        print("✓ Document type detection works")

        # Test state save/load
        state_file = Path(temp_dir) / "state.json"
        rag.save_state(str(state_file))
        assert state_file.exists()
        print("✓ State saved")

        rag.load_state(str(state_file))
        print("✓ State loaded")

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_utils():
    """Test utility functions."""
    from construction_rag.utils import (
        estimate_tokens,
        format_file_size,
        sanitize_filename,
        chunk_list
    )

    # Test token estimation
    tokens = estimate_tokens("This is a test sentence.")
    assert tokens > 0
    print(f"✓ Token estimation: {tokens} tokens")

    # Test file size formatting
    size_str = format_file_size(1024 * 1024 * 2.5)
    assert "MB" in size_str
    print(f"✓ File size formatting: {size_str}")

    # Test filename sanitization
    clean = sanitize_filename("file<>:name?.pdf")
    assert "<" not in clean
    print(f"✓ Filename sanitization: {clean}")

    # Test list chunking
    lst = list(range(10))
    chunks = chunk_list(lst, 3)
    assert len(chunks) == 4
    print(f"✓ List chunking: {len(chunks)} chunks")

    return True


def run_all_tests():
    """Run all basic tests."""
    print("\n" + "="*80)
    print("Construction RAG System - Basic Functionality Tests")
    print("="*80 + "\n")

    tests = [
        ("Module Imports", test_imports),
        ("Document Processor", test_document_processor_basics),
        ("IFC Parser", test_ifc_parser_basics),
        ("Vector Store", test_vector_store_basics),
        ("Embedding Handler", test_embedding_handler_basics),
        ("Query Engine", test_query_engine_basics),
        ("Construction RAG", test_construction_rag_basics),
        ("Utilities", test_utils),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n--- Testing: {name} ---")
        try:
            success = test_func()
            results.append((name, "PASSED" if success else "FAILED"))
            print(f"✓ {name}: PASSED\n")
        except Exception as e:
            results.append((name, "FAILED"))
            print(f"✗ {name}: FAILED")
            print(f"  Error: {str(e)}\n")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {name}: {status}")

    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
