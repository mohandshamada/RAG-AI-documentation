"""
Manual Testing Script
=====================

Run manual tests on the Construction RAG system with sample data.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("Construction RAG System - Manual Testing")
print("="*80)

# Test 1: Check imports
print("\n[Test 1] Checking imports...")
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
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize components
print("\n[Test 2] Initializing components...")
temp_dir = tempfile.mkdtemp()
print(f"Using temporary directory: {temp_dir}")

try:
    # Document Processor
    print("\n  [2.1] Document Processor...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    test_text = """
# Construction Specification

## Concrete Requirements

The concrete strength for all structural elements shall be 4000 PSI minimum.
All concrete shall be properly cured for at least 7 days.

### Reinforcement

Steel reinforcement shall be Grade 60 with appropriate cover.

## Foundation

Foundation depth shall be minimum 8 feet below grade.
Soil bearing capacity assumed at 3000 PSF.
    """

    chunks = processor._chunk_text(test_text, 300, 50)
    print(f"  ✓ Created {len(chunks)} chunks from test text")

    # Embedding Handler
    print("\n  [2.2] Embedding Handler...")
    embedder = EmbeddingHandler(model_name="test-model")
    embedder.provider = "fallback"
    embedder.embedding_dimension = 384
    print("  ✓ Embedding handler initialized (fallback mode)")

    # Vector Store
    print("\n  [2.3] Vector Store...")
    vector_store = VectorStore(
        db_path=temp_dir,
        embedding_dimension=384
    )
    print("  ✓ Vector store initialized")

    # Add sample documents
    print("\n  [2.4] Adding sample documents...")
    sample_docs = [
        "Concrete strength requirement is 4000 PSI for all structural elements.",
        "Steel reinforcement must be Grade 60 with proper cover.",
        "Foundation depth shall be minimum 8 feet below grade.",
        "Soil bearing capacity is 3000 PSF.",
        "All concrete must be cured for at least 7 days.",
    ]

    embeddings = embedder.generate_embeddings(sample_docs)
    metadatas = [
        {"file_name": "spec.pdf", "page": 1, "document_type": "specification"},
        {"file_name": "spec.pdf", "page": 2, "document_type": "specification"},
        {"file_name": "spec.pdf", "page": 3, "document_type": "specification"},
        {"file_name": "foundation_spec.pdf", "page": 1, "document_type": "specification"},
        {"file_name": "spec.pdf", "page": 1, "document_type": "specification"},
    ]

    ids = vector_store.add_documents(
        texts=sample_docs,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"  ✓ Added {len(ids)} documents to vector store")

    # Test 3: Query Engine
    print("\n[Test 3] Testing Query Engine...")
    query_engine = QueryEngine(
        vector_store=vector_store,
        embedding_handler=embedder,
        llm_api_key=None
    )
    print("  ✓ Query engine initialized")

    # Test query
    test_queries = [
        "What is the concrete strength requirement?",
        "What is the required steel grade?",
        "How deep should the foundation be?",
        "What is the soil bearing capacity?",
    ]

    for query in test_queries:
        response = query_engine.query(query, top_k=3)
        print(f"\n  Q: {query}")
        print(f"  A: {response['answer'][:150]}...")

    # Test 4: Full RAG System
    print("\n[Test 4] Testing Full RAG System...")
    rag = ConstructionRAG(
        vector_db_path=temp_dir + "_rag",
        embedding_model="test-model",
        chunk_size=500,
        chunk_overlap=100
    )

    # Force fallback mode
    rag.embedding_handler.provider = "fallback"
    rag.embedding_handler.embedding_dimension = 384

    print("  ✓ RAG system initialized")

    # Simulate document ingestion
    print("\n  [4.1] Simulating document ingestion...")
    construction_docs = [
        "The building shall have reinforced concrete columns spaced at 25 feet on center.",
        "All structural steel shall conform to ASTM A992 Grade 50.",
        "Exterior walls shall be 8-inch CMU with #5 reinforcement at 24 inches on center.",
        "Floor slabs shall be 6-inch thick with WWF6x6 W2.9xW2.9 reinforcement.",
        "The foundation system consists of spread footings bearing on undisturbed soil.",
    ]

    embs = rag.embedding_handler.generate_embeddings(construction_docs)
    metas = [
        {"file_name": "structural_specs.pdf", "page": i+1, "document_type": "specification"}
        for i in range(len(construction_docs))
    ]

    doc_id = "structural_specs_001"
    chunk_ids = rag.vector_store.add_documents(
        texts=construction_docs,
        embeddings=embs,
        metadatas=metas,
        document_id=doc_id
    )

    rag.ingested_documents[doc_id] = {
        "file_name": "structural_specs.pdf",
        "document_type": "specification",
        "num_chunks": len(construction_docs),
        "chunk_ids": chunk_ids
    }

    print(f"  ✓ Ingested {len(construction_docs)} chunks")

    # Test search
    print("\n  [4.2] Testing search...")
    search_queries = [
        "column spacing",
        "steel grade",
        "wall reinforcement",
        "slab thickness",
    ]

    for query in search_queries:
        results = rag.search(query, top_k=2)
        print(f"\n  Search: '{query}'")
        print(f"  Results: {len(results)} found")
        if results:
            print(f"  Top result: {results[0]['text'][:100]}...")

    # Test 5: Document Management
    print("\n[Test 5] Testing Document Management...")
    docs = rag.get_document_info()
    print(f"  ✓ Retrieved info for {len(docs)} documents")

    # Test state save/load
    print("\n  [5.1] Testing state save/load...")
    state_file = Path(temp_dir) / "test_state.json"
    rag.save_state(str(state_file))
    print(f"  ✓ State saved to {state_file}")

    rag.load_state(str(state_file))
    print("  ✓ State loaded successfully")

    # Test delete
    print("\n  [5.2] Testing document deletion...")
    initial_count = rag.vector_store.get_count()
    success = rag.delete_document(doc_id)
    final_count = rag.vector_store.get_count()

    print(f"  ✓ Deleted document (count: {initial_count} → {final_count})")

    # Test 6: Edge Cases
    print("\n[Test 6] Testing Edge Cases...")

    # Empty query
    print("  [6.1] Empty database query...")
    empty_results = rag.search("test", top_k=5)
    print(f"  ✓ Empty database returned {len(empty_results)} results")

    # Large text chunking
    print("  [6.2] Large text chunking...")
    large_text = "This is a test sentence. " * 1000
    large_chunks = processor._chunk_text(large_text, 500, 100)
    print(f"  ✓ Large text split into {len(large_chunks)} chunks")

    # Small chunk size
    print("  [6.3] Small chunk size...")
    small_chunks = processor._chunk_text("Test text.", 5, 1)
    print(f"  ✓ Small chunk size produced {len(small_chunks)} chunks")

    # Test 7: IFC Parser
    print("\n[Test 7] Testing IFC Parser...")
    ifc_parser = IFCParser()
    print("  ✓ IFC parser initialized")

    # Create fake IFC content
    fake_ifc_path = Path(temp_dir) / "test.ifc"
    fake_ifc_path.write_text("""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('test.ifc','2024-01-01T00:00:00',('Test'),'Test Org','TestApp','','');
FILE_SCHEMA(('IFC2X3'));
ENDSEC;
DATA;
#1=IFCPROJECT('123456',$,'Test Building Project',$,$,$,$,$,$);
#2=IFCWALL('234567',$,'Wall-External-001',$,$,$,$,$,$);
#3=IFCSLAB('345678',$,'Slab-Floor-L1',$,$,$,$,$);
#4=IFCCOLUMN('456789',$,'Column-C1',$,$,$,$,$);
ENDSEC;
END-ISO-10303-21;
""")

    ifc_chunks = ifc_parser.parse(str(fake_ifc_path), {"test": "data"})
    print(f"  ✓ IFC parser extracted {len(ifc_chunks)} chunks")

    if ifc_chunks:
        print(f"  Sample chunk: {ifc_chunks[0]['text'][:100]}...")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print("✓ All manual tests completed successfully!")
    print("\nTested components:")
    print("  ✓ Document Processor (chunking, splitting)")
    print("  ✓ Embedding Handler (fallback mode)")
    print("  ✓ Vector Store (add, search, delete)")
    print("  ✓ Query Engine (query, context assembly)")
    print("  ✓ Construction RAG (full system)")
    print("  ✓ IFC Parser (text parsing)")
    print("  ✓ State Management (save/load)")
    print("  ✓ Edge Cases (empty queries, large text, etc.)")
    print("="*80)

except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    if Path(temp_dir + "_rag").exists():
        shutil.rmtree(temp_dir + "_rag", ignore_errors=True)

print("\n✓ Cleanup complete")
sys.exit(0)
