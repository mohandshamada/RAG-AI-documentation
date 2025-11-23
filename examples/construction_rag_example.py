"""
Construction RAG System - Example Usage
========================================

This example demonstrates how to use the Construction RAG system for
helping construction engineers with daily tasks.

Features demonstrated:
1. Document ingestion (PDFs, IFC files)
2. Querying the system
3. Filtering by document type
4. Batch operations
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from construction_rag import ConstructionRAG


def main():
    print("=" * 80)
    print("Construction RAG System - Example Usage")
    print("=" * 80)

    # Initialize the RAG system
    print("\n1. Initializing Construction RAG System...")
    rag = ConstructionRAG(
        vector_db_path="./construction_vector_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        use_gpu=False,  # Set to True if you have GPU
        llm_api_key=os.getenv("OPENAI_API_KEY"),  # Optional: for LLM-powered answers
        llm_model="gpt-4"
    )

    print("✓ RAG system initialized")

    # Example 1: Ingest a single PDF specification
    print("\n2. Ingesting documents...")
    print("-" * 80)

    # Note: Replace these with actual file paths
    example_files = [
        "path/to/specification.pdf",
        "path/to/drawing.pdf",
        "path/to/building_model.ifc"
    ]

    # Check if example files exist
    available_files = [f for f in example_files if Path(f).exists()]

    if available_files:
        for file_path in available_files:
            try:
                result = rag.ingest_document(file_path)
                print(f"✓ Ingested: {result['file_name']}")
                print(f"  - Type: {result['document_type']}")
                print(f"  - Chunks: {result['num_chunks']}")
                print(f"  - Size: {result['file_size_mb']:.2f} MB")
            except Exception as e:
                print(f"✗ Error ingesting {file_path}: {str(e)}")
    else:
        print("No example files found. Using demo mode...")
        print("To use real files, update the example_files list with actual paths.")

    # Example 2: Ingest a directory of documents
    print("\n3. Ingesting from directory (if available)...")
    print("-" * 80)

    project_dir = "path/to/project/documents"
    if Path(project_dir).exists():
        results = rag.ingest_directory(
            project_dir,
            recursive=True,
            file_extensions=['.pdf', '.ifc']
        )

        successful = [r for r in results if r['status'] == 'success']
        print(f"✓ Ingested {len(successful)} documents from directory")
    else:
        print("Project directory not found. Skipping...")

    # Example 3: Query the system
    print("\n4. Querying the RAG system...")
    print("-" * 80)

    questions = [
        "What are the concrete strength requirements?",
        "What is the fire rating for the main structural elements?",
        "List all the walls on Level 1",
        "What materials are specified for the foundation?",
        "What are the dimensions of the main structural columns?"
    ]

    for question in questions:
        print(f"\nQ: {question}")

        try:
            response = rag.query(
                question=question,
                top_k=5,
                return_sources=True
            )

            print(f"A: {response['answer'][:300]}...")

            if response.get('sources'):
                print(f"\nSources ({len(response['sources'])}):")
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"  {i}. {source['file_name']} (Page {source.get('page', 'N/A')})")

        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 80)

    # Example 4: Semantic search (without LLM)
    print("\n5. Semantic search example...")
    print("-" * 80)

    search_query = "steel beam specifications"
    print(f"Searching for: {search_query}")

    try:
        results = rag.search(
            query=search_query,
            top_k=5
        )

        print(f"\nFound {len(results)} relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata'].get('file_name', 'Unknown')}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Preview: {result['text'][:150]}...")

    except Exception as e:
        print(f"Error: {str(e)}")

    # Example 5: Filter by document type
    print("\n6. Filtering by document type...")
    print("-" * 80)

    try:
        # Query only specifications
        response = rag.query(
            question="What are the quality control requirements?",
            top_k=5,
            filter_metadata={"document_type": "specification"}
        )

        print(f"Answer (from specifications only):")
        print(response['answer'][:300] + "...")

    except Exception as e:
        print(f"Error: {str(e)}")

    # Example 6: Get document information
    print("\n7. Document information...")
    print("-" * 80)

    docs = rag.get_document_info()
    print(f"Total documents ingested: {len(docs)}")

    for doc in docs[:5]:  # Show first 5
        print(f"  - {doc['file_name']}: {doc['num_chunks']} chunks ({doc['document_type']})")

    # Example 7: Save and load state
    print("\n8. Saving system state...")
    print("-" * 80)

    try:
        rag.save_state()
        print("✓ State saved successfully")
        print("  You can reload this state later with rag.load_state()")
    except Exception as e:
        print(f"Error saving state: {str(e)}")

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


def demo_without_files():
    """
    Demonstration mode without actual files.
    Shows how to use the API.
    """
    print("\n" + "=" * 80)
    print("DEMO MODE - API Usage Examples")
    print("=" * 80)

    print("\n# Initialize RAG system")
    print("""
rag = ConstructionRAG(
    vector_db_path="./construction_vector_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200
)
    """)

    print("\n# Ingest a PDF specification")
    print("""
result = rag.ingest_document(
    "specifications/concrete_specs.pdf",
    document_type="specification"
)
    """)

    print("\n# Ingest an IFC model")
    print("""
result = rag.ingest_document(
    "models/building_model.ifc",
    metadata={"project": "Hospital Building", "phase": "Design"}
)
    """)

    print("\n# Ingest a directory")
    print("""
results = rag.ingest_directory(
    "project_documents/",
    recursive=True
)
    """)

    print("\n# Query the system")
    print("""
response = rag.query(
    "What are the concrete strength requirements for the foundation?",
    top_k=5,
    return_sources=True
)
print(response['answer'])
    """)

    print("\n# Semantic search")
    print("""
results = rag.search(
    "fire resistance requirements",
    top_k=10
)
for result in results:
    print(result['text'])
    """)

    print("\n# Filter by document type")
    print("""
response = rag.query(
    "List all columns on Level 3",
    filter_metadata={"document_type": "ifc_model"}
)
    """)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Check if we have any real files to work with
    has_real_files = False

    # Try to detect if we're in a real project environment
    potential_dirs = ["documents", "specs", "drawings", "models"]
    for dir_name in potential_dirs:
        if Path(dir_name).exists():
            has_real_files = True
            break

    if has_real_files:
        main()
    else:
        print("\nNo document directories found. Running in demo mode...")
        demo_without_files()
        print("\n\nTo use with real files, update the example_files list in the script.")
