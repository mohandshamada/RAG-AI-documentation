#!/usr/bin/env python3
"""
Construction RAG Command Line Interface
========================================

Simple CLI for the Construction RAG system.

Usage:
    # Ingest documents
    python construction_rag_cli.py ingest /path/to/documents

    # Query the system
    python construction_rag_cli.py query "What are the concrete requirements?"

    # Interactive mode
    python construction_rag_cli.py interactive
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from construction_rag import ConstructionRAG


def cmd_ingest(args):
    """Ingest documents into the RAG system."""
    print("Initializing Construction RAG system...")

    rag = ConstructionRAG(
        vector_db_path=args.db_path,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_gpu=args.gpu,
        llm_api_key=os.getenv("OPENAI_API_KEY") or args.api_key,
        llm_model=args.llm_model
    )

    path = Path(args.path)

    if path.is_file():
        print(f"\nIngesting file: {path.name}")
        result = rag.ingest_document(str(path))

        if result['status'] == 'success':
            print(f"✓ Success!")
            print(f"  - Chunks: {result['num_chunks']}")
            print(f"  - Type: {result['document_type']}")
            print(f"  - Size: {result['file_size_mb']:.2f} MB")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")

    elif path.is_dir():
        print(f"\nIngesting directory: {path}")
        results = rag.ingest_directory(
            str(path),
            recursive=args.recursive
        )

        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        print(f"\n{'='*60}")
        print(f"✓ Successfully ingested: {len(successful)} documents")
        if failed:
            print(f"✗ Failed: {len(failed)} documents")

        total_chunks = sum(r['num_chunks'] for r in successful)
        print(f"Total chunks: {total_chunks}")
        print(f"{'='*60}")

    else:
        print(f"Error: {path} is not a valid file or directory")
        return

    # Save state
    rag.save_state()
    print("\n✓ State saved")


def cmd_query(args):
    """Query the RAG system."""
    print("Initializing Construction RAG system...")

    rag = ConstructionRAG(
        vector_db_path=args.db_path,
        embedding_model=args.embedding_model,
        use_gpu=args.gpu,
        llm_api_key=os.getenv("OPENAI_API_KEY") or args.api_key,
        llm_model=args.llm_model
    )

    # Load previous state
    rag.load_state()

    print(f"\nQuery: {args.question}")
    print(f"{'='*60}\n")

    response = rag.query(
        question=args.question,
        top_k=args.top_k,
        return_sources=args.show_sources
    )

    print(f"Answer:\n{response['answer']}\n")

    if args.show_sources and response.get('sources'):
        print(f"{'='*60}")
        print(f"Sources ({len(response['sources'])}):")
        for i, source in enumerate(response['sources'], 1):
            print(f"\n{i}. {source['file_name']}")
            print(f"   Type: {source['document_type']}")
            print(f"   Page: {source.get('page', 'N/A')}")
            print(f"   Score: {source.get('relevance_score', 'N/A')}")


def cmd_interactive(args):
    """Interactive query mode."""
    print("Initializing Construction RAG system...")

    rag = ConstructionRAG(
        vector_db_path=args.db_path,
        embedding_model=args.embedding_model,
        use_gpu=args.gpu,
        llm_api_key=os.getenv("OPENAI_API_KEY") or args.api_key,
        llm_model=args.llm_model
    )

    # Load previous state
    rag.load_state()

    docs = rag.get_document_info()
    print(f"\n{'='*60}")
    print(f"Loaded {len(docs)} documents")
    print(f"{'='*60}\n")

    print("Interactive mode - Type 'quit' or 'exit' to quit\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not question:
                continue

            print(f"\n{'='*60}\n")

            response = rag.query(
                question=question,
                top_k=args.top_k,
                return_sources=True
            )

            print(f"{response['answer']}\n")

            if response.get('sources'):
                print(f"Sources: {len(response['sources'])}")
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"  {i}. {source['file_name']} (Page {source.get('page', 'N/A')})")

            print(f"\n{'='*60}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


def cmd_info(args):
    """Show information about the RAG system."""
    print("Loading Construction RAG system...")

    rag = ConstructionRAG(
        vector_db_path=args.db_path,
        embedding_model=args.embedding_model,
        use_gpu=args.gpu
    )

    # Load previous state
    rag.load_state()

    docs = rag.get_document_info()

    print(f"\n{'='*60}")
    print(f"Construction RAG System Information")
    print(f"{'='*60}\n")

    print(f"Database Path: {args.db_path}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Total Documents: {len(docs)}\n")

    if docs:
        print("Documents:")
        for doc in docs:
            print(f"  - {doc['file_name']}")
            print(f"    Type: {doc['document_type']}")
            print(f"    Chunks: {doc['num_chunks']}")
            print(f"    Ingested: {doc['ingestion_date']}")
            print()

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Construction RAG System - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global arguments
    parser.add_argument(
        "--db-path",
        default="./construction_vector_db",
        help="Path to vector database (default: ./construction_vector_db)"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available"
    )
    parser.add_argument(
        "--api-key",
        help="LLM API key (or set OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4",
        help="LLM model to use (default: gpt-4)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory to ingest")
    ingest_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively process directories (default: True)"
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)"
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve (default: 5)"
    )
    query_parser.add_argument(
        "--show-sources",
        action="store_true",
        default=True,
        help="Show source information (default: True)"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Interactive query mode"
    )
    interactive_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve (default: 5)"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
