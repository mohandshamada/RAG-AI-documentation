# Construction RAG System

A complete RAG (Retrieval-Augmented Generation) system designed specifically for construction engineers to help with daily work involving specifications, drawings, and BIM data.

## Features

### Document Support
- **PDF Specifications**: Automatic extraction of text, tables, and structure
- **PDF Drawings**: Image and annotation extraction
- **IFC Files**: Complete BIM data parsing including:
  - Building elements (walls, slabs, columns, beams, etc.)
  - Properties and property sets
  - Materials and specifications
  - Spatial structure
  - Quantities and measurements

### Large File Handling
- Smart chunking strategy with configurable size and overlap
- Streaming processing for memory efficiency
- Preserves document structure (headers, sections)
- Batch processing for multiple documents

### Advanced RAG Capabilities
- Vector embeddings using multiple models:
  - Sentence Transformers (local, GPU-accelerated)
  - OpenAI embeddings
  - Hugging Face models
- Persistent vector database (ChromaDB)
- Semantic search with metadata filtering
- LLM-powered answer generation
- Source attribution and citations

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd RAG-AI-documentation

# Install core dependencies
pip install -r construction_rag_requirements.txt
```

### Optional Dependencies

```bash
# For IFC file support (BIM data)
pip install ifcopenshell

# For OpenAI embeddings and LLM
pip install openai

# For Anthropic Claude
pip install anthropic

# For GPU acceleration (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Initialize the RAG System

```python
from construction_rag import ConstructionRAG

rag = ConstructionRAG(
    vector_db_path="./construction_vector_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200,
    use_gpu=False,  # Set to True if you have GPU
    llm_api_key="your-openai-api-key",  # Optional
    llm_model="gpt-4"
)
```

### 2. Ingest Documents

```python
# Ingest a single PDF specification
result = rag.ingest_document(
    "specifications/concrete_specs.pdf",
    document_type="specification"
)

# Ingest a PDF drawing
result = rag.ingest_document(
    "drawings/floor_plan.pdf",
    document_type="drawing"
)

# Ingest an IFC model
result = rag.ingest_document(
    "models/building_model.ifc",
    metadata={"project": "Hospital Building", "phase": "Design"}
)

# Ingest an entire directory
results = rag.ingest_directory(
    "project_documents/",
    recursive=True,
    file_extensions=['.pdf', '.ifc']
)
```

### 3. Query the System

```python
# Ask a question
response = rag.query(
    "What are the concrete strength requirements for the foundation?",
    top_k=5,
    return_sources=True
)

print(response['answer'])
print(f"Sources: {len(response['sources'])}")

# Filter by document type
response = rag.query(
    "List all columns on Level 3",
    filter_metadata={"document_type": "ifc_model"}
)
```

### 4. Semantic Search

```python
# Search without LLM generation
results = rag.search(
    "fire resistance requirements",
    top_k=10
)

for result in results:
    print(f"{result['metadata']['file_name']}: {result['text'][:200]}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Construction RAG System                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   PDF Docs   │  │   Drawings   │  │  IFC Models  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │          │
│         └──────────────────┼──────────────────┘          │
│                           │                              │
│         ┌─────────────────▼─────────────────┐           │
│         │    Document Processor              │           │
│         │  - PyMuPDF for PDFs                │           │
│         │  - IFC Parser for BIM              │           │
│         │  - Smart Chunking                  │           │
│         └─────────────────┬─────────────────┘           │
│                           │                              │
│         ┌─────────────────▼─────────────────┐           │
│         │    Embedding Handler               │           │
│         │  - Sentence Transformers           │           │
│         │  - OpenAI / Custom Models          │           │
│         └─────────────────┬─────────────────┘           │
│                           │                              │
│         ┌─────────────────▼─────────────────┐           │
│         │    Vector Store (ChromaDB)         │           │
│         │  - Persistent Storage              │           │
│         │  - Metadata Filtering              │           │
│         │  - Similarity Search               │           │
│         └─────────────────┬─────────────────┘           │
│                           │                              │
│         ┌─────────────────▼─────────────────┐           │
│         │    Query Engine                    │           │
│         │  - Context Assembly                │           │
│         │  - LLM Integration                 │           │
│         │  - Source Attribution              │           │
│         └────────────────────────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### Document Processor
- Converts PDFs to structured markdown
- Preserves headers, tables, images
- Intelligent chunking strategies:
  - Split on headers (maintains context)
  - Split on paragraphs (natural boundaries)
  - Split on sentences (last resort)
- Configurable chunk size and overlap

### IFC Parser
- Extracts building elements and properties
- Parses material information
- Captures spatial structure
- Retrieves quantities and measurements
- Fallback mode for when `ifcopenshell` is not available

### Vector Store
- Uses ChromaDB for persistent storage
- Supports metadata filtering
- Batch operations for efficiency
- Automatic deduplication
- Fallback mode with in-memory storage

### Embedding Handler
- Multiple model support:
  - Sentence Transformers (recommended for local use)
  - OpenAI embeddings (best quality, requires API)
  - Any Hugging Face model
- GPU acceleration support
- Batch processing
- Configurable embedding dimensions

### Query Engine
- Semantic search using vector similarity
- Context assembly from relevant chunks
- LLM-powered answer generation
- Construction-specific prompting
- Source citation and attribution
- Fallback mode without LLM

## Configuration Options

### Embedding Models

```python
# Local model (no API key needed, GPU-accelerated)
embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384 dim
embedding_model="sentence-transformers/all-mpnet-base-v2"  # Better quality, 768 dim

# OpenAI (requires API key)
embedding_model="text-embedding-3-small"  # 1536 dim
embedding_model="text-embedding-3-large"  # 3072 dim, best quality

# Any Hugging Face model
embedding_model="BAAI/bge-small-en-v1.5"
```

### Chunking Strategies

```python
# Small chunks - better precision, more chunks
ConstructionRAG(chunk_size=500, chunk_overlap=100)

# Medium chunks - balanced (recommended)
ConstructionRAG(chunk_size=1000, chunk_overlap=200)

# Large chunks - better context, fewer chunks
ConstructionRAG(chunk_size=2000, chunk_overlap=400)
```

### LLM Integration

```python
# OpenAI
ConstructionRAG(
    llm_api_key="sk-...",
    llm_model="gpt-4",
    llm_provider="openai"
)

# Anthropic Claude
ConstructionRAG(
    llm_api_key="sk-ant-...",
    llm_model="claude-3-sonnet-20240229",
    llm_provider="anthropic"
)

# Without LLM (semantic search only)
ConstructionRAG(llm_api_key=None)
```

## Use Cases

### 1. Specification Review
```python
response = rag.query(
    "What are the reinforcement requirements for slabs?",
    filter_metadata={"document_type": "specification"}
)
```

### 2. BIM Model Queries
```python
response = rag.query(
    "List all structural columns and their dimensions",
    filter_metadata={"file_type": ".ifc"}
)
```

### 3. Drawing Analysis
```python
response = rag.query(
    "What is shown in the foundation detail?",
    filter_metadata={"document_type": "drawing"}
)
```

### 4. Cross-Document Search
```python
response = rag.query(
    "Are the concrete strengths in the spec consistent with the structural drawings?"
)
```

### 5. Material Takeoffs
```python
response = rag.query(
    "What materials are specified for external walls?",
    top_k=10
)
```

## Performance Tips

### For Large Projects

1. **Use GPU acceleration**:
   ```python
   rag = ConstructionRAG(use_gpu=True)
   ```

2. **Process documents in batches**:
   ```python
   results = rag.ingest_directory("documents/", recursive=True)
   ```

3. **Save and load state**:
   ```python
   # After ingestion
   rag.save_state()

   # Later sessions
   rag.load_state()
   ```

4. **Use appropriate chunk sizes**:
   - Smaller chunks for precise answers
   - Larger chunks for context-heavy questions

### Memory Management

The system uses streaming and batch processing to handle large files efficiently:
- PDFs are processed page by page
- IFC files are processed by element type
- Embeddings are generated in batches
- Vector storage is persistent (not in RAM)

## Troubleshooting

### Issue: "ifcopenshell not installed"
**Solution**: Install IFC support:
```bash
pip install ifcopenshell
```

### Issue: "chromadb not installed"
**Solution**: Install vector database:
```bash
pip install chromadb
```

### Issue: Out of memory with large PDFs
**Solution**: Reduce chunk size or process pages individually

### Issue: Slow embedding generation
**Solution**:
- Enable GPU acceleration (`use_gpu=True`)
- Use a smaller embedding model
- Process in smaller batches

### Issue: Poor answer quality
**Solution**:
- Increase `top_k` to retrieve more context
- Use a better LLM model (e.g., GPT-4 vs GPT-3.5)
- Adjust chunk size for better context
- Add more specific metadata filters

## API Reference

See individual module documentation:
- `construction_rag/core.py` - Main RAG system
- `construction_rag/document_processor.py` - PDF processing
- `construction_rag/ifc_parser.py` - IFC file handling
- `construction_rag/vector_store.py` - Vector database
- `construction_rag/embeddings.py` - Embedding generation
- `construction_rag/query_engine.py` - Query processing

## Examples

See `examples/construction_rag_example.py` for complete usage examples.

## License

This project extends the PyMuPDF RAG system and is subject to the same license terms (GNU AGPL 3.0 or Commercial License).

## Support

For issues and questions:
1. Check this documentation
2. Review example code
3. Check the troubleshooting section
4. Open an issue on GitHub

## Contributing

Contributions are welcome! Areas for improvement:
- Additional document formats (DWG, DXF, etc.)
- Enhanced construction-specific processing
- Additional LLM integrations
- Performance optimizations
- UI/web interface

## Roadmap

- [ ] DWG/DXF support for CAD drawings
- [ ] Enhanced construction terminology understanding
- [ ] Multi-language support
- [ ] Web-based interface
- [ ] API server deployment
- [ ] Advanced visualization of results
- [ ] Integration with construction management software
