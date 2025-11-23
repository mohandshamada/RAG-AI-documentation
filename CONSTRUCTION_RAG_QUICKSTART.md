# Construction RAG System - Quick Start Guide

Get started with the Construction RAG system in 5 minutes!

## 1. Install Dependencies

```bash
# Install core dependencies
pip install -r construction_rag_requirements.txt

# Optional: Install IFC support for BIM files
pip install ifcopenshell

# Optional: Install OpenAI for better embeddings and answers
pip install openai
```

## 2. Prepare Your Documents

Organize your construction documents in a folder:

```
project_documents/
├── specifications/
│   ├── concrete_specs.pdf
│   ├── steel_specs.pdf
│   └── general_conditions.pdf
├── drawings/
│   ├── foundation_plan.pdf
│   ├── floor_plans.pdf
│   └── sections.pdf
└── models/
    └── building_model.ifc
```

## 3. Ingest Documents (CLI)

```bash
# Ingest a single file
python construction_rag_cli.py ingest specifications/concrete_specs.pdf

# Ingest an entire directory
python construction_rag_cli.py ingest project_documents/ --recursive

# View system info
python construction_rag_cli.py info
```

## 4. Query the System (CLI)

```bash
# Ask a question
python construction_rag_cli.py query "What are the concrete strength requirements?"

# Interactive mode
python construction_rag_cli.py interactive
```

## 5. Use in Python Code

```python
from construction_rag import ConstructionRAG

# Initialize
rag = ConstructionRAG(
    vector_db_path="./construction_vector_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Ingest documents
rag.ingest_directory("project_documents/")

# Query
response = rag.query("What are the fire resistance requirements?")
print(response['answer'])

# View sources
for source in response['sources']:
    print(f"- {source['file_name']} (Page {source['page']})")
```

## 6. Advanced Usage

### Use OpenAI for Better Results

```python
import os

rag = ConstructionRAG(
    vector_db_path="./construction_vector_db",
    embedding_model="text-embedding-3-small",  # OpenAI embeddings
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    llm_model="gpt-4"
)
```

### Filter by Document Type

```python
# Query only IFC models
response = rag.query(
    "List all columns on Level 2",
    filter_metadata={"document_type": "ifc_model"}
)

# Query only specifications
response = rag.query(
    "What are the quality control procedures?",
    filter_metadata={"document_type": "specification"}
)
```

### GPU Acceleration

```python
rag = ConstructionRAG(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True  # Requires NVIDIA GPU and CUDA
)
```

## Common Use Cases

### 1. Specification Review
```python
response = rag.query("What are the reinforcement requirements for slabs?")
```

### 2. BIM Model Queries
```python
response = rag.query("What materials are used in external walls?")
```

### 3. Cross-Document Analysis
```python
response = rag.query(
    "Compare the concrete strengths in specs vs. structural drawings"
)
```

### 4. Material Takeoffs
```python
response = rag.query("List all steel beam sizes and quantities")
```

## Troubleshooting

### "No module named 'construction_rag'"
Make sure you're running from the repository root directory.

### "ifcopenshell not installed"
```bash
pip install ifcopenshell
```

### "chromadb not installed"
```bash
pip install chromadb
```

### Slow Performance
- Enable GPU: `use_gpu=True`
- Use smaller embedding model
- Reduce `top_k` parameter

## Next Steps

1. Read the full documentation: `CONSTRUCTION_RAG_README.md`
2. Check examples: `examples/construction_rag_example.py`
3. Customize chunk size and overlap for your documents
4. Set up API keys for OpenAI/Anthropic for better results

## Support

For issues:
1. Check the documentation
2. Review troubleshooting section
3. Open an issue on GitHub

Enjoy using the Construction RAG system!
