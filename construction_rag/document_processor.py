"""
Document Processor for Construction Documents
==============================================

Processes PDF specifications and drawings using PyMuPDF.
Implements smart chunking strategies for large files.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path to import pymupdf4llm
sys.path.insert(0, str(Path(__file__).parent.parent / "pymupdf4llm"))

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes construction documents (PDFs) with intelligent chunking.

    Features:
    - Markdown conversion using PyMuPDF
    - Preserves document structure (headers, tables, images)
    - Smart chunking for large documents
    - Metadata extraction
    - Construction-specific optimizations
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Try to import pymupdf4llm
        try:
            import pymupdf4llm
            self.pymupdf4llm = pymupdf4llm
            self.pymupdf_available = True
        except ImportError:
            logger.warning("pymupdf4llm not found in expected location, trying standard import")
            try:
                import pymupdf4llm
                self.pymupdf4llm = pymupdf4llm
                self.pymupdf_available = True
            except ImportError:
                logger.error("pymupdf4llm not available")
                self.pymupdf_available = False

    def process_pdf(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a PDF file and return chunks.

        Args:
            file_path: Path to PDF file
            metadata: Additional metadata

        Returns:
            List of document chunks with text and metadata
        """
        if not self.pymupdf_available:
            return self._process_pdf_fallback(file_path, metadata)

        logger.info(f"Processing PDF: {file_path}")

        try:
            # Convert PDF to markdown with page chunks
            md_result = self.pymupdf4llm.to_markdown(
                file_path,
                page_chunks=True,
                write_images=False,
                embed_images=False,
                show_progress=False
            )

            chunks = []

            # Process each page
            for page_data in md_result:
                page_num = page_data.get('metadata', {}).get('page', 0)
                page_text = page_data.get('text', '')

                # Check if page is large enough to need chunking
                if len(page_text) <= self.chunk_size:
                    # Single chunk for this page
                    chunks.append({
                        "text": page_text,
                        "metadata": {
                            **(metadata or {}),
                            "page": page_num,
                            "chunk_type": "page",
                            "has_tables": bool(page_data.get('tables')),
                            "has_images": bool(page_data.get('images')),
                            "toc_items": page_data.get('toc_items', [])
                        }
                    })
                else:
                    # Split page into multiple chunks
                    page_chunks = self._chunk_text(
                        page_text,
                        self.chunk_size,
                        self.chunk_overlap
                    )

                    for i, chunk_text in enumerate(page_chunks):
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                **(metadata or {}),
                                "page": page_num,
                                "chunk_type": "page_section",
                                "section_index": i,
                                "has_tables": bool(page_data.get('tables')),
                                "has_images": bool(page_data.get('images')),
                                "toc_items": page_data.get('toc_items', [])
                            }
                        })

            logger.info(f"Processed PDF into {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.error(f"Error processing PDF with pymupdf4llm: {str(e)}")
            return self._process_pdf_fallback(file_path, metadata)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks while preserving structure.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []

        # Try to split on headers first (markdown headers)
        if '\n#' in text:
            sections = self._split_on_headers(text)

            current_chunk = ""
            for section in sections:
                if len(current_chunk) + len(section) <= chunk_size:
                    current_chunk += section
                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                    # If section itself is too large, split it
                    if len(section) > chunk_size:
                        section_chunks = self._split_on_paragraphs(
                            section,
                            chunk_size,
                            chunk_overlap
                        )
                        chunks.extend(section_chunks[:-1])
                        current_chunk = section_chunks[-1] if section_chunks else ""
                    else:
                        current_chunk = section

            if current_chunk:
                chunks.append(current_chunk)
        else:
            # Split on paragraphs
            chunks = self._split_on_paragraphs(text, chunk_size, chunk_overlap)

        return chunks

    def _split_on_headers(self, text: str) -> List[str]:
        """Split text on markdown headers while keeping header with content."""
        import re

        # Find all headers
        header_pattern = r'\n(#{1,6}\s+.+?\n)'
        parts = re.split(header_pattern, text)

        sections = []
        current_section = parts[0] if parts else ""

        for i in range(1, len(parts), 2):
            if i < len(parts):
                header = parts[i]
                content = parts[i + 1] if i + 1 < len(parts) else ""
                sections.append(current_section)
                current_section = header + content

        if current_section:
            sections.append(current_section)

        return [s for s in sections if s.strip()]

    def _split_on_paragraphs(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text on paragraph boundaries."""
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(para) > chunk_size:
                    # Split long paragraph on sentences
                    sentences = self._split_on_sentences(para, chunk_size, chunk_overlap)
                    chunks.extend(sentences[:-1])
                    current_chunk = sentences[-1] + "\n\n" if sentences else para + "\n\n"
                else:
                    # Add overlap from previous chunk if possible
                    if chunks and chunk_overlap > 0:
                        overlap_text = chunks[-1][-chunk_overlap:]
                        current_chunk = overlap_text + "\n\n" + para + "\n\n"
                    else:
                        current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_on_sentences(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text on sentence boundaries."""
        import re

        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(sentence) > chunk_size:
                    # Split on character limit as last resort
                    for i in range(0, len(sentence), chunk_size - chunk_overlap):
                        chunks.append(sentence[i:i + chunk_size])
                    current_chunk = ""
                else:
                    if chunks and chunk_overlap > 0:
                        overlap_text = chunks[-1][-chunk_overlap:]
                        current_chunk = overlap_text + " " + sentence + " "
                    else:
                        current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _process_pdf_fallback(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fallback PDF processing using PyMuPDF directly.
        """
        logger.info("Using fallback PDF processor")

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            chunks = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if text.strip():
                    # Chunk if needed
                    if len(text) <= self.chunk_size:
                        chunks.append({
                            "text": text,
                            "metadata": {
                                **(metadata or {}),
                                "page": page_num + 1,
                                "chunk_type": "page",
                                "processing_method": "fallback"
                            }
                        })
                    else:
                        page_chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)
                        for i, chunk_text in enumerate(page_chunks):
                            chunks.append({
                                "text": chunk_text,
                                "metadata": {
                                    **(metadata or {}),
                                    "page": page_num + 1,
                                    "chunk_type": "page_section",
                                    "section_index": i,
                                    "processing_method": "fallback"
                                }
                            })

            doc.close()

            logger.info(f"Processed PDF (fallback) into {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.error(f"Fallback PDF processing failed: {str(e)}")

            # Last resort: return placeholder
            return [{
                "text": f"PDF file: {Path(file_path).name}\n\nCould not process PDF content.",
                "metadata": {
                    **(metadata or {}),
                    "chunk_type": "error",
                    "error": str(e)
                }
            }]
