"""
Utility functions for Construction RAG system
==============================================

Helper functions for common operations.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    path = Path(file_path)
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def find_construction_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[Path]:
    """
    Find construction-related files in a directory.

    Args:
        directory: Directory to search
        extensions: File extensions to look for (default: ['.pdf', '.ifc'])
        recursive: Whether to search subdirectories

    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.pdf', '.ifc']

    directory = Path(directory)
    files = []

    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        files.extend(directory.glob(f"{pattern}{ext}"))

    return sorted(files)


def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text.

    Uses rough approximation: 1 token ≈ 4 characters

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB", "250 KB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import re

    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')

    return filename


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def detect_language(text: str) -> str:
    """
    Detect the language of text (simplified).

    Args:
        text: Input text

    Returns:
        Language code (e.g., 'en', 'es', 'unknown')
    """
    # Simple heuristic - can be improved with langdetect library
    text_lower = text.lower()

    # Check for common English words
    english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a']
    english_count = sum(1 for word in english_words if word in text_lower)

    if english_count >= 3:
        return 'en'

    return 'unknown'


class ProgressTracker:
    """
    Simple progress tracker for batch operations.
    """

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description

        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=total, desc=description)
            self.use_tqdm = True
        except ImportError:
            self.use_tqdm = False

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items processed
        """
        self.current += n

        if self.use_tqdm:
            self.pbar.update(n)
        else:
            percent = (self.current / self.total) * 100
            print(f"{self.description}: {self.current}/{self.total} ({percent:.1f}%)")

    def close(self):
        """Close progress tracker."""
        if self.use_tqdm:
            self.pbar.close()


def validate_api_key(api_key: str, provider: str = "openai") -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate
        provider: API provider (openai, anthropic, etc.)

    Returns:
        True if format is valid
    """
    if not api_key:
        return False

    if provider == "openai":
        return api_key.startswith("sk-")
    elif provider == "anthropic":
        return api_key.startswith("sk-ant-")

    return True


def merge_metadata(*metadata_dicts) -> dict:
    """
    Merge multiple metadata dictionaries.

    Args:
        *metadata_dicts: Metadata dictionaries to merge

    Returns:
        Merged metadata dictionary
    """
    result = {}

    for metadata in metadata_dicts:
        if metadata:
            result.update(metadata)

    return result
